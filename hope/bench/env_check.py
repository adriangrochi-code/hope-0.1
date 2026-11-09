"""Prueba del entorno de entrenamiento (fase F0).

Mide en el dispositivo real:
  1. Rendimiento de multiplicación de matrices (bf16 y fp32) en TFLOPS.
  2. Atención con ``scaled_dot_product_attention``.
  3. Un paso completo de entrenamiento (forward + backward + AdamW) de un bloque
     de tamaño similar al backbone de HN-1.
  4. Memoria máxima usada, cuando el backend la informa.

Uso:
    python -m hope.bench.env_check                # mejor dispositivo disponible
    python -m hope.bench.env_check --device cpu
    python -m hope.bench.env_check --json resultado.json

Las cifras sirven para recalcular las estimaciones de costo de la §3.9 de
docs/ARQUITECTURA_HN1.md con datos medidos.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from hope.device import select_device, synchronize


def _timeit(fn, device, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - t0) / iters


def bench_matmul(device, dtype, n: int) -> dict:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    sec = _timeit(lambda: a @ b, device)
    return {"n": n, "dtype": str(dtype), "seconds": sec, "tflops": 2 * n**3 / sec / 1e12}


def bench_sdpa(device, dtype, batch: int, heads: int, seq: int, head_dim: int) -> dict:
    q, k, v = (torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype) for _ in range(3))
    sec = _timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), device)
    flops = 4 * batch * heads * seq * seq * head_dim / 2  # causal ≈ la mitad
    return {"seq": seq, "dtype": str(dtype), "seconds": sec, "tflops": flops / sec / 1e12}


class _Block(nn.Module):
    """Bloque de referencia: atención + SwiGLU, d=1024 (~10M parámetros)."""

    def __init__(self, d: int = 1024, heads: int = 16, hidden: int = 2048):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        g, u = self.w1(self.norm2(x)).chunk(2, dim=-1)
        return x + self.w2(F.silu(g) * u)


def bench_train_step(device, dtype, batch: int, seq: int) -> dict:
    block = _Block().to(device)
    opt = torch.optim.AdamW(block.parameters(), lr=1e-4)
    x = torch.randn(batch, seq, 1024, device=device)
    n_params = sum(p.numel() for p in block.parameters())

    def step():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            loss = block(x).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    sec = _timeit(step, device, warmup=2, iters=5)
    tokens = batch * seq
    # ≈ 6 FLOPs por parámetro y token (forward + backward), más la atención.
    flops = 6 * n_params * tokens + 3 * 4 * batch * seq * seq * 1024 / 2
    return {
        "params": n_params,
        "tokens_per_step": tokens,
        "seconds": sec,
        "tokens_per_second": tokens / sec,
        "effective_tflops": flops / sec / 1e12,
    }


def peak_memory_gb(device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "xpu" and hasattr(torch.xpu, "max_memory_allocated"):
        return torch.xpu.max_memory_allocated() / 1e9
    return None


def run(device_name: str | None = None, quick: bool = False) -> dict:
    device = select_device(device_name)
    big = device.type != "cpu" and not quick
    n = 4096 if big else 512
    seq = 2048 if big else 128
    batch = 8 if big else 2
    low = torch.bfloat16

    report: dict = {
        "device": str(device),
        "device_name": _device_name(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "matmul": [bench_matmul(device, torch.float32, n), bench_matmul(device, low, n)],
        "sdpa": bench_sdpa(device, low, batch, 16, seq, 64),
        "train_step": bench_train_step(device, low, batch, seq),
    }
    report["peak_memory_gb"] = peak_memory_gb(device)
    try:
        torch.compile  # noqa: B018 — solo comprobamos que existe
        report["torch_compile_available"] = True
    except AttributeError:
        report["torch_compile_available"] = False
    return report


def _device_name(device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "xpu":
        return torch.xpu.get_device_name(device)
    return platform.processor() or "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true", help="tamaños chicos (prueba de humo)")
    ap.add_argument("--json", default=None, help="guardar el reporte en este archivo")
    args = ap.parse_args()
    report = run(args.device, args.quick)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
