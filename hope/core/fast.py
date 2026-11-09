"""Aprovechar al máximo la GPU sin cambiar el modelo (ARQUITECTURA_HN1 §3.9.1).

En la A770 (16 GB, ~560 GB/s) lo que limita depende de la tarea:

* **Generar** (lote 1) está limitado por el ancho de banda y por lanzar muchos kernels
  chicos. Ayudan:
  - ``compile_model``: ``torch.compile`` fusiona operaciones elementales (menos idas y
    vueltas a memoria) y, con ``mode="reduce-overhead"``, graba grafos para no pagar el
    lanzamiento de cada kernel;
  - ``quantize_int8``: pesos en int8 con escala por fila: la mitad de memoria que bf16
    (deja lugar a Qwen en la misma GPU). La ganancia de velocidad requiere un kernel que
    fusione la decuantización (IPEX weight-only o ``torch.compile``); sin eso, solo ahorra memoria;
  - la parte global incremental y la generación especulativa (``hope/core/speculative.py``),
    que leen los pesos una vez por varios bytes.
* **Entrenar** debe estar limitado por cómputo, no por memoria: lotes grandes.
  - ``find_batch_size`` busca el lote más grande que entra;
  - ``TrainConfig.checkpointing`` recalcula activaciones en el paso hacia atrás (≈ +30 % de
    cómputo, mucho menos memoria → lotes más grandes);
  - ``TrainConfig.grad_accum`` acumula gradientes de varios lotes chicos.
"""

from __future__ import annotations

import contextlib
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _blocks(model) -> list[nn.Module]:
    out = [*model.encoder, *model.decoder, *model.backbone, model.core.block]
    if getattr(model, "vision", None) is not None:
        out += list(model.vision.encoder.blocks)
    return out


def compile_model(model, mode: str | None = None) -> int:
    """Compila el ``forward`` de cada bloque (formas dinámicas). Devuelve cuántos se compilaron.
    Si ``torch.compile`` no está disponible en el backend, deja el modelo como estaba."""
    n = 0
    for blk in _blocks(model):
        if "_hope_eager_forward" in blk.__dict__:
            continue
        try:
            compiled = torch.compile(blk.forward, dynamic=True, mode=mode)
        except Exception as e:                       # noqa: BLE001
            warnings.warn(f"torch.compile no disponible: {e}")
            return n
        blk.__dict__["_hope_eager_forward"] = blk.forward
        blk.forward = compiled
        n += 1
    return n


def uncompile_model(model) -> None:
    for blk in _blocks(model):
        eager = blk.__dict__.pop("_hope_eager_forward", None)
        if eager is not None:
            blk.__dict__.pop("forward", None)


@contextlib.contextmanager
def eager(model):
    """Quita la compilación temporalmente (p. ej. para guardar la estructura con pickle)."""
    saved = {id(b): b.__dict__.get("forward") for b in _blocks(model) if "_hope_eager_forward" in b.__dict__}
    for b in _blocks(model):
        if id(b) in saved:
            b.__dict__.pop("forward", None)
            b.__dict__["_hope_saved"] = b.__dict__.pop("_hope_eager_forward")
    try:
        yield model
    finally:
        for b in _blocks(model):
            if id(b) in saved:
                b.__dict__["_hope_eager_forward"] = b.__dict__.pop("_hope_saved")
                b.forward = saved[id(b)]


class Int8Linear(nn.Module):
    """Capa lineal con pesos int8 y una escala por fila de salida (solo inferencia)."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        self.register_buffer("qweight", torch.round(w / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(lin.weight.dtype))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    @property
    def weight(self) -> torch.Tensor:                      # para código que lee .weight (dispositivo, tipo)
        return self.qweight.to(self.scale.dtype) * self.scale

    def forward(self, x):
        return F.linear(x, self.qweight.to(x.dtype) * self.scale.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))


def quantize_int8(model, prefixes=("encoder", "decoder", "backbone", "core", "regions", "memory", "vision")) -> dict:
    """Reemplaza las capas lineales de esas partes por ``Int8Linear``. Devuelve bytes antes/después."""
    before = after = 0
    targets = []
    for name, mod in model.named_modules():
        if not name.startswith(prefixes):
            continue
        for child_name, child in mod.named_children():
            if isinstance(child, nn.Linear):
                targets.append((mod, child_name, child))
    for parent, child_name, lin in targets:
        before += lin.weight.numel() * lin.weight.element_size()
        q = Int8Linear(lin)
        after += q.qweight.numel() + q.scale.numel() * q.scale.element_size()
        setattr(parent, child_name, q)
    return {"capas": len(targets), "bytes_antes": before, "bytes_despues": after}


class QuantLinear(nn.Module):
    """Capa lineal cuantizada por grupos (simétrica), para inferir: 8 bits (int8) o 4 bits empaquetados
    (dos pesos por byte). Con 4 bits y grupos de 64, los pesos ocupan ~0,56 bytes por parámetro.
    Coincide exactamente con ``hope.train.qat.fake_quant``: lo entrenado con QAT se exporta sin
    sorpresas."""

    def __init__(self, lin: nn.Linear, bits: int = 4, group: int = 64):
        super().__init__()
        if bits not in (4, 8):
            raise ValueError("bits: 4 u 8")
        w = lin.weight.detach().float()
        out_f, in_f = w.shape
        g = group if in_f % group == 0 else in_f
        qmax = 2 ** (bits - 1) - 1
        wg = w.reshape(out_f, in_f // g, g)
        scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
        q = torch.round(wg / scale).clamp(-qmax, qmax).to(torch.int8).reshape(out_f, in_f)
        if bits == 4:
            if in_f % 2:
                raise ValueError("4 bits necesita un número par de entradas")
            u = (q + 8).to(torch.uint8)                          # -7..7 → 1..15
            q = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
        self.register_buffer("qweight", q)
        # Escala en fp32 (4 bytes cada 64 pesos): en fp16 las de pesos chicos se redondean o se anulan
        # (pesos de ~1e-6 en 8 bits perdían el 99 %) y la exportación dejaba de coincidir con la QAT.
        self.register_buffer("scale", scale.squeeze(-1).float())
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        self.bits, self.group, self.in_features, self.out_features = bits, g, in_f, out_f

    def dequantized(self) -> torch.Tensor:
        q = self.qweight
        if self.bits == 4:
            lo = (q & 0x0F).to(torch.int8) - 8
            hi = (q >> 4).to(torch.int8) - 8
            q = torch.stack([lo, hi], dim=-1).reshape(self.out_features, self.in_features)
        w = q.float().reshape(self.out_features, -1, self.group) * self.scale.float().unsqueeze(-1)
        return w.reshape(self.out_features, self.in_features)

    @property
    def weight(self) -> torch.Tensor:
        return self.dequantized()

    def forward(self, x):
        return F.linear(x, self.dequantized().to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def quantize(model, bits: int = 4, group: int = 64,
             prefixes=("encoder", "decoder", "backbone", "core", "regions", "memory", "vision", "connections")) -> dict:
    """Exporta las capas lineales internas a ``QuantLinear``. Devuelve bytes de pesos antes y después."""
    targets = []
    for name, mod in model.named_modules():
        if not name.startswith(prefixes):
            continue
        for child_name, child in mod.named_children():
            if isinstance(child, nn.Linear) and min(child.weight.shape) > 1:
                targets.append((mod, child_name, child))
    before = after = 0
    for parent, child_name, lin in targets:
        before += lin.weight.numel() * lin.weight.element_size()
        q = QuantLinear(lin, bits, group)
        after += q.qweight.numel() + q.scale.numel() * q.scale.element_size()
        setattr(parent, child_name, q)
    return {"capas": len(targets), "bits": bits, "bytes_antes": before, "bytes_despues": after}


def find_batch_size(try_step, start: int = 4, maximum: int = 256) -> int:
    """Duplica el lote mientras ``try_step(lote)`` no se quede sin memoria. ``try_step`` hace
    un paso completo (adelante + atrás) y lanza la excepción del backend si no entra."""
    def fits(b: int) -> bool:
        try:
            try_step(b)
            return True
        except (RuntimeError, MemoryError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(e, MemoryError):
                raise
            return False
        finally:
            for mod in (getattr(torch, "xpu", None), getattr(torch, "cuda", None)):
                if mod is not None and hasattr(mod, "empty_cache") and mod.is_available():
                    mod.empty_cache()

    best, b = 0, max(start, 1)
    while b <= maximum and fits(b):
        best = b
        b *= 2
    if best == 0:                                  # ni el lote inicial entra: probar más chicos
        b = max(start, 1) // 2
        while b >= 1 and not fits(b):
            b //= 2
        best = b
    return max(best, 1)
