"""Perfiles de hardware y calibración (pedido del usuario: entrenar la base en una GTX 1060 en casa
y continuar en la Arc A770).

* ``PROFILES``: datos de cada máquina (memoria, ancho de banda, cómputo efectivo supuesto,
  bf16 en hardware, si ``torch.compile`` sirve, precisión recomendada).
* ``calibrate(device)``: mide la máquina real (matmul fp32 y bf16, memoria, núcleos, RAM) y
  devuelve un perfil medido. El cómputo efectivo medido reemplaza al supuesto.
* ``recommend(perfil, meses)``: para cada preset, memoria de entrenamiento, si entra, texto que
  alcanza a ver en ese tiempo y velocidad de generación estimada por ancho de banda.

Uso::

    python -m hope.bench.hardware --perfil gtx1060      # tabla con los valores supuestos
    python -m hope.bench.hardware --calibrar            # mide esta máquina y recomienda
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace

import torch

from hope.bench.plan import budget, cost, meta_model
from hope.train.lowprec import BYTES_PER_PARAM


@dataclass
class Profile:
    name: str
    vram_gb: float
    bandwidth_gbs: float
    tflops_eff: float                 # cómputo efectivo de entrenamiento (supuesto hasta calibrar)
    bf16: bool
    compile_ok: bool
    precision: str
    ram_gb: float = 16.0
    cpu_cores: int = 6
    measured: bool = False


PROFILES = {
    # Pascal: sin bf16, fp16 a 1/64; ~4,4 TFLOPS fp32 de pico. Triton (torch.compile) no soporta Pascal.
    "gtx1060": Profile("GTX 1060 6 GB + i5-9400F + 16 GB", 6, 192, 1.5, False, False, "fp32-8bit", 16, 6),
    "gtx1060-3gb": Profile("GTX 1060 3 GB", 3, 192, 1.3, False, False, "fp32-8bit", 16, 6),
    "a770": Profile("Intel Arc A770 16 GB", 16, 560, 12.0, True, True, "bf16-8bit", 64, 8),
    "rtx4090": Profile("RTX 4090 24 GB", 24, 1008, 80.0, True, True, "bf16-8bit", 64, 16),
    "rtx5090": Profile("RTX 5090 32 GB", 32, 1792, 110.0, True, True, "bf16-8bit", 64, 16),
    "cpu": Profile("solo CPU", 0, 40, 0.1, False, False, "fp32-8bit", 16, 4),
}

PRESETS = ("hn1-cpu", "hn1-25m", "hn1-200m", "hn1-500m")


def _ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def match_profile(device_name: str, vram_gb: float = 0.0) -> Profile:
    """Perfil supuesto para el nombre que informa el controlador («NVIDIA GeForce GTX 1060 6GB»,
    «Intel(R) Arc(TM) A770 Graphics»). Se busca el número de modelo; si no se conoce, el de CPU."""
    words = device_name.upper().replace("(TM)", " ").replace("(R)", " ").split()
    if "1060" in words:
        return PROFILES["gtx1060-3gb" if 0 < vram_gb < 4.5 or "3GB" in words else "gtx1060"]
    for key, model in (("a770", "A770"), ("rtx4090", "4090"), ("rtx5090", "5090")):
        if model in words:
            return PROFILES[key]
    return PROFILES["cpu"]


def calibrate(device: str | None = None, n: int = 2048) -> Profile:
    """Mide esta máquina. Devuelve un perfil con el cómputo efectivo medido (≈ 40 % del matmul,
    que es lo que suele rendir un paso de entrenamiento completo frente a un matmul ideal)."""
    from hope.bench.env_check import bench_matmul
    from hope.device import select_device, supports_bf16
    dev = select_device(device)
    bf16 = supports_bf16(dev)
    mm = bench_matmul(dev, torch.bfloat16 if bf16 else torch.float32, n)
    vram = 0.0
    name = "CPU"
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        vram, name = props.total_memory / 1e9, props.name
        arch = f"sm_{props.major}{props.minor}"
        if arch not in torch.cuda.get_arch_list():
            raise RuntimeError(f"esta instalación de PyTorch no incluye {arch}: para una GTX 10xx instalá la "
                               "variante CUDA 12.6 (pip install torch --index-url https://download.pytorch.org/whl/cu126)")
    elif dev.type == "xpu":
        vram, name = torch.xpu.get_device_properties(dev).total_memory / 1e9, torch.xpu.get_device_name(dev)
    base = match_profile(name, vram)
    return replace(base, name=f"{name} (medido)", vram_gb=round(vram, 1) or base.vram_gb,
                   tflops_eff=round(0.4 * mm["tflops"], 2), bf16=bf16,
                   precision="bf16-8bit" if bf16 else "fp32-8bit", ram_gb=round(_ram_gb(), 1) or base.ram_gb,
                   cpu_cores=os.cpu_count() or base.cpu_cores, measured=True)


def recommend(p: Profile, months: float = 6.0, presets=PRESETS) -> list[dict]:
    rows = []
    for name in presets:
        c = cost(meta_model(name))
        mem = c.total_params * BYTES_PER_PARAM[p.precision] / 1e9
        # Con el optimizador descargado a la RAM, en la GPU quedan solo pesos y gradientes.
        w_and_g = 8.0 if p.precision.startswith("fp32") else 4.0
        mem_offload = c.total_params * w_and_g / 1e9
        state_ram = (mem - mem_offload)
        act = 1.5                                       # reserva típica de activaciones con checkpointing (GB)
        b = budget(c, months, p.tflops_eff)
        # Generación limitada por ancho de banda: pesos activos leídos por parche (~4,5 bytes).
        w_bytes = 4 if p.precision.startswith("fp32") else 2
        active_patch = (c.active_text_params * 4.5) * w_bytes
        gen_bytes_s = 0.5 * p.bandwidth_gbs * 1e9 / active_patch * 4.5 if p.bandwidth_gbs else 0.0
        rows.append({"preset": name, "parametros_M": round(c.total_params / 1e6, 1),
                     "memoria_entrenamiento_GB": round(mem + act, 2), "entra": mem + act <= 0.9 * p.vram_gb,
                     "entra_con_offload": mem_offload + act <= 0.9 * p.vram_gb and state_ram <= 0.5 * p.ram_gb,
                     "texto_GB_en_meses": round(b["gb_texto"]), "tokens_por_parametro": round(b["tokens_por_parametro"]),
                     "generacion_tokens_s_aprox": round(gen_bytes_s / 4)})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perfil", default="gtx1060", choices=list(PROFILES))
    ap.add_argument("--calibrar", action="store_true")
    ap.add_argument("--meses", type=float, default=6.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    p = calibrate(args.device) if args.calibrar else PROFILES[args.perfil]
    print(json.dumps({"perfil": asdict(p), "recomendacion": recommend(p, args.meses)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
