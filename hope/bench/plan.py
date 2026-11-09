"""Plan de entrenamiento por cómputo (ARQUITECTURA_HN1 §11.1).

Estima, para un modelo HN-1 y una GPU, cuánto texto y cuántas imágenes se pueden
entrenar en un tiempo dado. Regla usual: entrenar cuesta ≈ 6 FLOPs por parámetro
**activo** y por elemento procesado (2 hacia adelante + 4 hacia atrás).

* Parte local (bytes): codificador, decodificador y cabezas → por byte.
* Parte global (parches): backbone, memoria, núcleo × iteraciones medias y zonas
  (solo los especialistas elegidos: top-k de n) → por parche, ≈ 1 cada 4,5 bytes.
* Zona visual: codificador × parches de píxeles → por imagen; la atención cruzada
  se cuenta por parche de texto.

``effective_tflops`` debe salir de ``python -m hope.bench.env_check`` en la A770
(campo ``effective_tflops``). El valor por defecto (12) es una suposición.

Uso::

    python -m hope.bench.plan --months 6 --tflops 12
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from hope.core.hn1 import HN1Model, preset

SECONDS_PER_MONTH = 30.44 * 24 * 3600


def _n(module) -> int:
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


@dataclass
class Cost:
    total_params: int
    active_text_params: float        # por byte equivalente
    flops_per_byte: float
    flops_per_image: float
    tokens_per_image: int


def cost(model: HN1Model, bytes_per_patch: float = 4.5, mean_iters: float | None = None) -> Cost:
    c = model.cfg
    local = _n(model.encoder) + _n(model.decoder) + _n(model.head) + _n(model.uncertainty) + _n(model.to_local)
    it = mean_iters if mean_iters is not None else (1 + c.max_iters) / 2
    regions = 0.0
    for z in model.regions:
        experts = list(z.experts)
        k = min(z.top_k, len(experts))
        regions += _n(z) - sum(_n(e) for e in experts) + k * sum(_n(e) for e in experts) / len(experts)
    xattn = sum(_n(x) for x in model.vision.xattn.values()) if model.vision is not None else 0
    glob = _n(model.to_global) + _n(model.backbone) + _n(model.memory) + it * _n(model.core) + regions
    active = local + glob / bytes_per_patch
    vis_flops, tpi = 0.0, 0
    if model.vision is not None:
        enc = model.vision.encoder
        grid = (enc.image // enc.patch) ** 2
        vis_flops = 6.0 * (_n(enc) - _n(enc.proj) - _n(enc.norm)) * grid + 6.0 * _n(enc.proj) * enc.tokens_per_image()
        tpi = enc.tokens_per_image()
    return Cost(_n(model), active, 6.0 * active + 6.0 * xattn / bytes_per_patch, vis_flops, tpi)


def meta_model(name: str) -> HN1Model:
    with torch.device("meta"):
        return HN1Model(preset(name))


def budget(c: Cost, months: float, tflops: float, image_share: float = 0.0,
           caption_bytes: int = 300) -> dict:
    """``image_share``: fracción del cómputo dedicada a pares imagen–texto."""
    flops = months * SECONDS_PER_MONTH * tflops * 1e12
    text_bytes = flops * (1 - image_share) / c.flops_per_byte
    per_pair = c.flops_per_image + caption_bytes * c.flops_per_byte
    images = flops * image_share / per_pair if c.flops_per_image and image_share else 0.0
    tokens_eq = (text_bytes + images * caption_bytes) / 4.0       # ≈ 4 bytes por token de un BPE
    return {"meses": months, "tflops_efectivos": tflops, "bytes_texto": text_bytes,
            "gb_texto": text_bytes / 1e9, "imagenes": images, "tokens_equivalentes": tokens_eq,
            "tokens_por_parametro": tokens_eq / c.total_params}


def six_month_plan(tflops: float = 12.0) -> list[dict]:
    """Plan propuesto (§11.1): crecer de 200M a ~460M y agregar visión por etapas."""
    small, big = cost(meta_model("hn1-200m")), cost(meta_model("hn1-500m"))
    phases = [
        ("1. texto y código, hn1-200m", small, 2.0, 0.0),
        ("2. crecer a hn1-500m (texto); visión con imágenes procedurales", big, 1.0, 0.25),
        ("3. multimodal con descripciones de Qwen + ciclo maestro–alumno", big, 3.0, 0.30),
    ]
    out = []
    for name, c, months, share in phases:
        b = budget(c, months, tflops, share)
        out.append({"etapa": name, "parametros": c.total_params, "flops_por_byte": c.flops_per_byte,
                    "flops_por_imagen": c.flops_per_image, **b})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--months", type=float, default=6.0)
    ap.add_argument("--tflops", type=float, default=12.0, help="TFLOPS efectivos medidos con env_check")
    args = ap.parse_args(argv)
    small, big = cost(meta_model("hn1-200m")), cost(meta_model("hn1-500m"))
    print(json.dumps({
        "hn1-200m": {**small.__dict__, **budget(small, args.months, args.tflops)},
        "hn1-500m": {**big.__dict__, **budget(big, args.months, args.tflops, 0.3)},
        "frena_500m_vs_200m": big.flops_per_byte / small.flops_per_byte,
        "plan_6_meses": six_month_plan(args.tflops)}, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
