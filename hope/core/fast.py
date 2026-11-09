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


def find_batch_size(try_step, start: int = 4, maximum: int = 256) -> int:
    """Duplica el lote mientras ``try_step(lote)`` no se quede sin memoria. ``try_step`` hace
    un paso completo (adelante + atrás) y lanza la excepción del backend si no entra."""
    best, b = 0, start
    while b <= maximum:
        try:
            try_step(b)
            best = b
            b *= 2
        except (RuntimeError, MemoryError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(e, MemoryError):
                raise
            break
        finally:
            for mod in (getattr(torch, "xpu", None), getattr(torch, "cuda", None)):
                if mod is not None and hasattr(mod, "empty_cache") and mod.is_available():
                    mod.empty_cache()
    return max(best, 1)
