"""Selección de dispositivo portable: Intel XPU (Arc), CUDA o CPU."""

from __future__ import annotations

import torch


def select_device(preferred: str | None = None) -> torch.device:
    """Devuelve el mejor dispositivo disponible.

    Orden: el preferido (si está disponible) → xpu → cuda → cpu.
    """
    if preferred:
        if preferred == "cpu" or _available(preferred):
            return torch.device(preferred)
        raise RuntimeError(f"Dispositivo '{preferred}' no disponible")
    for kind in ("xpu", "cuda"):
        if _available(kind):
            return torch.device(kind)
    return torch.device("cpu")


def _available(kind: str) -> bool:
    if kind == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if kind == "cuda":
        return torch.cuda.is_available()
    return kind == "cpu"


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
