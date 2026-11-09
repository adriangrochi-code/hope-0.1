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


def supports_bf16(device: torch.device) -> bool:
    """bf16 en hardware: Intel Arc (XMX) y NVIDIA desde Ampere (capacidad 8.0). Las Pascal (GTX 10xx)
    y Turing no lo tienen: ahí se entrena en fp32 (modo ``fp32-8bit``)."""
    if device.type == "xpu":
        return True
    if device.type == "cuda":
        return torch.cuda.get_device_capability(device)[0] >= 8
    return False


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
