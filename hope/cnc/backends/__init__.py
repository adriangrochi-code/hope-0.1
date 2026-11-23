"""Controladores de tiempo real a los que Hope CNC envía el programa ya revisado.

* ``sim`` — simulador (sin hardware): usa el planificador para avanzar en tiempo virtual.
* ``grbl`` — grbl / grblHAL / FluidNC por puerto serie (USB) o TCP (Ethernet/Wi-Fi), con el
  protocolo de conteo de caracteres para mantener lleno el búfer del controlador.
* ``linuxcnc`` — LinuxCNC por su API de Python (puerto paralelo en PCs viejas, Mesa en modernas).
"""

from __future__ import annotations

from hope.cnc.backends.base import Backend, BackendError, Status


def make_backend(cfg: dict, machine=None) -> Backend:
    kind = str(cfg.get("tipo", "sim")).lower()
    if kind == "sim":
        from hope.cnc.backends.sim import SimBackend
        return SimBackend(machine)
    if kind in ("grbl", "grblhal", "fluidnc"):
        from hope.cnc.backends.grbl import GrblBackend
        return GrblBackend.from_config(cfg)
    if kind == "linuxcnc":
        from hope.cnc.backends.linuxcnc import LinuxCNCBackend
        return LinuxCNCBackend(cfg.get("ini"))
    raise BackendError(f"controlador «{kind}» desconocido (sim, grbl, linuxcnc)")


__all__ = ["Backend", "BackendError", "Status", "make_backend"]
