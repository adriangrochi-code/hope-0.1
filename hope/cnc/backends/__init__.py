"""Controladores de tiempo real a los que Hope CNC envía el programa ya revisado.

* ``sim`` — simulador (sin hardware): usa el planificador para avanzar en tiempo virtual.
* ``grbl`` — grbl / grblHAL / FluidNC por puerto serie (USB) o TCP (Ethernet/Wi-Fi), con el
  protocolo de conteo de caracteres para mantener lleno el búfer del controlador.
* ``linuxcnc`` — LinuxCNC por su API de Python (puerto paralelo en PCs viejas, Mesa en modernas).

QUÉ ES
    El punto de entrada del paquete: reexporta la interfaz común (``base``) y ofrece
    ``make_backend``, la fábrica que elige el controlador según la configuración.

CÓMO ENCAJA
    El perfil de máquina (``hope.cnc.profiles``) trae una sección ``controlador`` con ``tipo``,
    ``puerto``, ``baudios`` y ``bufer``; ``Machine.controller`` la guarda y ``hope.cnc.__main__`` llama a
    ``make_backend(m.controller, m)``. El backend que sale se le pasa al ``Controller``
    (``hope.cnc.controller``), que revisa el programa, pide permiso al guardián si la máquina es real
    y recién ahí llama a ``stream``.

    Los módulos de cada controlador se importan **solo cuando se piden**: así no hace falta tener
    pyserial (grbl) ni el módulo ``linuxcnc`` instalados para usar el simulador, y el arranque en
    PCs viejas es más rápido.

EJEMPLO
    >>> b = make_backend({"tipo": "grbl", "puerto": "COM3", "baudios": 115200, "bufer": 128})
    >>> b.connect(); print(b.banner); print(b.status().state)
"""

from __future__ import annotations

from hope.cnc.backends.base import Backend, BackendError, Status

# ==================================================================================================
# ============================================= FÁBRICA ============================================
# ==================================================================================================

def make_backend(cfg: dict, machine=None) -> Backend:
    """Crea el controlador que pide la configuración.

    Parámetros:
        cfg:     la sección ``controlador`` del perfil de máquina. Claves:
                 ``tipo`` («sim» por defecto, «grbl»/«grblhal»/«fluidnc», «linuxcnc»; sin distinguir
                 mayúsculas); para grbl ``puerto``, ``baudios``, ``bufer``, ``ejes``; para LinuxCNC
                 ``ini`` (opcional, solo informativo).
        machine: la ``Machine`` efectiva; solo la usa el simulador (para recorridos, velocidades y
                 orígenes). None = máquina genérica.

    Devuelve un ``Backend`` todavía sin conectar (llamar a ``connect()``), salvo grbl, que ya abre el
    puerto al crearse.

    Errores: ``BackendError`` si el tipo es desconocido, falta el puerto, falta pyserial o falta
    LinuxCNC.
    """
    kind = str(cfg.get("tipo", "sim")).lower()
    if kind == "sim":
        from hope.cnc.backends.sim import SimBackend
        return SimBackend(machine)
    if kind in ("grbl", "grblhal", "fluidnc"):                # los tres hablan el mismo protocolo de grbl 1.1
        from hope.cnc.backends.grbl import GrblBackend
        return GrblBackend.from_config(cfg)
    if kind == "linuxcnc":
        from hope.cnc.backends.linuxcnc import LinuxCNCBackend
        return LinuxCNCBackend(cfg.get("ini"))
    raise BackendError(f"controlador «{kind}» desconocido (sim, grbl, linuxcnc)")


__all__ = ["Backend", "BackendError", "Status", "make_backend"]
