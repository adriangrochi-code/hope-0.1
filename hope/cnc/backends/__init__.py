"""Real-time controllers that Hope CNC hands the already-checked program to.

* ``sim`` — simulator (no hardware): uses the planner to advance in virtual time.
* ``grbl`` — grbl / grblHAL / FluidNC over a serial port (USB) or TCP (Ethernet/Wi-Fi), using the
  character-counting protocol to keep the controller's buffer full.
* ``linuxcnc`` — LinuxCNC through its Python API (parallel port on old PCs, Mesa on modern ones).

WHAT IT IS
    The package entry point: re-exports the common interface (``base``) and provides
    ``make_backend``, the factory that picks the controller according to the configuration.

HOW IT FITS
    The machine profile (``hope.cnc.profiles``) has a ``controlador`` section with ``tipo``,
    ``puerto``, ``baudios`` and ``bufer``; ``Machine.controller`` stores it and ``hope.cnc.__main__`` calls
    ``make_backend(m.controller, m)``. The resulting backend is passed to the ``Controller``
    (``hope.cnc.controller``), which checks the program, asks the guardian for permission if the machine
    is real, and only then calls ``stream``.

    Each controller's module is imported **only when requested**: that way neither pyserial (grbl)
    nor the ``linuxcnc`` module has to be installed to use the simulator, and startup on old PCs is
    faster.

EXAMPLE
    >>> b = make_backend({"tipo": "grbl", "puerto": "COM3", "baudios": 115200, "bufer": 128})
    >>> b.connect(); print(b.banner); print(b.status().state)
"""

from __future__ import annotations

from hope.cnc.backends.base import Backend, BackendError, Status

# ==================================================================================================
# ============================================= FACTORY ============================================
# ==================================================================================================

def make_backend(cfg: dict, machine=None) -> Backend:
    """Create the controller the configuration asks for.

    Parameters:
        cfg:     the ``controlador`` section of the machine profile. Keys:
                 ``tipo`` («sim» by default, «grbl»/«grblhal»/«fluidnc», «linuxcnc»; case
                 insensitive); for grbl ``puerto``, ``baudios``, ``bufer``, ``ejes``; for LinuxCNC
                 ``ini`` (optional, informational only).
        machine: the effective ``Machine``; only the simulator uses it (for travel, speeds and
                 offsets). None = generic machine.

    Returns a ``Backend`` that is not yet connected (call ``connect()``), except grbl, which already
    opens the port when created.

    Raises: ``BackendError`` if the type is unknown, the port is missing, pyserial is missing or
    LinuxCNC is missing.
    """
    kind = str(cfg.get("tipo", "sim")).lower()
    if kind == "sim":
        from hope.cnc.backends.sim import SimBackend
        return SimBackend(machine)
    if kind in ("grbl", "grblhal", "fluidnc"):                # all three speak the same grbl 1.1 protocol
        from hope.cnc.backends.grbl import GrblBackend
        return GrblBackend.from_config(cfg)
    if kind == "linuxcnc":
        from hope.cnc.backends.linuxcnc import LinuxCNCBackend
        return LinuxCNCBackend(cfg.get("ini"))
    raise BackendError(f"controlador «{kind}» desconocido (sim, grbl, linuxcnc)")


__all__ = ["Backend", "BackendError", "Status", "make_backend"]
