"""LinuxCNC por su API de Python (módulo ``linuxcnc``, viene con LinuxCNC).

LinuxCNC hace el tiempo real: en PCs viejas genera los pulsos por el **puerto paralelo** (la misma
placa de conexión que usaba Mach3, configurada con StepConf); en PCs modernas, con tarjetas **Mesa**
(7i96S, 7i76E) por Ethernet o con una Raspberry Pi 4/5. Hope CNC le entrega el programa traducido
(dialecto ``linuxcnc``) como archivo y lo ejecuta en modo automático.
"""

from __future__ import annotations

import tempfile
import time

from hope.cnc.backends.base import Backend, BackendError, Status, StreamResult

AXES = ("X", "Y", "Z", "A", "B", "C")


class LinuxCNCBackend(Backend):
    name = "linuxcnc"
    physical = True

    def __init__(self, ini: str | None = None, module=None):
        if module is None:
            try:
                import linuxcnc as module
            except ImportError:
                raise BackendError("el módulo «linuxcnc» no está: este controlador necesita LinuxCNC instalado "
                                   "y en marcha (linuxcnc.org)") from None
        self.lc = module
        self.ini = ini
        self.stat = module.stat()
        self.cmd = module.command()

    def status(self) -> Status:
        self.stat.poll()
        state = {self.lc.RCS_DONE: "Idle", self.lc.RCS_EXEC: "Run"}.get(self.stat.state, "Error")
        if self.stat.paused:
            state = "Hold"
        if self.stat.estop:
            state = "Alarm"
        pos = {a: float(v) for a, v in zip(AXES, self.stat.actual_position)}
        off = {a: float(v) for a, v in zip(AXES, self.stat.g5x_offset)}
        return Status(state, pos, {a: pos[a] - off[a] for a in pos}, feed=float(self.stat.current_vel) * 60.0)

    def stream(self, lines, progress=None, stop_on_error=True) -> StreamResult:
        with tempfile.NamedTemporaryFile("w", suffix=".ngc", delete=False, encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")
            path = f.name
        self.cmd.mode(self.lc.MODE_AUTO)
        self.cmd.wait_complete()
        self.cmd.program_open(path)
        self.cmd.auto(self.lc.AUTO_RUN, 0)
        errors = []
        err = self.lc.error_channel()
        while True:
            time.sleep(0.1)
            self.stat.poll()
            e = err.poll()
            if e:
                errors.append(str(e[1]))
                if stop_on_error:
                    self.cmd.abort()
                    return StreamResult(len(lines), self.stat.motion_line, errors, aborted=True)
            if progress:
                progress(len(lines), self.stat.motion_line, len(lines))
            if self.stat.interp_state == self.lc.INTERP_IDLE and self.stat.state == self.lc.RCS_DONE:
                return StreamResult(len(lines), len(lines), errors)

    def hold(self) -> None:
        self.cmd.auto(self.lc.AUTO_PAUSE)

    def resume(self) -> None:
        self.cmd.auto(self.lc.AUTO_RESUME)

    def reset(self) -> None:
        self.cmd.abort()
