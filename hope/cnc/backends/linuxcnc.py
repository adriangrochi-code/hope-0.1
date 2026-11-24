"""LinuxCNC through its Python API (the ``linuxcnc`` module, shipped with LinuxCNC).

LinuxCNC does the real-time work: on old PCs it generates the pulses through the **parallel port**
(the same breakout board Mach3 used, configured with StepConf); on modern PCs, with **Mesa** cards
(7i96S, 7i76E) over Ethernet or with a Raspberry Pi 4/5. Hope CNC hands it the translated program
(``linuxcnc`` dialect) as a file and runs it in automatic mode.

HOW IT FITS
    ``make_backend({"tipo": "linuxcnc"})`` creates a ``LinuxCNCBackend``. LinuxCNC must be installed
    **and running** (with its .ini configuration loaded): this module only talks to it through the
    Python API, it does not start it. Unlike grbl, sending does not need to be metered here: LinuxCNC
    reads the whole file and plans on its own.

    API used (see «Python Interface» in the LinuxCNC documentation):
        ``linuxcnc.stat()``          reads the state (``poll()`` must be called before reading).
        ``linuxcnc.command()``       commands: mode, open program, run, pause, abort.
        ``linuxcnc.error_channel()`` errors reported by LinuxCNC (interpreter, limits, etc.).

UNITS
    LinuxCNC reports velocity in machine units **per second**; here it is converted to mm/min (× 60)
    to match grbl and the interface. A configuration in mm is assumed.

EXAMPLE
    >>> b = make_backend({"tipo": "linuxcnc"})
    >>> b.stream(["G21 G90", "G0 X10", "M2"]).ok
    True
"""

from __future__ import annotations

import tempfile
import time

from hope.cnc.backends.base import Backend, BackendError, Status, StreamResult

# Axis order in LinuxCNC's position vectors (``actual_position``, ``g5x_offset``):
# always 9 values X Y Z A B C U V W; only the first 6 are used here.
AXES = ("X", "Y", "Z", "A", "B", "C")


# ==================================================================================================
# =========================================== CONTROLLER ===========================================
# ==================================================================================================

class LinuxCNCBackend(Backend):
    """LinuxCNC controller (parallel port, Mesa, Raspberry Pi) through its Python API."""

    name = "linuxcnc"
    physical = True

    def __init__(self, ini: str | None = None, module=None):
        """Attach to the running LinuxCNC.

        Parameters:
            ini:    path to the machine's .ini (kept only for reference: the running LinuxCNC
                    already has its own loaded).
            module: the ``linuxcnc`` module to use; None = import the real one. Tests pass a fake one.

        Raises ``BackendError`` if the ``linuxcnc`` module is not installed.
        """
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
        """State translated to grbl's names (Idle, Run, Hold, Alarm; «Error» for anything else).

        Priority: emergency stop (Alarm) > pause (Hold) > task state. The work position is computed
        by subtracting the active G5x offset (it does not include G92 or tool length compensation).
        The feed comes in mm/s and is converted to mm/min.
        """
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
        """Write the program to a temporary .ngc, open it in LinuxCNC and run it to the end.

        Parameters: as in ``Backend.stream``. ``progress`` is called every 0.1 s with the line LinuxCNC
        is executing (``motion_line``) as «acknowledged». With ``stop_on_error`` the first message on
        the error channel aborts the program.

        Blocks until the interpreter is idle and the task is done. The temporary file is not deleted
        (``delete=False``): LinuxCNC keeps reading it while running and it stays around for review.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".ngc", delete=False, encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")
            path = f.name
        self.cmd.mode(self.lc.MODE_AUTO)                # a program can only run in automatic mode
        self.cmd.wait_complete()
        self.cmd.program_open(path)
        self.cmd.auto(self.lc.AUTO_RUN, 0)              # 0 = start from the first line
        errors = []
        err = self.lc.error_channel()
        while True:
            time.sleep(0.1)                             # poll at 10 Hz: enough for the progress bar, light on CPU
            self.stat.poll()
            e = err.poll()                              # None or (kind, text)
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
        """Pause the program (like AXIS's pause button)."""
        self.cmd.auto(self.lc.AUTO_PAUSE)

    def resume(self) -> None:
        """Resume a paused program."""
        self.cmd.auto(self.lc.AUTO_RESUME)

    def reset(self) -> None:
        """Abort the program and stop motion. It does not replace the physical emergency stop."""
        self.cmd.abort()
