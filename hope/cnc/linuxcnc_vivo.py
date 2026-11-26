"""LinuxCNC as a live connection: TISMA's screen in front of a running LinuxCNC.

What it is
----------
``LinuxCncLink`` is the ``links.Link`` for ``controlador.tipo: linuxcnc``. LinuxCNC does the real
time (parallel port with StepConf on old PCs — 32 or 64-bit Linux, the most precise way to use a
Mach3 breakout board —, Mesa cards, Raspberry Pi) and TISMA is the operator screen: DRO, Cycle
Start / Feed Hold / Stop, jog, MDI, homing, overrides, offsets, probing routines and the program
check before moving, exactly as with grbl.

How it fits
-----------
``links.make_link`` creates it. It talks to the LinuxCNC that is already running (started with
its .ini, any GUI or ``linuxcnc -r``) through LinuxCNC's own Python module:

    linuxcnc.stat()          state, read with ``poll()`` (positions, modes, overrides, probe…)
    linuxcnc.command()       commands: modes, MDI, program open/run/pause/step/abort, jog, home,
                             e-stop reset / machine on, overrides, coolant
    linuxcnc.error_channel() errors and operator messages

The program goes as a temporary ``.ngc`` file in the ``linuxcnc`` dialect (the same translation the
checks were made on); LinuxCNC's ``motion_line`` is then the index of the running line, which is
what the progress bar needs. ``backends/linuxcnc.py`` stays for the command line (``enviar``).

Units and conventions
---------------------
A LinuxCNC configuration in mm is assumed (as in backends/linuxcnc.py). LinuxCNC reports speeds
per second: converted to mm/min. Position vectors are indexed by axis letter (X Y Z A B C U V W),
whatever the number of joints. Offsets: the active one comes in ``stat``; the six G54…G59 are read
from the parameter (.var) file named in the .ini (parameters 5221… every 20), G92 from 5211….

Safety
------
LinuxCNC keeps its own e-stop chain (the physical button wired to it): «Alarm» here while
LinuxCNC is in e-stop or the machine is off; «desbloquear» (unlock) sends e-stop reset and machine
on, which LinuxCNC only accepts if the physical e-stop is released.

Example
-------
::

    link = make_link(machine, {"tipo": "linuxcnc"})
    link.unlock()                  # e-stop reset + machine on (F1/F2 in AXIS)
    link.mdi("G0 X10")
"""

from __future__ import annotations

import os
import tempfile
import time

from hope.cnc.backends.base import BackendError
from hope.cnc.links import Link, _job
from hope.cnc.machine import Machine

# Index of each axis letter in LinuxCNC's position vectors.
AXIS_INDEX = {a: i for i, a in enumerate("XYZABCUVW")}
# First parameter of each work offset in the .var file (X; Y = +1 … C = +5), and G92's.
WCS_PARAMS = {"G54": 5221, "G55": 5241, "G56": 5261, "G57": 5281, "G58": 5301, "G59": 5321, "G92": 5211}
# Override ranges (fraction), like grbl's: feed and spindle 10–200 %, rapid 1–100 %.
OV_RANGE = {"avance": (10, 200), "rapido": (1, 100), "husillo": (10, 200)}


class LinuxCncLink(Link):
    """Live connection to a running LinuxCNC (see the module header).

    Parameters:
        machine: machine profile (axis names, limits: the jog checks them like grbl's $J).
        cfg:     the ``controlador`` section (``ini``: optional path, only to find the .var file if
                 LinuxCNC does not report its own .ini).
        module:  the ``linuxcnc`` module; None = import the real one (tests pass a fake one).

    Raises:
        BackendError: the ``linuxcnc`` module is not installed (LinuxCNC not present).
    """

    physical = True
    name = "LinuxCNC"

    def __init__(self, machine: Machine, cfg: dict | None = None, module=None):
        if module is None:
            try:
                import linuxcnc as module
            except ImportError:
                raise BackendError("el módulo «linuxcnc» no está: hace falta LinuxCNC instalado y en marcha "
                                   "(linuxcnc.org); TISMA se conecta a él") from None
        self.lc, self.m, self.cfg = module, machine, dict(cfg or {})
        self.stat = module.stat()
        self.cmd = module.command()
        self.err = module.error_channel()
        self.job = _job()
        self.message = ""
        self.alarm = ""
        self.ov = {"avance": 100, "rapido": 100, "husillo": 100}
        self._program = None             # path of the temporary .ngc running
        self._probe = None               # last probe result reported (see last_probe)
        self._last_probed = None

    def set_machine(self, machine: Machine) -> None:
        """Another machine profile (limits for the jog)."""
        self.m = machine

    # --- helpers ---------------------------------------------------------------------------------------
    def _poll(self):
        """Fresh state, and pending errors/messages read into ``alarm``/``message``."""
        self.stat.poll()
        while True:
            e = self.err.poll()
            if not e:
                break
            kind, text = e[0], str(e[1])
            if kind in (getattr(self.lc, "NML_ERROR", -1), getattr(self.lc, "OPERATOR_ERROR", -2)):
                self.alarm = text
                if self.job["activo"]:
                    self.job.update(activo=False, error=text)
            else:
                self.message = text
        return self.stat

    def _mode(self, mode) -> None:
        """Switches LinuxCNC's task mode (MANUAL / AUTO / MDI) if it is not there already."""
        s = self._poll()
        if s.task_mode != mode:
            self.cmd.mode(mode)
            self.cmd.wait_complete()

    def _require_on(self, what: str) -> None:
        """BackendError if LinuxCNC is in e-stop / off, or (for what moves) a program is running."""
        s = self._poll()
        if s.estop or s.task_state != self.lc.STATE_ON:
            raise BackendError(f"{what}: LinuxCNC está en parada de emergencia o apagado (pulsá «Desbloquear»)")

    def _busy(self) -> bool:
        s = self.stat
        return s.interp_state != self.lc.INTERP_IDLE or s.state == self.lc.RCS_EXEC

    # --- state -----------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """See :meth:`Link.snapshot`, translated from ``linuxcnc.stat``.

        State: Alarm (e-stop or machine off) > Hold (paused) > Run (interpreter busy or motion
        executing) / Jog (moving in manual mode) > Idle. Work offset (``wco``) = G5x + G92 + tool
        length in Z, so machine − wco is the DRO's work position.
        """
        s = self._poll()
        axes = self.m.axis_names
        pos = {a: float(s.actual_position[AXIS_INDEX[a]]) for a in axes}
        g5x, g92 = s.g5x_offset, s.g92_offset
        tlo = s.tool_offset[2] if len(s.tool_offset) > 2 else 0.0
        wco = {a: float(g5x[AXIS_INDEX[a]]) + float(g92[AXIS_INDEX[a]]) + (float(tlo) if a == "Z" else 0.0) for a in axes}
        if s.estop or s.task_state != self.lc.STATE_ON:
            state = "Alarm"
            alarm = self.alarm or ("parada de emergencia de LinuxCNC" if s.estop else "LinuxCNC apagado")
        else:
            alarm = self.alarm
            if s.paused or s.interp_state == self.lc.INTERP_PAUSED:
                state = "Hold"
            elif self._busy():
                state = "Jog" if s.task_mode == self.lc.MODE_MANUAL else "Run"
            else:
                state = "Idle"
        if self.job["activo"]:
            self.job["confirmadas"] = max(self.job["confirmadas"], int(s.motion_line) - 1)
            if s.interp_state == self.lc.INTERP_IDLE and s.state != self.lc.RCS_EXEC and s.task_mode == self.lc.MODE_AUTO:
                self.job.update(activo=False, terminado=self.job["error"] is None, confirmadas=self.job["total"])
        speed, direction = self._spindle(s)
        homed = list(getattr(s, "homed", ()) or ())
        return {"estado": state, "mpos": pos, "wco": wco, "avance": round(float(s.current_vel) * 60.0, 1),
                "husillo": speed, "husillo_dir": direction, "refrigerante": bool(s.flood or s.mist),
                "ov": dict(self.ov), "referenciado": bool(homed) and all(homed[:max(1, len(axes))]),
                "trabajo": dict(self.job), "mensaje": self.message, "alarma": alarm,
                "wcs": self._wcs_name(s), "conexion": "LinuxCNC", "pausa_g4": False,
                "bloque_a_bloque": self.single}

    def _spindle(self, s) -> tuple:
        """(rpm, "cw"/"ccw"/"off") — LinuxCNC 2.8+ has a ``spindle`` list, older ones flat fields."""
        sp = getattr(s, "spindle", None)
        if sp:
            speed, direction = float(sp[0].get("speed", 0.0)), int(sp[0].get("direction", 0))
        else:
            speed, direction = float(getattr(s, "spindle_speed", 0.0)), int(getattr(s, "spindle_direction", 0))
        return abs(speed), "cw" if direction > 0 else "ccw" if direction < 0 else "off"

    @staticmethod
    def _wcs_name(s) -> str:
        """Active work offset: ``g5x_index`` 1–6 = G54–G59, 7–9 = G59.1–G59.3."""
        i = int(getattr(s, "g5x_index", 1))
        return f"G5{3 + i}" if 1 <= i <= 6 else f"G59.{i - 6}"

    def offsets(self) -> dict:
        """G54–G59 and G92 from LinuxCNC's parameter file (written by LinuxCNC on every change).

        Path: ``[RS274NGC] PARAMETER_FILE`` of the running .ini (``stat.ini_filename``), relative
        to the .ini's folder. Returns {} if it cannot be read (the app then uses the profile's).
        """
        s = self._poll()
        ini = getattr(s, "ini_filename", "") or self.cfg.get("ini") or ""
        var = _parameter_file(ini)
        if not var:
            return {}
        try:
            with open(var, encoding="ascii", errors="replace") as f:
                params = {}
                for ln in f:
                    p = ln.split()
                    if len(p) >= 2 and p[0].isdigit():
                        params[int(p[0])] = float(p[1])
        except OSError:
            return {}
        out = {}
        for w, base in sorted(WCS_PARAMS.items()):
            out[w] = {a: params.get(base + AXIS_INDEX[a], 0.0) for a in self.m.axis_names if AXIS_INDEX[a] < 6}
        return out

    def modal(self) -> str:
        """Active G codes as text («G1 G17 G21 G90 G94 G54…»), from ``stat.gcodes`` (×10 values)."""
        s = self._poll()
        codes = []
        for v in getattr(s, "gcodes", ()) or ():
            if v is None or v < 0:
                continue
            codes.append(f"G{v // 10}" + (f".{v % 10}" if v % 10 else ""))
        return " ".join(codes)

    def tool(self) -> int | None:
        return int(getattr(self._poll(), "tool_in_spindle", 0)) or None

    def last_probe(self) -> dict | None:
        """See :meth:`Link.last_probe`: ``probed_position`` (latched by LinuxCNC at the trigger) and
        ``probe_tripped``. A result counts as new when the probed position changes."""
        s = self._poll()
        pp = tuple(float(v) for v in getattr(s, "probed_position", ()) or ())
        if pp and pp != self._last_probed:
            self._last_probed = pp
            n = (self._probe["n"] if self._probe else 0) + 1
            self._probe = {"n": n, "pos": {a: pp[AXIS_INDEX[a]] for a in self.m.axis_names},
                           "ok": bool(getattr(s, "probe_tripped", True))}
        return None if self._probe is None else dict(self._probe)

    # --- program -----------------------------------------------------------------------------------------
    def start(self, lines: list[str]) -> None:
        """Writes the program to a temporary .ngc, opens it in automatic mode and runs it (or one
        line, in single block). Raises BackendError if LinuxCNC is off or busy."""
        self._require_on("ciclo")
        if self._busy():
            raise BackendError("ciclo: LinuxCNC está ocupado")
        body = "\n".join(lines)
        if "M2" not in body and "M30" not in body:
            body += "\nM2"                      # LinuxCNC warns about a program without an end
        fd, path = tempfile.mkstemp(suffix=".ngc", prefix="tisma_")
        with os.fdopen(fd, "w", encoding="ascii", errors="replace") as f:
            f.write(body + "\n")
        self._program = path
        self.alarm = self.message = ""
        self._mode(self.lc.MODE_AUTO)
        self.cmd.program_open(path)
        self.cmd.wait_complete()
        self.job = _job(len(lines))
        self.job.update(activo=True, enviadas=len(lines))
        self.cmd.auto(self.lc.AUTO_STEP if self.single else self.lc.AUTO_RUN, 0)

    def hold(self) -> None:
        """Feed Hold: LinuxCNC's pause (decelerates, keeps the position)."""
        self.cmd.auto(self.lc.AUTO_PAUSE)

    def resume(self) -> None:
        """Cycle Start during a pause or an M0/M1: resume (one more line in single block)."""
        self.cmd.auto(self.lc.AUTO_STEP if self.single else self.lc.AUTO_RESUME)

    def stop(self) -> None:
        """Stop: LinuxCNC's abort (stops motion, spindle and coolant; the program ends)."""
        self.cmd.abort()
        if self.job["activo"]:
            self.job.update(activo=False, error="detenido por el operador")

    def clear_job(self) -> None:
        self.job = _job()

    # --- manual --------------------------------------------------------------------------------------------
    def mdi(self, line: str) -> None:
        """One MDI line in LinuxCNC's MDI mode. Waits until LinuxCNC accepted it (≤ 5 s)."""
        self._require_on("MDI")
        if self.job["activo"] or (self.stat.task_mode == self.lc.MODE_AUTO and self._busy()):
            raise BackendError("hay un programa en marcha")
        self._mode(self.lc.MODE_MDI)
        self.cmd.mdi(line)
        self.cmd.wait_complete(5.0)

    def jog(self, moves: dict, feed: float) -> None:
        """Incremental jog of each axis (teleop mode: axes, not joints), checked against the travel
        like grbl's $J. ``feed`` in mm/min."""
        self._require_on("jog")
        s = self.stat
        for a, d in moves.items():
            ax = self.m.axes.get(a)
            target = float(s.actual_position[AXIS_INDEX[a]]) + d
            if ax is not None and ((ax.min is not None and target < ax.min - 1e-3)
                                   or (ax.max is not None and target > ax.max + 1e-3)):
                raise BackendError(f"el jog sale del recorrido en {a}")
        self._mode(self.lc.MODE_MANUAL)
        if hasattr(self.cmd, "teleop_enable"):
            self.cmd.teleop_enable(1)
        for a, d in sorted(moves.items()):
            self.cmd.jog(self.lc.JOG_INCREMENT, False, AXIS_INDEX[a], feed / 60.0, d)

    def jog_cancel(self) -> None:
        for a in self.m.axis_names:
            self.cmd.jog(self.lc.JOG_STOP, False, AXIS_INDEX[a])

    def home(self) -> None:
        """Ref All Home: LinuxCNC homes every joint with the .ini's sequence. Waits up to 3 min.
        Raises BackendError on an error from LinuxCNC or if it does not finish."""
        self._require_on("referenciar")
        self._mode(self.lc.MODE_MANUAL)
        if hasattr(self.cmd, "teleop_enable"):
            self.cmd.teleop_enable(0)            # homing works on joints
        self.alarm = ""
        self.cmd.home(-1)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 180:
            s = self._poll()
            if self.alarm:
                raise BackendError(f"referenciado: {self.alarm}")
            homed = list(getattr(s, "homed", ()) or ())
            n = int(getattr(s, "joints", len(self.m.axis_names)) or len(self.m.axis_names))
            if homed and all(homed[:n]):
                return
            time.sleep(0.1)
        raise BackendError("referenciado: LinuxCNC no terminó en 3 minutos")

    def unlock(self) -> None:
        """E-stop reset and machine on (F1, F2 in AXIS). LinuxCNC refuses while the physical e-stop
        is pressed; then the state stays «Alarm»."""
        self.cmd.state(self.lc.STATE_ESTOP_RESET)
        self.cmd.wait_complete()
        self.cmd.state(self.lc.STATE_ON)
        self.cmd.wait_complete()
        self.alarm = ""

    def override(self, kind: str, action: str) -> None:
        """Feed / rapid / spindle override: «100», «+10», «-10», «+1», «-1», «50», «25»; ranges in
        OV_RANGE (LinuxCNC also clamps to its .ini limits)."""
        if kind not in OV_RANGE:
            return
        lo, hi = OV_RANGE[kind]
        cur = self.ov[kind]
        val = cur + int(action) if action[:1] in "+-" else int(action)
        self.ov[kind] = max(lo, min(hi, val))
        frac = self.ov[kind] / 100.0
        if kind == "avance":
            self.cmd.feedrate(frac)
        elif kind == "rapido":
            self.cmd.rapidrate(frac)
        else:
            self.cmd.spindleoverride(frac)

    def coolant_toggle(self) -> None:
        s = self._poll()
        self.cmd.flood(self.lc.FLOOD_OFF if s.flood else self.lc.FLOOD_ON)


def _parameter_file(ini: str) -> str:
    """Path of the .var file from ``[RS274NGC] PARAMETER_FILE`` of ``ini`` ("" if not found)."""
    if not ini or not os.path.exists(ini):
        return ""
    section = ""
    try:
        with open(ini, encoding="utf-8", errors="replace") as f:
            for ln in f:
                t = ln.split("#")[0].strip()
                if t.startswith("[") and t.endswith("]"):
                    section = t[1:-1].strip().upper()
                elif section == "RS274NGC" and "=" in t:
                    k, v = (x.strip() for x in t.split("=", 1))
                    if k.upper() == "PARAMETER_FILE" and v:
                        return v if os.path.isabs(v) else os.path.join(os.path.dirname(ini), v)
    except OSError:
        return ""
    return ""
