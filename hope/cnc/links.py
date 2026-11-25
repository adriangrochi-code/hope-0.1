"""Live connections for the app (hope/cnc/app): real-time simulator, grbl and «ok» firmwares.

WHAT IT IS
----------
A *link* is the object that talks to the machine controller while the app is open. Unlike
``backends`` (batch sending from the console: the program is sent and we wait for it to finish), a
live connection handles **at the same time** sending the program, the status every 200 ms, jog, MDI,
homing and the on-the-fly feed/spindle adjustments (overrides), as Mach3 does. All links share the
:class:`Link` interface, so the app neither knows nor cares which controller is on the other side.

* :class:`SimLink` — simulator: runs the planner blocks in real time (acceleration, corners,
  overrides, M0 pauses, feed hold, jog, homing). Useful to learn the app and to try programs without
  a machine. Its own thread advances the simulation clock every ``tick``.
* :class:`GrblLink` — grbl / grblHAL / FluidNC: a reader thread dispatches ``ok``/``error``/status and
  another one requests ``?``; the program is sent with **character counting** (the controller's
  receive buffer is filled without overflowing it). On-the-fly adjustments are «real-time» bytes
  (0x90…0x9D), jog is ``$J=``, homing is ``$H``.
* :class:`OkLink` — Marlin / Smoothieware / RepRapFirmware (MPCNC, Lowrider, 3D printer boards):
  **line by line** protocol, each line waits for its ``ok``. Position is requested with ``M114``,
  feed is adjusted with ``M220`` and stopping uses ``M410``/``M108`` (EMERGENCY_PARSER).

HOW IT FITS
-----------
* Used by: ``hope/cnc/session.py`` (:class:`CncSession`), which in turn serves the web interface
  (``hope/cnc/app/server.py``). The session polls :meth:`Link.snapshot` several times per second and
  calls ``start``/``hold``/``resume``/``stop``/``jog``/… when the operator presses a button.
* Uses: ``hope.cnc.gcode`` (interpreter, for the simulator), ``hope.cnc.planner`` (simulator speed
  profiles) and ``hope.cnc.backends.grbl`` (``open_port`` opens the serial/TCP port, ``describe``
  turns ``error:N``/``ALARM:N`` into text).
* :func:`make_link` picks the link according to the ``controlador`` section of the machine profile.

CONVENTIONS
-----------
* Units: millimetres (mm), mm/s inside the simulator, mm/min in what is displayed (``avance``) and in
  the F word, seconds for times, integer percentage (100 = no adjustment) for overrides.
* States (``snapshot()["estado"]``): grbl's names — ``Idle`` (at rest), ``Run`` (running),
  ``Hold`` (held), ``Jog``, ``Home`` (homing), ``Alarm``, plus ``Desconectado`` (disconnected).
  The other links mimic those names so the interface treats them all the same.
* Job dictionary (``_job``): ``activo`` (active), ``enviadas`` (lines sent), ``confirmadas``
  (lines acknowledged with ``ok``), ``total``, ``error`` (text or None) and ``terminado`` (finished).
* Concurrency: each physical link has **two locks**. ``lock`` (RLock) protects the state shared
  between the reader thread, the polling thread and the interface requests; ``wlock`` serializes
  writes to the port so two threads never interleave bytes of different lines. They are separate so
  the polling thread can write ``?`` without waiting for the reader.
* Texts for the operator are in Spanish; commands to the firmware, exactly as it expects them.

EXAMPLE
-------
::

    from hope.cnc.links import make_link
    link = make_link(machine)                     # according to machine.controller: {"tipo": "grbl", ...}
    link.start(["G21 G90", "G0 X10 Y10", "G1 X50 F600", "M30"])
    while link.snapshot()["trabajo"]["activo"]:
        time.sleep(0.2)                           # meanwhile the interface shows status/position
    link.close()
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque

from hope.cnc.backends.base import BackendError
from hope.cnc.backends.grbl import describe, open_port
from hope.cnc.gcode import Interpreter, Op
from hope.cnc.machine import Machine, as_controller
from hope.cnc.planner import distance_at, plan

# ==================================================================================================
# ================================== GRBL REAL-TIME COMMANDS =======================================
# ==================================================================================================

# grbl 1.1 real-time overrides: (kind, action) -> byte to send.
#
# grbl pulls these bytes out of the G-code stream as soon as they arrive at the port (in the receive
# interrupt): they take NO room in the 128-byte buffer, carry NO newline and get NO «ok» reply.
# That is why they can be sent at any time, even with a full buffer, and are not counted in ``pending``.
#
#   feed     (F of G1/G2/G3): 0x90 = 100 %, 0x91 = +10 %, 0x92 = −10 %, 0x93 = +1 %, 0x94 = −1 %.
#                             Range accepted by grbl: 10 %–200 % (anything beyond is clamped).
#   rapid    (G0):            0x95 = 100 %, 0x96 = 50 %, 0x97 = 25 %. Only those three values.
#   spindle  (S):             0x99 = 100 %, 0x9A = +10 %, 0x9B = −10 %, 0x9C = +1 %, 0x9D = −1 %.
#                             Range 10 %–200 %. (0x9E, stop spindle during hold, is not used.)
#
# The simulator uses this same table to validate (kind, action) and mimics those ranges.
OVERRIDES = {
    ("avance", "100"): 0x90, ("avance", "+10"): 0x91, ("avance", "-10"): 0x92, ("avance", "+1"): 0x93,
    ("avance", "-1"): 0x94, ("rapido", "100"): 0x95, ("rapido", "50"): 0x96, ("rapido", "25"): 0x97,
    ("husillo", "100"): 0x99, ("husillo", "+10"): 0x9A, ("husillo", "-10"): 0x9B, ("husillo", "+1"): 0x9C,
    ("husillo", "-1"): 0x9D,
}
# Jog cancel (grbl 1.1). Decelerates with the configured acceleration and discards the rest of the
# ``$J=`` motion in the buffer; the machine returns to Idle without losing position (unlike Ctrl-X).
JOG_CANCEL = 0x85
# Toggle flood coolant (M8/M9) in real time, without waiting for the buffer to drain.
# (0xA1 would be the mist one, M7; the app only handles flood.)
FLOOD_TOGGLE = 0xA0


# --- job ------------------------------------------------------------------------------------------
def _job(total=0):
    """New dictionary with the progress of a program (see «Conventions» in the header).

    total: number of lines in the program (0 = no program).
    Returns an inactive job: whoever starts the cycle turns ``activo`` on.
    """
    return {"activo": False, "enviadas": 0, "confirmadas": 0, "total": total, "error": None, "terminado": False}


# ==================================================================================================
# ===================================== COMMON INTERFACE (Link) ====================================
# ==================================================================================================

class Link:
    """Common interface of the live connections. The app only talks through these methods.

    Subclasses override what their controller can do; whatever they do not override stays as «does
    nothing» (``...``) or raises ``NotImplementedError`` if it is essential. All methods are called
    from the interface threads and must return quickly (except those that wait for a controller
    reply, such as ``home``, which document their maximum time).

    Class attributes:
        physical: True if it moves a real machine (the session asks for the guardian's approval).
        name:     short name shown as «conexión» (connection) if the controller did not identify itself.
        single:   single-block mode (Mach3's Single BLK) on or off.

    Errors: invalid requests (machine busy, command rejected, no reply) raise
    :class:`BackendError` with a Spanish text ready to show to the operator.
    """

    physical = True
    name = "base"

    # Opens the connection and starts the background threads. Idempotent: calling it twice does not
    # duplicate threads.
    def open(self) -> None: ...

    # Stops the background threads and closes the port. Does not stop the machine (that is ``stop``).
    def close(self) -> None: ...

    def snapshot(self) -> dict:
        """Status snapshot for the interface. Called several times per second: must not block.

        Returns a new dictionary (the interface may keep it without it changing underneath) with:
            estado (str), mpos (machine position per axis, mm), wco (active work origin, mm),
            avance (actual mm/min), husillo (rpm), husillo_dir ("cw"/"ccw"/"off"), refrigerante (bool),
            ov ({"avance", "rapido", "husillo"} in %), referenciado (bool), trabajo (see ``_job``),
            mensaje, alarma, wcs ("G54"…), conexion (name/banner), pausa_g4 (bool),
            bloque_a_bloque (bool) and, in the simulator, avance_prog (programmed mm/min).
        """
        raise NotImplementedError

    def offsets(self) -> dict[str, dict[str, float]]:
        """Work origins of the controller: ``{"G54": {"X": mm, ...}, ..., "G92": {...}}``.

        Returns {} if the controller does not report them (the app then uses its own).
        """
        return {}

    def start(self, lines: list[str]) -> None:
        """Starts a program (Cycle Start). ``lines``: already translated G-code lines, one per element.

        Returns right away; sending continues in the background and is followed via ``snapshot()["trabajo"]``.
        Raises BackendError if a program is already running or the machine is not at rest.
        """
        raise NotImplementedError

    # Feed Hold: decelerate without losing position. Continued with ``resume``.
    def hold(self) -> None: ...

    # Cycle Start during a hold or an M0/M1 pause: continue.
    def resume(self) -> None: ...

    # Stop: abort the program, turn the spindle off and leave the machine at rest.
    def stop(self) -> None: ...

    def jog(self, moves: dict[str, float], feed: float) -> None:
        """Incremental manual motion.

        moves: displacement per axis in mm, e.g. ``{"X": 1.0, "Z": -0.1}``.
        feed:  speed in mm/min.
        Raises BackendError if the machine is busy or the motion goes beyond the travel.
        """
        raise NotImplementedError

    # Cuts a jog in progress (on releasing the continuous-jog key).
    def jog_cancel(self) -> None: ...

    def home(self) -> None:
        """Homing (Mach3's Ref All Home). May wait until the cycle ends."""
        raise NotImplementedError

    # Clear the alarm lock without homing ($X in grbl, M999 in Marlin).
    def unlock(self) -> None: ...

    def mdi(self, line: str) -> None:
        """Runs a single loose G-code line (MDI). Raises BackendError if a program is running."""
        raise NotImplementedError

    # On-the-fly adjustment: kind = "avance" | "rapido" | "husillo"; action = OVERRIDES key ("+10", "100"…).
    def override(self, kind: str, action: str) -> None: ...

    # Turns coolant on/off (Mach3's Flood button), even with a program running.
    def coolant_toggle(self) -> None: ...

    def last_probe(self) -> dict | None:
        """Result of the last probing move (G38.2/G38.3) as the controller reported it.

        Returns ``{"n": counter, "pos": {axis: machine mm}, "ok": touched?}`` or ``None`` if the
        controller does not report probe results (then ``palpado`` uses the position where the
        machine stopped). ``n`` grows by one with every report, so a caller can tell a fresh result
        from the previous one.
        """
        return None

    # Single block off by default: the program runs straight through.
    single = False

    def set_single(self, on: bool) -> None:
        """Single block (Mach3's Single BLK): each Cycle Start runs one line and then pauses."""
        self.single = bool(on)

    def modal(self) -> str:
        """Text with the active modal codes (G90 G21 G54…), or "" if the link does not know them."""
        return ""

    def tool(self) -> int | None:
        """Number of the active tool (T), or None if the link does not know it."""
        return None

    def clear_job(self) -> None:
        """Rewind (Mach3): forgets the progress of the last program if none is running."""
        job = getattr(self, "job", None)
        if job is not None and not job["activo"]:
            self.job = _job()


# ==================================================================================================
# ====================================== SIMULATOR (SimLink) =======================================
# ==================================================================================================

class SimLink(Link):
    """Real-time simulator: a «machine» that lives in memory and moves like the real one.

    It interprets the G-code with the app's own interpreter (``Interpreter``), runs it through the
    same planner (``plan``: acceleration, cornering speed) and then advances the planned blocks
    following the clock, so the position, speed and times shown on screen are those the real machine
    would have. It mimics grbl: same states, same override ranges, feed hold, M0, jog and homing.

    Parameters:
        machine:   machine profile (axes, travels, speeds, accelerations).
        speed:     clock multiplier. 1.0 = real time; 10 = ten times faster (useful to try long
                   programs). Very high (>100) makes big jumps per tick and the view looks «jumpy»,
                   but the final result is the same.
        tick:      how often the simulation advances, in seconds. 0.02 s = 50 Hz, finer than the
                   interface polling (200 ms). Smaller uses more CPU (beware on old PCs);
                   larger makes the position advance in jumps.
        autostart: start the simulation thread already in the constructor (the tests turn it off and
                   call ``step`` by hand to control time).

    All the state is touched with ``self.lock`` held: the simulation thread and the interface
    requests run in parallel.
    """

    physical = False
    name = "simulador"

    def __init__(self, machine: Machine, speed: float = 1.0, tick: float = 0.02, autostart: bool = True):
        self.m = as_controller(machine)               # receives the program already translated
        self.speed = speed
        self.tick = tick
        self.lock = threading.RLock()
        self.interp = Interpreter(self.m)             # modal state and origins, like the controller's
        self._pos = dict(self.interp.pos)             # simulated physical position (the interpreter runs ahead)
        self.state = "Idle"
        self.homed = False
        self.ov = {"avance": 100, "rapido": 100, "husillo": 100}   # overrides in %
        self.spindle = ("off", 0.0)                   # (direction "cw"/"ccw"/"off", programmed rpm)
        self.coolant = False
        # Queue of what is left to run, in order: ("block", planned block) for motions and
        # ("op", Op) for events that do not move (spindle, coolant, G4, M0, tool change, end).
        self.items: deque = deque()
        self.cur = None                               # element being run (same format as items) or None
        self.t_in = 0.0                               # seconds elapsed inside ``cur``
        self.v = 0.0                                  # current speed along the path, mm/s
        self.job = _job()
        self.jogging = False
        self.message = ""                             # text for the operator (e.g. that of an M0)
        self.alarm = ""
        self._home_left = 0.0                         # seconds left of the simulated homing
        self._last_line = None                        # program line of the last element started
        self._single_go = False                       # single block: permission to start one more line
        self._run = True                              # False stops the simulation thread
        self._thread = None
        # Probing in the simulator. ``probe_surface(start, target)`` returns where a probe moving in
        # a straight line from ``start`` to ``target`` (machine {axis: mm}) touches, or None if it
        # touches nothing. Without one (None) the interpreter's assumption is used: a downward Z
        # probe touches the work Z0, any other probe at the end of its travel. Tests (and a future
        # «virtual stock») set it to measure real shapes.
        self.probe_surface = None
        self._probe = None                            # last result (see Link.last_probe)
        if autostart:
            self.open()

    @property
    def pos(self) -> dict[str, float]:
        """Simulated machine position, mm per axis (the real one, not what the interpreter already read)."""
        return self._pos

    def open(self) -> None:
        """Starts the simulation thread (daemon: does not prevent the program from closing)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="hope-cnc-sim")
            self._thread.start()

    def close(self) -> None:
        """Asks the simulation thread to finish on the next tick."""
        self._run = False

    def _loop(self):
        """Simulation thread: every ``tick`` it advances the real time elapsed (× ``speed``).

        Time is measured with ``time.monotonic`` instead of assuming the sleep lasted exactly ``tick``:
        on a loaded PC the sleep gets longer and, this way, the simulation does not fall behind the clock.
        """
        last = time.monotonic()
        while self._run:
            time.sleep(self.tick)
            now = time.monotonic()
            self.step((now - last) * self.speed)
            last = now

    # --- execution --------------------------------------------------------------------------------
    def _queue(self, ops: list[Op], line_base: int = 0):
        """Plans ``ops`` and appends to ``self.items`` the blocks interleaved with their events.

        ``plan`` returns the motion blocks and, separately, the ``events`` list with pairs
        (index of the block before which it happens, Op). Here they are merged back in program
        order: an M3 between two G1 runs when the first one ends, as on the machine.
        ``line_base`` is not used (kept for signature compatibility).
        """
        events: list = []
        pl = plan(ops, self.m, events=events)
        ev = deque(events)
        for i, b in enumerate(pl.blocks):
            while ev and ev[0][0] == i:
                self.items.append(("op", ev.popleft()[1]))
            self.items.append(("block", b))
        while ev:                                     # events after the last motion (M5, M30…)
            self.items.append(("op", ev.popleft()[1]))

    def step(self, dt: float) -> None:
        """Advances the simulation by ``dt`` seconds of machine time.

        Several short blocks may end in the same step: the time left over when one ends is used in
        the next one (``while dt`` loop), so the total duration does not depend on ``tick``.
        """
        with self.lock:
            if self.state == "Home":
                self._home_left -= dt
                if self._home_left <= 0:
                    # Simulated homing: Z up (maximum, like grbl which raises Z first) and the rest
                    # at the travel minimum; 0 if the axis has no configured limit.
                    for a, ax in self.m.axes.items():
                        self.pos[a] = (ax.max if ax.max is not None else 0.0) if a == "Z" else \
                            (ax.min if ax.min is not None else 0.0)
                    self.homed, self.state = True, "Idle"
                return
            while dt > 1e-12 and self.state in ("Run", "Jog"):
                if self.cur is None:
                    if not self.items:
                        self._finish()
                        return
                    nxt = self.items[0][1]
                    # Single block: when moving to a new program line it holds, unless a Cycle Start
                    # (resume) gave permission for one more line (_single_go).
                    if self.single and self.job["activo"] and self._last_line is not None \
                            and nxt.line != self._last_line:
                        if not self._single_go:
                            self.state, self.v = "Hold", 0.0     # single block: waits for Cycle Start
                            return
                        self._single_go = False
                    self._last_line = nxt.line
                    self.cur, self.t_in = self.items.popleft(), 0.0
                kind, it = self.cur
                if self.job["activo"]:
                    # Everything before the running line is already «acknowledged» (like grbl's oks).
                    self.job["confirmadas"] = max(self.job["confirmadas"], it.line - 1)
                if kind == "block":
                    # The override stretches or shrinks the block time: at 50 % each real second
                    # advances half a second of the planned profile. G0 uses the rapid one, the rest the feed one.
                    scale = (self.ov["rapido"] if it.kind == "rapid" else self.ov["avance"]) / 100.0
                    self.t_in += dt * scale
                    if self.t_in >= it.time:
                        # Finished: it lands on the exact end point (no accumulated error) and the
                        # leftover time, converted back to real seconds, goes on with the next element.
                        dt = (self.t_in - it.time) / scale
                        for i, a in enumerate(self.m.axis_names):
                            self.pos[a] = it.end[i]
                        self.cur = None
                    else:
                        d = distance_at(it, self.t_in)   # mm travelled inside the block according to its profile
                        # Instantaneous speed by a 1 ms finite difference (for the feed shown on screen).
                        self.v = (distance_at(it, self.t_in + 1e-3) - d) / 1e-3 * scale
                        for i, a in enumerate(self.m.axis_names):
                            self.pos[a] = it.start[i] + it.unit[i] * d   # unit = unit direction vector
                        dt = 0.0
                else:
                    dt = self._event(it, dt)

    def _event(self, op: Op, dt: float) -> float:
        """Runs a motionless event and returns the time (s) left over for what follows.

        * dwell (G4): consumes ``op.value`` seconds; not affected by the override, same as in grbl.
        * spindle / coolant: change the state instantly.
        * pause (M0/M1) and tool (M6): hold with a message until Cycle Start.
        * end (M2/M30): discards whatever is left in the queue.
        """
        if op.kind == "dwell":
            self.t_in += dt
            if self.t_in < float(op.value):
                return 0.0
            dt, self.cur = self.t_in - float(op.value), None
            return dt
        self.cur = None
        if op.kind == "spindle":
            self.spindle = op.value
        elif op.kind == "coolant":
            self.coolant = bool(op.value)
        elif op.kind in ("pause", "tool"):
            self.message = str(op.value) if op.kind == "pause" else f"cambiar a la herramienta T{op.value}"
            self.state, self.v = "Hold", 0.0
            return 0.0
        elif op.kind == "end":
            self.items.clear()
        return dt

    def _finish(self):
        """The queue emptied: the program (or the jog) ended and the machine returns to rest."""
        self.cur, self.v = None, 0.0
        if self.job["activo"]:
            self.job.update(activo=False, terminado=True, confirmadas=self.job["total"])
        self.state = "Idle"
        self.jogging = False

    # --- interface --------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """See :meth:`Link.snapshot`. ``avance`` is converted from mm/s to mm/min; spindle includes the override."""
        with self.lock:
            return {"estado": self.state, "mpos": dict(self.pos), "wco": {a: self.interp._off(a) for a in self.pos},
                    "avance": round(self.v * 60.0, 1), "husillo": self.spindle[1] * self.ov["husillo"] / 100.0
                    if self.spindle[0] != "off" else 0.0, "husillo_dir": self.spindle[0],
                    "refrigerante": self.coolant, "ov": dict(self.ov), "referenciado": self.homed,
                    "trabajo": dict(self.job), "mensaje": self.message, "alarma": self.alarm,
                    "wcs": self.interp.wcs, "conexion": self.name,
                    "pausa_g4": bool(self.cur and self.cur[0] == "op" and self.cur[1].kind == "dwell"),
                    "avance_prog": round(self.cur[1].v_nominal * 60.0, 1)
                    if self.cur and self.cur[0] == "block" and self.cur[1].kind == "feed" else None,
                    "bloque_a_bloque": self.single}

    def set_machine(self, machine: Machine) -> None:
        """Another machine (profile): new limits, speeds and origins; the position is kept."""
        with self.lock:
            self.m = as_controller(machine)
            pos = dict(self._pos)
            self.interp = Interpreter(self.m)
            self._pos = {a: pos.get(a, 0.0) for a in self.m.axis_names}   # new axes start at 0
            self.interp.pos = dict(self._pos)

    def modal(self) -> str:
        """Active modal codes of the simulator's interpreter (G90 G21 G54 G17…)."""
        from hope.cnc.gcode import modal_text
        with self.lock:
            return modal_text(self.interp)

    def tool(self) -> int | None:
        """Active tool according to the simulator's interpreter."""
        return self.interp.tool

    def offsets(self) -> dict[str, dict[str, float]]:
        """Copy of the interpreter's G54…G59 origins plus the G92 offset."""
        with self.lock:
            out = {w: dict(v) for w, v in self.interp.offsets.items()}
            out["G92"] = dict(self.interp.g92)
            return out

    def _idle(self, what: str):
        """Raises BackendError («<what>: la máquina no está en reposo») if the state is not Idle."""
        if self.state != "Idle":
            raise BackendError(f"{what}: la máquina no está en reposo ({self.state})")

    def _run_text(self, text: str) -> list[Op]:
        """Interprets ``text`` from the current physical position and returns the resulting operations.

        Reuses the interpreter (keeps modal state and origins, like a real controller between one MDI
        and the next) but empties its previous output. Raises BackendError with all errors joined.
        """
        it = self.interp
        it.ops, it.diags, it.ended = [], [], False
        it.pos = dict(self._pos)
        it.run(text)
        if any(d.level == "error" for d in it.diags):
            raise BackendError("; ".join(str(d) for d in it.diags if d.level == "error"))
        return it.ops

    def start(self, lines: list[str]) -> None:
        """See :meth:`Link.start`. Interprets and plans the whole program at once before starting."""
        with self.lock:
            self._idle("ciclo")
            ops = self._run_text("\n".join(lines))
            self.items.clear()
            self._queue(ops)
            self.job = _job(len(lines))
            self.job.update(activo=True, enviadas=len(lines))   # in the simulator «sending» is instantaneous
            self.message = ""
            self._last_line, self._single_go = None, False
            self.state = "Run"

    def hold(self) -> None:
        """Immediate hold (the simulator stops dead; grbl would do it with its acceleration)."""
        with self.lock:
            if self.state == "Run":
                self.state, self.v = "Hold", 0.0

    def resume(self) -> None:
        """Continues from a hold or an M0; in single block it enables one more line."""
        with self.lock:
            if self.state == "Hold":
                self.message = ""
                self._single_go = self.single
                self.state = "Run"

    def stop(self) -> None:
        """Aborts: empties the queue, turns off the spindle (keeps the programmed rpm) and the coolant."""
        with self.lock:
            self.items.clear()
            self.cur, self.v = None, 0.0
            if self.job["activo"]:
                self.job.update(activo=False, error="detenido por el operador")
            self.spindle = ("off", self.spindle[1])
            self.coolant = False
            self.state = "Idle"

    def jog(self, moves: dict[str, float], feed: float) -> None:
        """See :meth:`Link.jog`. A new jog replaces the one in progress (continuous jog).

        Checks the travels before moving (like grbl's soft limits with ``$J=``: error 15).
        ``feed`` arrives in mm/min and is planned as a G1.
        """
        with self.lock:
            if self.state == "Jog":
                self.jog_cancel()
            self._idle("jog")
            target = dict(self.pos)
            for a, d in moves.items():
                target[a] += d
            ax_bad = [a for a in moves if not self._inside(a, target[a])]
            if ax_bad:
                raise BackendError(f"el jog sale del recorrido en {', '.join(ax_bad)}")
            self.items.clear()
            self._queue([Op("feed", 0, dict(self.pos), target, feed)])
            self.state, self.jogging = "Jog", True

    def _inside(self, a: str, v: float) -> bool:
        """True if coordinate ``v`` (mm) is inside the travel of axis ``a`` (1 µm tolerance)."""
        ax = self.m.axes[a]
        return (ax.min is None or v >= ax.min - 1e-6) and (ax.max is None or v <= ax.max + 1e-6)

    def jog_cancel(self) -> None:
        """Cuts the jog wherever it is (no deceleration ramp) and returns to Idle."""
        with self.lock:
            if self.state == "Jog":
                self.items.clear()
                self.cur, self.v = None, 0.0
                self.state, self.jogging = "Idle", False

    def home(self) -> None:
        """Simulated homing: 1.5 s in Home state and then each axis to its end (see ``step``).

        Does not wait for it to finish (unlike grbl): the interface sees the Home state in the snapshot.
        """
        with self.lock:
            if self.state not in ("Idle", "Alarm"):
                raise BackendError("referenciar: la máquina no está en reposo")
            self.state, self._home_left, self.alarm = "Home", 1.5, ""

    def unlock(self) -> None:
        """Like ``$X``: leaves Alarm without homing."""
        with self.lock:
            if self.state == "Alarm":
                self.state, self.alarm = "Idle", ""

    def mdi(self, line: str) -> None:
        """See :meth:`Link.mdi`. If the line moves nothing (M3, G54, M8…) it is applied without going through Run."""
        with self.lock:
            self._idle("MDI")
            ops = self._run_text(line)
            for op in ops:
                if op.kind == "probe" and not self._sim_probe(op):
                    return                                 # G38.2 without contact: alarm, nothing moves
            self.items.clear()
            self._queue(ops)
            if self.items and all(k == "op" and it.kind != "dwell" for k, it in self.items):
                while self.items:                          # no motion: applied right away
                    self._event(self.items.popleft()[1], 0.0)
            if self.items:
                self.state = "Run"

    def _sim_probe(self, op: Op) -> bool:
        """Decides where a simulated probing move stops (``op.end``) and records the result.

        With ``probe_surface`` the contact is that function's answer; without it, the end the
        interpreter planned. No contact: ``G38.3`` goes the whole way and reports «not touched»;
        ``G38.2`` / ``G31`` raise an alarm like grbl (ALARM:5) — here before moving, which is
        enough for the routines and the tests. Returns False if it raised the alarm.
        """
        target = dict(op.value) if isinstance(op.value, dict) else dict(op.end)
        contact = self.probe_surface(dict(op.start), target) if self.probe_surface else dict(op.end)
        n = (self._probe["n"] if self._probe else 0) + 1
        if contact is None:
            self._probe = {"n": n, "pos": dict(target), "ok": False}
            if op.source != "G38.3":
                self.state, self.alarm = "Alarm", "sensado sin contacto (G38.2): la sonda no tocó nada"
                return False
            op.end = target
            return True
        op.end = {a: float(contact.get(a, op.end[a])) for a in op.end}
        self._probe = {"n": n, "pos": dict(op.end), "ok": True}
        return True

    def last_probe(self) -> dict | None:
        """See :meth:`Link.last_probe`."""
        with self.lock:
            return None if self._probe is None else {"n": self._probe["n"], "pos": dict(self._probe["pos"]),
                                                      "ok": self._probe["ok"]}

    def override(self, kind: str, action: str) -> None:
        """See :meth:`Link.override`. Mimics grbl's ranges: feed/spindle 10–200 %, rapid 25–100 %."""
        with self.lock:
            if (kind, action) not in OVERRIDES:
                raise BackendError(f"ajuste desconocido: {kind} {action}")
            lo, hi = (10, 200) if kind != "rapido" else (25, 100)
            if action in ("100", "50", "25"):              # absolute values
                self.ov[kind] = int(action)
            else:                                          # relative steps (+10, −1…), clamped to the range
                self.ov[kind] = max(lo, min(hi, self.ov[kind] + int(action)))

    def coolant_toggle(self) -> None:
        """Toggles the coolant instantly."""
        with self.lock:
            self.coolant = not self.coolant


# ==================================================================================================
# ============================== GRBL / GRBLHAL / FLUIDNC (GrblLink) ===============================
# ==================================================================================================

# grbl 1.1 status report (reply to «?»), e.g.:
#     <Run|MPos:10.000,5.000,-1.000|FS:600,12000|Ov:100,100,100|A:SF|WCO:0.000,0.000,0.000>
# Group 1: state (may carry a substate: «Hold:0», «Door:1»). Group 2: «name:value» fields joined by «|».
STATUS = re.compile(r"<([^|>]+)\|(.*)>")


class GrblLink(Link):
    """grbl 1.1 / grblHAL / FluidNC over a serial or TCP port, with two background threads.

    PROTOCOL (why it is done this way):

    * **Character counting.** grbl has a 128-byte serial receive buffer (on the Arduino Uno;
      grblHAL and FluidNC usually have 1024 or more). If we waited for each line's ``ok`` before
      sending the next one, the controller's planner would run out of blocks on short segments
      (curves made of segments) and the machine would stutter. If we sent without control, the
      buffer would overflow and characters would be lost (cut lines = wrong motions). The solution
      grbl recommends is counting: the bytes of each sent line are recorded (``pending``) and another
      one is sent only if the sum fits in the buffer; each ``ok``/``error`` frees the oldest line,
      because grbl replies in order.
    * **Real-time commands** (``?``, ``!``, ``~``, Ctrl-X, overrides 0x90…): grbl pulls them out of
      the stream on receipt, they take no buffer room and get no ``ok``; they are sent at any time.
    * **Status**: the polling thread sends ``?`` every ``poll`` seconds; the ``<…>`` reply is
      processed by the reader thread.
    * **Loose commands** (``$H``, ``$J=``, ``$#``, MDI): only with the machine free of a program; the
      ``ok`` is delivered to whoever is waiting (``waiter``), together with the lines arriving before.

    Parameters:
        port:      port object (``write``/``readline``/``close``), see ``backends.grbl.open_port``.
        machine:   machine profile; the axis names are taken from it, in the order in which grbl
                   reports the coordinates.
        rx_buffer: size of the controller's receive buffer, in bytes. 128 for grbl 1.1 on the Uno
                   (default value, safe for all). grblHAL/FluidNC accept 1024 or more and with that
                   they feed programs with many short segments better. If set LARGER than the real
                   one, the controller loses characters and runs corrupted lines: do not overdo it.
                   Smaller only costs smoothness.
        poll:      ``?`` status polling period, in seconds. 0.2 s (5 Hz) is what grbl recommends for
                   the interface. Going below ~0.05 s loads the port and the controller's processor;
                   raising it makes the on-screen position update in jumps.
        autostart: open (start threads) already in the constructor.
    """

    name = "grbl"

    def __init__(self, port, machine: Machine, rx_buffer: int = 128, poll: float = 0.2, autostart: bool = True):
        self.port = port
        self.m = machine
        self.axes = machine.axis_names
        self.rx = int(rx_buffer)
        self.poll = poll
        self.wlock = threading.Lock()                 # one port write at a time (see header)
        self.lock = threading.RLock()                 # state shared between reader, interface and commands
        # Last state reported by grbl. mpos always in machine coordinates (mm).
        self.st = {"estado": "Desconectado", "mpos": {a: 0.0 for a in self.axes}, "avance": 0.0, "husillo": 0.0,
                   "husillo_dir": "off", "refrigerante": False, "ov": {"avance": 100, "rapido": 100, "husillo": 100},
                   "pines": ""}
        self.wco = {a: 0.0 for a in self.axes}       # active work origin (WCO = MPos − WPos), mm
        self.homed = False
        self.banner = ""                              # «Grbl 1.1h ['$' for help]» or similar, on connect/reset
        self.message = ""
        self.alarm = ""
        self.probe = None                             # last «[PRB:…]» (see last_probe)
        self.lines: list[str] = []                    # program in progress, already clean (no comments or spaces)
        self.pending: deque[int] = deque()            # bytes of each sent line that has not had ok/error yet
        self.job = _job()
        # Loose command waiting for a reply: {"event": Event, "lines": [replies], "error": text|None}.
        self.waiter: dict | None = None
        self._tokens = 0                              # single block: lines that may still be sent
        self.log: deque[str] = deque(maxlen=200)      # latest replies (without the status reports)
        self._run = False                             # False stops the reader and polling threads
        if autostart:
            self.open()

    @classmethod
    def from_config(cls, cfg: dict, machine: Machine) -> GrblLink:
        """Creates the link from the ``controlador`` section of the profile.

        Keys: ``puerto`` (required: ``/dev/ttyUSB0``, ``COM3`` or ``tcp://host:port``),
        ``baudios`` (115200, grbl 1.1's) and ``bufer`` (bytes of the receive buffer, see the class).
        Raises BackendError if the port is missing or cannot be opened.
        """
        if not cfg.get("puerto"):
            raise BackendError("controlador grbl sin «puerto» (p. ej. /dev/ttyUSB0, COM3 o tcp://192.168.1.50:23)")
        return cls(open_port(str(cfg["puerto"]), int(cfg.get("baudios", 115200))), machine,
                   rx_buffer=int(cfg.get("bufer", 128)))

    # --- threads ----------------------------------------------------------------------------------
    def open(self) -> None:
        """Wakes grbl up and starts the reader and polling threads (daemon). Idempotent."""
        if self._run:
            return
        self._run = True
        # Two newlines: they close any half line left in grbl's buffer (e.g. from a previous cut
        # session) and «wake it up», as grbl's stream.py does.
        self._write(b"\r\n\r\n")
        threading.Thread(target=self._reader, daemon=True, name="hope-cnc-grbl-rx").start()
        threading.Thread(target=self._poller, daemon=True, name="hope-cnc-grbl-poll").start()

    def close(self) -> None:
        """Stops the threads and closes the port (an error while closing is only written to the log)."""
        self._run = False
        try:
            self.port.close()
        except Exception as e:                             # noqa: BLE001
            self.log.append(f"al cerrar: {e}")

    def _write(self, data: bytes) -> None:
        """Writes raw bytes to the port holding ``wlock``: lines and real-time bytes do not interleave."""
        with self.wlock:
            self.port.write(data)

    def _poller(self):
        """Polling thread: sends ``?`` every ``poll`` s. If the port fails, marks Desconectado and exits."""
        while self._run:
            try:
                self._write(b"?")
            except Exception as e:                          # noqa: BLE001
                self.alarm = f"conexión perdida: {e}"
                self.st["estado"] = "Desconectado"
                return
            time.sleep(self.poll)

    def _reader(self):
        """Reader thread: reads line by line and hands them to :meth:`handle`.

        ``readline`` returns empty when its timeout expires (0.2 s in ``open_port``); then it
        sleeps 2 ms so as not to spin idle, and tries again.
        """
        while self._run:
            try:
                raw = self.port.readline()
            except Exception as e:                          # noqa: BLE001
                self.alarm = f"conexión perdida: {e}"
                self.st["estado"] = "Desconectado"
                return
            if not raw:
                time.sleep(0.002)
                continue
            self.handle(raw.decode("ascii", errors="replace").strip())

    # --- protocol ---------------------------------------------------------------------------------
    def handle(self, line: str) -> None:
        """Processes a line received from grbl (called from the reader thread; public for the tests).

        Line kinds:
            ``<…>``         status report → :meth:`_status` (not stored in the log: 5 arrive per second).
            ``ok``/``error:N``  acknowledges the oldest line: the program's if there is one, else the command's.
            ``ALARM:N``     alarm (limit switch, soft limit, reset while moving…): aborts everything.
            ``Grbl …``      startup banner: the controller was reset and emptied its buffer.
            ``[MSG:…]``     message for the operator (e.g. «Caution: Unlocked»).
            others          ``[G54:…]``, ``$N=…``… are part of the reply to a loose command.
        """
        if not line:
            return
        with self.lock:
            if line.startswith("<"):
                self._status(line)
                return
            self.log.append(line)
            if line == "ok" or line.startswith("error:"):
                if self.job["activo"] and self.pending:
                    self.pending.popleft()                 # grbl replies in order: the oldest is freed
                    self.job["confirmadas"] += 1
                    if line != "ok":
                        # A rejected line: grbl goes on with those already in its buffer, which may
                        # depend on it (e.g. a G1 without the G90 that failed). For safety we abort
                        # with a reset (Ctrl-X), which discards the buffer and the planner.
                        n = self.job["confirmadas"]
                        self.job.update(activo=False, error=f"línea {n} «{self.lines[n - 1]}»: {describe(line)}")
                        self._write(b"\x18")
                        return
                    if self.job["confirmadas"] == self.job["total"]:
                        self.job.update(activo=False, terminado=True)
                    self._fill()                           # room was freed in the buffer: send more
                elif self.waiter is not None:
                    self.waiter["error"] = None if line == "ok" else describe(line)
                    self.waiter["event"].set()
            elif line.startswith("ALARM:"):
                self.alarm = describe(line)
                self.st["estado"] = "Alarm"
                if self.job["activo"]:
                    self.job.update(activo=False, error=self.alarm)
                if self.waiter is not None:
                    self.waiter["error"] = self.alarm
                    self.waiter["event"].set()
            elif line.lower().startswith(("grbl", "fluidnc")) or "grbl" in line.lower():
                # Banner: the controller started or was reset (Ctrl-X, cable, watchdog). Everything
                # in its buffer was lost, so no more oks are expected for those lines.
                self.banner = line
                self.pending.clear()
                if self.job["activo"]:
                    self.job.update(activo=False, error="el controlador se reinició")
            elif line.startswith("[MSG:"):
                self.message = line[5:].rstrip("]").strip()
            elif line.startswith("[PRB:"):
                # Probe result «[PRB:x,y,z:1]»: machine position latched when the probe input
                # switched (not where it stopped after decelerating) and 1/0 = touched or not.
                body = line[5:].rstrip("]")
                coords, _, flag = body.rpartition(":")
                n = (self.probe["n"] if self.probe else 0) + 1
                self.probe = {"n": n, "pos": {a: float(v) for a, v in zip(self.axes, coords.split(","))},
                              "ok": flag.strip() == "1"}
                if self.waiter is not None:
                    self.waiter["lines"].append(line)
            elif self.waiter is not None:
                self.waiter["lines"].append(line)

    def last_probe(self) -> dict | None:
        """See :meth:`Link.last_probe` (from grbl's ``[PRB:…]`` line)."""
        with self.lock:
            return None if self.probe is None else {"n": self.probe["n"], "pos": dict(self.probe["pos"]),
                                                     "ok": self.probe["ok"]}

    def _status(self, line: str) -> None:
        """Updates ``self.st`` with a ``<state|field:value|…>`` report (see STATUS).

        grbl 1.1 fields that are used:
            MPos / WPos  machine or work position (depending on ``$10``); the machine one is always
                         stored: MPos = WPos + WCO.
            WCO          active work origin. grbl sends it only every 10–30 reports (to save
                         bandwidth), so the last one is remembered.
            FS / F       actual feed (mm/min) and spindle rpm; «F» only if there is no variable spindle.
            Ov           overrides in % (feed, rapid, spindle); it also arrives only now and then.
            A            accessories on: S = spindle clockwise, C = counter-clockwise, F = flood,
                         M = mist. grbl omits it if nothing is on, but always sends it together
                         with Ov, so «Ov without A» means «everything off».
            Pn           active inputs (X/Y/Z limit switches, P probe, D door, H hold…).
        """
        m = STATUS.match(line)
        if not m:
            return
        self.st["estado"] = m.group(1).split(":")[0]       # «Hold:0» → «Hold»
        fields = dict(f.split(":", 1) for f in m.group(2).split("|") if ":" in f)
        vec = lambda s: {a: float(v) for a, v in zip(self.axes, s.split(","))}
        if "WCO" in fields:
            self.wco = vec(fields["WCO"])
        if "MPos" in fields:
            self.st["mpos"] = vec(fields["MPos"])
        elif "WPos" in fields:
            w = vec(fields["WPos"])
            self.st["mpos"] = {a: w[a] + self.wco.get(a, 0.0) for a in w}
        fs = fields.get("FS") or fields.get("F")
        if fs:
            p = fs.split(",")
            self.st["avance"] = float(p[0])
            self.st["husillo"] = float(p[1]) if len(p) > 1 else 0.0
        if "Ov" in fields:
            f, r, s = (int(x) for x in fields["Ov"].split(","))
            self.st["ov"] = {"avance": f, "rapido": r, "husillo": s}
        if "Ov" in fields or "A" in fields:
            acc = fields.get("A", "")
            self.st["husillo_dir"] = "cw" if "S" in acc else "ccw" if "C" in acc else "off"
            self.st["refrigerante"] = "F" in acc or "M" in acc
        self.st["pines"] = fields.get("Pn", "")

    def _fill(self) -> None:
        """Sends program lines while they fit in grbl's buffer (character counting).

        Called with ``lock`` held: on start, after each ``ok`` and on resume. Each line takes
        ``len + 1`` bytes (the newline counts). In single block, besides, only one line is sent per
        «token» (``_tokens``, one per Cycle Start) and only with the buffer empty, so the machine
        really stops between one line and the next.
        """
        while self.job["enviadas"] < self.job["total"]:
            if self.single:                        # single block: one line per Cycle Start
                if self.pending or self._tokens <= 0:
                    break
                self._tokens -= 1
            s = self.lines[self.job["enviadas"]]
            if sum(self.pending) + len(s) + 1 > self.rx:   # does not fit: wait for the next ok
                break
            self._write((s + "\n").encode("ascii"))
            self.pending.append(len(s) + 1)
            self.job["enviadas"] += 1

    def _command(self, line: str, timeout: float = 5.0) -> list[str]:
        """Sends a loose command and waits for its ``ok``. Returns the lines that arrived before the ok.

        line:    command without newline (``$H``, ``$#``, ``G0 X0``…).
        timeout: maximum seconds to wait.
        Raises BackendError if a program is running (its ``ok`` would be mixed up with the program's),
        if another command is already waiting, if it does not reply in time or if it replies with
        ``error:N``/``ALARM:N``. The wait is done without ``lock`` so the reader can reply.
        """
        with self.lock:
            if self.job["activo"]:
                raise BackendError("hay un programa en marcha")
            if self.waiter is not None:
                raise BackendError("el controlador está atendiendo otro comando")
            self.waiter = {"event": threading.Event(), "lines": [], "error": None}
            w = self.waiter
            self._write((line + "\n").encode("ascii"))
        try:
            if not w["event"].wait(timeout):
                raise BackendError(f"sin respuesta a «{line}»")
            if w["error"]:
                raise BackendError(w["error"])
            return w["lines"]
        finally:
            with self.lock:
                self.waiter = None

    # --- interface --------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """See :meth:`Link.snapshot`. Copies ``self.st`` (the inner dicts too) plus the job.

        ``wcs`` is always reported as G54: grbl does not say which system is active in the status
        report (``$G`` would have to be requested), but it does give its offset (``wco``), which is what matters.
        """
        with self.lock:
            return {**{k: (dict(v) if isinstance(v, dict) else v) for k, v in self.st.items()},
                    "wco": dict(self.wco), "referenciado": self.homed, "trabajo": dict(self.job),
                    "mensaje": self.message, "alarma": self.alarm, "wcs": "G54", "conexion": self.banner or self.name,
                    "pausa_g4": False, "bloque_a_bloque": self.single}

    def offsets(self) -> dict[str, dict[str, float]]:
        """Requests ``$#`` and returns G54…G59, G28, G30 and G92 (mm). The lines have the form
        ``[G54:0.000,0.000,0.000]``; the others (``[TLO:…]``, ``[PRB:…]``) are ignored.
        Raises BackendError if a program is running or it does not reply (see :meth:`_command`)."""
        out = {}
        for line in self._command("$#"):
            m = re.match(r"\[(G5[4-9]|G28|G30|G92):([^\]]+)\]", line)
            if m:
                out[m.group(1)] = {a: float(v) for a, v in zip(self.axes, m.group(2).split(","))}
        return out

    def start(self, lines: list[str]) -> None:
        """See :meth:`Link.start`. Cleans the lines and starts sending with character counting.

        Cleaning: ``( … )`` and ``; …`` comments are removed, spaces too (grbl ignores them and this
        way more lines fit in the buffer), and everything is upper-cased. A line that does not fit in
        the buffer by itself could never be sent, so the whole program is rejected before moving anything.
        Raises BackendError if another program is running or the machine is not Idle (Alarm, Hold, Jog…).
        """
        clean = []
        for ln in lines:
            s = re.sub(r"\(.*?\)|;.*", "", ln).strip().upper().replace(" ", "")
            if s:
                if len(s) + 1 > self.rx:
                    raise BackendError(f"línea más larga que el búfer del controlador: {s[:40]}…")
                clean.append(s)
        with self.lock:
            if self.job["activo"]:
                raise BackendError("ya hay un programa en marcha")
            if self.st["estado"] not in ("Idle",):
                raise BackendError(f"la máquina no está en reposo ({self.st['estado']})")
            self.lines = clean
            self.pending.clear()
            self.job = _job(len(clean))
            self.job["activo"] = bool(clean)
            self.job["terminado"] = not clean
            self.message = ""
            self._tokens = 1                       # single block: the Cycle Start that started it is worth one line
            self._fill()

    def hold(self) -> None:
        """Feed Hold: real-time byte ``!``. grbl decelerates with its acceleration and does not lose position."""
        self._write(b"!")

    def set_machine(self, machine: Machine) -> None:
        """Changes the profile (the reported axes remain those of the profile it was opened with)."""
        self.m = machine

    def set_single(self, on: bool) -> None:
        """See :meth:`Link.set_single`."""
        with self.lock:
            self.single = bool(on)
            if not self.single and self.job["activo"]:
                self._fill()                       # turning single block off resumes continuous sending

    def resume(self) -> None:
        """Cycle Start: byte ``~`` (leaves Hold). In single block, besides, it enables the next line."""
        self.message = ""
        self._write(b"~")
        with self.lock:
            if self.single and self.job["activo"]:
                self._tokens += 1
                self._fill()

    def stop(self) -> None:
        """Stop: hold (``!``), wait for it to decelerate and reset grbl (Ctrl-X = 0x18).

        The reset discards the buffer and the planner, which is what cuts the program. If it were
        reset while moving, grbl would not know where the machine ended up (ALARM:3, homing needed);
        that is why the hold comes first, with a 0.25 s wait. With very high feeds and low acceleration
        the machine may take longer to stop, and then alarm 3 does appear.
        """
        self._write(b"!")
        time.sleep(0.25)                                   # decelerate before resetting: position is not lost
        with self.lock:
            if self.job["activo"]:
                self.job.update(activo=False, error="detenido por el operador")
            self.pending.clear()
        self._write(b"\x18")

    def jog(self, moves: dict[str, float], feed: float) -> None:
        """Jog with ``$J=G91 G21 X… F…`` (incremental, in mm, without touching the program's modal state).

        grbl checks the soft limits and replies ``error:15`` if the jog goes beyond the travel (it
        arrives as BackendError). The ``ok`` arrives when the motion is accepted, not when it ends, so
        it does not block during the jog. Axes with a near-zero displacement are omitted.
        """
        axes = " ".join(f"{a}{v:.4f}" for a, v in moves.items() if abs(v) > 1e-9)
        if axes:
            self._command(f"$J=G91 G21 {axes} F{feed:.0f}")

    def jog_cancel(self) -> None:
        """Cuts the jog with the real-time byte 0x85 (decelerates with a ramp, no alarm)."""
        self._write(bytes([JOG_CANCEL]))

    def home(self) -> None:
        """Homing ``$H``. grbl replies ``ok`` only when the cycle ends: waits up to 180 s
        (large, slow machines). Raises BackendError if it fails (ALARM:6…9) or does not reply."""
        self._command("$H", timeout=180.0)
        self.homed = True

    def unlock(self) -> None:
        """``$X``: clears the alarm lock without homing (the position may not be reliable)."""
        self._command("$X")
        self.alarm = ""

    def mdi(self, line: str) -> None:
        """See :meth:`Link.mdi`. Removes comments; waits for the ``ok`` up to 30 s."""
        s = re.sub(r"\(.*?\)|;.*", "", line).strip()
        if s:
            self._command(s, timeout=30.0)

    def override(self, kind: str, action: str) -> None:
        """Sends the real-time byte from OVERRIDES. The new value shows up in the next report (Ov)."""
        b = OVERRIDES.get((kind, action))
        if b is None:
            raise BackendError(f"ajuste desconocido: {kind} {action}")
        self._write(bytes([b]))

    def coolant_toggle(self) -> None:
        """Toggles flood coolant with the real-time byte 0xA0."""
        self._write(bytes([FLOOD_TOGGLE]))


# ==================================================================================================
# ======================== MARLIN / SMOOTHIEWARE / REPRAPFIRMWARE (OkLink) =========================
# ==================================================================================================

# Position in the reply to M114, e.g. (Marlin):
#     X:10.00 Y:5.00 Z:-1.00 E:0.00 Count X:800 Y:400 Z:-400
# Captures (axis, value in mm). What comes after «Count» are motor steps and is discarded beforehand.
POS = re.compile(r"\b([XYZABC]):\s*(-?\d+(?:\.\d+)?)")

# Program pause on an «ok» controller (Marlin and friends): M0, M00, M1 or M01 followed by a
# space, text or end of line. The text that follows is the message for the operator.
PAUSE_CODE = re.compile(r"\s*[Mm]0*[01](?![0-9.])")


class OkLink(Link):
    """Marlin, Smoothieware, RepRapFirmware and any controller that replies «ok» per line (MPCNC,
    Lowrider, routers with 3D printer boards). Sends line by line and requests M114 for the position.

    Differences from grbl (firmware limitations, not the app's): there is no immediate hold (Feed Hold
    stops sending and the machine finishes what it has in its buffer), Stop uses M410 (needs
    EMERGENCY_PARSER in Marlin), M0 waits for M108 to continue, and feed is adjusted with M220.

    PROTOCOL (why it is done this way):

    * **One line at a time.** These firmwares do not offer reliable character counting: the serial
      buffer size varies with the board and the build, and Marlin does not separate real-time
      commands. The safe way is what 3D printing programs do: send a line and wait for its ``ok``.
      The firmware's planner stays full because the ``ok`` of a motion arrives when it enters the
      queue, not when it ends.
    * **A single «turn».** Since the ``ok`` reply does not say which line it belongs to, there is
      never more than one line in flight; ``inflight`` remembers whose it is: ``"job"`` (program),
      ``"poll"`` (M114 from the polling thread), ``"cmd"`` (loose command with ``waiter``) or
      ``"extra"`` (M220/M8/M9 requested while running, queued in ``extra`` and interleaved between
      program lines).
    * **M108 / M410 out of turn.** The firmware is blocked running M0 or with a full queue, and a
      normal command would not be read until it frees up. Marlin built with EMERGENCY_PARSER
      recognizes them as soon as they arrive at the port (like grbl's real-time bytes); without that
      option M108 does not wake an M0 and M410 only acts when the queue empties.

    Parameters:
        port:      port object (``write``/``readline``/``close``), see ``backends.grbl.open_port``.
        machine:   machine profile (axis names).
        poll:      position polling period (M114), in seconds. 0.5 s by default: each M114 takes the
                   turn and delays the next program line, so polling more often makes the program
                   slower (and stutters on short segments); polling less often makes the on-screen
                   position update in jumps.
        autostart: open (start threads) already in the constructor.
    """

    name = "ok"

    def __init__(self, port, machine: Machine, poll: float = 0.5, autostart: bool = True):
        self.port = port
        self.m = machine
        self.axes = machine.axis_names
        self.poll = poll
        self.wlock = threading.Lock()                 # one port write at a time
        self.lock = threading.RLock()                 # state shared between reader, polling and interface
        self.pos = {a: 0.0 for a in self.axes}        # last M114 position, mm (see note in handle)
        self.state = "Desconectado"                   # becomes Idle with the first line that arrives
        self.homed = False
        self.ov = {"avance": 100, "rapido": 100, "husillo": 100}   # only «avance» is sent (M220)
        self.spindle_on = False                       # deduced from the M3/M4/M5 sent through MDI
        self.coolant = False
        self.message = ""
        self.alarm = ""
        self.banner = ""
        self.lines: list[str] = []
        self.job = _job()
        self.inflight: str | None = None          # "job" | "poll" | "cmd" | "extra": who awaits the next ok
        self.extra: deque[str] = deque()          # M220, M8/M9…: go in the next gap, without breaking the turn
        self.waiter: dict | None = None               # loose command: {"event": Event, "error": text|None}
        self.held = False                             # held: no more program lines are sent
        self.log: deque[str] = deque(maxlen=200)      # latest replies (without the M114 ones)
        self._run = False
        if autostart:
            self.open()

    @classmethod
    def from_config(cls, cfg: dict, machine: Machine) -> OkLink:
        """Creates the link from the ``controlador`` section of the profile: ``puerto`` (required) and
        ``baudios`` (115200 by default; Marlin usually uses 115200 or 250000). Raises BackendError if
        the port is missing or cannot be opened."""
        if not cfg.get("puerto"):
            raise BackendError("controlador sin «puerto» (p. ej. /dev/ttyUSB0, COM3 o tcp://192.168.1.50:23)")
        return cls(open_port(str(cfg["puerto"]), int(cfg.get("baudios", 115200))), machine)

    # --- threads ----------------------------------------------------------------------------------
    def open(self) -> None:
        """Starts the reader and polling threads (daemon). Idempotent."""
        if self._run:
            return
        self._run = True
        threading.Thread(target=self._reader, daemon=True, name="hope-cnc-ok-rx").start()
        threading.Thread(target=self._poller, daemon=True, name="hope-cnc-ok-poll").start()

    def close(self) -> None:
        """Stops the threads and closes the port (an error while closing is only written to the log)."""
        self._run = False
        try:
            self.port.close()
        except Exception as e:                             # noqa: BLE001
            self.log.append(f"al cerrar: {e}")

    def set_machine(self, machine: Machine) -> None:
        """Changes the profile (the reported axes remain those of the profile it was opened with)."""
        self.m = machine

    def _write(self, line: str) -> None:
        """Writes a line (adds the newline) holding ``wlock``. Non-ASCII characters become «?»."""
        with self.wlock:
            self.port.write((line + "\n").encode("ascii", errors="replace"))

    def _poller(self):
        """Polling thread: every ``poll`` s, if the turn is free, sends what is queued or an M114.

        It also works as a «pump»: if an ``extra`` was left waiting because the turn was busy, it is
        sent here. It does not request M114 while a loose command awaits its turn (that would give the
        M114 priority).
        """
        while self._run:
            time.sleep(self.poll)
            with self.lock:
                if self.inflight is None and self.extra:
                    self._pump()
                elif self.inflight is None and self.waiter is None:
                    self.inflight = "poll"
                    try:
                        self._write("M114")
                    except Exception as e:                 # noqa: BLE001
                        self.alarm, self.state = f"conexión perdida: {e}", "Desconectado"
                        return

    def _reader(self):
        """Reader thread: reads line by line and hands them to :meth:`handle` (same as in GrblLink)."""
        while self._run:
            try:
                raw = self.port.readline()
            except Exception as e:                         # noqa: BLE001
                self.alarm, self.state = f"conexión perdida: {e}", "Desconectado"
                return
            if not raw:
                time.sleep(0.002)
                continue
            self.handle(raw.decode("ascii", errors="replace").strip())

    # --- protocol ---------------------------------------------------------------------------------
    def handle(self, line: str) -> None:
        """Processes a received line (called from the reader thread; public for the tests).

        Line kinds:
            ``X:… Y:… Z:…``   reply to M114 → ``self.pos``. Beware: Marlin reports the target position
                              of the last queued move, which may be ahead of the real one.
            ``ok…``           closes the turn (``inflight``) and sends what comes next (``_pump``).
            ``Error…``/``!!`` line rejected or firmware halted (Smoothie «!!», Marlin «kill()»).
            ``start``/banner  the firmware started or was reset: the running program was lost.
            ``echo:…``        messages for the operator; ``echo:busy: processing`` (Marlin's notice
                              that it is still working on a long command) is ignored.
        """
        if not line:
            return
        low = line.lower()
        with self.lock:
            if self.state == "Desconectado":
                self.state = "Idle"                        # something arrived: the controller is alive
            if "x:" in low and "y:" in low and not low.startswith("ok"):
                for a, v in POS.findall(re.split(r"count", line, flags=re.IGNORECASE)[0]):   # «Count X:…» are steps
                    if a in self.pos:
                        self.pos[a] = float(v)
                return
            self.log.append(line)
            if low.startswith("ok"):                       # «ok», «ok T:…», «ok N12» depending on the firmware
                kind, self.inflight = self.inflight, None
                if kind == "job":
                    self.job["confirmadas"] += 1
                    if self.job["confirmadas"] >= self.job["total"]:
                        self.job.update(activo=False, terminado=True)
                        self.state = "Idle"
                    self._next()
                elif kind == "cmd" and self.waiter is not None:
                    self.waiter["event"].set()
                self._pump()                               # after an M114 or an M220 the program continues
            elif low.startswith(("error", "!!")):
                msg = line
                if self.inflight == "job":
                    # Error on a program line: sending is cut (the firmware does not go on by itself).
                    self.inflight = None
                    self.job.update(activo=False, error=f"línea {self.job['confirmadas'] + 1}: {msg}")
                    self.state = "Idle"
                elif self.inflight in ("poll", "extra"):
                    # An M114 or an M220/M8 failed: not serious, the operator is told and the program goes on.
                    self.inflight = None
                    self.message = msg
                    self._pump()
                elif self.waiter is not None:
                    self.waiter["error"] = msg
                    self.waiter["event"].set()
                    self.inflight = None
                # «!!» or «kill»: the firmware is halted and asks for M999 (see unlock); shown as an alarm.
                self.alarm = msg if low.startswith("!!") or "kill" in low else self.alarm
            elif low.startswith(("start", "marlin", "smoothie", "reprapfirmware", "fw_name")):
                self.banner = line
                if self.job["activo"]:
                    self.job.update(activo=False, error="el controlador se reinició")
            elif low.startswith("echo:") and "busy" not in low:
                self.message = line[5:].strip()

    def _pump(self) -> None:
        """With the lock held: if nothing awaits an ok, sends the queued commands first and then the program."""
        if self.inflight is not None:
            return
        if self.extra and not self.held:
            self.inflight = "extra"
            self._write(self.extra.popleft())
            return
        self._next()

    def _send_soon(self, line: str) -> None:
        """Queues ``line`` in ``extra`` for the next free turn (sends it right away if the turn is free)."""
        with self.lock:
            self.extra.append(line)
            self._pump()

    def _next(self) -> None:
        """Sends the next program line (one at a time: the controller acknowledges each one).

        Called with ``lock`` held. If the line is M0/M1 (program pause), the hold is set before
        sending it: the firmware does not reply ``ok`` until the operator continues (M108, see
        :meth:`resume`), and the text following the M0 is shown as a message.
        """
        if not self.job["activo"] or self.held or self.inflight is not None:
            return
        i = self.job["enviadas"]
        if i >= self.job["total"]:
            return
        s = self.lines[i]
        self.job["enviadas"] += 1
        self.inflight = "job"
        # M0/M00/M1/M01 followed by something that is not a digit: this way M03/M05 (with a leading
        # zero, as some CAMs write them) or M104/M106 are NOT taken for a pause.
        pausa = PAUSE_CODE.match(s)
        if pausa:
            self.held, self.state, self.message = True, "Hold", s[pausa.end():].strip() or "pausa"
        self._write(s)

    def _command(self, line: str, timeout: float = 10.0) -> None:
        """Sends a loose command in the next free turn and waits for its ``ok``.

        line:    command without newline.
        timeout: maximum seconds to wait for the ``ok`` (once sent).
        First it waits up to 2 s for the turn to be freed (normally taken by an M114 from the polling
        thread, which replies right away). Raises BackendError if a program is running, if the turn
        is not freed («el controlador no responde»), if the ok does not arrive in time or if an error arrives.
        """
        with self.lock:
            if self.job["activo"]:
                raise BackendError("hay un programa en marcha")
            self.waiter = {"event": threading.Event(), "error": None}
            w = self.waiter
        deadline = time.monotonic() + 2.0
        while True:                                        # wait for the turn to be freed (poll)
            with self.lock:
                if self.inflight is None:
                    self.inflight = "cmd"
                    self._write(line)
                    break
            if time.monotonic() > deadline:
                with self.lock:
                    self.waiter = None
                raise BackendError("el controlador no responde")
            time.sleep(0.01)
        try:
            if not w["event"].wait(timeout):
                raise BackendError(f"sin respuesta a «{line}»")
            if w["error"]:
                raise BackendError(w["error"])
        finally:
            with self.lock:
                self.waiter = None

    # --- interface --------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """See :meth:`Link.snapshot`. These firmwares do not report actual feed, rpm or work origin
        in the position query: those fields are 0 and the position is M114's."""
        with self.lock:
            return {"estado": self.state, "mpos": dict(self.pos), "wco": {a: 0.0 for a in self.pos}, "avance": 0.0,
                    "husillo": 0.0, "husillo_dir": "cw" if self.spindle_on else "off", "refrigerante": self.coolant,
                    "ov": dict(self.ov), "referenciado": self.homed, "trabajo": dict(self.job),
                    "mensaje": self.message, "alarma": self.alarm, "wcs": "G54", "conexion": self.banner or self.name,
                    "pausa_g4": False, "bloque_a_bloque": self.single}

    def start(self, lines: list[str]) -> None:
        """See :meth:`Link.start`. Removes comments (if the line was only a comment it is sent as is,
        and an empty one is dropped) and sends the first line; the rest go one by one with each ``ok``.
        Raises BackendError if a program is already running."""
        clean = [re.sub(r"\(.*?\)|;.*", "", ln).strip() or ln.strip() for ln in lines]
        clean = [c for c in clean if c]
        with self.lock:
            if self.job["activo"]:
                raise BackendError("ya hay un programa en marcha")
            self.lines, self.job, self.held = clean, _job(len(clean)), False
            self.job["activo"] = bool(clean)
            self.job["terminado"] = not clean
            self.state = "Run" if clean else "Idle"
            self.message = ""
            self._next()

    def hold(self) -> None:
        """«Soft» Feed Hold: stops sending lines. The machine finishes what it already has in its queue
        (Marlin has no immediate hold over serial), so it may keep moving for a while."""
        with self.lock:
            if self.job["activo"]:
                self.held, self.state = True, "Hold"

    def resume(self) -> None:
        """Cycle Start: resumes sending lines. If the hold was an M0/M1 (there is a message and the M0
        line is still awaiting its ok), sends M108 out of turn so the firmware leaves the pause."""
        with self.lock:
            was_m0 = self.message and self.held
            self.held, self.message = False, ""
            if self.job["activo"]:
                self.state = "Run"
                if was_m0 and self.inflight == "job":
                    self._write("M108")                    # Marlin: continue after M0
                if self.single:
                    self.held = False
                self._next()

    def stop(self) -> None:
        """Stop: cuts sending and sends, out of turn, M410 (quick stop: empties the motion queue) and
        M5 (spindle off). Without EMERGENCY_PARSER, Marlin only reads M410 when room frees up in its
        buffer, and the machine may lose its position (homing is advisable)."""
        with self.lock:
            if self.job["activo"]:
                self.job.update(activo=False, error="detenido por el operador")
            self.held, self.state = False, "Idle"
            self._write("M410")                            # quick stop (Marlin with EMERGENCY_PARSER)
            self._write("M5")

    def jog(self, moves: dict[str, float], feed: float) -> None:
        """Jog with ``G91`` + ``G1 X… F…`` + ``G90`` (relative and back to absolute, even if the G1 fails).

        These firmwares have no ``$J`` nor jog cancel: each step is a complete G1. The firmware
        does not check the travels unless it has soft endstops enabled.
        """
        axes = " ".join(f"{a}{v:.4f}" for a, v in moves.items() if abs(v) > 1e-9)
        if axes:
            self._command("G91")
            try:
                self._command(f"G1 {axes} F{feed:.0f}", timeout=60.0)
            finally:
                self._command("G90")

    def home(self) -> None:
        """Homing ``G28``. The firmware replies ``ok`` when it finishes: waits up to 180 s."""
        self._command("G28", timeout=180.0)
        self.homed = True

    def unlock(self) -> None:
        """``M999``: brings the firmware out of the halted state (after a «!!» or a kill) without resetting it."""
        self._command("M999")
        self.alarm = ""

    def mdi(self, line: str) -> None:
        """See :meth:`Link.mdi`. Waits for the ok up to 60 s and remembers whether the spindle was left on
        (M3/M4/M5), because the firmware does not report it."""
        s = re.sub(r"\(.*?\)|;.*", "", line).strip()
        if s:
            self._command(s, timeout=60.0)
            up = s.upper()
            if up.startswith(("M3", "M4")):
                self.spindle_on = True
            elif up.startswith("M5"):
                self.spindle_on = False

    def override(self, kind: str, action: str) -> None:
        """Feed adjustment with ``M220 S<percentage>`` (10–300 %, the range Marlin accepts without
        surprises). Rapids and spindle have no equivalent command: raises BackendError."""
        if kind != "avance":
            raise BackendError("este controlador solo ajusta el avance (M220)")
        pct = 100 if action == "100" else max(10, min(300, self.ov["avance"] + int(action)))
        self.ov["avance"] = pct
        self._send_soon(f"M220 S{pct}")                    # in the next gap: never two lines in flight

    def coolant_toggle(self) -> None:
        """Toggles M8/M9 (in Marlin it requires COOLANT_FLOOD); sent in the next gap of the program."""
        self.coolant = not self.coolant
        self._send_soon("M8" if self.coolant else "M9")


# ==================================================================================================
# ============================================ FACTORY =============================================
# ==================================================================================================

def make_link(machine: Machine, cfg: dict | None = None) -> Link:
    """Creates the live link that matches the profile's controller.

    machine: machine profile.
    cfg:     ``controlador`` section (if None, ``machine.controller`` is used). Key ``tipo``:
             ``sim`` (default; ``velocidad`` = clock multiplier), ``grbl``/``grblhal``/
             ``fluidnc`` (see GrblLink.from_config) or ``ok``/``marlin``/``smoothie``/``reprap``/``rrf``
             (see OkLink.from_config).
    Raises BackendError if the type is not handled live (e.g. LinuxCNC, which goes in batch) or if the
    port cannot be opened.
    """
    cfg = cfg if cfg is not None else machine.controller
    kind = str(cfg.get("tipo", "sim")).lower()
    if kind == "sim":
        return SimLink(machine, speed=float(cfg.get("velocidad", 1.0)))
    if kind in ("grbl", "grblhal", "fluidnc"):
        return GrblLink.from_config(cfg, machine)
    if kind in ("ok", "marlin", "smoothie", "smoothieware", "reprap", "rrf"):
        return OkLink.from_config(cfg, machine)
    raise BackendError(f"la app todavía no maneja el controlador «{kind}» en vivo (sim, grbl u ok); "
                       "para LinuxCNC usar «python -m hope.cnc enviar»")
