"""Hope CNC app session: the Mach3 «control panel» behind the screen.

Ties together the machine, the live connection (:mod:`hope.cnc.links`), the loaded program (checked
and translated) and the guardian. Usage flow, as in Mach3:

1. On opening, the machine is in **RESET** (disabled): the button blinks. Pressing it enables the
   machine for the operator (jog, MDI, homing, zeroing). On a real machine that is
   ``hardware.control`` in manual mode: the guardian asks for approval and the operator grants it on
   the same screen (it goes into the audit log).
2. **Load G-code** → it is checked (errors, limits, time) and the toolpath is drawn.
3. **Cycle Start** → it is checked again with the controller's real offsets and, on a real machine,
   the guardian asks for approval of THAT program (hash + envelope). While running:
   Feed Hold, Stop and the feed/spindle overrides.

Pressing RESET again (or Esc) stops everything and disables the machine.

What it is
----------
:class:`CncSession` is the state of ONE TISMA CNC screen (the modern replacement for Mach3): which
machine is selected, whether it is enabled, which program is loaded, which approval is waiting for
the operator, the message log, the MDI history, the LED buttons (Single BLK, Block Delete, M1),
«Set Next Line» / «Run From Here», the recent files and the slicer-style profiles
(machine + material + consumable + process). Every screen button is a method of this class.

How it fits in
--------------
* **Used by**: :mod:`hope.cnc.app.server` (the screen's HTTP server). Each browser request calls a
  method (``reset``, ``load``, ``cycle_start``, ``jog``, ``mdi``…) and returns its ``dict`` as
  JSON; the screen requests ``state()`` several times per second.
  Errors for the operator come out as :class:`SessionError` (the server shows them as-is).
* **Uses**:

  - :mod:`hope.cnc.links` (``Link``): the connection to the controller (simulator, grbl, Marlin…):
    ``start``, ``hold``, ``resume``, ``stop``, ``jog``, ``mdi``, ``snapshot`` (live state).
  - :mod:`hope.cnc.gcode`: interprets the G-code (``interpret``) and keeps the modal state (``Interpreter``).
  - :mod:`hope.cnc.planner` and :mod:`hope.cnc.check`: motion plan (times) and checking
    (errors, limits, envelope).
  - :mod:`hope.cnc.dialect`: translates the program into the controller's dialect (``emit``).
  - :mod:`hope.cnc.profiles`: slicer-style profiles (imported lazily, only if enabled).
  - ``guardian`` (optional): asks for human approval before moving a real machine.

Conventions
-----------
* Units: mm, mm/min, seconds, RPM (like the rest of :mod:`hope.cnc`).
* «Original lines» = lines of the user's G-code file (1, 2, 3…; 0 = before starting).
  «Emitted lines» = what is sent to the controller, already translated (``emitidas[0]`` is the
  ``G21 G90 …`` header). Each emitted line records which original line it came from; an arc or a
  circle may produce several emitted lines. «Sent lines» = what was actually sent in this Cycle
  Start (it may have a «Run From Here» preamble at the beginning); ``run_map`` maps them back to
  the emitted lines to show progress and remaining time.
* Safety: with a physical connection (``link.physical``) NOTHING moves without the guardian, and the
  guardian requires a person to approve on the screen (manual mode when pressing RESET; each program
  —text hash + envelope— when pressing Cycle Start). The simulator asks for no approval.
* Concurrency: the HTTP server handles requests in threads; methods that change several things
  together (load, start, approve, profiles) take ``self.lock`` (an ``RLock``, reentrant: that is why
  ``cycle_start`` can call ``load`` inside the lock). Single-step buttons (Feed Hold, Stop, jog)
  deliberately do NOT take it: a Feed Hold must not wait for a load to finish.
* User-facing texts and the keys of the returned ``dict`` are in Spanish (the screen reads them).

Usage example
-------------
::

    from hope.cnc.links import SimLink
    from hope.cnc.session import CncSession

    s = CncSession(maquina, SimLink(maquina))        # simulator: no approval asked
    s.reset()                                        # enables (on a real one returns {"aprobar": …})
    s.load("tapa.nc", open("tapa.nc").read())        # checks, plans and translates
    s.set_next_line(120, from_here=True)             # optional: «Run From Here» from line 120
    s.cycle_start()                                  # starts
    s.state()["programa"]["tiempo_restante"]         # seconds remaining
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from hope.cnc.backends.base import BackendError
from hope.cnc.check import check_program, report_text
from hope.cnc.dialect import HEADER, HEADERS, dialect_for, emit
from hope.cnc.gcode import Interpreter, Program, interpret, linearize, modal_text
from hope.cnc.links import Link
from hope.cnc.machine import ROTARY, WCS, Machine, as_controller
from hope.cnc.planner import plan

# ==================================================================================================
# ======================================== CONFIGURATION ===========================================
# ==================================================================================================

# Maximum number of toolpath points sent to the 3D view (points, not lines: an arc is split into
# several). Past that number the drawing is cut and flagged with «recortado». Higher = the whole
# program is shown even if huge, but the JSON grows (~40 bytes per point) and an old browser
# (Firefox 52 on an XP PC) chokes drawing it. 250 000 ≈ 10 MB, still manageable.
MAX_PATH_POINTS = 250_000

# Brand name shown on the screen and in messages.
BRAND = "TISMA CNC"

# Incremental jog steps offered by the screen (mm), like Mach3's «Step». The chosen step reaches
# ``CncSession.jog`` as ``step``; without a step (None) the jog is continuous while the button is
# held. Adding values only adds buttons; 10 mm is already a lot to press without looking.
JOG_STEPS = (0.001, 0.01, 0.1, 1.0, 10.0)


# Error for the operator: the message (in Spanish) is shown as-is on the screen. Raised when a button
# cannot run right now (machine in RESET, program running, no program, MDI line with errors,
# approval denied…). It does not signal a program bug but an action that does not fit the current
# state. Inherits from ValueError so the server treats it as an invalid request (not as an internal
# failure).
class SessionError(ValueError):
    pass


# ==================================================================================================
# ======================================= CUMULATIVE TIMES =========================================
# ==================================================================================================

def _view_point_function(m: Machine, g54: dict):
    """Returns a function mapping a machine position ({axis: value}) to view [x, y, z], in G54 (mm).

    With rotary axes the view shows the WORKPIECE, not the machine: the point where the tool cuts is
    rotated by the opposite of the axis angle, as if the workpiece stood still. That way an engraving
    around a cylinder (4th axis) appears wrapped on the cylinder, as it will turn out.

    RS274/NGC convention: A rotates about X, B about Y and C about Z (right-hand rule).
    Rotating (u, v) by an angle θ: u' = u·cos θ − v·sin θ, v' = u·sin θ + v·cos θ; here θ = −angle.
    For A the rotating plane is (Y, Z); for B, (Z, X); for C, (X, Y). Applied in order A, B, C.
    The rotation axis passes through the work zero: 4-axis CAMs put Y0 Z0 on the indexer's axis."""
    rot = [a for a in m.axis_names if a in ROTARY]
    gx, gy, gz = g54.get("X", 0.0), g54.get("Y", 0.0), g54.get("Z", 0.0)

    def view(q: dict) -> list:
        x, y, z = q.get("X", 0.0) - gx, q.get("Y", 0.0) - gy, q.get("Z", 0.0) - gz
        for a in rot:
            ang = -math.radians(q.get(a, 0.0) - g54.get(a, 0.0))
            c, s = math.cos(ang), math.sin(ang)
            if a == "A":
                y, z = y * c - z * s, y * s + z * c
            elif a == "B":
                z, x = z * c - x * s, z * s + x * c
            else:
                x, y = x * c - y * s, x * s + y * c
        return [round(x, 4), round(y, 4), round(z, 4)]
    return view


def _cumulative_times(prog, pl, emitted: list, origins: list[int]) -> list[float]:
    """Cumulative time (s) by confirmed lines: ``cum[k]`` = seconds taken by the first k emitted
    lines (``cum[0]`` = 0, ``len(cum) = len(emitted) + 1``). Gives the remaining time and the live progress bar.

    The motion time of each operation (from the plan) is split among the motion lines it emitted
    (a full circle comes out as several G1); G4 dwells add their duration on their own line.

    Parameters:
        prog: interpreted ``Program`` (``prog.ops`` = operations).
        pl: plan from :func:`hope.cnc.planner.plan`; ``pl.time_by_op[k]`` = motion seconds of
            operation k (with accelerations).
        emitted: emitted lines ``[(text, original_line), …]`` (line 0 is the header).
        origins: for each emitted line, index of the operation that produced it (-1 = header), as
            recorded by :func:`hope.cnc.dialect.emit`.

    Returns the ``cum`` list (s). Why this way: the controller confirms LINES (grbl's «ok»), not
    operations; with this table progress is a simple ``cum[confirmed]`` lookup on every
    ``state()``, without walking the program (with 100 000 lines it matters on an old PC)."""
    cum = [0.0]
    if not emitted:
        return cum
    # First pass: how many MOTION lines each operation emitted, to split its time.
    moves_per_op: dict[int, int] = {}
    for (t, _), k in zip(emitted, origins):
        if t[:2] in ("G0", "G1", "G2", "G3") or t.startswith(("G18 G", "G19 G")):    # G38.2 matches via «G3»
            moves_per_op[k] = moves_per_op.get(k, 0) + 1
    ops = prog.ops
    acc = 0.0
    # Second pass: add up. Lines that are neither motion nor G4 (M3, F, G10…) add no time.
    for (t, _), k in zip(emitted, origins):       # cum[k + 1] = up to line k inclusive (0 = header)
        n = moves_per_op.get(k)
        if n:
            acc += pl.time_by_op.get(k, 0.0) / n
        elif k >= 0 and ops[k].kind == "dwell" and t.startswith("G4"):
            acc += float(ops[k].value)
        cum.append(acc)
    return cum


# ==================================================================================================
# ============================================ SESSION =============================================
# ==================================================================================================

class CncSession:
    """State of one TISMA CNC screen and the logic of every button (see the module header).

    Main attributes (the rest are explained in ``__init__``):
        m: EFFECTIVE machine in use (with profiles: machine + material + consumable + process).
        link: connection to the controller (``Link``).
        enabled: False = RESET (nothing can be moved by hand and no program can be started).
        program: loaded program (dict, see :meth:`load`) or None.
        pending: guardian approval waiting for the operator, or None.
        lock: ``RLock`` that serializes compound operations (see «Concurrency» in the module).
    """

    def __init__(self, machine: Machine, link: Link, guardian=None, machine_path: str = "",
                 recent_dir: str | None = None, profiles=None, selection: dict | None = None, link_factory=None):
        """``profiles`` (ProfileStore) and ``selection`` enable the slicer-style profiles: the effective
        machine comes from machine + material + consumable + process. ``link_factory(machine)`` creates
        the connection when another machine with a different controller is chosen.

        Parameters:
            machine: configured machine (if there are profiles, it is replaced by the effective one).
            link: connection already open to the controller (or the simulator).
            guardian: safety guardian (``guardian.core``). Required if ``link.physical``: without
                it, :meth:`_guard` refuses to move the machine.
            machine_path: machine configuration file (only shown in :meth:`config`).
            recent_dir: «recent files» folder (list + a copy of each file). None = in memory only.
            profiles: ``ProfileStore`` from :mod:`hope.cnc.profiles`, or None (profiles disabled).
            selection: saved selection ``{"maquina", "material", "consumible", "proceso", "espesor"}``;
                fixed up with ``fix_selection`` (deleted or incompatible profiles).
            link_factory: ``f(machine) -> Link`` to switch controllers when choosing another machine.
        """
        self.m = machine
        self.link = link
        self.profiles = profiles
        self.link_factory = link_factory
        self.selection: dict | None = None
        self.profile_summary: dict | None = None
        if profiles is not None:
            # Lazy import: profiles are optional and are not loaded if unused.
            from hope.cnc.profiles import effective, fix_selection
            self.selection = fix_selection(profiles, selection or {})
            self.m, self.profile_summary = effective(profiles, self.selection)
        self.guardian = guardian
        self.machine_path = machine_path
        self.id = secrets.token_hex(6)    # identifies this session in guardian requests (audit)
        self.enabled = False              # starts in RESET, like Mach3: enabling is the operator's decision
        self.manual_grant = None          # manual-mode approval (real machine)
        self.program = None               # dict with the loaded program
        self.pending = None               # approval waiting for the operator {"id", "tipo", "resumen"}
        self.started_at = None            # time.monotonic() of the last Cycle Start (for «transcurrido», elapsed)
        self.paused_time = 0.0
        self.log: deque[tuple[float, str]] = deque(maxlen=300)    # (epoch time, message); the screen shows 40
        self.mdi_history: deque[str] = deque(maxlen=30)
        self.lock = threading.RLock()
        self.mdi_interp = Interpreter(as_controller(self.m))    # MDI modal state (G90/G91, units…)
        # Mach3 LED buttons: Single BLK, Block Delete, M1 Optional Stop
        self.opts = {"bloque_a_bloque": False, "borrado_bloque": False, "parada_opcional": False}
        self.next_line = 0                # «Set Next Line» / «Run From Here» (0 = from the start)
        self.from_here = False            # True = «Run From Here»: prepare (raise Z, go, spindle, lower)
        self.run_map: list[int] | None = None   # sent line → index into the program's emitted lines
        self.recent_dir = Path(recent_dir) if recent_dir else None
        self.recent: list[dict] = self._read_recent()
        self._recent_mem: dict[str, str] = {}   # sha → text, when there is no recent-files folder
        self.say(f"{BRAND} listo. Pulsá RESET para habilitar la máquina.")

    # --- utilities --------------------------------------------------------------------------------
    def say(self, msg: str) -> None:
        """Appends a message to the log shown on the screen (with the current time, epoch in s)."""
        self.log.append((time.time(), msg))

    def _require_enabled(self, what: str):
        """Raises :class:`SessionError` if the machine is in RESET. ``what`` = button name (for the
        message). It is the first line of everything that moves the machine or changes its state."""
        if not self.enabled:
            raise SessionError(f"{what}: la máquina está en RESET (pulsá RESET para habilitarla)")

    def _machine_real(self) -> Machine:
        """Copy of the machine with the offsets (G54…G59, G92) the controller has RIGHT NOW.

        Offsets can change in the controller (G10 via MDI, «Zero X», plasma probing) without going
        through the configuration: checking with the saved ones would give false limits and envelope.
        If the controller does not answer or reports no offsets, the machine is used as-is."""
        try:
            real = self.link.offsets()
        except BackendError:
            real = {}
        if not real:
            return self.m
        return dataclasses.replace(self.m, offsets={**self.m.offsets, **{k: dict(v) for k, v in real.items()}})

    def _guard(self, action):
        """None if allowed; otherwise leaves the approval pending for the screen.

        Parameters:
            action: ``guardian.core.Action`` describing what is to be done.

        Returns None (go ahead) or the id of the request the operator has to approve.
        Raises :class:`SessionError` if there is no guardian with a real machine, or if the guardian
        denies it with no option to approve.

        Why: moving a real machine requires a person's approval (project rule). In the simulator
        (``link.physical`` False) there is no risk and nothing is asked."""
        if not self.link.physical:
            return None
        if self.guardian is None:
            raise SessionError("sin guardián no se mueve una máquina real (configs/guardian.yaml)")
        d = self.guardian.decide(action)
        if d.allowed:
            return None
        if not d.pending_request:
            raise SessionError("; ".join(d.reasons))
        return d.pending_request

    # ==============================================================================================
    # ================================ RESET AND GUARDIAN APPROVALS ================================
    # ==============================================================================================

    # --- RESET (enable / disable) -----------------------------------------------------------------
    def reset(self) -> dict:
        """Mach3 RESET button: toggles between disabled and enabled machine.

        * Enabled → stops everything (``link.stop``), forgets the manual approval and goes to RESET.
        * In RESET → enables it. With a real machine it asks the guardian for ``hardware.control`` in
          manual mode; if approval is needed it returns ``{"ok": True, "aprobar": {...}}`` and the screen
          shows the dialog for the operator to approve (:meth:`approve`) or cancel (:meth:`cancel`).

        Returns ``{"ok": True, "habilitada": bool}`` or ``{"ok": True, "aprobar": …}``.
        Raises :class:`SessionError` if the guardian is missing or denies it."""
        with self.lock:
            if self.enabled:
                self.link.stop()
                self.enabled = False
                self.manual_grant = None
                self.say("RESET: máquina deshabilitada.")
                return {"ok": True, "habilitada": False}
            if self.link.physical:
                # Lazy import: only with a real machine (the simulator does not need the guardian).
                from guardian.core import Action
                act = Action("hardware.control", {"dispositivo": "cnc", "modo": "manual", "maquina": self.m.name,
                                                  "controlador": self.link.name, "sesion": self.id},
                             requested_by="hope.cnc.app", justification="habilitar la máquina para el operador")
                req = self._guard(act)
                if req:
                    self.pending = {"id": req, "tipo": "manual", "accion": act,
                                    "resumen": f"Habilitar «{self.m.name}» para jog, MDI y referenciado"}
                    return {"ok": True, "aprobar": self._pending_public()}
            self.enabled = True
            self.say("Máquina habilitada.")
            return {"ok": True, "habilitada": True}

    def _pending_public(self) -> dict | None:
        """The pending approval without the internal data (the ``Action`` and the lines to send), for the
        screen: only id, type («manual» or «programa»), summary and check report."""
        p = self.pending
        return None if p is None else {k: v for k, v in p.items() if k in ("id", "tipo", "resumen", "informe")}

    def approve(self, request_id: str) -> dict:
        """The operator approves on the screen the request the guardian created.

        Parameters:
            request_id: request id (the screen takes it from ``pendiente``/``aprobar``).

        Records the approval in the guardian (it goes into its audit log), asks it again with that id
        and, if it now allows it: type «manual» → enables the machine; type «programa» → starts the
        EXACT lines that were shown when approval was requested (:meth:`_launch` checks the hash).

        Raises :class:`SessionError` if there is no request with that id or the guardian still denies it."""
        with self.lock:
            p = self.pending
            if p is None or p["id"] != request_id:
                raise SessionError("no hay ninguna aprobación pendiente con ese número")
            self.guardian.approve(request_id, "operador (app Hope CNC)")
            d = self.guardian.decide(p["accion"], approval_id=request_id)
            self.pending = None
            if not d.allowed:
                raise SessionError("; ".join(d.reasons))
            if p["tipo"] == "manual":
                self.enabled = True
                self.say("Máquina habilitada (aprobado por el operador).")
                return {"ok": True, "habilitada": True}
            return self._launch(p["lineas"], p["hash"], p.get("mapa"))

    def cancel(self) -> dict:
        """The operator rejects the pending approval: it is reported to the guardian as denied (it goes
        into the audit log) and discarded. Returns ``{"ok": True}``."""
        with self.lock:
            if self.pending and self.guardian is not None:
                self.guardian.deny(self.pending["id"], "operador (app Hope CNC)")
            self.pending = None
            return {"ok": True}

    # ==============================================================================================
    # ========================================== PROGRAM ===========================================
    # ==============================================================================================

    # --- loading and toolpath ---------------------------------------------------------------------
    def load(self, name: str, text: str) -> dict:
        """Loads a G-code program: interprets, plans, checks and translates it into the dialect.

        Parameters:
            name: file name (for display and for recent files).
            text: full G-code text.

        Stores in ``self.program`` a dict with: ``nombre`` (name), ``texto`` (text), ``lineas``
        (original lines), ``informe`` (report from :func:`check_program` + ``sha256`` of the text),
        ``programa`` (interpreted ``Program``), ``emitidas`` (``[(text, original_line)]`` in the
        controller's dialect), ``acumulado`` (times, see :func:`_cumulative_times`), ``maquina`` (with
        the real offsets), ``perforaciones`` (indices of the emitted lines that fire the torch),
        ``referencia`` (the same emitted lines in grbl syntax) and ``dialecto``.

        Returns ``{"ok", "informe", "texto_informe"}``. A program with errors is loaded anyway (to
        see them), but cannot be started. Raises :class:`SessionError` if one is running.

        It is called again from Cycle Start, the Block Delete / M1 buttons and the profiles, because
        they all change what gets executed."""
        with self.lock:
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("no se puede cargar otro programa con uno en marcha")
            m = self._machine_real()
            prog = interpret(text, m, block_delete=self.opts["borrado_bloque"],
                             optional_stop=self.opts["parada_opcional"])
            # A single plan for everything (the emitted lines used to be re-interpreted and re-planned:
            # twice the load time, which on an old PC with 100 000 lines means tens of seconds).
            pl = plan(prog, m)
            report = check_program(prog, m, pl)
            report["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()    # what the guardian approves
            if self.profile_summary:
                report["perfil"] = self._profile_line()
            dialect = dialect_for(m)
            origins: list[int] = []        # filled by emit(): emitted i → index of the operation that produced it
            emitted = emit(prog, m, dialect, origins) if prog.ok else []
            # grbl version of the same (same line count): «correr desde aquí» (Run From Here) replays it
            reference = emit(prog, m, "grbl") if prog.ok and dialect != "grbl" else emitted
            cum = _cumulative_times(prog, pl, emitted, origins)
            # Emitted lines that fire the torch (plasma): the «pierces done» counter in state().
            pierce_idx = [i for i, (t, _) in enumerate(emitted) if t.startswith(("M3", "M4"))]
            self.program = {"nombre": name, "texto": text, "lineas": text.splitlines(), "informe": report,
                            "programa": prog, "emitidas": emitted, "acumulado": cum, "maquina": m,
                            "perforaciones": pierce_idx, "referencia": reference, "dialecto": dialect}
            self.started_at = None
            if self.next_line > len(self.program["lineas"]):    # «Set Next Line» from a longer program
                self.next_line = 0
            self._remember(name, text, report["sha256"])
            self.say(f"Cargado «{name}»: " + report_text(report).splitlines()[1])
            return {"ok": True, "informe": report, "texto_informe": report_text(report)}

    def unload(self) -> dict:
        """Unloads the program («Cerrar G-code» / Close G-Code button). Raises :class:`SessionError` if running."""
        with self.lock:
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("hay un programa en marcha")
            self.program = None
            return {"ok": True}

    def toolpath(self) -> dict:
        """Toolpath in G54 coordinates for the view: [[x, y, z, type, line], …] (type 0 rapid, 1 cut).

        Types: 0 rapid (G0), 1 cut (spindle/torch on), 3 feed without cutting (probing, plunges in
        air), -1 «jump» (starts a new stroke: not joined to the previous point).
        ``line`` = original line, to highlight the current line in the view. Arcs are split into
        segments (``linearize``, minimum tolerance 0.01 mm so as not to generate millions of points).

        Returns ``{"puntos", "perforaciones" ([x, y, z] where the torch fires in plasma),
        "recortado" (True if MAX_PATH_POINTS was exceeded), "limites" (see _limits_work)}``. Does not
        take the lock: it only reads ``self.program``, which is replaced whole on load (never half-modified)."""
        p = self.program
        if p is None:
            return {"puntos": [], "limites": self._limits_work()}
        m = p["maquina"]
        g54 = m.offset("G54")
        view = _view_point_function(m, g54)
        rot = [a for a in m.axis_names if a in ROTARY]
        pts, pierces = [], []
        on = False                         # spindle/torch on at this point of the program
        op_pos: dict[str, float] = {}      # position (machine) at the end of the last move
        for op in p["programa"].ops:
            if op.kind == "spindle":
                if op.value[0] != "off" and not on and m.is_plasma:
                    pierces.append(view(op_pos))
                on = op.value[0] != "off"
                continue
            if op.kind not in ("rapid", "feed", "arc", "probe"):
                continue
            # 0 rapid, 1 cut (spindle/torch on), 3 feed without cutting (probing, plunges in air)
            kind = 0 if op.kind == "rapid" else 1 if on and op.kind != "probe" else 3
            # If the move does not start where the previous one ended (offset change, G92, G28…),
            # a jump point (-1) is added so the view does not draw a line that does not exist.
            start = view(op.start)
            if not pts or any(abs(pts[-1][i] - start[i]) > 1e-9 for i in range(3)):
                pts.append([*start, -1, op.line])
            seq = linearize(op, max(m.arc_tolerance, 0.01)) if op.kind == "arc" else [op.end]
            prev = op.start
            for q in seq:
                # With rotary axes, a long rotation is split every 5°: otherwise «A0 → A360» without
                # moving X would start and end at the same point and the turn would not show.
                steps = 1
                if rot:
                    steps = max(1, math.ceil(max(abs(q.get(a, 0.0) - prev.get(a, 0.0)) for a in rot) / 5.0))
                for k in range(1, steps + 1):
                    t = k / steps
                    r = q if k == steps else {a: prev.get(a, 0.0) + (q.get(a, 0.0) - prev.get(a, 0.0)) * t
                                              for a in q}
                    pts.append([*view(r), kind, op.line])
                prev = q
            op_pos = op.end
            if len(pts) > MAX_PATH_POINTS:
                break
        return {"puntos": pts, "perforaciones": pierces, "recortado": len(pts) > MAX_PATH_POINTS,
                "limites": self._limits_work(), "rotativos": rot}

    def _limits_work(self) -> dict:
        """Travel of each axis in G54 coordinates (mm): ``{"X": [min, max], …}`` (None = no limit).
        The view draws it as the table box."""
        g54 = self.m.offset("G54")
        return {a: [None if ax.min is None else ax.min - g54[a], None if ax.max is None else ax.max - g54[a]]
                for a, ax in self.m.axes.items()}

    # ==============================================================================================
    # =========================================== CYCLE ============================================
    # ==============================================================================================

    # --- Cycle Start ------------------------------------------------------------------------------
    def cycle_start(self) -> dict:
        """Mach3 Cycle Start button (Alt+R): starts, resumes or advances one block.

        * In Feed Hold → resumes. In single-block mode with the program running → runs the next line.
        * Program already running → does nothing.
        * Otherwise, reloads the program with the real offsets (G54 may have changed since it was
          loaded), builds the lines to send (:meth:`_lines_to_run`, with «Set Next Line» / «Run From
          Here») and, on a real machine, asks the guardian to approve THAT program: text hash,
          envelope in machine coordinates and, if partial, from which line plus the hash of the
          EXACT lines that will be sent (preamble included).

        Returns ``{"ok", "en_marcha"}`` or ``{"ok", "aprobar"}`` if waiting for the operator.
        Raises :class:`SessionError` in RESET, without a program or with errors."""
        with self.lock:
            self._require_enabled("Cycle Start")
            st = self.link.snapshot()
            if st["estado"] == "Hold" or (self.link.single and st["trabajo"]["activo"]):
                self.link.resume()
                self.say("Reanudado." if not self.link.single else "Bloque siguiente.")
                return {"ok": True}
            if st["trabajo"]["activo"]:
                return {"ok": True}
            if self.program is None:
                raise SessionError("no hay programa cargado (Cargar G-code)")
            # check again with the real offsets: G54 may have changed since it was loaded
            r = self.load(self.program["nombre"], self.program["texto"])["informe"]    # RLock: reentrant
            if not r["runnable"]:
                raise SessionError("el programa tiene errores: " + "; ".join(r["errores"][:3]))
            lines, run_map = self._lines_to_run()
            partial = self.next_line > 0
            if self.link.physical:
                from guardian.core import Action
                params = {"dispositivo": "cnc", "maquina": self.m.name, "controlador": self.link.name,
                          "programa_sha256": r["sha256"], "envolvente_maquina": r["envolvente_maquina"]}
                if partial:
                    # What is sent is not the whole file: the guardian also approves the actual lines.
                    params.update(desde_linea=self.next_line, preparacion=self.from_here,
                                  lineas_sha256=hashlib.sha256("\n".join(lines).encode()).hexdigest())
                act = Action("hardware.control", params, requested_by="hope.cnc.app",
                             justification=f"ejecutar «{self.program['nombre']}» ({r['lineas']} líneas)"
                                           + (f" desde la línea {self.next_line}" if partial else ""))
                req = self._guard(act)
                if req:
                    # The ALREADY built lines are stored: what is approved is exactly what is sent.
                    self.pending = {"id": req, "tipo": "programa", "accion": act, "lineas": lines, "mapa": run_map,
                                    "hash": r["sha256"], "informe": r,
                                    "resumen": f"Ejecutar «{self.program['nombre']}»" +
                                               (f" desde la línea {self.next_line}" +
                                                (" (Run From Here: sube Z, va al punto y baja)" if self.from_here
                                                 else "") if partial else "")}
                    return {"ok": True, "aprobar": self._pending_public()}
            return self._launch(lines, r["sha256"], run_map)

    # --- «Set Next Line» / «Run From Here» --------------------------------------------------------
    def _lines_to_run(self) -> tuple[list[str], list[int]]:
        """Program to send according to «Set Next Line» / «Run From Here», and the map to emitted lines.

        Returns ``(lines, map)``: ``lines`` = texts to send to the controller; ``map[j]`` = index in
        ``program["emitidas"]`` of sent line j (preamble lines point to the first resumed one).
        With ``next_line == 0`` it is the whole program and the map is the identity.

        How it resumes from original line N:

        1. Finds the first emitted line coming from an original line >= N.
        2. Replays the previous emitted lines (in grbl syntax, ``referencia``) to know the state at
           that point: position, offsets, spindle, coolant, feed.
        3. Plasma: if that point falls in the middle of a cut (torch on), it backs up to the pierce
           (M3) of that cut, because cutting cannot continue without striking the arc.
        4. «Set Next Line» sends header + from there on, as-is (like Mach3: the operator is
           responsible for the machine being in a safe place).
           «Run From Here» prepends a safe preamble: raises Z to the top (G53), moves in XY to the
           point, starts the spindle (not in plasma: the program itself fires the torch) and coolant,
           lowers Z with a slow G1 and restores the feed F.

        Raises :class:`SessionError` if nothing is left to run from that line."""
        p = self.program
        emitted = p["emitidas"]
        texts = [t for t, _ in emitted]
        ref = [t for t, _ in p.get("referencia") or emitted]      # same length; in grbl syntax
        header = HEADERS.get(p.get("dialecto"), HEADER)
        n = self.next_line
        if n <= 0:
            return texts, list(range(len(texts)))
        # From 1: emitted line 0 is the header (original line 0), never the resume point.
        i = next((k for k in range(1, len(emitted)) if emitted[k][1] >= n), None)
        if i is None:
            raise SessionError(f"desde la línea {n} no queda nada que ejecutar")
        mc = as_controller(p["maquina"])

        def replay(upto: int) -> Interpreter:
            """Interprets the first ``upto`` emitted lines (grbl version) and returns the interpreter: its
            state is the machine's state just before emitted line ``upto``."""
            it = Interpreter(mc)
            it.run("\n".join(ref[:upto]))
            return it

        it = replay(i)
        if self.m.is_plasma and it.spindle[0] != "off":
            # in the middle of a cut: go back to that cut's pierce (cannot continue with the arc off)
            k = max(j for j in range(1, i) if ref[j].startswith(("M3", "M4")))
            # Also back up over the preceding emitted lines from the SAME original line as the M3 (e.g.
            # the positioning and plunge the dialect adds before piercing).
            while k - 1 >= 1 and emitted[k - 1][1] == emitted[k][1]:
                k -= 1
            i = k
            it = replay(i)
        if not self.from_here:
            # Header + the rest; the header maps to emitted line i so progress starts there.
            return [header, *texts[i:]], [i] + list(range(i, len(texts)))
        g54 = it.offsets["G54"]
        work = {a: it.pos[a] - g54[a] for a in it.axes}    # target in G54 work coordinates (mm)
        prep = [header]
        zax = self.m.axes.get("Z")
        if zax is not None and zax.max is not None:
            # First raise Z to the top: the XY move must not hit the workpiece or the clamps.
            if p.get("dialecto") == "marlin":                # no G53: the top in work coordinates
                prep.append(f"G0 Z{zax.max - g54['Z']:.4f}")
            else:
                prep.append(f"G53 G0 Z{zax.max:g}")
        prep.append("G0 " + " ".join(f"{a}{work[a]:.4f}" for a in it.axes if a != "Z"))
        if it.spindle[0] != "off" and not self.m.is_plasma:
            prep.append(("M3" if it.spindle[0] == "cw" else "M4") + (f" S{it.rpm:g}" if it.rpm else ""))
        for c in sorted(it.coolant):
            prep.append("M7" if c == "niebla" else "M8")
        if "Z" in work:
            # Lower with G1 (not G0): at the program feed but never faster than the Z axis maximum;
            # 300 mm/min if the program has not set F yet.
            plunge = min(it.feed or 300.0, zax.max_rate if zax else 300.0)
            prep.append(f"G1 Z{work['Z']:.4f} F{plunge:g}")
        if it.feed:
            prep.append(f"F{it.feed:g}")
        # The whole preamble maps to emitted line i (the screen shows «at line N» from the start).
        return prep + texts[i:], [i] * len(prep) + list(range(i, len(texts)))

    def _launch(self, lines: list[str], sha: str, run_map: list[int] | None = None) -> dict:
        """Sends the lines to the controller and starts the clock. Called by :meth:`cycle_start` (no
        approval needed) and :meth:`approve` (after approving).

        Parameters:
            lines: lines to send (from :meth:`_lines_to_run`).
            sha: hash of the program text the lines were built from.
            run_map: sent → emitted map (see ``self.run_map``).

        Raises :class:`SessionError` if the loaded program is no longer that one (another was loaded
        while the approval was waiting): nothing other than what was approved is ever sent. After
        starting, «Set Next Line» is cleared: the next Cycle Start begins from the start again."""
        if self.program is None or self.program["informe"]["sha256"] != sha:
            raise SessionError("el programa cambió: volvé a pulsar Cycle Start")
        self.link.start(lines)
        self.run_map = run_map
        self.started_at = time.monotonic()    # monotonic: a clock change does not alter «transcurrido» (elapsed)
        self.say(f"Cycle Start: «{self.program['nombre']}»" +
                 (f" desde la línea {self.next_line}." if self.next_line else "."))
        self.next_line, self.from_here = 0, False
        return {"ok": True, "en_marcha": True}

    # --- Mach3 buttons: Single BLK, Block Delete, M1, Set Next Line, Run From Here ----------------
    def toggle(self, name: str) -> dict:
        """Toggles a Mach3 LED button.

        Parameters:
            name: ``"bloque_a_bloque"`` (Single BLK: each Cycle Start runs one line),
                ``"borrado_bloque"`` (Block Delete: skips lines starting with «/») or
                ``"parada_opcional"`` (M1 Optional Stop: M1 pauses like M0).

        Block Delete and M1 change what gets executed, so the program is reloaded (if not
        running). Returns ``{"ok", "opciones"}``. Raises :class:`SessionError` if the name does not exist."""
        if name not in self.opts:
            raise SessionError(f"opción desconocida: {name}")
        with self.lock:
            self.opts[name] = not self.opts[name]
            if name == "bloque_a_bloque":
                self.link.set_single(self.opts[name])
            elif self.program is not None and not self.link.snapshot()["trabajo"]["activo"]:
                self.load(self.program["nombre"], self.program["texto"])     # changes what gets executed
            return {"ok": True, "opciones": dict(self.opts)}

    def set_next_line(self, line: int, from_here: bool = False) -> dict:
        """Mach3 «Fijar línea siguiente» (Set Next Line) or «Correr desde aquí» (Run From Here).

        Parameters:
            line: ORIGINAL line of the file (1 = first; 0 = from the start, cancels the previous one).
            from_here: True = «Run From Here» (the next Cycle Start prepares the machine first: raises Z,
                goes to the point, spindle, lowers; see :meth:`_lines_to_run`). No effect with line 0.

        Only marks where the NEXT Cycle Start begins (moves nothing). Clears the progress of the
        previous job (like Mach3 «Rewind»). Raises :class:`SessionError` without a program, with one
        running or with the line outside the file."""
        with self.lock:
            if self.program is None:
                raise SessionError("no hay programa cargado")
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("hay un programa en marcha")
            n = int(line)
            if not 0 <= n <= len(self.program["lineas"]):
                raise SessionError(f"línea fuera del programa (1 a {len(self.program['lineas'])})")
            self.next_line, self.from_here = n, bool(from_here) and n > 0
            self.link.clear_job()
            if n:
                self.say(f"{'Run From Here' if self.from_here else 'Set Next Line'}: el próximo Cycle Start "
                         f"empieza en la línea {n}.")
            return {"ok": True, "linea": n, "desde_aqui": self.from_here}

    # ==============================================================================================
    # ======================================== RECENT FILES ========================================
    # ==============================================================================================
    # The last 10 loaded programs are kept: ``recientes.json`` (list with name, sha256, date and
    # line count) and a copy of each text in ``<sha256[:16]>.nc``. That way «Recientes» (Recent)
    # reopens the SAME text even if the original file was moved or changed.

    # --- recent files -----------------------------------------------------------------------------
    def _read_recent(self) -> list[dict]:
        """Reads ``recientes.json`` (at most 10). Missing or damaged file → empty list (not serious)."""
        if self.recent_dir is None:
            return []
        try:
            return json.loads((self.recent_dir / "recientes.json").read_text(encoding="utf-8"))[:10]
        except (OSError, ValueError):
            return []

    def _remember(self, name: str, text: str, sha: str) -> None:
        """Puts the program first in recent files (no duplicates by sha256), saves its text and deletes
        the texts that dropped off the list. A disk error is only reported in the log: it does not
        prevent loading the program."""
        self.recent = [r for r in self.recent if r["sha"] != sha]
        self.recent.insert(0, {"nombre": name, "sha": sha, "fecha": time.time(), "lineas": text.count("\n") + 1})
        self.recent = self.recent[:10]
        keep = {r["sha"] for r in self.recent}
        if self.recent_dir is None:
            self._recent_mem[sha] = text
            self._recent_mem = {k: v for k, v in self._recent_mem.items() if k in keep}
            return
        try:
            self.recent_dir.mkdir(parents=True, exist_ok=True)
            (self.recent_dir / f"{sha[:16]}.nc").write_text(text, encoding="utf-8")
            (self.recent_dir / "recientes.json").write_text(json.dumps(self.recent, ensure_ascii=False), encoding="utf-8")
            for f in self.recent_dir.glob("*.nc"):
                if not any(f.stem == k[:16] for k in keep):
                    f.unlink()
        except OSError as e:
            self.say(f"no se pudo guardar en recientes: {e}")

    def recent_files(self) -> dict:
        """Recent-files list for the menu: ``{"recientes": [{"nombre", "sha", "fecha", "lineas"}, …]}``."""
        return {"recientes": self.recent}

    def load_recent(self, sha: str) -> dict:
        """Reloads a recent file by its sha256 (like :meth:`load`). Raises :class:`SessionError` if
        it is no longer on the list or its copy was lost."""
        r = next((x for x in self.recent if x["sha"] == sha), None)
        if r is None:
            raise SessionError("ese archivo ya no está en recientes")
        if self.recent_dir is None:
            text = self._recent_mem.get(sha)
        else:
            f = self.recent_dir / f"{sha[:16]}.nc"
            text = f.read_text(encoding="utf-8") if f.is_file() else None
        if text is None:
            raise SessionError("no se encontró el texto del archivo reciente")
        return self.load(r["nombre"], text)

    # ==============================================================================================
    # =============================== RUNNING: REWIND, FEED HOLD, STOP =============================
    # ==============================================================================================
    # No lock and no requirement that the machine be enabled: stopping or pausing must ALWAYS respond
    # instantly, even while another thread is loading a program. (The real emergency stop is the
    # physical one: no program replaces it.)

    def rewind(self) -> dict:
        """Mach3 «Rewind»: forgets the progress of the last job (the bar goes back to 0)."""
        self.link.clear_job()
        self.started_at = None
        return {"ok": True}

    def feed_hold(self) -> dict:
        """Mach3 «Feed Hold» (space bar): decelerates on a ramp and stays paused; Cycle Start resumes."""
        self.link.hold()
        self.say("Feed Hold.")
        return {"ok": True}

    def stop(self) -> dict:
        """Mach3 «Stop» (Alt+S): stops the program and discards what was left to send."""
        self.link.stop()
        self.say("Stop.")
        return {"ok": True}

    # ==============================================================================================
    # ==================================== MANUAL: JOG AND MDI =====================================
    # ==============================================================================================

    # --- jog, homing, unlock ----------------------------------------------------------------------
    def jog(self, axis: str, direction: int, step: float | None, fast: bool, percent: float = 30.0) -> dict:
        """Moves an axis by hand (jog), like the keyboard arrows in Mach3.

        Parameters:
            axis: axis («X», «y»… converted to uppercase).
            direction: > 0 positive, otherwise negative.
            step: step in mm (incremental, see JOG_STEPS) or None/0 = continuous until released.
            fast: True = at the axis maximum speed (Shift in Mach3); False = ``percent``.
            percent: % of the axis maximum speed (1–100; Mach3 «Slow Jog Rate»).

        Continuous jog requests a move up to the axis soft limit (or 1000 mm if it has none) and is
        cut with :meth:`jog_stop` on release: if the connection to the screen is lost, it does not go
        past the limit. Raises :class:`SessionError` in RESET, with an unknown axis or already at the limit."""
        self._require_enabled("jog")
        axis = axis.upper()
        if axis not in self.m.axes:
            raise SessionError(f"eje {axis} no configurado")
        ax = self.m.axes[axis]
        feed = ax.max_rate * (1.0 if fast else max(1.0, min(100.0, percent)) / 100.0)    # mm/min
        pos = self.link.snapshot()["mpos"][axis]
        sign = 1 if direction > 0 else -1
        if step:
            d = sign * float(step)
        else:                                               # continuous: up to the limit (cut on release)
            lim = ax.max if sign > 0 else ax.min
            d = (lim - pos) if lim is not None else sign * 1000.0
            if sign * d <= 1e-6:
                raise SessionError(f"{axis} ya está en el límite")
        self.link.jog({axis: d}, feed)
        return {"ok": True}

    def jog_stop(self) -> dict:
        """Cuts the current jog (when the button or key is released)."""
        self.link.jog_cancel()
        return {"ok": True}

    def home(self) -> dict:
        """Mach3 «Ref All Home»: homes all axes. Runs in a separate thread because it may take a
        minute and the HTTP request must not be left waiting; the result shows up in the log."""
        self._require_enabled("referenciar")
        self.say("Referenciando todos los ejes…")
        threading.Thread(target=self._safe, args=(self.link.home, "referenciado"), daemon=True).start()
        return {"ok": True}

    def _safe(self, fn, what):
        """Runs ``fn()`` (in a background thread) and reports in the log whether it finished or failed.
        ``what`` = task name for the message. Catches :class:`BackendError` (a thread has nobody to
        raise the exception to)."""
        try:
            fn()
            self.say(f"{what.capitalize()} terminado.")
        except BackendError as e:
            self.say(f"Error en {what}: {e}")

    def unlock(self) -> dict:
        """Unlocks the controller after an alarm (``$X`` in grbl). Does not require enabling: its
        very purpose is to get out of the alarm."""
        self.link.unlock()
        return {"ok": True}

    # --- MDI --------------------------------------------------------------------------------------
    def mdi(self, line: str) -> dict:
        """Runs one MDI line (Mach3 Manual Data Input, Alt+2).

        Parameters:
            line: one G-code line (max. 200 characters; empty = does nothing).

        Before sending it, checks it as a one-line program (:meth:`_check_mdi`): a G0 outside the
        limits is rejected here, not on the machine. Stores the line in the history (without repeating
        the last one). Raises :class:`SessionError` in RESET, if too long or if it has errors."""
        self._require_enabled("MDI")
        line = line.strip()
        if not line:
            return {"ok": True}
        if len(line) > 200:
            raise SessionError("línea MDI demasiado larga")
        r = self._check_mdi(line)
        if r["errores"]:
            raise SessionError(r["errores"][0].replace("línea 1: ", ""))
        self.link.mdi(line)
        if not self.mdi_history or self.mdi_history[-1] != line:
            self.mdi_history.append(line)
        self.say(f"MDI: {line}")
        return {"ok": True}

    def _check_mdi(self, line: str) -> dict:
        """Checks the line with the MDI modal state (G91 still applies on the next line, as in the
        controller) and with the real offsets; on error, the modal state does not change.

        Works on a COPY of the MDI interpreter with the position and offsets the controller reports
        now; only if the line has no errors does the copy become the MDI state.
        Returns the report from :func:`check_program`."""
        import copy
        try:
            real = self.link.offsets()
        except BackendError:
            real = {}
        it = copy.deepcopy(self.mdi_interp)
        for w, v in real.items():
            if w in it.offsets:
                it.offsets[w] = {a: float(v.get(a, 0.0)) for a in it.axes}
            elif w == "G92":
                it.g92 = {a: float(v.get(a, 0.0)) for a in it.axes}
        it.pos = dict(self.link.snapshot()["mpos"])
        it.ops, it.diags, it.ended, it.line = [], [], False, 0    # forget the previous MDI line's leftovers
        it.run(line)
        m = dataclasses.replace(it.m, offsets={w: dict(o) for w, o in it.offsets.items()})
        r = check_program(Program(it.ops, it.diags, 1, [], True), m)
        if not r["errores"]:
            self.mdi_interp = it
        return r

    # --- zeroing, spindle, overrides --------------------------------------------------------------
    def zero(self, axis: str) -> dict:
        """Mach3 «Zero X/Y/Z»: the current position becomes 0 in G54 (``G10 L20 P1``) for one axis or
        for all (``axis`` = «todo»/«all»). Goes through :meth:`mdi`, so it requires an enabled machine."""
        axes = list(self.m.axis_names) if axis.lower() in ("todo", "all") else [axis.upper()]
        if any(a not in self.m.axes for a in axes):
            raise SessionError(f"eje {axis} no configurado")
        return self.mdi("G10 L20 P1 " + " ".join(f"{a}0" for a in axes))

    def goto_zero(self) -> dict:
        """Like Mach3 «Goto Z» but cautious: raises Z to the maximum and goes to X0 Y0 (does not lower Z).

        It deliberately does not lower Z: plunging in G0 down to the work zero is the classic way to
        break an end mill. Waits for the rise to finish before moving XY (up to 120 s)."""
        self._require_enabled("ir a cero")
        zmax = self.m.axes["Z"].max if "Z" in self.m.axes else None
        if zmax is not None:
            self.mdi(f"G53 G0 Z{zmax:g}")
            self._wait_idle()
        return self.mdi("G90 G0 X0 Y0")

    def _wait_idle(self, timeout: float = 120.0):
        """Waits (blocking, polling every 50 ms) for the controller to become «Idle». ``timeout`` in s;
        on expiry it returns without error (the caller goes on: the controller queues the next MDI)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.link.snapshot()["estado"] == "Idle":
                return
            time.sleep(0.05)

    def override(self, kind: str, action: str) -> dict:
        """Override while running (Mach3 «FRO»/«SRO»). ``kind``: «avance» (feed), «rapido» (rapid) or
        «husillo» (spindle); ``action``: «100», «+10», «-10», «+1», «-1» (or «50»/«25» for rapids).
        See ``links``."""
        self.link.override(kind, action)
        return {"ok": True}

    def spindle(self, on: bool, rpm: float | None = None) -> dict:
        """Spindle button: turns on clockwise (M3) at ``rpm`` (by default half the machine maximum)
        or off (M5). Goes through :meth:`mdi`: requires an enabled machine and is checked."""
        if on:
            rpm = rpm or self.m.spindle_max / 2
            return self.mdi(f"M3 S{rpm:g}")
        return self.mdi("M5")

    # ==============================================================================================
    # ========================================== PLASMA ============================================
    # ==============================================================================================

    # --- plasma -----------------------------------------------------------------------------------
    def _plasma(self):
        """The machine's plasma parameters (``machine.Plasma``). Raises :class:`SessionError` if the
        machine is not a plasma one."""
        if not self.m.is_plasma or self.m.plasma is None:
            raise SessionError("la máquina no es de plasma (tipo: plasma en la configuración)")
        return self.m.plasma

    def _sequence(self, lines: list[str], what: str):
        """Runs several MDI lines one after another in a background thread, waiting for «Idle» between
        each (e.g. G38.2 must finish touching the sheet before the G10 that sets the zero). Returns
        right away; the result shows up in the log (see :meth:`_safe`)."""
        def run():
            """Thread body: each line through :meth:`mdi` (checked), waiting for the previous one to finish."""
            for ln in lines:
                self._wait_idle()
                self.mdi(ln)
            self._wait_idle()
        threading.Thread(target=self._safe, args=(run, what), daemon=True).start()

    def plasma_probe(self) -> dict:
        """Height sensing (IHS) in place: lowers until it touches the sheet and sets Z0 at the surface.

        Sequence: G91 → ``G38.2 Z-<probe_depth> F<probe_feed>`` (lowers until the sensor touches) →
        G90 → ``G10 L20 P1 Z-<float_offset>`` (subtracts the floating-head travel: how far the torch
        rises until the switch trips) → ``G0 Z<travel_height>`` (travel height).
        All measurements in mm, from the machine's plasma configuration."""
        pl = self._plasma()
        self._require_enabled("sensar altura")
        self.say("Sensando la altura de la chapa…")
        self._sequence(["G91", f"G38.2 Z-{pl.probe_depth:g} F{pl.probe_feed:g}", "G90",
                        f"G10 L20 P1 Z-{pl.float_offset:g}", f"G0 Z{pl.travel_height:g}"], "sensado")
        return {"ok": True}

    def test_fire(self) -> dict:
        """Test fire: torch on for 0.3 s in place (to check the arc and the air pressure)."""
        self._plasma()
        self._require_enabled("disparo de prueba")
        if self.link.snapshot()["trabajo"]["activo"]:
            raise SessionError("hay un programa en marcha")
        self.say("Disparo de prueba.")
        self._sequence(["M3", "G4 P0.3", "M5"], "disparo de prueba")
        return {"ok": True}

    def coolant(self) -> dict:
        """Coolant button (Mach3 M8/M9). With a program running it uses the controller's real-time
        command (an MDI cannot be inserted in the middle of the program); otherwise M8 or M9 via MDI."""
        self._require_enabled("refrigerante")
        st = self.link.snapshot()
        if st["trabajo"]["activo"]:
            self.link.coolant_toggle()
            return {"ok": True}
        return self.mdi("M9" if st["refrigerante"] else "M8")

    # ==============================================================================================
    # ====================================== STATE FOR THE SCREEN ==================================
    # ==============================================================================================

    # --- state for the screen ---------------------------------------------------------------------
    def state(self) -> dict:
        """Everything the screen shows, in a single dict (it requests it several times per second).

        Main keys: ``estado`` (controller snapshot: Idle/Run/Hold/Alarm, positions, current job),
        ``trabajo_pos`` (position in work coordinates, mm), ``programa`` (name, current line, times
        in s, progress 0–1, errors/warnings), ``plasma`` (torch and pierces done/total),
        ``pendiente`` (waiting approval), ``mensajes`` (last 40 messages), ``opciones`` (LEDs),
        ``linea_siguiente``, ``modal``, ``herramienta`` (tool) and ``perfiles`` (profiles).

        Progress: the controller reports ``confirmadas`` = SENT lines already confirmed. With
        ``run_map`` they become an index into the emitted lines, and with ``acumulado`` seconds. Does
        not take the lock: it only reads, and a value from an instant ago is good enough to display."""
        st = self.link.snapshot()
        wpos = {a: st["mpos"][a] - st["wco"].get(a, 0.0) for a in st["mpos"]}    # machine − active offset
        job = st["trabajo"]
        prog = self.program
        info = None
        if prog is not None:
            emitted, cum = prog["emitidas"], prog["acumulado"]
            ran = job["activo"] or job["terminado"] or job["error"]    # there is (or was) a job to show
            done = min(job["confirmadas"], len(emitted)) if ran else 0
            rm = self.run_map
            if ran and rm:
                # Sent → emitted (skipping the «Run From Here» preamble); finished = everything.
                done = len(emitted) if job["terminado"] else rm[min(job["confirmadas"], len(rm) - 1)]
            src = 0                            # current original line (0 = nothing yet)
            if ran and emitted:
                src = emitted[min(done, len(emitted) - 1)][1]
            total = cum[-1] if cum else 0.0
            elapsed = (time.monotonic() - self.started_at) if self.started_at and job["activo"] else None
            # Note: when starting midway, «tiempo_restante» and «progreso» count from the resume emitted
            # line (cum[done]), so they reflect what is really left.
            info = {"nombre": prog["nombre"], "lineas_total": len(prog["lineas"]), "linea_actual": src,
                    "tiempo_total": round(total, 1),
                    "tiempo_restante": round(max(0.0, total - cum[min(done, len(cum) - 1)]), 1) if cum else 0.0,
                    "transcurrido": None if elapsed is None else round(elapsed, 1),
                    "progreso": (cum[min(done, len(cum) - 1)] / total) if total else 0.0,
                    "runnable": prog["informe"]["runnable"], "errores": prog["informe"]["errores"][:50],
                    "avisos": prog["informe"]["avisos"][:50], "informe": report_text(prog["informe"]),
                    "sha256": prog["informe"]["sha256"]}
        plasma = None
        if self.m.is_plasma:
            idx = prog["perforaciones"] if prog else []
            ran = prog is not None and (job["activo"] or job["terminado"] or job["error"])
            k = job["confirmadas"]
            if ran and self.run_map:
                k = len(prog["emitidas"]) if job["terminado"] else self.run_map[min(k, len(self.run_map) - 1)]
            # Pierces done = M3s among the emitted lines before the current one (on resume, skipped ones count).
            done_n = sum(1 for i in idx if i < k) if ran else 0
            plasma = {"antorcha": st.get("husillo_dir", "off") != "off", "perforaciones_total": len(idx),
                      "perforaciones_hechas": min(done_n, len(idx))}
        return {"maquina": self.m.name, "marca": BRAND, "tipo": self.m.kind, "plasma": plasma,
                "ejes": list(self.m.axis_names), "habilitada": self.enabled,
                "fisica": self.link.physical, "estado": st, "trabajo_pos": {a: round(v, 4) for a, v in wpos.items()},
                "programa": info, "pendiente": self._pending_public(),
                "mensajes": [{"t": t, "m": m} for t, m in list(self.log)[-40:]],
                "mdi_historial": list(self.mdi_history), "opciones": dict(self.opts),
                "linea_siguiente": self.next_line, "desde_aqui": self.from_here,
                "modal": self.link.modal() or modal_text(self.mdi_interp), "herramienta": self._tool_info(st),
                "perfiles": None if self.selection is None else {"seleccion": self.selection,
                                                                 "resumen": self.profile_summary}}

    def _tool_info(self, st: dict) -> dict:
        """Current tool for the screen: number (the controller's; if it does not report one, the MDI
        modal state's), diameter and length (mm) from the tool table, name, and ``cambio`` = True if
        the controller is waiting for a manual tool change (M6)."""
        n = self.link.tool()
        if n is None:
            n = self.mdi_interp.tool
        t = self.m.tool(n) if n is not None else None
        return {"n": n or 0, "diametro": t.diameter if t else 0.0, "largo": t.length if t else 0.0,
                "nombre": t.name if t else "", "cambio": str(st.get("mensaje", "")).startswith("cambiar a la herramienta")}

    def program_lines(self) -> dict:
        """Text of the loaded program, line by line (for the screen's G-code list)."""
        p = self.program
        return {"nombre": p["nombre"], "lineas": p["lineas"]} if p else {"nombre": "", "lineas": []}

    # ==============================================================================================
    # ==================================== SLICER-STYLE PROFILES ===================================
    # ==============================================================================================
    # As in a 3D slicer: the effective machine comes from four profiles (machine, material,
    # consumable, process) plus the sheet thickness. Each profile may inherit from another. See
    # :mod:`hope.cnc.profiles`. Changing the selection recomputes the machine and reloads the program.

    # --- slicer-style profiles --------------------------------------------------------------------
    def _profile_line(self) -> str:
        """One-line summary of the chosen profiles («Acero + 45 A + Corte fino: …»)."""
        sel, s = self.selection, self.profile_summary
        names = " + ".join(x for x in (sel.get("material"), sel.get("consumible"), sel.get("proceso")) if x)
        return f"{names}: {s['texto']}"

    def _selection_file(self):
        """File where the selection is remembered between sessions (``<user_dir>/seleccion.json``), or
        None if the profile store has no user folder."""
        d = getattr(self.profiles, "user_dir", None)
        return None if d is None else Path(d) / "seleccion.json"

    def choose_profile(self, kind: str | None = None, name: str | None = None, thickness: float | None = None) -> dict:
        """Chooses a profile (like a slicer's drop-downs) or the thickness; recomputes everything.

        Parameters:
            kind: profile type («maquina», «material», «consumible», «proceso») or None.
            name: name of the profile of that type.
            thickness: sheet thickness (mm) or None to leave it unchanged.

        Without arguments it only recomputes (used when the profile in use was edited). Choosing
        another machine clears material/consumable/process (they may not fit it) and ``fix_selection``
        picks compatible ones. Applies the new machine (:meth:`_apply_machine`), saves the selection and
        reloads the program. Returns ``{"ok", "seleccion", "resumen"}``. Raises :class:`SessionError`
        without profiles, with a program running or with a nonexistent/invalid profile."""
        from hope.cnc.profiles import KINDS, ProfileError, effective, fix_selection
        if self.profiles is None:
            raise SessionError("los perfiles no están activos")
        with self.lock:
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("no se cambian perfiles con un programa en marcha")
            sel = dict(self.selection)
            if kind is not None:
                if kind not in KINDS:
                    raise SessionError(f"tipo de perfil desconocido: {kind}")
                if name not in self.profiles.names(kind):
                    raise SessionError(f"no existe el perfil «{name}»")
                sel[kind] = name
                if kind == "maquina":
                    sel = {"maquina": name, "espesor": sel.get("espesor")}
            if thickness is not None:
                sel["espesor"] = float(thickness)
            try:
                sel = fix_selection(self.profiles, sel)
                m, summary = effective(self.profiles, sel)
            except (ProfileError, ValueError) as e:
                raise SessionError(str(e)) from None
            self._apply_machine(m)
            self.selection, self.profile_summary = sel, summary
            f = self._selection_file()
            if f is not None:
                try:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(sel, ensure_ascii=False), encoding="utf-8")
                except OSError as e:
                    self.say(f"no se pudo guardar la selección: {e}")
            if self.program is not None:
                self.load(self.program["nombre"], self.program["texto"])
            self.say(f"Perfiles: {sel['maquina']} · {self._profile_line()}")
            return {"ok": True, "seleccion": sel, "resumen": summary}

    def _apply_machine(self, m: Machine) -> None:
        """Switches to the effective machine ``m``.

        * Different controller (and there is a ``link_factory``): stops, closes the connection, opens a
          new one and goes to RESET.
        * Different machine with the same controller: goes to RESET. For safety: what was approved for
          one machine does not hold for another; the operator re-enables it deliberately.
        * Same machine (material, process… changed): stays enabled.

        Always resets the MDI modal state (axes, units or offsets may change)."""
        changed_machine = m.name != self.m.name
        if m.controller != self.m.controller and self.link_factory is not None:
            if self.enabled:
                self.link.stop()
            self.link.close()
            self.link = self.link_factory(m)
            self.enabled = False
            self.say(f"Conexión nueva: {self.link.name}. Pulsá RESET para habilitar.")
        elif changed_machine and self.enabled:
            self.enabled = False                  # another machine: re-enable deliberately
            self.say("Máquina cambiada: pulsá RESET para habilitarla.")
        self.m = m
        if hasattr(self.link, "set_machine"):    # the simulator needs the new machine (limits, speeds)
            self.link.set_machine(m)
        self.mdi_interp = Interpreter(as_controller(m))

    def profile_list(self) -> dict:
        """Profiles available for the drop-downs: those of each type compatible with the current
        machine type (mill, plasma…), all machines, the selection, the summary and, for plasma, the
        thicknesses offered by the chosen material and consumable. ``{"activos": False}`` if
        profiles are disabled."""
        from hope.cnc.profiles import KIND_NAMES, KINDS, thickness_options
        if self.profiles is None:
            return {"activos": False}
        kind = self.m.kind
        out = {k: self.profiles.list(k, kind) for k in KINDS}
        out["maquina"] = self.profiles.list("maquina")    # machines are not filtered by type
        esp = []
        if self.m.is_plasma and self.selection.get("material"):
            esp = thickness_options(self.profiles.resolve("material", self.selection["material"]),
                                    self.profiles.resolve("consumible", self.selection["consumible"])
                                    if self.selection.get("consumible") else None)
        return {"activos": True, "tipos": out, "nombres": KIND_NAMES, "seleccion": self.selection,
                "resumen": self.profile_summary, "espesores": esp}

    def profile_detail(self, kind: str, name: str) -> dict:
        """Profile data for the editor: ``crudo`` (raw, as written in its file), ``datos`` (resolved,
        with inheritance applied), ``padre`` (resolved parent), ``propios`` (keys it overrides),
        ``origen`` (factory or user) and ``esquema`` (fields with their units).
        Raises :class:`SessionError` if it does not exist or profiles are disabled."""
        from hope.cnc.profiles import SCHEMA, ProfileError
        if self.profiles is None:
            raise SessionError("los perfiles no están activos")
        try:
            raw, origin = self.profiles.raw(kind, name)
            data = self.profiles.resolve(kind, name)
            parent = self.profiles.resolve(kind, raw["hereda"]) if raw.get("hereda") else None
            own = self.profiles.own_keys(kind, name)
        except ProfileError as e:
            raise SessionError(str(e)) from None
        return {"tipo": kind, "nombre": name, "origen": origin, "hereda": raw.get("hereda"), "datos": data,
                "padre": parent, "propios": own, "esquema": SCHEMA[kind], "crudo": raw}

    def profile_save(self, kind: str, name: str, data: dict, inherits: str | None) -> dict:
        """Saves a user profile (``data`` = own values, ``inherits`` = parent profile or None).
        If it is one of the profiles in use, recomputes the effective machine. Raises
        :class:`SessionError` if the data is invalid."""
        from hope.cnc.profiles import ProfileError
        if self.profiles is None:
            raise SessionError("los perfiles no están activos")
        try:
            out = self.profiles.save(kind, name, data, inherits)
        except (ProfileError, ValueError) as e:
            raise SessionError(str(e)) from None
        self.say(f"Perfil guardado: {name}")
        if self.selection and self.selection.get(kind) == name or kind == "maquina" and self.selection["maquina"] == name:
            self.choose_profile()                 # the profile in use changed: recompute
        return {"ok": True, "perfil": out}

    def profile_delete(self, kind: str, name: str) -> dict:
        """Deletes a user profile; if it was in use, ``choose_profile`` picks another compatible one.
        Raises :class:`SessionError` if it cannot be deleted (factory, nonexistent, profiles disabled:
        ``AttributeError`` when ``self.profiles`` is None)."""
        from hope.cnc.profiles import ProfileError
        try:
            self.profiles.delete(kind, name)
        except (ProfileError, AttributeError) as e:
            raise SessionError(str(e)) from None
        if self.selection.get(kind) == name:
            self.choose_profile()
        return {"ok": True}

    def profile_import(self, kind: str, text: str) -> dict:
        """Imports a profile of type ``kind`` from text (a file shared by another user).
        Raises :class:`SessionError` if the text is not a valid profile."""
        from hope.cnc.profiles import ProfileError
        try:
            out = self.profiles.import_text(kind, text)
        except (ProfileError, ValueError, AttributeError) as e:
            raise SessionError(str(e)) from None
        return {"ok": True, "perfil": out}

    # ==============================================================================================
    # ================================ CONFIGURATION AND OFFSETS ===================================
    # ==============================================================================================

    def config(self) -> dict:
        """Effective machine configuration for the «Config» screen (read-only): axes (steps/mm, max.
        speed mm/min, acceleration mm/s², limits mm), junction deviation and arc tolerance (mm),
        spindle (RPM), tool table (mm), controller, jog steps and, for plasma, heights (mm), pierce
        delay (s), plunge feed (mm/min) and probing options."""
        m = self.m
        return {"nombre": m.name, "tipo": m.kind, "archivo": self.machine_path,
                "ejes": {a: {"pasos_mm": ax.steps_per_mm, "vel_max": ax.max_rate, "acel": ax.accel, "min": ax.min,
                             "max": ax.max} for a, ax in m.axes.items()},
                "desviacion_union": m.junction_deviation, "tolerancia_arco": m.arc_tolerance,
                "husillo": {"rpm_min": m.spindle_min, "rpm_max": m.spindle_max},
                "herramientas": {n: {"largo": t.length, "diametro": t.diameter, "nombre": t.name}
                                 for n, t in sorted(m.tools.items())},
                "controlador": m.controller, "pasos_jog": JOG_STEPS, "marca": BRAND,
                "plasma": None if m.plasma is None or not m.is_plasma else {
                    "altura_perforacion": m.plasma.pierce_height, "altura_corte": m.plasma.cut_height,
                    "retardo_perforacion": m.plasma.pierce_delay, "altura_traslado": m.plasma.travel_height,
                    "avance_bajada": m.plasma.plunge_feed, "perforacion_auto": m.plasma.auto_pierce,
                    "sensado": m.plasma.probe, "holgura_flotante": m.plasma.float_offset}}

    def offsets_table(self) -> dict:
        """Offset table (G54…G59 and G92 if present) as the controller has it RIGHT NOW; those it does
        not report come from the configuration. ``{"ok": False, "mensaje"}`` if the controller does not answer."""
        try:
            real = self.link.offsets()
        except BackendError as e:
            return {"ok": False, "mensaje": str(e)}
        rows = {w: real.get(w, self.m.offset(w)) for w in WCS}
        if "G92" in real:
            rows["G92"] = real["G92"]
        return {"ok": True, "origenes": rows}

