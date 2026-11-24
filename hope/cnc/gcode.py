"""G-code interpreter (RS274/NGC with the habits of Mach3, LinuxCNC and grbl).

Turns the text into a list of operations (:class:`Op`) in **machine coordinates** and mm, with
per-line diagnostics. It moves nothing: the result is used by the planner, the check and the
translation for each controller.

Supported (v0):

* motion: G0, G1, G2/G3 (IJK incremental or absolute with G90.1, or R; extra turns with P; helical
  arcs), G4 (P in seconds), G28/G30 (with intermediate point), G53;
* modes: G17/G18/G19, G20/G21, G90/G91, G90.1/G91.1, G94, G61/G61.1/G64, G98/G99;
* coordinates: G54–G59, G10 L2/L20, G92/G92.1, G43 Hn (tool table), G43.1, G49;
* canned cycles in G17 and G90: G81, G82, G83 (peck), G73 (chip break), G80 — expanded into
  simple moves, so they also work on grbl, which does not have them;
* M0/M1, M2/M30, M3/M4/M5 with S, M6 with T, M7/M8/M9; comments ``( )`` and ``;``, ``%``, N, ``/``.

Also: subprograms ``O1000`` … ``M99`` called with ``M98 P1000 [Ln]`` (with repetitions and up to
:data:`MAX_CALL_DEPTH` levels); Mach3's own codes G12/G13 (circle), G15/G16 (polar coordinates)
and G52 (temporary origin); probing G38.2/G38.3 and G31 (Mach3's); G28.1/G30.1 (store reference
position).

Not supported (clear error): parameters and expressions (``#1``, ``[ ]``), radius compensation
G41/G42, inverse time G93, scaling G50/G51 and probing away G38.4/G38.5. Mach3 user macros (M100
and up) are ignored with a warning: they have to be rewritten.

How it fits
-----------
* Uses ``hope.cnc.machine`` (axes, G54–G59 origins, tool table, plasma, ``job_*`` profiles).
* Used by ``planner.py`` (timings with acceleration), ``check.py`` (limits, spindle, envelope),
  ``dialect.py`` (translation to grbl/LinuxCNC/Marlin), ``session.py`` and ``links.py`` (interface and
  live modal state), ``controller.py``, ``backends/sim.py``, ``medir.py``, ``tool.py`` and ``__main__.py``.

Conventions
-----------
* All :class:`Op` positions are in **machine coordinates** and mm (degrees on A/B/C): work
  origins, G92, G52, tool length and inches are already applied. That way the planner and the
  limit check need to know nothing about modes.
* Work coordinate = machine − (origin G54..G59 + G92 + G52 + tool length in Z).
* Feeds (F) in mm/min; times (G4 P, G82 P) in seconds, as in LinuxCNC and grbl.
* Errors do not stop interpretation: they are recorded in :attr:`Program.diags` with their line
  number and it moves on to the next (like a CAM's «verify»), to show all of them at once.

Usage example::

    from hope.cnc.gcode import interpret
    from hope.cnc.machine import load_machine
    prog = interpret("G21 G90\\nG0 X10 Y10\\nG1 Z-1 F300\\nG2 X20 Y10 R5\\nM30", load_machine(None))
    if not prog.ok:
        for d in prog.errors:
            print(d)            # «línea 4: …» (line 4: …)
    for op in prog.ops:
        print(op.kind, op.end)  # rapid {'X': 10.0, 'Y': 10.0, 'Z': 0.0} …
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from hope.cnc.machine import ROTARY, WCS, Machine

# ==================================================================================================
# ================================== CONFIGURATION AND TABLES ======================================
# ==================================================================================================

INCH = 25.4                     # mm per inch (G20). Exact by definition.
# Distance (mm) above the bottom of the previous peck at which G83/G73 stop the rapid on the way back
# down, so as not to hit chips left in the hole. 0.254 mm = 0.010" as in LinuxCNC. Typical
# 0.1–1. Too small = the drill may hit chips at rapid; too large = time wasted at feed.
PECK_CLEARANCE = 0.254          # mm G83/G73 leave above the previous bottom (0.010" as in LinuxCNC)

# --- RS274/NGC modal groups -----------------------------------------------------------------------
# Motion codes (group 1). G80 cancels the canned cycle; 81/82/83/73 are canned cycles.
MOTION = {"0", "1", "2", "3", "80", "81", "82", "83", "73"}
CANNED = {"81", "82", "83", "73"}
# Modal groups: two G codes of the same group in one block are an error (RS274/NGC, table 4). The last
# group is the non-modal ones (group 0): it is left out of that check (see Interpreter.block).
GROUPS = [MOTION, {"17", "18", "19"}, {"90", "91"}, {"90.1", "91.1"}, {"93", "94"}, {"20", "21"},
          {"40", "41", "42"}, {"43", "43.1", "49"}, {"98", "99"}, {"54", "55", "56", "57", "58", "59"},
          {"61", "61.1", "64"}, {"15", "16"},
          {"4", "10", "12", "13", "28", "28.1", "30", "30.1", "31", "38.2", "38.3", "52", "53", "92", "92.1"}]
# Codes recognized but not implemented: rejected with a message that says what to do, instead of
# an «unknown G» that confuses someone coming from Mach3.
UNSUPPORTED_G = {
    "50": "G50/G51 (escala de Mach3) no está soportada",
    "51": "G50/G51 (escala de Mach3) no está soportada",
    "41": "compensación de radio G41/G42 no soportada en v0: que la haga el CAM",
    "42": "compensación de radio G41/G42 no soportada en v0: que la haga el CAM",
    "93": "modo de avance de tiempo inverso (G93) no soportado en v0",
    "38.4": "G38.4/G38.5 (sensado alejándose) no soportados: usar G38.2 o G31",
    "38.5": "G38.4/G38.5 (sensado alejándose) no soportados: usar G38.2 o G31",
}
PROBES = {"31", "38.2", "38.3"}  # G31 = Mach3 probing; G38.2 = alarm if no contact; G38.3 = no alarm

# --- interpreter safety limits --------------------------------------------------------------------
# Maximum executed lines (counting every subprogram repetition). Protects against an M98 that calls
# itself or a huge L: G-code is not trusted. 5 million is enough for 3D reliefs of several hundred
# thousand lines with repetitions. Higher = a malicious program hangs the check (and, on old PCs,
# the interface) for longer.
MAX_STEPS = 5_000_000            # executed lines (with subprograms and repetitions): stops infinite loops
# Maximum nested subprograms (M98 inside M98). Fanuc/Mach3 use 4–10; 16 is plenty.
MAX_CALL_DEPTH = 16

# --- regular expressions and helper tables --------------------------------------------------------
LABEL = re.compile(r"\s*[Oo](\d+)")      # «O1000» at the start of the line: subprogram label
KNOWN_G = set().union(*GROUPS) | set(UNSUPPORTED_G)
# For each arc plane: (axis a, axis b, normal axis, center letter on a, on b, on the normal).
# G18 is ZX (not XZ) so that clockwise/counterclockwise matches RS274/NGC looking from +Y.
PLANES = {"17": ("X", "Y", "Z", "I", "J", "K"), "18": ("Z", "X", "Y", "K", "I", "J"),
          "19": ("Y", "Z", "X", "J", "K", "I")}
WORD = re.compile(r"([A-Z])([-+]?(?:\d+\.?\d*|\.\d+))")  # one word: letter + number («X-1.5», «F.5»)
# Performance: a whole valid block in one go (if it does not match, the error is located with WORD)
BLOCK_OK = re.compile(r"(?:[A-Z][-+]?(?:\d+\.?\d*|\.\d+))*")


# ==================================================================================================
# ============================================ RESULTS =============================================
# ==================================================================================================

@dataclass
class Diag:
    """A diagnostic: error (the program cannot be run) or warning («aviso»: it runs, but check it)."""
    level: str          # "error" | "aviso"
    line: int           # line number in the file (from 1)
    msg: str

    def __str__(self):
        return f"línea {self.line}: {self.msg}"


@dataclass
class Arc:
    """Geometry of an already-resolved G2/G3 arc (center, radius and swept angle).

    The start and end points are in the :class:`Op`; the axis normal to the plane (Z in G17) is
    interpolated linearly along the arc: that gives the helix.
    """
    center: tuple[float, float]       # in the plane (machine coordinates)
    plane: tuple[str, str, str]       # axes (a, b, normal)
    radius: float                     # mm
    sweep: float                      # radians; negative = clockwise (G2)


@dataclass
class Op:
    """One elementary operation, already in machine coordinates. It is the only thing the planner,
    the check and the translation see.

    ``start``/``end`` are ``{axis: mm}`` with all the machine's axes. ``frame`` is the G54 origin in
    effect: the translation emits coordinates relative to G54 (subtracts ``frame``), so G55..G59, G92,
    G52 and the tool length are «baked» into the numbers and the controller does not need them.
    """
    kind: str                 # rapid | feed | arc | probe | dwell | spindle | coolant | tool | pause | wcs | end
    line: int
    start: dict[str, float] = field(default_factory=dict)
    end: dict[str, float] = field(default_factory=dict)
    feed: float = 0.0         # mm/min (0 for rapid)
    arc: Arc | None = None
    exact_stop: bool = False  # G61: stop at the end of the move
    value: object = None      # seconds (dwell), (dir, rpm) (spindle), set (coolant), n (tool), text (pause)
    source: str = ""          # e.g. "G83" if it comes from an expanded canned cycle
    frame: dict[str, float] | None = None   # G54 origin in effect (to translate with origins that change mid-run)


@dataclass
class Program:
    """Result of interpreting a complete program."""
    ops: list[Op]
    diags: list[Diag]
    lines: int                # last line processed (number of lines if it reached the end)
    tools: list[int]          # tools used with M6, in order of appearance
    ended: bool               # M2/M30 was found

    @property
    def errors(self) -> list[Diag]:
        """Only the errors (with at least one, the program must not be run)."""
        return [d for d in self.diags if d.level == "error"]

    @property
    def warnings(self) -> list[Diag]:
        """Only the warnings."""
        return [d for d in self.diags if d.level == "aviso"]

    @property
    def ok(self) -> bool:
        """True if there are no errors (there may be warnings)."""
        return not self.errors


# Error in a G-code line. Interpreter.run catches it and records it as a diagnostic with the line
# number; outside the interpreter only strip_line and parse_words raise it. (A comment and not a
# docstring, to keep the original «pass» without ruff flagging it as unnecessary.)
class GcodeError(ValueError):
    pass


# ==================================================================================================
# ======================================= READING LINES ============================================
# ==================================================================================================

_FAST_KEYS = frozenset("XYZABCF")    # words allowed by Interpreter.block's fast path
_GKEY_CACHE: dict[str, str] = {}


def _gkey(v: str) -> str:
    """Normalized G/M number: «01», «1.0» and «1» → «1»; «38.20» → «38.2».

    Performance: a program repeats the same few codes thousands of times; they are stored already
    converted (the cache stays small: only the distinct texts that appear)."""
    k = _GKEY_CACHE.get(v)
    if k is None:
        k = _GKEY_CACHE[v] = f"{float(v):g}"     # «:g» drops extra zeros: 1.0 → «1», 38.20 → «38.2»
    return k


def strip_line(raw: str) -> tuple[str, str]:
    """Strips comments; returns (upper-case code without spaces, MSG message if any).

    Accepted comments: ``( … )`` (RS274/NGC; nesting is accepted, as in Mach3) and ``;`` to the end
    of the line (grbl/LinuxCNC). ``(MSG, text)`` keeps the text: Mach3/LinuxCNC show it to the
    operator and here it is used in the next M0/M1 pause. Spaces are removed because in RS274/NGC
    they mean nothing (``G 0 1 X 1 0`` is ``G01X10``).

    Raises:
        GcodeError: unclosed parenthesis.
    """
    if "(" not in raw and ";" not in raw and ")" not in raw:
        # Performance: the vast majority of a CAM's lines have no comments
        return raw.upper().replace(" ", "").replace("\t", ""), ""
    msg = ""
    out, depth, cur = [], 0, []      # remaining code, parenthesis level, text of the current comment
    for ch in raw:
        if ch == ";" and not depth:
            break                    # «;» starts a comment only outside parentheses: «(T1 D=6; CR=0)» is one comment
        if ch == "(":
            depth += 1
            cur = []
        elif ch == ")" and depth:
            depth -= 1
            text = "".join(cur).strip()
            if text.upper().startswith("MSG,"):
                msg = text[4:].strip()
        elif depth:
            cur.append(ch)
        else:
            out.append(ch)
    if depth:
        raise GcodeError("comentario sin cerrar")
    return "".join(out).upper().replace(" ", "").replace("\t", ""), msg


def parse_words(code: str) -> list[tuple[str, str]]:
    """Splits a clean block (no comments or spaces, upper case) into words.

    Args:
        code: e.g. ``"G1X10.5Y-2F300"``.

    Returns:
        ``[("G", "1"), ("X", "10.5"), ("Y", "-2"), ("F", "300")]``; the numbers stay as text
        (``_gkey`` and ``float`` convert them later depending on the letter).

    Raises:
        GcodeError: parameters/expressions (``#``, ``[ ]``) or text that is not a valid word
            (it says which, so the operator can find it).
    """
    if "#" in code or "[" in code:
        raise GcodeError("parámetros y expresiones (#, [ ]) no soportados en v0")
    if BLOCK_OK.fullmatch(code):                      # performance: valid block → all words at once
        return WORD.findall(code)
    # Slow path (only with errors): walk word by word to point at the leftover text.
    words, pos = [], 0
    for m in WORD.finditer(code):
        if m.start() != pos:
            raise GcodeError(f"texto no reconocido «{code[pos:m.start()]}»")
        words.append((m.group(1), m.group(2)))
        pos = m.end()
    if pos != len(code):
        raise GcodeError(f"texto no reconocido «{code[pos:]}»")
    return words


# ==================================================================================================
# ========================================= INTERPRETER ============================================
# ==================================================================================================

class Interpreter:
    """RS274/NGC state machine: walks the program block by block, keeps the modal state (plane,
    units, absolute/incremental, origin, tool, spindle…) and generates :class:`Op`.

    Used once per program (:func:`interpret`) or kept alive to track the modal state in the
    interface (``session.py``, :func:`modal_text`).

    Args:
        machine: machine configuration.
        start: initial position in machine coordinates (``{axis: mm}``); missing axes at 0.
        block_delete: True = skip lines starting with ``/`` (Mach3's «Block Delete» button).
        optional_stop: True = M1 pauses; False = M1 is ignored.
    """

    def __init__(self, machine: Machine, start: dict[str, float] | None = None, block_delete: bool = True,
                 optional_stop: bool = True):
        self.m = machine
        self.axes = machine.axis_names
        self.pos = {a: float((start or {}).get(a, 0.0)) for a in self.axes}   # current position (machine, mm)
        self.wcs = "G54"
        self.offsets = {w: machine.offset(w) for w in WCS}                 # copy: G10 changes it mid-run
        self.g92 = {a: 0.0 for a in self.axes}
        self.g52 = {a: 0.0 for a in self.axes}      # Mach3 temporary origin (G52)
        self.polar: tuple[float, float] | None = None   # G16: center (work) of the polar coordinates
        self.polar_ra = [0.0, 0.0]                  # last radius and angle (degrees)
        self._jump = None                           # pending M98/M99 jump (resolved by run)
        self.tlo = 0.0                              # tool length offset in Z (mm), G43
        self.scale = 1.0                            # mm per program unit: 1 (G21) or 25.4 (G20)
        self.absolute = True                        # G90 / G91
        self.arc_absolute = False                   # G90.1 / G91.1 (IJK absolute or relative to the start)
        self.plane = "17"
        self.motion = "0"                           # modal motion code in effect
        self.feed: float | None = None              # mm/min (already converted from inches)
        self.spindle = ("off", 0.0)                 # ("cw" | "ccw" | "off", rpm)
        self.rpm = 0.0
        self.coolant: set[str] = set()
        self.selected_tool: int | None = None       # selected T (not loaded yet)
        self.tool: int | None = None                # tool loaded with M6
        self.exact = False                          # G61 / G61.1 (exact stop)
        self.retract_r = False                  # G99
        # Modal values of the canned cycles (in work units, mm). Zinit = Z when entering the cycle,
        # where G98 returns to.
        self.cycle = {"Z": None, "R": None, "Q": None, "P": 0.0, "Zinit": None}
        self.block_delete = block_delete
        self.optional_stop = optional_stop      # M1 stops only if enabled (Mach3's «M1 Optional Stop» button)
        self.ops: list[Op] = []
        self.diags: list[Diag] = []
        self.tools: list[int] = []
        self.line = 0
        self.message = ""                       # last (MSG, …) read, for the next pause
        self.ended = False
        # G28/G30 positions in machine coordinates (from «origenes: G28/G30» in the configuration).
        self.home = {"28": machine.offset("G28"), "30": machine.offset("G30")}
        self._frame: dict[str, float] = {}         # last copy of the G54 origin (shared between operations)
        # rotary axes with «vuelta_corta» (shortest rotation on G0); empty on most machines, so the
        # check in _move costs nothing for them
        self._shortest = [a for a in self.axes if a in ROTARY and machine.axes[a].shortest_g0]

    # --- utilities -----------------------------------------------------------------
    def _diag(self, level: str, msg: str):
        """Records a diagnostic on the current line."""
        self.diags.append(Diag(level, self.line, msg))

    def _off(self, a: str) -> float:
        """Total offset of axis ``a`` between work and machine (mm): machine = work + _off.

        Adds the active G54..G59 origin, G92, G52 and, in Z, the tool length (G43): with a longer
        tool the same work Z is higher in machine coordinates, so the tip reaches exactly the same
        place.
        """
        return self.offsets[self.wcs][a] + self.g92[a] + self.g52[a] + (self.tlo if a == "Z" else 0.0)

    def _s(self, a: str) -> float:
        """Unit factor of the axis: rotary axes are in degrees and G20 does not convert them."""
        return 1.0 if a in ROTARY else self.scale

    def work(self, a: str, machine_value: float | None = None) -> float:
        """Work coordinate (mm) of axis ``a`` at the current position or at ``machine_value``."""
        return (self.pos[a] if machine_value is None else machine_value) - self._off(a)

    def _target(self, w: dict[str, float], machine_coords: bool = False) -> dict[str, float]:
        """Target in machine coordinates from the program's axis words.

        Args:
            w: ``{axis: value}`` in program units (inches with G20).
            machine_coords: True for G53 (the values are already machine values, without origins).

        Returns:
            Complete position (axes not named stay where they are). In G91 the value is added to the
            current position; in G90 the origin offset is added to it.
        """
        t = dict(self.pos)
        for a in self.axes:
            if a in w:
                v = w[a] * self._s(a)
                t[a] = v if machine_coords else (v + self._off(a) if self.absolute else self.pos[a] + v)
        return t

    def _move(self, kind: str, target: dict[str, float], source: str = "", feed: float | None = None):
        """Adds a straight move (``"rapid"`` or ``"feed"``) to ``target`` (machine, mm).

        A zero-length move produces no operation (CAMs repeat points and that would make the planner
        stop at each one). For ``feed`` it applies the profile feed (``job_feed``/``job_plunge``)
        only to program lines (empty ``source``): internal sequences (cycles, piercing) keep their
        own feed.

        Raises:
            GcodeError: feed move without F defined.
        """
        pos = self.pos
        if kind == "rapid" and self.absolute and not source and self._shortest:
            # «Shortest Rot on G0» (Mach3): the rotary axis reaches the same angle turning the short way.
            # The new position stays continuous (e.g. 350° → 450°, not 90°) so the controller, which
            # does not wrap, gets a target it can reach turning +100°.
            target = dict(target)
            for a in self._shortest:
                delta = target[a] - pos[a]
                target[a] = pos[a] + ((delta + 180.0) % 360.0 - 180.0)
        for a in self.axes:                                # does any axis move? (plain loop: faster)
            if abs(target[a] - pos[a]) >= 1e-9:
                break
        else:
            return
        if kind == "feed":
            feed = self.feed if feed is None else feed
            if not feed:
                raise GcodeError("movimiento con avance (G1/G2/G3) sin F")
            # With motion in XY it is cutting (job_feed); if it only goes down in Z it is a plunge (job_plunge).
            if not source and any(abs(target[a] - self.pos[a]) > 1e-9 for a in ("X", "Y") if a in self.axes):
                feed = self._job_feed(feed)
            elif not source and "Z" in self.axes and target["Z"] < self.pos["Z"] - 1e-9:
                feed = self._job_feed(feed, self.m.job_plunge)
        # Performance and memory: self.pos is never modified in place (it is always replaced), so the
        # operation keeps the same dictionary without copying it; and the G54 origin is shared by all
        # operations while it does not change (before: one copy per line, 100,000 copies in a 3D relief).
        g54 = self.offsets["G54"]
        if self._frame != g54:
            self._frame = dict(g54)
        end = dict(target)
        self.ops.append(Op(kind, self.line, pos, end, feed or 0.0,
                           exact_stop=self.exact, source=source, frame=self._frame))
        self.pos = end

    def _job_feed(self, feed: float, value: float | None = None) -> float:
        """Profile cutting feed (only with spindle/torch on and motion in the plane).

        Args:
            feed: program F (mm/min).
            value: profile feed to use (``job_plunge``); ``None`` = ``job_feed``.

        Returns:
            In «reemplazar» (replace) mode, the profile's; in «limitar» (cap), the smaller of the two.
            Without a profile or with the spindle off (e.g. dry runs), the program F as is.
        """
        jf = self.m.job_feed if value is None else value
        if not jf or self.spindle[0] == "off":
            return feed
        return jf if self.m.job_feed_mode == "reemplazar" else min(feed, jf)

    def _work_target(self, axis: str, work_value: float) -> dict[str, float]:
        """Current position with a single axis moved to ``work_value`` in work coordinates (mm)."""
        return {**self.pos, axis: work_value + self._off(axis)}

    # --- interpretation ----------------------------------------------------------
    def run(self, text: str) -> Program:
        """Interprets the complete program and returns the :class:`Program`.

        Besides calling :meth:`block` on each line, it handles what belongs to the «file»: ``%``, ``/``
        (block delete), ``O…`` labels and M98/M99 subprogram jumps. Each line's errors are recorded and
        it carries on (it does not raise :class:`GcodeError`).
        """
        lines = text.splitlines()
        # First pass: where each «O<n>» is (the first occurrence wins), so that M98 can call
        # subprograms defined further down, as in Fanuc/Mach3.
        labels: dict[int, int] = {}
        for i, raw in enumerate(lines):
            m = LABEL.match(raw)
            if m:
                labels.setdefault(int(m.group(1)), i)
        stack: list[list[int]] = []                   # [return, remaining repetitions, start of the O]
        i = steps = 0
        while i < len(lines) and not self.ended:
            steps += 1
            if steps > MAX_STEPS:
                self._diag("error", "demasiadas líneas ejecutadas: ¿subprograma que se llama sin fin?")
                break
            self.line, raw = i + 1, lines[i]
            i += 1
            s = raw.strip()
            if not s or s.startswith("%"):            # «%» marks tape start/end (Fanuc): ignored
                continue
            if s.startswith("/") and self.block_delete:
                continue
            m = LABEL.match(s)
            if m:                                     # «O1000 (name)»: label, not executed
                s = s[m.end():].strip()
            try:
                code, msg = strip_line(s.lstrip("/"))
                if msg:
                    self.message = msg
                if code:
                    self.block(parse_words(code))
            except GcodeError as e:
                self._diag("error", str(e))
                self._jump = None                     # a line with an error does not jump
            if self._jump:
                jump, self._jump = self._jump, None
                if jump[0] == "call":
                    _, num, times = jump              # M98 P<num> L<times>
                    if num not in labels:
                        self._diag("error", f"M98 P{num}: no existe el subprograma O{num}")
                    elif len(stack) >= MAX_CALL_DEPTH:
                        self._diag("error", "subprogramas anidados demasiado profundo")
                    elif times > 0:
                        stack.append([i, times - 1, labels[num] + 1])
                        i = labels[num] + 1
                elif stack:                           # M99: return (or repeat if any remain)
                    if stack[-1][1] > 0:
                        stack[-1][1] -= 1
                        i = stack[-1][2]
                    else:
                        i = stack.pop()[0]
                else:
                    self._diag("aviso", "M99 en el programa principal: se toma como fin")
                    self.ended = True
        return Program(self.ops, self.diags, self.line, self.tools, self.ended)

    def block(self, words: list[tuple[str, str]]):
        """Executes a block (a line already split into words by :func:`parse_words`).

        Follows the RS274/NGC order of execution (section 3.8 of the standard): first units, F, S,
        T, spindle/coolant/change M codes, G4, plane, tool length, origin, modes, then the
        non-modal ones (G10, G28, G92…) and finally the motion and the stop/end M codes. So, for
        example, ``G20 G0 X1`` moves 1 inch and ``S1000 M3 G1 X10`` cuts with the spindle already on.
        One deliberate exception: G20/G21 is applied before F so that ``G20 F10`` means 10 in/min.

        Raises:
            GcodeError: repeated letter, axis not configured, incompatible codes or missing data.
        """
        gs, ms, w = [], [], {}            # G codes, M codes, other words {letter: value}
        for letter, val in words:
            if letter == "G":
                gs.append(_gkey(val))
            elif letter == "M":
                ms.append(_gkey(val))
            elif letter == "N":           # line number: does nothing
                continue
            elif letter in w:
                raise GcodeError(f"la letra {letter} aparece dos veces en el bloque")
            elif letter in "XYZABC" and letter not in self.axes:
                raise GcodeError(f"eje {letter} no configurado en la máquina")
            elif letter == "E":
                raise GcodeError("E es de impresora 3D (extrusor): este programa no es para una CNC")
            else:
                w[letter] = float(val)
        # --- fast path -------------------------------------------------------------------------
        # Performance: the vast majority of a CAM's lines are «G1 X… Y… Z… F…» (or only axes, with
        # the modal G1). If the block has at most one G0/G1, only axes and F, no M and no polar
        # coordinates, none of the ~40 modal checks below can apply: it moves directly. The result
        # is identical to the full path (the tests check this).
        if not ms and len(gs) <= 1 and self.polar is None:
            g = gs[0] if gs else self.motion
            if g in ("0", "1") and _FAST_KEYS.issuperset(w):
                if "F" in w:
                    if w["F"] < 0:
                        raise GcodeError("F negativo")
                    self.feed = w["F"] * self.scale
                if gs:
                    if self.motion in CANNED:               # leaving a canned cycle clears its values
                        self.cycle = {"Z": None, "R": None, "Q": None, "P": 0.0, "Zinit": None}
                    self.motion = g
                axis_w = {a: v for a, v in w.items() if a != "F"}
                if axis_w:
                    self._move("rapid" if g == "0" else "feed", self._target(axis_w))
                return
        # --- full path -------------------------------------------------------------------------
        for g in gs:
            if g in UNSUPPORTED_G:
                raise GcodeError(UNSUPPORTED_G[g])
            if g not in KNOWN_G:
                raise GcodeError(f"G{g} desconocido")
        for grp in GROUPS[:-1] if len(gs) > 1 else ():   # with a single G there can be no conflict
            if len([g for g in gs if g in grp]) > 1:
                raise GcodeError("dos códigos G del mismo grupo modal en un bloque: " + ", ".join(
                    "G" + g for g in gs if g in grp))
        gset = set(gs)
        axis_w = {a: v for a, v in w.items() if a in self.axes}

        # RS274/NGC order of execution
        if "93" in gset:
            raise GcodeError(UNSUPPORTED_G["93"])
        if "20" in gset:
            self.scale = INCH
        if "21" in gset:
            self.scale = 1.0
        if "F" in w:
            if w["F"] < 0:
                raise GcodeError("F negativo")
            self.feed = w["F"] * self.scale
        if "S" in w:
            if w["S"] < 0:
                raise GcodeError("S negativo")
            self.rpm = w["S"]
            # Changing S with the spindle running changes the speed immediately (no new M3), as in Mach3.
            if self.spindle[0] != "off" and not ({"3", "4", "5"} & set(ms)):
                self._spindle(self.spindle[0])
        if "T" in w:
            self.selected_tool = int(w["T"])        # T only selects; loading is M6
        self._mcodes_pre(ms)
        if "4" in gset:
            p = w.get("P")
            if p is None or p < 0:
                raise GcodeError("G4 necesita P ≥ 0 (segundos)")
            if p > 1000:
                # Mach3 interprets P in ms or in s depending on a configuration checkbox: a huge P is
                # usually a program written for milliseconds.
                self._diag("aviso", f"G4 P{p:g}: en este intérprete P son segundos (Mach3 puede usar ms)")
            self.ops.append(Op("dwell", self.line, value=float(p)))
        for g in ("17", "18", "19"):
            if g in gset:
                self.plane = g
        if "43" in gset:
            # G43 without H uses the loaded tool (LinuxCNC does the same).
            n = int(w["H"]) if "H" in w else self.tool
            t = self.m.tool(n) if n is not None else None
            if t is None:
                raise GcodeError(f"G43: la herramienta {n} no está en la tabla (herramientas: en la configuración)")
            self.tlo = t.length
        if "43.1" in gset:
            # G43.1 Z…: dynamic length (no table). The Z is consumed: it is not a move.
            if "Z" not in axis_w:
                raise GcodeError("G43.1 necesita Z")
            self.tlo = axis_w.pop("Z") * self.scale
        if "49" in gset:
            self.tlo = 0.0
        for g in ("54", "55", "56", "57", "58", "59"):
            if g in gset:
                self.wcs = "G" + g
        if "61" in gset or "61.1" in gset:
            self.exact = True
        if "64" in gset:
            self.exact = False
        if "90" in gset:
            self.absolute = True
        if "91" in gset:
            self.absolute = False
        if "90.1" in gset:
            self.arc_absolute = True
        if "91.1" in gset:
            self.arc_absolute = False
        if "98" in gset:
            self.retract_r = False
        if "99" in gset:
            self.retract_r = True

        # --- non-modal (group 0) ---------------------------------------------------------------
        # uses_axes: those that «consume» the block's X Y Z words (no modal move afterwards).
        nonmodal = gset & ({"10", "12", "13", "28", "28.1", "30", "30.1", "52", "53", "92", "92.1"} | PROBES)
        uses_axes = nonmodal & ({"10", "28", "30", "52", "92"} | PROBES)
        if nonmodal & PROBES:
            self._probe(next(iter(nonmodal & PROBES)), axis_w)
        elif "10" in gset:
            self._g10(w, axis_w)
        elif "28" in gset or "30" in gset:
            self._g28("28" if "28" in gset else "30", axis_w)
        elif "28.1" in gset or "30.1" in gset:
            self.home["28" if "28.1" in gset else "30"] = dict(self.pos)
        elif "92" in gset:
            # G92: «the current position is called v». It stores the offset that makes
            # work = machine − (origin + G92 + G52 + tlo) equal v; that is why G52 is subtracted too
            # (Mach3 adds G52 to the origin: without subtracting it, the coordinate ended up off by G52).
            if not axis_w:
                raise GcodeError("G92 necesita al menos un eje")
            for a, v in axis_w.items():
                self.g92[a] = (self.pos[a] - self.offsets[self.wcs][a] - self.g52[a] - (self.tlo if a == "Z" else 0)
                               - v * self._s(a))
        elif "92.1" in gset:
            self.g92 = {a: 0.0 for a in self.axes}
        elif "52" in gset:
            # Mach3/Fanuc G52: shifts the work origin by (X, Y, Z) relative to the active G54..G59,
            # without touching the origin table. G52 X0 Y0 Z0 cancels it.
            if not axis_w:
                raise GcodeError("G52 necesita al menos un eje")
            for a, v in axis_w.items():
                self.g52[a] = v * self._s(a)
        elif "12" in gset or "13" in gset:
            self._pocket("12" if "12" in gset else "13", w)
        if "16" in gset:
            # Mach3 G16: the polar center is the (work) point where the tool is.
            self.polar = (self.work("X"), self.work("Y"))
            self.polar_ra = [0.0, 0.0]
        if "15" in gset:
            self.polar = None
        if self.polar is not None and ("X" in axis_w or "Y" in axis_w) and not uses_axes:
            axis_w = self._from_polar(axis_w, gs)

        # --- motion ----------------------------------------------------------------------------
        motion = [g for g in gs if g in MOTION]
        if motion:
            self.motion = motion[0]
            if self.motion in CANNED and self.cycle["Zinit"] is None:
                self.cycle["Zinit"] = self.work("Z") if "Z" in self.axes else None   # height for G98
            if self.motion not in CANNED:
                self.cycle = {"Z": None, "R": None, "Q": None, "P": 0.0, "Zinit": None}
        if not uses_axes and axis_w:
            if "53" in gset:
                if self.motion not in ("0", "1"):
                    raise GcodeError("G53 solo con G0 o G1")
                self._move("rapid" if self.motion == "0" else "feed", self._target(axis_w, machine_coords=True))
            elif self.motion == "80":
                if axis_w:
                    raise GcodeError("ejes con G80 activo: falta G0/G1")
            elif self.motion in ("0", "1"):
                self._move("rapid" if self.motion == "0" else "feed", self._target(axis_w))
            elif self.motion in ("2", "3"):
                self._arc(w, axis_w)
            else:
                self._cycle(w, axis_w)       # with a modal cycle, each line with X/Y is another hole
        # --- subprograms and stops (after the motion) ------------------------------------------
        if "98" in ms:
            if "P" not in w:
                raise GcodeError("M98 necesita P (número del subprograma O…)")
            self._jump = ("call", int(w["P"]), int(w.get("L", 1)))
        elif "99" in ms:
            self._jump = ("ret",)
        self._mcodes_post(ms)

    # --- M ------------------------------------------------------------------------------
    def _spindle(self, d: str):
        """Turns the spindle or torch on (``"cw"``/``"ccw"``) or off (``"off"``).

        Warns if S exceeds the maximum or is 0 (on mill/router/laser). On plasma with automatic
        piercing, the first M3 triggers the :meth:`_pierce` sequence and M5 rises to the travel
        height (if the CAM did not) so as not to drag the torch across the sheet.
        """
        pl = self.m.plasma if self.m.is_plasma else None
        if d != "off" and self.rpm <= 0 and self.m.job_rpm and pl is None:
            self.rpm = float(self.m.job_rpm)            # the profile supplies the missing S
        if pl is None:
            if d != "off" and self.rpm > self.m.spindle_max:
                self._diag("aviso", f"S{self.rpm:g} supera el máximo del husillo ({self.m.spindle_max:g} rpm)")
            if d != "off" and self.rpm <= 0:
                self._diag("aviso", "husillo encendido con S0")
        was_on = self.spindle[0] != "off"
        if pl is not None and pl.auto_pierce and d != "off" and not was_on:
            self._pierce(pl)
            return
        self.spindle = (d, self.rpm)
        self.ops.append(Op("spindle", self.line, value=self.spindle))
        if pl is not None and pl.auto_pierce and d == "off" and was_on and self.work("Z") < pl.travel_height:
            self._move("rapid", self._work_target("Z", pl.travel_height), source="perforación")

    def _pierce(self, pl):
        """Plasma pierce sequence: [probing] → pierce height → torch → delay → cut.

        With probing (IHS): if it is below the pierce height it rises to the travel height, moves down
        with G38.2 up to ``probe_depth`` looking for the sheet and sets G54 Z0 with ``G10 L20 Z-slack``:
        when the floating head switch trips, the nozzle is already ``float_offset`` mm below the
        contact point; declaring that position as Z = −slack leaves Z0 right on the sheet.
        This corrects warped sheets or unlevel tables on every pierce.
        """
        src = "perforación"
        if pl.probe:
            if self.wcs != "G54":
                raise GcodeError("el sensado automático de plasma trabaja en G54")
            if self.work("Z") < pl.pierce_height:
                self._move("rapid", self._work_target("Z", pl.travel_height), source=src)
            # The target is passed in program units (/ _s) because _probe multiplies again.
            self._probe("38.2", {"Z": (self.work("Z") - pl.probe_depth) / self._s("Z")}, feed=pl.probe_feed, source=src)
            self._set_wcs(20, 1, {"Z": -pl.float_offset}, source=src)
        self._move("rapid", self._work_target("Z", pl.pierce_height), source=src)
        self.spindle = ("cw", self.rpm)
        self.ops.append(Op("spindle", self.line, value=self.spindle, source=src))
        if pl.pierce_delay:
            self.ops.append(Op("dwell", self.line, value=float(pl.pierce_delay), source=src))
        self._move("feed", self._work_target("Z", pl.cut_height), source=src, feed=pl.plunge_feed)

    # --- probing (G38.2 / G38.3 / Mach3 G31) ------------------------------------------------
    def _probe(self, g: str, axis_w: dict, feed: float | None = None, source: str = ""):
        """Probing ends wherever it touches. For planning, contact is assumed at the expected surface
        (Z0 of the origin in effect, minus the floating head slack on plasma); for the limit check the
        full travel (``value``) is used, which is the worst that can happen.

        Args:
            g: ``"38.2"``, ``"38.3"`` or ``"31"``.
            axis_w: probing target (program units).
            feed: feed (mm/min); ``None`` = modal F.
            source: internal sequence requesting it (``"perforación"``); with ``source`` the target is
                always absolute in work coordinates, even if the program is in G91.

        Raises:
            GcodeError: no axes, no F or no travel.
        """
        if not axis_w:
            raise GcodeError(f"G{g} necesita el eje y la distancia de sensado")
        feed = feed if feed is not None else self.feed
        if not feed:
            raise GcodeError(f"G{g} sin F")
        target = self._target(axis_w) if source == "" else {**self.pos, **{a: v * self._s(a) + self._off(a)
                                                                            for a, v in axis_w.items()}}
        moved = [a for a in self.axes if abs(target[a] - self.pos[a]) > 1e-9]
        if not moved:
            raise GcodeError(f"G{g} sin recorrido")
        end = dict(target)
        if moved == ["Z"] and target["Z"] < self.pos["Z"]:
            # Vertical probing downward (the typical case: finding the workpiece surface).
            flo = self.m.plasma.float_offset if self.m.is_plasma and self.m.plasma else 0.0
            surface = self._off("Z") - flo                       # work Z0 in machine coordinates
            if target["Z"] <= surface <= self.pos["Z"]:
                end["Z"] = surface
            elif g != "38.3" and target["Z"] > surface:
                self._diag("aviso", f"G{g}: el recorrido no llega a Z0 del origen: puede no tocar (alarma)")
        else:
            self._diag("aviso", f"G{g}: sensado fuera de Z; se planifica con el recorrido completo")
        self.ops.append(Op("probe", self.line, dict(self.pos), end, feed, value=target,
                           source=source or ("G31" if g == "31" else "G" + g), frame=dict(self.offsets["G54"])))
        self.pos = end

    def _set_wcs(self, lv: int, p: int, vals: dict[str, float], source: str = ""):
        """Changes a work origin (G10 L2 / L20 logic).

        Args:
            lv: 2 = ``vals`` is the origin in machine coordinates; 20 = ``vals`` is the work coordinate
                the current position must have (the origin is computed: machine − G92 − G52 − tlo − v).
            p: 0 = active origin; 1–6 = G54–G59.
            vals: ``{axis: mm}`` already converted to mm.
            source: internal sequence requesting it (for the translation).

        If G54 changes, a ``"wcs"`` :class:`Op` is emitted: the translation works relative to G54, so
        the controller has to receive the change at the same point of the program (e.g. after probing).
        """
        wcs = self.wcs if p == 0 else WCS[p - 1]
        for a, v in vals.items():
            # L20: «the current position is v» → origin = machine − (G92 + G52 + tlo) − v, with the same
            # terms _off() adds (without G52 it ended up off by G52 if one was active).
            self.offsets[wcs][a] = v if lv == 2 else (self.pos[a] - self.g92[a] - self.g52[a]
                                                      - (self.tlo if a == "Z" else 0) - v)
        if wcs == "G54":                 # the controller must learn about it too (probing mid-run)
            self.ops.append(Op("wcs", self.line, value={a: self.offsets["G54"][a] for a in vals},
                               source=source or f"G10 L{lv}", frame=dict(self.offsets["G54"]),
                               end={"L": lv, **{a: (self.pos[a] - self.offsets["G54"][a]) for a in vals}}))

    def _mcodes_pre(self, ms: list[str]):
        """M codes that act BEFORE the block's motion: M6, M3/M4/M5, M7/M8/M9 (RS274/NGC order).

        Stop/end and subprogram M codes are left to :meth:`_mcodes_post`/:meth:`run`. Mach3's own
        M codes and user macros are ignored with a warning: there is no way to know what they did.
        """
        for mc in ms:
            n = float(mc)
            if mc == "6":
                if self.selected_tool is None:
                    raise GcodeError("M6 sin T")
                self.tool = self.selected_tool
                if self.tool not in self.tools:
                    self.tools.append(self.tool)
                self.ops.append(Op("tool", self.line, value=self.tool))
            elif mc in ("3", "4", "5"):
                self._spindle({"3": "cw", "4": "ccw", "5": "off"}[mc])
            elif mc in ("7", "8"):
                self.coolant.add("niebla" if mc == "7" else "inundación")
                self.ops.append(Op("coolant", self.line, value=set(self.coolant)))
            elif mc == "9":
                self.coolant = set()
                self.ops.append(Op("coolant", self.line, value=set()))
            elif mc in ("98", "99"):
                continue                                 # after the block (run)
            elif mc in ("0", "1", "2", "30"):
                continue                                 # after the motion
            elif mc in ("48", "49", "47"):
                # Mach3 M47 (repeat from the start), M48/M49 (enable/disable overrides).
                self._diag("aviso", f"M{mc} (Mach3) se ignora")
            elif n >= 100:
                self._diag("aviso", f"M{mc}: macro de usuario de Mach3; se ignora (reescribirla para este control)")
            else:
                self._diag("aviso", f"M{mc} desconocido; se ignora")

    def _mcodes_post(self, ms: list[str]):
        """M codes that act AFTER the motion: M0/M1 (pause, with the last MSG) and M2/M30 (end).

        At the end the spindle is turned off if it was left on (M2/M30 do so on every controller), and
        the program is marked as finished: whatever follows is not interpreted.
        """
        for mc in ms:
            if mc == "1" and not self.optional_stop:
                continue
            if mc in ("0", "1"):
                self.ops.append(Op("pause", self.line, value=self.message or ("pausa" if mc == "0" else
                                                                             "pausa opcional")))
                self.message = ""
            elif mc in ("2", "30"):
                if self.spindle[0] != "off":
                    self.spindle = ("off", self.rpm)
                    self.ops.append(Op("spindle", self.line, value=self.spindle))
                self.ops.append(Op("end", self.line))
                self.ended = True

    # --- non-modal G codes ----------------------------------------------------------
    def _from_polar(self, axis_w: dict, gs: list) -> dict:
        """Mach3 G16: X = radius, Y = angle (degrees) around the point where it was activated.

        Converts (radius, angle) to Cartesian work X/Y: X = cx + r·cos(θ), Y = cy + r·sin(θ),
        with θ in degrees from +X counterclockwise. If the block carries only one of the two, the other
        is the last one used (modal), as in Mach3: «Y90» rotates at the same radius.

        Returns:
            ``axis_w`` with Cartesian X/Y in program units (``_target`` scales them again, which is
            why they are divided by ``scale``).

        Raises:
            GcodeError: outside G17/G90 or with arcs.
        """
        motion = next((g for g in gs if g in MOTION), self.motion)
        if self.plane != "17" or not self.absolute:
            raise GcodeError("coordenadas polares (G16) solo en G17 y G90")
        if motion in ("2", "3"):
            raise GcodeError("coordenadas polares (G16) solo con G0, G1 o ciclos de taladrado")
        r = axis_w.get("X", self.polar_ra[0] / self.scale) * self.scale     # radius in mm
        ang = axis_w.get("Y", self.polar_ra[1])                             # degrees: not scaled
        self.polar_ra = [r, ang]
        cx, cy = self.polar
        out = dict(axis_w)
        out["X"] = (cx + r * math.cos(math.radians(ang))) / self.scale
        out["Y"] = (cy + r * math.sin(math.radians(ang))) / self.scale
        return out

    def _pocket(self, g: str, w: dict):
        """Mach3 G12/G13: moves out in X by distance I, cuts a circle (clockwise or counterclockwise)
        centered on the starting point and returns to the center.

        Expanded into: straight feed to (X+I), a 360° arc (G12 clockwise = sweep −2π, G13 = +2π) and
        a feed back to the center. That way grbl and LinuxCNC, which lack G12/G13, understand it.

        Raises:
            GcodeError: outside G17, without I (or I0), or without F.
        """
        if self.plane != "17":
            raise GcodeError("G12/G13 solo en el plano XY (G17)")
        if "I" not in w or w["I"] == 0:
            raise GcodeError(f"G{g} necesita I (radio del círculo)")
        if not self.feed:
            raise GcodeError(f"G{g} sin F")
        r = w["I"] * self.scale
        center = dict(self.pos)
        src = "G" + g
        self._move("feed", {**self.pos, "X": center["X"] + r}, source=src)
        sweep = (-2 if g == "12" else 2) * math.pi
        self.ops.append(Op("arc", self.line, dict(self.pos), dict(self.pos), self._job_feed(self.feed),
                           exact_stop=self.exact, arc=Arc((center["X"], center["Y"]), ("X", "Y", "Z"), abs(r), sweep),
                           source=src, frame=dict(self.offsets["G54"])))
        self._move("feed", center, source=src)


    def _g10(self, w, axis_w):
        """G10 L2/L20 P0..6: sets a work origin (see :meth:`_set_wcs`).

        Raises:
            GcodeError: another L (e.g. L1 = tool table, not supported) or P outside 0–6.
        """
        lv, p = int(w.get("L", -1)), int(w.get("P", 0))
        if lv not in (2, 20):
            raise GcodeError("G10: solo L2 y L20 (orígenes de trabajo)")
        if not 0 <= p <= 6:
            raise GcodeError("G10: P entre 0 (actual) y 6 (G59)")
        self._set_wcs(lv, p, {a: v * self._s(a) for a, v in axis_w.items()})

    def _g28(self, which: str, axis_w):
        """G28/G30: go to the stored reference position (``origenes: G28/G30`` or G28.1/G30.1).

        With axes (``G28 Z5``) it first rapids to that intermediate point (work coordinates) and
        then takes ONLY those axes to the reference, as in RS274/NGC; without axes it takes all of
        them. The intermediate point exists to raise Z before crossing the workpiece.
        """
        if axis_w:
            self._move("rapid", self._target(axis_w), source="G" + which)
            home = dict(self.pos)
            for a in axis_w:
                home[a] = self.home[which][a]
        else:
            home = dict(self.home[which])
        self._move("rapid", home, source="G" + which)

    # --- arcs ----------------------------------------------------------------------
    def _arc(self, w, axis_w):
        """G2 (clockwise) / G3 (counterclockwise) in the active plane, with IJK center or radius R.

        Axes outside the plane (Z in G17) go linearly from start to end: helical arc.
        ``P`` (LinuxCNC) adds full turns: P2 = one extra turn before reaching the end.

        Raises:
            GcodeError: no F, no center or R, impossible R, zero radius, or different start and end
                radii (center miscalculated by the CAM).
        """
        a, b, n, ia, ib, _ = PLANES[self.plane]
        if a not in self.axes or b not in self.axes:
            raise GcodeError(f"arco en G{self.plane}: faltan los ejes {a}/{b}")
        if not self.feed:
            raise GcodeError("arco G2/G3 sin F")
        cw = self.motion == "2"
        end = self._target(axis_w)
        x, y = end[a] - self.pos[a], end[b] - self.pos[b]          # chord from start to end, in the plane
        if "R" in w:
            # --- radius format (same as grbl, mc_arc / gc_execute_block) ---
            # The center lies on the perpendicular bisector of the chord, at a distance
            # d = sqrt(r² − (c/2)²) from the midpoint (c = chord length). With
            # h = −2·d / c the center relative to the start is (x − y·h, y + x·h) / 2: the midpoint plus
            # the scaled perpendicular vector (−y, x). The sign picks the side: for G2 with positive R,
            # the short arc (≤ 180°); negative R asks for the long arc (> 180°), and the side flips.
            r = w["R"] * self.scale
            if abs(x) < 1e-9 and abs(y) < 1e-9:
                raise GcodeError("arco con R y el mismo punto inicial y final: usar I/J/K")
            h = 4.0 * r * r - x * x - y * y                       # (2d)² = (2r)² − c²
            if h < 0:
                # The chord is longer than the diameter. If only slightly (CAM rounding, within the
                # arc tolerance) an exact half circle is taken; otherwise the R is wrong.
                if h < -(4 * abs(r) * self.m.arc_tolerance + 1e-6):
                    raise GcodeError(f"R{w['R']:g} demasiado chico para llegar al punto final")
                h = 0.0
            h = -math.sqrt(h) / math.hypot(x, y)
            if not cw:
                h = -h
            if r < 0:
                h, r = -h, -r
            ci, cj = 0.5 * (x - y * h), 0.5 * (y + x * h)
            ca, cb = self.pos[a] + ci, self.pos[b] + cj
            radius = r
        else:
            # --- center format (IJK) ---
            if ia not in w and ib not in w:
                raise GcodeError(f"arco sin {ia}/{ib} ni R")
            if self.arc_absolute:
                # G90.1: IJK are work coordinates of the center (a missing letter = current position).
                ca = w.get(ia, self.work(a)) * self.scale + self._off(a)
                cb = w.get(ib, self.work(b)) * self.scale + self._off(b)
            else:
                # G91.1 (default, like Mach3 «IJ incremental»): IJK relative to the start point.
                ca = self.pos[a] + w.get(ia, 0.0) * self.scale
                cb = self.pos[b] + w.get(ib, 0.0) * self.scale
            radius = math.hypot(self.pos[a] - ca, self.pos[b] - cb)
            r_end = math.hypot(end[a] - ca, end[b] - cb)
            if radius < 1e-9:
                raise GcodeError("arco de radio cero")
            # Same criterion as LinuxCNC: allowed difference 0.005 mm or 0.1 % of the radius (whichever
            # is larger). More than that means a miscalculated center and the arc would end with a jump.
            if abs(radius - r_end) > max(0.005, 0.001 * radius):
                raise GcodeError(f"arco inválido: radio inicial {radius:.4f} y final {r_end:.4f} distintos")
        # Swept angle: difference of the start and end polar angles around the center, brought to the
        # requested direction (negative = clockwise). Start = end gives a full circle (±2π), not a null arc.
        a0 = math.atan2(self.pos[b] - cb, self.pos[a] - ca)
        a1 = math.atan2(end[b] - cb, end[a] - ca)
        sweep = a1 - a0
        if cw and sweep >= -1e-9:
            sweep -= 2 * math.pi
        elif not cw and sweep <= 1e-9:
            sweep += 2 * math.pi
        turns = int(w.get("P", 1))
        if turns < 1:
            raise GcodeError("P del arco (vueltas) debe ser ≥ 1")
        sweep += math.copysign(2 * math.pi * (turns - 1), sweep)  # extra turns, in the same direction
        self.ops.append(Op("arc", self.line, dict(self.pos), dict(end), self._job_feed(self.feed), exact_stop=self.exact,
                           arc=Arc((ca, cb), (a, b, n), radius, sweep), frame=dict(self.offsets["G54"])))
        self.pos = dict(end)

    # --- canned cycles --------------------------------------------------------------
    def _cycle(self, w, axis_w):
        """Expands a canned drilling cycle into simple moves (for controllers without cycles).

        Sequence for each hole (Z, R and Q in work coordinates, modal across lines):

        1. if the tool is below R, rapid up to R (do not cross the workpiece flush with it);
        2. rapid to the hole's X/Y; 3. rapid to R;
        4. G81: feed down to Z. G82: same + dwell P seconds at the bottom (cleans the flat bottom).
           G83 (deep peck): goes down Q at a time, retracts to R after each peck to clear the chips and
           rapids back down to :data:`PECK_CLEARANCE` above the previous bottom.
           G73 (chip break): goes down Q at a time and backs off only :data:`PECK_CLEARANCE` (faster,
           for long-chip materials);
        5. retract: G99 → to R; G98 → to the higher of the cycle's initial Z and R.

        Raises:
            GcodeError: outside G17/G90, no Z or R, Z above R, Q ≤ 0 in G83/G73, or no F.
        """
        g = self.motion
        if self.plane != "17":
            raise GcodeError("ciclos fijos solo en G17 en v0")
        if not self.absolute:
            raise GcodeError("ciclos fijos en G91 no soportados en v0")
        for k in ("Z", "R", "Q"):
            if k in w:
                self.cycle[k] = w[k] * self.scale
        if "P" in w:
            self.cycle["P"] = w["P"]              # seconds: not scaled
        z, r, q = self.cycle["Z"], self.cycle["R"], self.cycle["Q"]
        if z is None or r is None:
            raise GcodeError(f"G{g} necesita Z (fondo) y R (plano de retracción)")
        if z > r:
            raise GcodeError(f"G{g}: el fondo Z{z:g} está por encima de R{r:g}")
        if g in ("83", "73") and (q is None or q <= 0):
            raise GcodeError(f"G{g} necesita Q > 0 (profundidad de cada picada)")
        if not self.feed:
            raise GcodeError(f"G{g} sin F")
        src = "G" + g
        off_z = self._off("Z")
        zinit = self.cycle["Zinit"] if self.cycle["Zinit"] is not None else self.work("Z")
        mz = lambda v: v + off_z                                  # work → machine (Z)
        if self.work("Z") < r:
            self._move("rapid", {**self.pos, "Z": mz(r)}, src)
        xy = {k: v for k, v in axis_w.items() if k != "Z"}       # the block's Z is the bottom, not a target
        self._move("rapid", self._target(xy), src)
        self._move("rapid", {**self.pos, "Z": mz(r)}, src)
        if g in ("81", "82"):
            self._move("feed", {**self.pos, "Z": mz(z)}, src)
        else:
            depth = r                                             # bottom reached so far (work)
            while depth > z + 1e-9:
                nxt = max(depth - q, z)                           # the last peck may be shorter
                if g == "83" and depth < r:
                    self._move("rapid", {**self.pos, "Z": mz(depth + PECK_CLEARANCE)}, src)
                self._move("feed", {**self.pos, "Z": mz(nxt)}, src)
                depth = nxt
                if depth > z + 1e-9:
                    self._move("rapid", {**self.pos, "Z": mz(r if g == "83" else depth + PECK_CLEARANCE)}, src)
        if g == "82" and self.cycle["P"]:
            self.ops.append(Op("dwell", self.line, value=float(self.cycle["P"]), source=src))
        self._move("rapid", {**self.pos, "Z": mz(r if self.retract_r else max(zinit, r))}, src)


# ==================================================================================================
# ======================================== PUBLIC FUNCTIONS ========================================
# ==================================================================================================

def interpret(text: str, machine: Machine, start: dict[str, float] | None = None, block_delete: bool = True,
              optional_stop: bool = True) -> Program:
    """Interprets a complete G-code program (shortcut for ``Interpreter(...).run(text)``).

    Args:
        text: the program (text).
        machine: the machine (see ``hope.cnc.machine``).
        start: initial position in machine coordinates (mm); everything at 0 by default.
        block_delete: skip lines with ``/``.
        optional_stop: whether M1 pauses.

    Returns:
        :class:`Program` with operations and diagnostics. Does not raise on G-code errors.
    """
    return Interpreter(machine, start, block_delete, optional_stop).run(text)


def modal_text(it: Interpreter) -> str:
    """Modal state like Mach3's active G-codes line (e.g. «G0 G17 G21 G90 G94 G54 G49 G98 G64»).

    Adds spindle (M3/M4/M5), coolant (M7/M8/M9), tool, F (in program units) and S.
    Shown by the interface while running.
    """
    g = [f"G{it.motion}", f"G{it.plane}", "G20" if it.scale != 1.0 else "G21", "G90" if it.absolute else "G91",
         "G94", it.wcs, "G43" if it.tlo else "G49", "G99" if it.retract_r else "G98", "G61" if it.exact else "G64"]
    m = {"cw": "M3", "ccw": "M4"}.get(it.spindle[0], "M5")
    cool = " ".join(sorted("M7" if c == "niebla" else "M8" for c in it.coolant)) or "M9"
    return " ".join(g) + f" {m} {cool} T{it.tool or 0}" + (f" F{it.feed / it.scale:g}" if it.feed else "") + \
        (f" S{it.rpm:g}" if it.rpm else "")


def linearize(op: Op, tolerance: float) -> list[dict[str, float]]:
    """End points of the chords of an arc (the chord error does not exceed ``tolerance``).

    Geometry: a chord spanning an angle θ deviates from the arc by at most the sagitta
    ``r·(1 − cos(θ/2))``. Setting it equal to the tolerance, θ = 2·acos(1 − tol/r); it uses
    ``ceil(|sweep| / θ)`` equal chords. With tolerance ≥ radius (tiny arcs) quarter turns are used.
    Axes outside the plane (helix) are interpolated linearly with the same parameter t.

    Args:
        op: ``"arc"`` operation (with ``op.arc``).
        tolerance: maximum error (mm), normally ``machine.arc_tolerance``.

    Returns:
        List of complete positions ``{axis: mm}`` (without the start point; the last one is exactly
        ``op.end``, so as not to accumulate rounding error between operations).

    Used by the planner (timings), the limit check and the translation for controllers that have
    no arcs or handle them badly.
    """
    arc = op.arc
    a, b, _ = arc.plane
    r = arc.radius
    step = 2 * math.acos(max(-1.0, 1 - tolerance / r)) if tolerance < r else math.pi / 2
    n = max(1, math.ceil(abs(arc.sweep) / max(step, 1e-6)))
    a0 = math.atan2(op.start[b] - arc.center[1], op.start[a] - arc.center[0])
    pts = []
    for k in range(1, n + 1):
        t = k / n
        p = {ax: op.start[ax] + (op.end[ax] - op.start[ax]) * t for ax in op.start}
        if k < n:
            ang = a0 + arc.sweep * t
            p[a] = arc.center[0] + r * math.cos(ang)
            p[b] = arc.center[1] + r * math.sin(ang)
        else:
            p = dict(op.end)
        pts.append(p)
    return pts
