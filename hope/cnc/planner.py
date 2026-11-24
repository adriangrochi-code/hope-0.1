"""Motion planner with lookahead.

Same model as grbl/grblHAL, so the estimated time matches what the machine will do:

* each linear move is a block with nominal speed = min(F, each axis' limit along the direction)
  and acceleration = min(axis_accel / |component|);
* the speed at each corner comes from the **junction deviation**: the sharper the corner, the
  slower; collinear segments do not slow down;
* backward and forward pass: a block is never entered faster than it can decelerate in time;
* trapezoidal profile per block (accelerate, cruise, decelerate) with exact times.

Mandatory stops (speed 0): start and end, pauses, spindle/coolant/tool changes, exact stop mode
(G61) and changes between rapid and feed if ``stop_on_rapid`` is requested.

How it fits
-----------
* **Uses:** ``gcode`` (the already interpreted operations, ``Op``/``Program``, and ``linearize``
  to split arcs into chords) and ``machine`` (maximum speed and acceleration of each axis,
  ``junction_deviation`` and ``arc_tolerance``).
* **Used by:** ``check`` (the report's estimated time), ``session`` (per-line and per-operation
  times for the screen), ``backends.sim`` (the simulator advances according to the plan), ``links``
  (position and speed at each instant with ``distance_at``) and ``medir`` (performance benchmark).

Units and conventions
---------------------
* Distances in **mm**, speeds in **mm/s**, accelerations in **mm/s²**, times in **s**.
  The G-code feed F comes in mm/min and is converted to mm/s here (``feed / 60``).
* Block positions are **machine** coordinates (G54… offsets already applied by the interpreter),
  as tuples in the order of ``machine.axis_names``.
* Arcs are approximated with chords (maximum error ``machine.arc_tolerance``, 0.002 mm by default);
  each chord is a block, just like grbl does internally.
* The machine parameter that matters most here: ``machine.junction_deviation`` (mm, default 0.01,
  like grbl). See ``junction_speed``.

Usage example
-------------
::

    from hope.cnc.gcode import interpret
    from hope.cnc.machine import load_machine
    from hope.cnc.planner import plan

    m = load_machine("configs/cnc_maquina.yaml")
    p = plan(interpret(open("pieza.nc").read(), m), m)
    print(p.total_time, "s;", len(p.blocks), "tramos;", p.feed_length, "mm de corte")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field  # Plan

from hope.cnc.gcode import Op, Program, linearize
from hope.cnc.machine import ROTARY, Machine

# ==================================================================================================
# ======================================== BLOCKS AND PLAN =========================================
# ==================================================================================================


class Block:
    """A straight segment of the plan (arcs are split into several).

    A class with ``__slots__`` instead of a dataclass: a 3D relief has hundreds of thousands of
    segments and this way each one takes about half the memory and is created faster (it matters on
    a 512 MB PC).

    ``build_blocks`` creates it with the geometry and the limits (``v_nominal``, ``accel``); ``plan``
    later fills in ``v_entry``, ``v_exit``, ``v_peak`` and ``time``. The meaning and units of each
    attribute are commented in ``__init__``."""

    __slots__ = ("accel", "end", "kind", "length", "line", "op", "start", "stop_after", "time", "unit",
                 "v_entry", "v_exit", "v_nominal", "v_peak")

    def __init__(self, line: int, kind: str, start: tuple[float, ...], end: tuple[float, ...], length: float,
                 unit: tuple[float, ...], v_nominal: float, accel: float, v_entry: float = 0.0,
                 v_exit: float = 0.0, v_peak: float = 0.0, time: float = 0.0, stop_after: bool = False,
                 op: int = -1):
        """Store the data as is (no validation, for performance: ``build_blocks`` builds them)."""
        self.line = line                 # line of the original program
        self.kind = kind                 # "rapid" | "feed"
        self.start = start               # start position (machine, in machine.axis_names order)
        self.end = end                   # end position
        self.length = length             # mm
        self.unit = unit                 # direction (unit vector)
        self.v_nominal = v_nominal       # mm/s: the fastest F and every axis allow in this direction
        self.accel = accel               # mm/s²: the acceleration the most limiting axis allows
        self.v_entry = v_entry           # mm/s on entry (from the lookahead)
        self.v_exit = v_exit             # mm/s on exit
        self.v_peak = v_peak             # maximum mm/s reached within the segment
        self.time = time                 # s the segment takes
        self.stop_after = stop_after     # mandatory stop at the end (G61, pause, M3…)
        self.op = op                     # index of the operation (Program.ops) it comes from

    def __repr__(self) -> str:
        """Short summary for debugging (line, kind, length in mm and time in s)."""
        return f"Block(line={self.line}, kind={self.kind!r}, length={self.length:.4f}, time={self.time:.4f})"


@dataclass
class Plan:
    """Result of ``plan``: the blocks with their speeds and times, plus the program totals.

    ``time_by_line`` and ``time_by_op`` add up only motion time (no G4); the screen uses them to show
    how much is left and for «Correr desde aquí» (Run From Here)."""

    blocks: list[Block]
    move_time: float           # s
    dwell_time: float          # s
    pauses: int                # M0/M1 and tool changes (they wait for the operator)
    feed_length: float         # mm of cutting, linear axes X Y Z only
    rapid_length: float        # mm of rapids, linear axes only
    time_by_line: dict[int, float]
    time_by_op: dict[int, float]   # s of motion per operation (index into Program.ops)
    # Degrees turned by each rotary axis (A, B, C) at cutting feed. Kept separate because adding
    # degrees to millimetres means nothing to the operator (time does mix them: see plan()).
    rotary_travel: dict = field(default_factory=dict)   # {axis: degrees}; empty if there are no rotary axes

    @property
    def total_time(self) -> float:
        """Total estimated time in s: motion + G4 dwells. It does not include waiting for the operator
        (M0/M1/M6), which cannot be known in advance: that is reported separately in ``pauses``."""
        return self.move_time + self.dwell_time


# ==================================================================================================
# ======================================== BUILDING BLOCKS =========================================
# ==================================================================================================


def build_blocks(program: Program | list[Op], machine: Machine,
                 events: list | None = None) -> tuple[list[Block], float, int]:
    """Linear blocks of the program. With ``events``, operations without motion are recorded as
    ``(number of blocks before, op)`` (the simulator uses it to execute them at the right moment).

    Parameters:
        program: the interpreted program or directly its list of operations.
        machine: where each axis' ``vmax``/``accel`` and the arc tolerance come from.
        events:  list (normally empty) to append the operations without motion to, or None.

    Returns ``(blocks, G4 time in s, number of operator pauses)``. Blocks come out with entry/exit
    speeds at 0: ``plan`` computes them.

    Each operation that does not move (spindle, coolant, G4, M0, M6…) sets ``stop_after`` on the
    previous block: the machine must be standing still when it executes, just like in grbl."""
    ops = program.ops if isinstance(program, Program) else program
    axes = machine.axis_names
    # Performance: each axis' limits in lists, in the same order as the segment coordinates,
    # to avoid looking up machine.axes[name] hundreds of thousands of times.
    vmaxs = [machine.axes[a].vmax for a in axes]
    accels = [machine.axes[a].accel for a in axes]
    rng = range(len(axes))
    sqrt = math.sqrt                                   # local names: faster than math.sqrt in the loop
    blocks: list[Block] = []
    append = blocks.append
    dwell, pauses = 0.0, 0

    # --- one straight segment ---------------------------------------------------------------------

    def add(kind, line, s, e, feed, exact, k):
        """Append the straight segment ``s``→``e`` (axis→mm dicts). ``feed`` in mm/min (only counts
        for "feed"), ``exact``: stop at the end (G61/probing), ``k``: operation index."""
        s = tuple([s[a] for a in axes])
        e = tuple([e[a] for a in axes])
        d = [e[i] - s[i] for i in rng]
        L = sqrt(sum([x * x for x in d]))
        if L < 1e-9:                                   # zero-length segment: adds no time and no corner
            return
        inv = 1.0 / L
        u = tuple([x * inv for x in d])
        # Segment speed and acceleration: each axis contributes its share according to the direction.
        # If the X axis does 50 % of the travel, the segment can go at twice X's maximum speed (and it
        # is limited by whichever axis runs short first). Same as grbl.
        v = acc = math.inf
        for i in rng:
            c = u[i] if u[i] >= 0 else -u[i]
            if c > 1e-12:                              # axis that barely moves: no limit (and avoids dividing by 0)
                vi = vmaxs[i] / c
                ai = accels[i] / c
                if vi < v:                             # noqa: PLR1730  (an if is faster than min())
                    v = vi
                if ai < acc:                           # noqa: PLR1730
                    acc = ai
        if kind == "feed":
            f = feed / 60.0                            # mm/min → mm/s
            if f < v:                                  # noqa: PLR1730
                v = f
        # Rapids (G0) have no F: they go as fast as the axes allow, as in any controller.
        append(Block(line, kind, s, e, L, u, v, acc, stop_after=exact, op=k))

    # --- walk through the operations --------------------------------------------------------------

    for k, op in enumerate(ops):
        if op.kind in ("rapid", "feed"):
            add(op.kind, op.line, op.start, op.end, op.feed, op.exact_stop, k)
        elif op.kind == "probe":                        # probing: feed to the assumed contact, and stop
            add("feed", op.line, op.start, op.end, op.feed, True, k)
        elif op.kind == "arc":
            # The arc is traversed as short chords, just like grbl (which also splits them into
            # segments): the corners between chords are nearly collinear and barely slow down.
            prev = op.start
            for p in linearize(op, machine.arc_tolerance):
                add("feed", op.line, prev, p, op.feed, False, k)
                prev = p
            if blocks and op.exact_stop:
                blocks[-1].stop_after = True
        else:
            # Operation without motion: executed with the machine standing still.
            if events is not None:
                events.append((len(blocks), op))
            if op.kind == "dwell":
                dwell += float(op.value)               # G4 P: seconds
            elif op.kind in ("pause", "tool"):
                pauses += 1
            if blocks:
                blocks[-1].stop_after = True
    return blocks, dwell, pauses


# ==================================================================================================
# ================================ PHYSICS: CORNERS AND TRAPEZOIDS =================================
# ==================================================================================================


def junction_speed(prev: Block, cur: Block, deviation: float) -> float:
    """Maximum speed (mm/s) to go from ``prev`` to ``cur`` without fully stopping at the corner.

    grbl's **junction deviation** model. Instead of a fixed per-axis «jerk», the machine is imagined
    to follow not the sharp corner but a circular arc tangent to both segments, whose point closest to
    the corner lies ``deviation`` mm from it. Traversing that arc at speed v requires a centripetal
    acceleration v²/R; at most the segment's acceleration is allowed, so:

        v = √(a · R)        with   R = δ · sin(θ/2) / (1 − sin(θ/2))

    where δ = ``deviation`` (mm), a = the smaller acceleration of the two segments (mm/s²) and θ is
    the corner's interior angle (180° = go straight on, 0° = go back the way it came). sin(θ/2) comes
    from the cosine with the half-angle identity: sin(θ/2) = √((1 − cos θ)/2), and
    cos θ = −(u_prev · u_cur) (minus sign because the interior angle is between the reversed incoming
    direction and the outgoing one). So it is computed with a dot product, no trigonometry: fast and
    without ``acos``.

    In practice the machine DOES go through the sharp corner (the arc is only a model to choose the
    speed); with a small δ the tolerated following error is small.

    Parameters:
        prev, cur:  consecutive blocks (with ``unit``, ``v_nominal`` and ``accel`` already computed).
        deviation:  ``machine.junction_deviation`` in mm. Typical 0.01–0.05; 0 = stop at every corner.

    Returns the speed in mm/s, never higher than the nominal speed of either segment."""
    cos_t = -sum(p * c for p, c in zip(prev.unit, cur.unit))
    vmax = min(prev.v_nominal, cur.v_nominal)
    if cos_t < -0.999999:                          # collinear
        return vmax                                # (R → ∞: no corner, only the nominal speed limits)
    if cos_t > 0.999999:                           # reversal
        return 0.0                                 # (R → 0: must stop and leave the other way)
    sin_half = math.sqrt(0.5 * (1.0 - cos_t))      # sin(θ/2), half-angle identity
    acc = min(prev.accel, cur.accel)
    return min(vmax, math.sqrt(acc * deviation * sin_half / (1.0 - sin_half)))   # √(a · R)


def distance_at(b: Block, t: float) -> float:
    """Distance travelled within the block ``t`` seconds after it starts (trapezoidal profile).

    Used to know where the tool is at each instant (simulation and on-screen animation, ``links``).
    The block must already be planned (``v_entry``, ``v_exit``, ``v_peak`` and ``time`` computed by
    ``plan``). ``t`` in s; returns mm, between 0 and ``b.length``.

    The trapezoid has three phases with constant acceleration ``a``:

    1. acceleration, from ``v_entry`` to ``v_peak``:  t1 = (vp − vi)/a,   d = vi·t + ½·a·t²
    2. cruise at ``v_peak`` (may last 0, «triangular» profile):  d = d1 + vp·(t − t1)
    3. deceleration, from ``v_peak`` to ``v_exit``:  t3 = (vp − vf)/a,   d = d1 + vp·t2 + vp·td − ½·a·td²"""
    if t <= 0:
        return 0.0
    if t >= b.time:
        return b.length
    a, vi, vf, vp = b.accel, b.v_entry, b.v_exit, b.v_peak
    t1, t3 = (vp - vi) / a, (vp - vf) / a          # duration of acceleration and deceleration (s)
    t2 = max(0.0, b.time - t1 - t3)                # cruise; max() for rounding in triangular profiles
    if t < t1:
        return vi * t + 0.5 * a * t * t
    d1 = vi * t1 + 0.5 * a * t1 * t1               # mm travelled when acceleration ends
    if t < t1 + t2:
        return d1 + vp * (t - t1)
    td = t - t1 - t2                               # s spent decelerating so far
    return min(b.length, d1 + vp * t2 + vp * td - 0.5 * a * td * td)   # min(): do not overshoot due to rounding


def _trapezoid(b: Block) -> None:
    """Compute the block's ``v_peak`` (mm/s) and ``time`` (s) with its ``v_entry``/``v_exit`` already set.

    With v² = v0² + 2·a·d (constant-acceleration motion):

    * distance to accelerate from vi to vm:  d_acc = (vm² − vi²) / (2a)
    * distance to decelerate from vm to vf:  d_dec = (vm² − vf²) / (2a)

    If they fit in the segment (d_acc + d_dec ≤ L) the profile is a **trapezoid**: it reaches the
    nominal speed and holds it for the rest. Otherwise it is a **triangle**: it accelerates up to a
    peak vp and decelerates right away. vp comes from equating both phases with d1 + d2 = L:

        vp² = vi² + 2a·d1 = vf² + 2a·d2    →    vp² = (2a·L + vi² + vf²) / 2

    The ``max`` with vi² and vf² covers rounding errors (the lookahead already guarantees it is possible)."""
    a, L, vi, vf, vm = b.accel, b.length, b.v_entry, b.v_exit, b.v_nominal
    d_acc = max(0.0, (vm * vm - vi * vi) / (2 * a))
    d_dec = max(0.0, (vm * vm - vf * vf) / (2 * a))
    if d_acc + d_dec <= L:
        # trapezoid: time accelerating + time decelerating + cruise at nominal speed
        b.v_peak = vm
        b.time = (vm - vi) / a + (vm - vf) / a + (L - d_acc - d_dec) / vm
    else:
        # triangle: does not reach the nominal speed
        vp = math.sqrt(max(vi * vi, vf * vf, (2 * a * L + vi * vi + vf * vf) / 2))
        b.v_peak = vp
        b.time = (vp - vi) / a + (vp - vf) / a


# ==================================================================================================
# ============================================ PLANNER =============================================
# ==================================================================================================


def plan(program: Program | list[Op], machine: Machine, stop_on_rapid: bool = False,
         events: list | None = None) -> Plan:
    """Plan the whole program and return the ``Plan`` with exact times.

    Parameters:
        program:       interpreted program (or its list of operations).
        machine:       per-axis limits and ``junction_deviation``.
        stop_on_rapid: True = come to a full stop when switching from G0 to G1 or back (some
                       controllers do; grbl does not). Default False.
        events:        see ``build_blocks`` (operations without motion, for the simulator).

    Steps (the same as grbl's planner, but in one go over the whole program instead of with a
    16–32 block buffer; the result is the same except for odd programs with a huge number of tiny
    consecutive segments):

    1. **Corners**: the top speed at each junction with ``junction_speed`` (0 if there is a stop).
    2. **Backward pass** (from end to start): the entry speed of a block cannot exceed the one that
       allows decelerating within it down to the exit speed: v_entry ≤ √(v_exit² + 2·a·L). That
       way a corner or a stop is never reached too fast, even if the stop is many short segments ahead.
    3. **Forward pass** (from start to end): the exit speed cannot exceed what is reached by
       accelerating from the entry speed: v_exit ≤ √(v_entry² + 2·a·L).
    4. **Trapezoid** of each block with ``_trapezoid`` → time of each segment.

    Raises no errors of its own: ``machine`` already rejects a machine with zero acceleration on load."""
    blocks, dwell, pauses = build_blocks(program, machine, events)
    n = len(blocks)
    vj = [0.0] * (n + 1)                           # speed at the junction before block i (vj[n] = end)
    # vj[0] = 0 (starts at rest) and vj[n] = 0 (ends at rest) are not touched in the corner loop.
    for i in range(1, n):
        p, c = blocks[i - 1], blocks[i]
        if p.stop_after or (stop_on_rapid and p.kind != c.kind):
            continue                               # mandatory stop: vj[i] stays at 0
        vj[i] = junction_speed(p, c, machine.junction_deviation)
    for i in range(n - 1, -1, -1):                 # backward: able to decelerate in time
        b = blocks[i]
        vj[i] = min(vj[i], math.sqrt(vj[i + 1] ** 2 + 2 * b.accel * b.length))
    for i in range(n):                             # forward: able to accelerate in time
        b = blocks[i]
        vj[i + 1] = min(vj[i + 1], math.sqrt(vj[i] ** 2 + 2 * b.accel * b.length))
        b.v_entry, b.v_exit = vj[i], vj[i + 1]
        _trapezoid(b)
    # --- totals per line and per operation --------------------------------------------------------
    by_line: dict[int, float] = {}
    by_op: dict[int, float] = {}
    for b in blocks:
        by_line[b.line] = by_line.get(b.line, 0.0) + b.time
        by_op[b.op] = by_op.get(b.op, 0.0) + b.time
    # --- travel: linear mm and degrees of each rotary axis, separately ----------------------------
    # A block's length mixes mm and degrees (that is how grbl and Mach3 handle feed: F applies to the
    # whole vector, with degrees as if they were mm). To report to the operator they are separated: the
    # linear part is length × |linear component of the direction|, and each rotary, length × |its part|.
    axes = machine.axis_names
    lin = [i for i, a in enumerate(axes) if a not in ROTARY]
    rot = [(i, a) for i, a in enumerate(axes) if a in ROTARY]
    feed_mm = rapid_mm = 0.0
    turned = {a: 0.0 for _, a in rot}
    for b in blocks:
        if rot:
            part = b.length * math.sqrt(sum([b.unit[i] * b.unit[i] for i in lin]))
            if b.kind == "feed":
                for i, a in rot:
                    turned[a] += b.length * abs(b.unit[i])
        else:
            part = b.length                            # 3 axes: everything is linear (fast path)
        if b.kind == "feed":
            feed_mm += part
        else:
            rapid_mm += part
    return Plan(blocks, sum(b.time for b in blocks), dwell, pauses, feed_mm, rapid_mm, by_line, by_op,
                {a: v for a, v in turned.items() if v > 1e-9})
