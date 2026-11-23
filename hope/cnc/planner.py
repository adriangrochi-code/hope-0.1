"""Planificador de movimiento con anticipación (lookahead).

Mismo modelo que grbl/grblHAL, para que el tiempo estimado coincida con lo que hará la máquina:

* cada movimiento lineal es un bloque con velocidad nominal = min(F, límite de cada eje según la
  dirección) y aceleración = min(acel_eje / |componente|);
* la velocidad en cada esquina sale de la **desviación de unión** (junction deviation): cuanto más
  cerrada la esquina, más lenta; tramos colineales no frenan;
* pasada hacia atrás y hacia adelante: nunca se entra a un bloque más rápido de lo que se puede
  frenar a tiempo;
* perfil trapezoidal por bloque (acelera, crucero, frena) con tiempos exactos.

Paradas obligatorias (velocidad 0): inicio y fin, pausas, cambios de husillo/refrigerante/herramienta,
modo de parada exacta (G61) y cambios entre rápido y avance si se pide ``stop_on_rapid``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hope.cnc.gcode import Op, Program, linearize
from hope.cnc.machine import Machine


@dataclass
class Block:
    line: int
    kind: str                  # rapid | feed
    start: tuple[float, ...]
    end: tuple[float, ...]
    length: float
    unit: tuple[float, ...]
    v_nominal: float           # mm/s
    accel: float               # mm/s²
    v_entry: float = 0.0
    v_exit: float = 0.0
    v_peak: float = 0.0
    time: float = 0.0
    stop_after: bool = False


@dataclass
class Plan:
    blocks: list[Block]
    move_time: float           # s
    dwell_time: float          # s
    pauses: int                # M0/M1 y cambios de herramienta (esperan al operador)
    feed_length: float         # mm
    rapid_length: float        # mm
    time_by_line: dict[int, float]

    @property
    def total_time(self) -> float:
        return self.move_time + self.dwell_time


def _limits(machine: Machine, unit: tuple[float, ...], axes: tuple[str, ...]) -> tuple[float, float]:
    v, a = math.inf, math.inf
    for i, ax in enumerate(axes):
        c = abs(unit[i])
        if c > 1e-12:
            v = min(v, machine.axes[ax].vmax / c)
            a = min(a, machine.axes[ax].accel / c)
    return v, a


def build_blocks(program: Program | list[Op], machine: Machine) -> tuple[list[Block], float, int]:
    ops = program.ops if isinstance(program, Program) else program
    axes = machine.axis_names
    blocks: list[Block] = []
    dwell, pauses = 0.0, 0

    def add(kind, line, s, e, feed, exact):
        s = tuple(s[a] for a in axes)
        e = tuple(e[a] for a in axes)
        d = [e[i] - s[i] for i in range(len(axes))]
        L = math.sqrt(sum(x * x for x in d))
        if L < 1e-9:
            return
        u = tuple(x / L for x in d)
        v, acc = _limits(machine, u, axes)
        if kind == "feed":
            v = min(v, feed / 60.0)
        blocks.append(Block(line, kind, s, e, L, u, v, acc, stop_after=exact))

    for op in ops:
        if op.kind in ("rapid", "feed"):
            add(op.kind, op.line, op.start, op.end, op.feed, op.exact_stop)
        elif op.kind == "arc":
            prev = op.start
            for p in linearize(op, machine.arc_tolerance):
                add("feed", op.line, prev, p, op.feed, False)
                prev = p
            if blocks and op.exact_stop:
                blocks[-1].stop_after = True
        else:
            if op.kind == "dwell":
                dwell += float(op.value)
            elif op.kind in ("pause", "tool"):
                pauses += 1
            if blocks:
                blocks[-1].stop_after = True
    return blocks, dwell, pauses


def junction_speed(prev: Block, cur: Block, deviation: float) -> float:
    cos_t = -sum(p * c for p, c in zip(prev.unit, cur.unit))
    vmax = min(prev.v_nominal, cur.v_nominal)
    if cos_t < -0.999999:                          # colineales
        return vmax
    if cos_t > 0.999999:                           # vuelta atrás
        return 0.0
    sin_half = math.sqrt(0.5 * (1.0 - cos_t))
    acc = min(prev.accel, cur.accel)
    return min(vmax, math.sqrt(acc * deviation * sin_half / (1.0 - sin_half)))


def _trapezoid(b: Block) -> None:
    a, L, vi, vf, vm = b.accel, b.length, b.v_entry, b.v_exit, b.v_nominal
    d_acc = max(0.0, (vm * vm - vi * vi) / (2 * a))
    d_dec = max(0.0, (vm * vm - vf * vf) / (2 * a))
    if d_acc + d_dec <= L:
        b.v_peak = vm
        b.time = (vm - vi) / a + (vm - vf) / a + (L - d_acc - d_dec) / vm
    else:
        vp = math.sqrt(max(vi * vi, vf * vf, (2 * a * L + vi * vi + vf * vf) / 2))
        b.v_peak = vp
        b.time = (vp - vi) / a + (vp - vf) / a


def plan(program: Program | list[Op], machine: Machine, stop_on_rapid: bool = False) -> Plan:
    blocks, dwell, pauses = build_blocks(program, machine)
    n = len(blocks)
    vj = [0.0] * (n + 1)                           # velocidad en la unión antes del bloque i (vj[n] = fin)
    for i in range(1, n):
        p, c = blocks[i - 1], blocks[i]
        if p.stop_after or (stop_on_rapid and p.kind != c.kind):
            continue
        vj[i] = junction_speed(p, c, machine.junction_deviation)
    for i in range(n - 1, -1, -1):                 # hacia atrás: poder frenar a tiempo
        b = blocks[i]
        vj[i] = min(vj[i], math.sqrt(vj[i + 1] ** 2 + 2 * b.accel * b.length))
    for i in range(n):                             # hacia adelante: poder acelerar a tiempo
        b = blocks[i]
        vj[i + 1] = min(vj[i + 1], math.sqrt(vj[i] ** 2 + 2 * b.accel * b.length))
        b.v_entry, b.v_exit = vj[i], vj[i + 1]
        _trapezoid(b)
    by_line: dict[int, float] = {}
    for b in blocks:
        by_line[b.line] = by_line.get(b.line, 0.0) + b.time
    return Plan(blocks, sum(b.time for b in blocks), dwell, pauses,
                sum(b.length for b in blocks if b.kind == "feed"),
                sum(b.length for b in blocks if b.kind == "rapid"), by_line)
