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
from dataclasses import dataclass  # Plan

from hope.cnc.gcode import Op, Program, linearize
from hope.cnc.machine import Machine


class Block:
    """Un tramo recto del plan (los arcos se dividen en varios).

    Clase con ``__slots__`` en vez de dataclass: un relieve 3D tiene cientos de miles de tramos y así
    cada uno ocupa cerca de la mitad de memoria y se crea más rápido (importa en una PC de 512 MB)."""

    __slots__ = ("accel", "end", "kind", "length", "line", "op", "start", "stop_after", "time", "unit",
                 "v_entry", "v_exit", "v_nominal", "v_peak")

    def __init__(self, line: int, kind: str, start: tuple[float, ...], end: tuple[float, ...], length: float,
                 unit: tuple[float, ...], v_nominal: float, accel: float, v_entry: float = 0.0,
                 v_exit: float = 0.0, v_peak: float = 0.0, time: float = 0.0, stop_after: bool = False,
                 op: int = -1):
        self.line = line                 # línea del programa original
        self.kind = kind                 # "rapid" | "feed"
        self.start = start               # posición inicial (máquina, en el orden de machine.axis_names)
        self.end = end                   # posición final
        self.length = length             # mm
        self.unit = unit                 # dirección (vector unitario)
        self.v_nominal = v_nominal       # mm/s: lo más rápido que permite F y cada eje en esta dirección
        self.accel = accel               # mm/s²: la aceleración que permite el eje más limitante
        self.v_entry = v_entry           # mm/s al entrar (sale del lookahead)
        self.v_exit = v_exit             # mm/s al salir
        self.v_peak = v_peak             # mm/s máxima alcanzada dentro del tramo
        self.time = time                 # s que tarda el tramo
        self.stop_after = stop_after     # parada obligatoria al terminar (G61, pausa, M3…)
        self.op = op                     # índice de la operación (Program.ops) de la que sale

    def __repr__(self) -> str:
        return f"Block(line={self.line}, kind={self.kind!r}, length={self.length:.4f}, time={self.time:.4f})"


@dataclass
class Plan:
    blocks: list[Block]
    move_time: float           # s
    dwell_time: float          # s
    pauses: int                # M0/M1 y cambios de herramienta (esperan al operador)
    feed_length: float         # mm
    rapid_length: float        # mm
    time_by_line: dict[int, float]
    time_by_op: dict[int, float]   # s de movimiento por operación (índice en Program.ops)

    @property
    def total_time(self) -> float:
        return self.move_time + self.dwell_time


def build_blocks(program: Program | list[Op], machine: Machine,
                 events: list | None = None) -> tuple[list[Block], float, int]:
    """Bloques lineales del programa. Con ``events`` se anotan las operaciones sin movimiento como
    ``(cantidad de bloques antes, op)`` (lo usa el simulador para ejecutarlas en su momento)."""
    ops = program.ops if isinstance(program, Program) else program
    axes = machine.axis_names
    # Rendimiento: límites de cada eje en listas, en el mismo orden que las coordenadas del tramo,
    # para no buscar machine.axes[nombre] cientos de miles de veces.
    vmaxs = [machine.axes[a].vmax for a in axes]
    accels = [machine.axes[a].accel for a in axes]
    rng = range(len(axes))
    sqrt = math.sqrt
    blocks: list[Block] = []
    append = blocks.append
    dwell, pauses = 0.0, 0

    def add(kind, line, s, e, feed, exact, k):
        s = tuple([s[a] for a in axes])
        e = tuple([e[a] for a in axes])
        d = [e[i] - s[i] for i in rng]
        L = sqrt(sum([x * x for x in d]))
        if L < 1e-9:                                   # tramo nulo: no aporta tiempo ni esquina
            return
        inv = 1.0 / L
        u = tuple([x * inv for x in d])
        # Velocidad y aceleración del tramo: cada eje aporta su parte según la dirección. Si el eje X
        # hace el 50 % del recorrido, el tramo puede ir al doble de la velocidad máxima de X (y lo
        # limita el eje que se quede corto primero). Igual que grbl.
        v = acc = math.inf
        for i in rng:
            c = u[i] if u[i] >= 0 else -u[i]
            if c > 1e-12:
                vi = vmaxs[i] / c
                ai = accels[i] / c
                if vi < v:                             # noqa: PLR1730  (un if es más rápido que min())
                    v = vi
                if ai < acc:                           # noqa: PLR1730
                    acc = ai
        if kind == "feed":
            f = feed / 60.0                            # mm/min → mm/s
            if f < v:                                  # noqa: PLR1730
                v = f
        append(Block(line, kind, s, e, L, u, v, acc, stop_after=exact, op=k))

    for k, op in enumerate(ops):
        if op.kind in ("rapid", "feed"):
            add(op.kind, op.line, op.start, op.end, op.feed, op.exact_stop, k)
        elif op.kind == "probe":                        # sensado: avance hasta el contacto supuesto, y parar
            add("feed", op.line, op.start, op.end, op.feed, True, k)
        elif op.kind == "arc":
            prev = op.start
            for p in linearize(op, machine.arc_tolerance):
                add("feed", op.line, prev, p, op.feed, False, k)
                prev = p
            if blocks and op.exact_stop:
                blocks[-1].stop_after = True
        else:
            if events is not None:
                events.append((len(blocks), op))
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


def distance_at(b: Block, t: float) -> float:
    """Distancia recorrida dentro del bloque a los ``t`` segundos de empezarlo (perfil trapezoidal)."""
    if t <= 0:
        return 0.0
    if t >= b.time:
        return b.length
    a, vi, vf, vp = b.accel, b.v_entry, b.v_exit, b.v_peak
    t1, t3 = (vp - vi) / a, (vp - vf) / a
    t2 = max(0.0, b.time - t1 - t3)
    if t < t1:
        return vi * t + 0.5 * a * t * t
    d1 = vi * t1 + 0.5 * a * t1 * t1
    if t < t1 + t2:
        return d1 + vp * (t - t1)
    td = t - t1 - t2
    return min(b.length, d1 + vp * t2 + vp * td - 0.5 * a * td * td)


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


def plan(program: Program | list[Op], machine: Machine, stop_on_rapid: bool = False,
         events: list | None = None) -> Plan:
    blocks, dwell, pauses = build_blocks(program, machine, events)
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
    by_op: dict[int, float] = {}
    for b in blocks:
        by_line[b.line] = by_line.get(b.line, 0.0) + b.time
        by_op[b.op] = by_op.get(b.op, 0.0) + b.time
    return Plan(blocks, sum(b.time for b in blocks), dwell, pauses,
                sum(b.length for b in blocks if b.kind == "feed"),
                sum(b.length for b in blocks if b.kind == "rapid"), by_line, by_op)
