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

Cómo encaja
-----------
* **Usa:** ``gcode`` (las operaciones ya interpretadas, ``Op``/``Program``, y ``linearize`` para
  partir los arcos en cuerdas) y ``machine`` (velocidad máxima y aceleración de cada eje,
  ``junction_deviation`` y ``arc_tolerance``).
* **Lo usan:** ``check`` (tiempo estimado del informe), ``session`` (tiempos por línea y por
  operación para la pantalla), ``backends.sim`` (el simulador avanza según el plan), ``links``
  (posición y velocidad en cada instante con ``distance_at``) y ``medir`` (prueba de rendimiento).

Unidades y convenciones
-----------------------
* Distancias en **mm**, velocidades en **mm/s**, aceleraciones en **mm/s²**, tiempos en **s**.
  El avance F del G-code viene en mm/min y se pasa a mm/s acá (``feed / 60``).
* Las posiciones de los bloques son coordenadas de **máquina** (los orígenes G54… ya aplicados por
  el intérprete), como tuplas en el orden de ``machine.axis_names``.
* Los arcos se aproximan con cuerdas (error máximo ``machine.arc_tolerance``, 0.002 mm por defecto);
  cada cuerda es un bloque, igual que hace grbl internamente.
* Parámetro de la máquina que más pesa acá: ``machine.junction_deviation`` (mm, por defecto 0.01,
  como grbl). Ver ``junction_speed``.

Ejemplo de uso
--------------
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
# ======================================== BLOQUES Y PLAN ==========================================
# ==================================================================================================


class Block:
    """Un tramo recto del plan (los arcos se dividen en varios).

    Clase con ``__slots__`` en vez de dataclass: un relieve 3D tiene cientos de miles de tramos y así
    cada uno ocupa cerca de la mitad de memoria y se crea más rápido (importa en una PC de 512 MB).

    Lo crea ``build_blocks`` con la geometría y los límites (``v_nominal``, ``accel``); ``plan``
    completa después ``v_entry``, ``v_exit``, ``v_peak`` y ``time``. Los significados y unidades de
    cada atributo están comentados en ``__init__``."""

    __slots__ = ("accel", "end", "kind", "length", "line", "op", "start", "stop_after", "time", "unit",
                 "v_entry", "v_exit", "v_nominal", "v_peak")

    def __init__(self, line: int, kind: str, start: tuple[float, ...], end: tuple[float, ...], length: float,
                 unit: tuple[float, ...], v_nominal: float, accel: float, v_entry: float = 0.0,
                 v_exit: float = 0.0, v_peak: float = 0.0, time: float = 0.0, stop_after: bool = False,
                 op: int = -1):
        """Guarda los datos tal cual (sin validar, por rendimiento: los arma ``build_blocks``)."""
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
        """Resumen corto para depurar (línea, tipo, largo en mm y tiempo en s)."""
        return f"Block(line={self.line}, kind={self.kind!r}, length={self.length:.4f}, time={self.time:.4f})"


@dataclass
class Plan:
    """Resultado de ``plan``: los bloques con sus velocidades y tiempos, más los totales del programa.

    ``time_by_line`` y ``time_by_op`` suman solo tiempo de movimiento (sin G4); la pantalla los usa
    para mostrar cuánto falta y para «Correr desde aquí»."""

    blocks: list[Block]
    move_time: float           # s
    dwell_time: float          # s
    pauses: int                # M0/M1 y cambios de herramienta (esperan al operador)
    feed_length: float         # mm de corte, solo ejes lineales X Y Z
    rapid_length: float        # mm de rápidos, solo ejes lineales
    time_by_line: dict[int, float]
    time_by_op: dict[int, float]   # s de movimiento por operación (índice en Program.ops)
    # Grados girados por cada eje rotativo (A, B, C) en avance de corte. Va aparte porque sumar
    # grados con milímetros no significa nada para el operador (el tiempo sí los mezcla: ver plan()).
    rotary_travel: dict = field(default_factory=dict)   # {eje: grados}; vacío si no hay rotativos

    @property
    def total_time(self) -> float:
        """Tiempo total estimado en s: movimiento + pausas G4. No incluye lo que espera al operador
        (M0/M1/M6), que no se puede saber de antemano: eso se informa aparte en ``pauses``."""
        return self.move_time + self.dwell_time


# ==================================================================================================
# ===================================== ARMADO DE LOS BLOQUES ======================================
# ==================================================================================================


def build_blocks(program: Program | list[Op], machine: Machine,
                 events: list | None = None) -> tuple[list[Block], float, int]:
    """Bloques lineales del programa. Con ``events`` se anotan las operaciones sin movimiento como
    ``(cantidad de bloques antes, op)`` (lo usa el simulador para ejecutarlas en su momento).

    Parámetros:
        program: el programa interpretado o directamente su lista de operaciones.
        machine: de dónde salen ``vmax``/``accel`` de cada eje y la tolerancia de arcos.
        events:  lista (normalmente vacía) donde agregar las operaciones sin movimiento, o None.

    Devuelve ``(bloques, tiempo de G4 en s, cantidad de pausas del operador)``. Los bloques salen
    con velocidades de entrada/salida en 0: las calcula ``plan``.

    Cada operación que no mueve (husillo, refrigerante, G4, M0, M6…) marca ``stop_after`` en el
    bloque anterior: la máquina tiene que estar quieta cuando se ejecuta, igual que en grbl."""
    ops = program.ops if isinstance(program, Program) else program
    axes = machine.axis_names
    # Rendimiento: límites de cada eje en listas, en el mismo orden que las coordenadas del tramo,
    # para no buscar machine.axes[nombre] cientos de miles de veces.
    vmaxs = [machine.axes[a].vmax for a in axes]
    accels = [machine.axes[a].accel for a in axes]
    rng = range(len(axes))
    sqrt = math.sqrt                                   # nombres locales: más rápidos que math.sqrt en el bucle
    blocks: list[Block] = []
    append = blocks.append
    dwell, pauses = 0.0, 0

    # --- un tramo recto ---------------------------------------------------------------------------

    def add(kind, line, s, e, feed, exact, k):
        """Agrega el tramo recto ``s``→``e`` (dicts eje→mm). ``feed`` en mm/min (solo cuenta en
        "feed"), ``exact``: parar al final (G61/sensado), ``k``: índice de la operación."""
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
            if c > 1e-12:                              # eje que casi no se mueve: no limita (y evita dividir por 0)
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
        # Los rápidos (G0) no tienen F: van a lo que den los ejes, como en cualquier controlador.
        append(Block(line, kind, s, e, L, u, v, acc, stop_after=exact, op=k))

    # --- recorrido de las operaciones -------------------------------------------------------------

    for k, op in enumerate(ops):
        if op.kind in ("rapid", "feed"):
            add(op.kind, op.line, op.start, op.end, op.feed, op.exact_stop, k)
        elif op.kind == "probe":                        # sensado: avance hasta el contacto supuesto, y parar
            add("feed", op.line, op.start, op.end, op.feed, True, k)
        elif op.kind == "arc":
            # El arco se recorre como cuerdas cortas, igual que grbl (que también los parte en
            # segmentos): las esquinas entre cuerdas son casi colineales y casi no frenan.
            prev = op.start
            for p in linearize(op, machine.arc_tolerance):
                add("feed", op.line, prev, p, op.feed, False, k)
                prev = p
            if blocks and op.exact_stop:
                blocks[-1].stop_after = True
        else:
            # Operación sin movimiento: se ejecuta con la máquina quieta.
            if events is not None:
                events.append((len(blocks), op))
            if op.kind == "dwell":
                dwell += float(op.value)               # G4 P: segundos
            elif op.kind in ("pause", "tool"):
                pauses += 1
            if blocks:
                blocks[-1].stop_after = True
    return blocks, dwell, pauses


# ==================================================================================================
# ================================ FÍSICA: ESQUINAS Y TRAPECIOS ====================================
# ==================================================================================================


def junction_speed(prev: Block, cur: Block, deviation: float) -> float:
    """Velocidad máxima (mm/s) para pasar de ``prev`` a ``cur`` sin frenar del todo en la esquina.

    Modelo de **desviación de unión** de grbl (junction deviation). En vez de una «sacudida» (jerk)
    fija por eje, se imagina que la máquina no sigue la esquina en punta sino un arco circular
    tangente a los dos tramos, cuyo punto más cercano a la esquina queda a ``deviation`` mm de ella.
    Recorrer ese arco a velocidad v pide una aceleración centrípeta v²/R; se permite como mucho la
    aceleración del tramo, así que:

        v = √(a · R)        con   R = δ · sen(θ/2) / (1 − sen(θ/2))

    donde δ = ``deviation`` (mm), a = la menor aceleración de los dos tramos (mm/s²) y θ es el ángulo
    interno de la esquina (180° = seguir derecho, 0° = volver por donde vino). sen(θ/2) sale del
    coseno con la identidad del ángulo mitad: sen(θ/2) = √((1 − cos θ)/2), y cos θ = −(u_prev · u_cur)
    (el signo menos porque el ángulo interno es entre la dirección de llegada invertida y la de
    salida). Así se calcula con un producto escalar, sin trigonometría: rápido y sin ``acos``.

    En la práctica la máquina SÍ pasa por la punta (el arco es solo un modelo para elegir la
    velocidad); con δ chico el error de seguimiento que se tolera es chico.

    Parámetros:
        prev, cur:  bloques consecutivos (con ``unit``, ``v_nominal`` y ``accel`` ya calculados).
        deviation:  ``machine.junction_deviation`` en mm. Típico 0.01–0.05; 0 = parar en cada esquina.

    Devuelve la velocidad en mm/s, nunca mayor que la nominal de ninguno de los dos tramos."""
    cos_t = -sum(p * c for p, c in zip(prev.unit, cur.unit))
    vmax = min(prev.v_nominal, cur.v_nominal)
    if cos_t < -0.999999:                          # colineales
        return vmax                                # (R → ∞: no hay esquina, solo manda la nominal)
    if cos_t > 0.999999:                           # vuelta atrás
        return 0.0                                 # (R → 0: hay que parar y salir para el otro lado)
    sin_half = math.sqrt(0.5 * (1.0 - cos_t))      # sen(θ/2), identidad del ángulo mitad
    acc = min(prev.accel, cur.accel)
    return min(vmax, math.sqrt(acc * deviation * sin_half / (1.0 - sin_half)))   # √(a · R)


def distance_at(b: Block, t: float) -> float:
    """Distancia recorrida dentro del bloque a los ``t`` segundos de empezarlo (perfil trapezoidal).

    Sirve para saber dónde está la herramienta en cada instante (simulación y animación en
    pantalla, ``links``). El bloque tiene que estar ya planificado (``v_entry``, ``v_exit``,
    ``v_peak`` y ``time`` calculados por ``plan``). ``t`` en s; devuelve mm, entre 0 y ``b.length``.

    El trapecio tiene tres fases con aceleración constante ``a``:

    1. aceleración, de ``v_entry`` a ``v_peak``:  t1 = (vp − vi)/a,   d = vi·t + ½·a·t²
    2. crucero a ``v_peak`` (puede durar 0, perfil «triangular»):  d = d1 + vp·(t − t1)
    3. frenado, de ``v_peak`` a ``v_exit``:  t3 = (vp − vf)/a,   d = d1 + vp·t2 + vp·td − ½·a·td²"""
    if t <= 0:
        return 0.0
    if t >= b.time:
        return b.length
    a, vi, vf, vp = b.accel, b.v_entry, b.v_exit, b.v_peak
    t1, t3 = (vp - vi) / a, (vp - vf) / a          # duración de la aceleración y del frenado (s)
    t2 = max(0.0, b.time - t1 - t3)                # crucero; max() por redondeo en perfiles triangulares
    if t < t1:
        return vi * t + 0.5 * a * t * t
    d1 = vi * t1 + 0.5 * a * t1 * t1               # mm recorridos al terminar de acelerar
    if t < t1 + t2:
        return d1 + vp * (t - t1)
    td = t - t1 - t2                               # s que lleva frenando
    return min(b.length, d1 + vp * t2 + vp * td - 0.5 * a * td * td)   # min(): no pasarse por redondeo


def _trapezoid(b: Block) -> None:
    """Calcula ``v_peak`` (mm/s) y ``time`` (s) del bloque con sus ``v_entry``/``v_exit`` ya fijadas.

    Con v² = v0² + 2·a·d (movimiento con aceleración constante):

    * distancia para acelerar de vi a vm:  d_acc = (vm² − vi²) / (2a)
    * distancia para frenar de vm a vf:    d_dec = (vm² − vf²) / (2a)

    Si entran en el tramo (d_acc + d_dec ≤ L) el perfil es un **trapecio**: llega a la nominal y la
    mantiene el resto. Si no, es un **triángulo**: acelera hasta un pico vp y frena enseguida. vp sale
    de igualar las dos fases con d1 + d2 = L:

        vp² = vi² + 2a·d1 = vf² + 2a·d2    →    vp² = (2a·L + vi² + vf²) / 2

    El ``max`` con vi² y vf² cubre errores de redondeo (el lookahead ya garantiza que se puede)."""
    a, L, vi, vf, vm = b.accel, b.length, b.v_entry, b.v_exit, b.v_nominal
    d_acc = max(0.0, (vm * vm - vi * vi) / (2 * a))
    d_dec = max(0.0, (vm * vm - vf * vf) / (2 * a))
    if d_acc + d_dec <= L:
        # trapecio: tiempo acelerando + tiempo frenando + crucero a la nominal
        b.v_peak = vm
        b.time = (vm - vi) / a + (vm - vf) / a + (L - d_acc - d_dec) / vm
    else:
        # triángulo: no llega a la nominal
        vp = math.sqrt(max(vi * vi, vf * vf, (2 * a * L + vi * vi + vf * vf) / 2))
        b.v_peak = vp
        b.time = (vp - vi) / a + (vp - vf) / a


# ==================================================================================================
# ========================================= PLANIFICADOR ===========================================
# ==================================================================================================


def plan(program: Program | list[Op], machine: Machine, stop_on_rapid: bool = False,
         events: list | None = None) -> Plan:
    """Planifica todo el programa y devuelve el ``Plan`` con tiempos exactos.

    Parámetros:
        program:       programa interpretado (o su lista de operaciones).
        machine:       límites por eje y ``junction_deviation``.
        stop_on_rapid: True = parar del todo al pasar de G0 a G1 o al revés (algunos controladores
                       lo hacen; grbl no). Por defecto False.
        events:        ver ``build_blocks`` (operaciones sin movimiento, para el simulador).

    Pasos (los mismos que el planificador de grbl, pero de una sola vez sobre todo el programa en
    vez de con un búfer de 16–32 bloques; el resultado es igual salvo programas raros con
    muchísimos tramos minúsculos seguidos):

    1. **Esquinas**: la velocidad tope en cada unión con ``junction_speed`` (0 si hay parada).
    2. **Pasada hacia atrás** (del final al principio): la velocidad al entrar a un bloque no puede
       superar la que permite frenar dentro de él hasta la velocidad de salida:
       v_entrada ≤ √(v_salida² + 2·a·L). Así nunca se llega a una esquina o a una parada demasiado
       rápido, aunque la parada esté muchos tramos cortos más adelante.
    3. **Pasada hacia adelante** (del principio al final): la velocidad al salir no puede superar
       la que se alcanza acelerando desde la de entrada: v_salida ≤ √(v_entrada² + 2·a·L).
    4. **Trapecio** de cada bloque con ``_trapezoid`` → tiempo de cada tramo.

    No lanza errores propios: ``machine`` ya rechaza al cargarla una máquina con aceleración 0."""
    blocks, dwell, pauses = build_blocks(program, machine, events)
    n = len(blocks)
    vj = [0.0] * (n + 1)                           # velocidad en la unión antes del bloque i (vj[n] = fin)
    # vj[0] = 0 (arranca quieta) y vj[n] = 0 (termina quieta) no se tocan en el bucle de esquinas.
    for i in range(1, n):
        p, c = blocks[i - 1], blocks[i]
        if p.stop_after or (stop_on_rapid and p.kind != c.kind):
            continue                               # parada obligatoria: vj[i] queda en 0
        vj[i] = junction_speed(p, c, machine.junction_deviation)
    for i in range(n - 1, -1, -1):                 # hacia atrás: poder frenar a tiempo
        b = blocks[i]
        vj[i] = min(vj[i], math.sqrt(vj[i + 1] ** 2 + 2 * b.accel * b.length))
    for i in range(n):                             # hacia adelante: poder acelerar a tiempo
        b = blocks[i]
        vj[i + 1] = min(vj[i + 1], math.sqrt(vj[i] ** 2 + 2 * b.accel * b.length))
        b.v_entry, b.v_exit = vj[i], vj[i + 1]
        _trapezoid(b)
    # --- totales por línea y por operación ---------------------------------------------------------
    by_line: dict[int, float] = {}
    by_op: dict[int, float] = {}
    for b in blocks:
        by_line[b.line] = by_line.get(b.line, 0.0) + b.time
        by_op[b.op] = by_op.get(b.op, 0.0) + b.time
    # --- recorridos: mm lineales y grados de cada eje rotativo, por separado -----------------------
    # El largo de un bloque mezcla mm y grados (así lo hacen grbl y Mach3 para el avance: F se aplica
    # al vector entero, con los grados como si fueran mm). Para informar al operador se separan: la
    # parte lineal es largo × |componente lineal de la dirección|, y cada rotativo, largo × |su parte|.
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
            part = b.length                            # 3 ejes: todo es lineal (camino rápido)
        if b.kind == "feed":
            feed_mm += part
        else:
            rapid_mm += part
    return Plan(blocks, sum(b.time for b in blocks), dwell, pauses, feed_mm, rapid_mm, by_line, by_op,
                {a: v for a, v in turned.items() if v > 1e-9})
