"""Intérprete de G-code (RS274/NGC con los hábitos de Mach3, LinuxCNC y grbl).

Convierte el texto en una lista de operaciones (:class:`Op`) en **coordenadas de máquina** y mm,
con diagnósticos por línea. No mueve nada: el resultado lo usan el planificador, la revisión y la
traducción a cada controlador.

Soportado (v0):

* movimiento: G0, G1, G2/G3 (IJK incremental o absoluto con G90.1, o R; vueltas extra con P; arcos
  helicoidales), G4 (P en segundos), G28/G30 (con punto intermedio), G53;
* modos: G17/G18/G19, G20/G21, G90/G91, G90.1/G91.1, G94, G61/G61.1/G64, G98/G99;
* coordenadas: G54–G59, G10 L2/L20, G92/G92.1, G43 Hn (tabla de herramientas), G43.1, G49;
* ciclos fijos en G17 y G90: G81, G82, G83 (picado), G73 (rompe viruta), G80 — se expanden en
  movimientos simples, así funcionan también en grbl, que no los tiene;
* M0/M1, M2/M30, M3/M4/M5 con S, M6 con T, M7/M8/M9; comentarios ``( )`` y ``;``, ``%``, N, ``/``.

También: subprogramas ``O1000`` … ``M99`` llamados con ``M98 P1000 [Ln]`` (con repeticiones y hasta
:data:`MAX_CALL_DEPTH` niveles); los códigos propios de Mach3 G12/G13 (círculo), G15/G16 (coordenadas
polares) y G52 (origen temporal); sensado G38.2/G38.3 y G31 (el de Mach3); G28.1/G30.1 (guardar
posición de referencia).

No soportado (error claro): parámetros y expresiones (``#1``, ``[ ]``), compensación de radio
G41/G42, tiempo inverso G93, escala G50/G51 y sensado alejándose G38.4/G38.5. Las macros de usuario
de Mach3 (M100 en adelante) se ignoran con un aviso: hay que reescribirlas.

Cómo encaja
-----------
* Usa ``hope.cnc.machine`` (ejes, orígenes G54–G59, tabla de herramientas, plasma, perfiles ``job_*``).
* Lo usan ``planner.py`` (tiempos con aceleración), ``check.py`` (límites, husillo, envolvente),
  ``dialect.py`` (traducción a grbl/LinuxCNC/Marlin), ``session.py`` y ``links.py`` (interfaz y estado
  modal en vivo), ``controller.py``, ``backends/sim.py``, ``medir.py``, ``tool.py`` y ``__main__.py``.

Convenciones
------------
* Todas las posiciones de :class:`Op` están en **coordenadas de máquina** y mm (grados en A/B/C):
  los orígenes de trabajo, G92, G52, el largo de herramienta y las pulgadas ya están aplicados.
  Así el planificador y la revisión de límites no necesitan saber nada de modos.
* Coordenada de trabajo = máquina − (origen G54..G59 + G92 + G52 + largo de herramienta en Z).
* Avances (F) en mm/min; tiempos (G4 P, G82 P) en segundos, como LinuxCNC y grbl.
* Los errores no cortan la interpretación: se anotan en :attr:`Program.diags` con su número de línea
  y se sigue con la próxima (como el «verificar» de un CAM), para mostrar todos de una vez.

Ejemplo de uso::

    from hope.cnc.gcode import interpret
    from hope.cnc.machine import load_machine
    prog = interpret("G21 G90\\nG0 X10 Y10\\nG1 Z-1 F300\\nG2 X20 Y10 R5\\nM30", load_machine(None))
    if not prog.ok:
        for d in prog.errors:
            print(d)            # «línea 4: …»
    for op in prog.ops:
        print(op.kind, op.end)  # rapid {'X': 10.0, 'Y': 10.0, 'Z': 0.0} …
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from hope.cnc.machine import ROTARY, WCS, Machine

# ==================================================================================================
# ===================================== CONFIGURACIÓN Y TABLAS =====================================
# ==================================================================================================

INCH = 25.4                     # mm por pulgada (G20). Exacto por definición.
# Distancia (mm) a la que G83/G73 vuelven a bajar en rápido antes del fondo de la picada anterior,
# para no chocar con la viruta que quedó en el agujero. 0.254 mm = 0.010" como LinuxCNC. Típico
# 0.1–1. Muy chico = la broca puede tocar viruta en rápido; muy grande = tiempo perdido en avance.
PECK_CLEARANCE = 0.254          # mm que el G83/G73 deja antes del fondo anterior (0.010" como LinuxCNC)

# --- grupos modales de RS274/NGC ------------------------------------------------------------------
# Códigos de movimiento (grupo 1). G80 cancela el ciclo fijo; 81/82/83/73 son ciclos fijos.
MOTION = {"0", "1", "2", "3", "80", "81", "82", "83", "73"}
CANNED = {"81", "82", "83", "73"}
# Grupos modales: dos G del mismo grupo en un bloque son un error (RS274/NGC, tabla 4). El último
# grupo son los no modales (grupo 0): no entra en esa comprobación (ver Interpreter.block).
GROUPS = [MOTION, {"17", "18", "19"}, {"90", "91"}, {"90.1", "91.1"}, {"93", "94"}, {"20", "21"},
          {"40", "41", "42"}, {"43", "43.1", "49"}, {"98", "99"}, {"54", "55", "56", "57", "58", "59"},
          {"61", "61.1", "64"}, {"15", "16"},
          {"4", "10", "12", "13", "28", "28.1", "30", "30.1", "31", "38.2", "38.3", "52", "53", "92", "92.1"}]
# Códigos reconocidos pero no implementados: se rechazan con un mensaje que dice qué hacer, en vez de
# un «G desconocido» que confunde a quien viene de Mach3.
UNSUPPORTED_G = {
    "50": "G50/G51 (escala de Mach3) no está soportada",
    "51": "G50/G51 (escala de Mach3) no está soportada",
    "41": "compensación de radio G41/G42 no soportada en v0: que la haga el CAM",
    "42": "compensación de radio G41/G42 no soportada en v0: que la haga el CAM",
    "93": "modo de avance de tiempo inverso (G93) no soportado en v0",
    "38.4": "G38.4/G38.5 (sensado alejándose) no soportados: usar G38.2 o G31",
    "38.5": "G38.4/G38.5 (sensado alejándose) no soportados: usar G38.2 o G31",
}
PROBES = {"31", "38.2", "38.3"}  # G31 = sensado de Mach3; G38.2 = con alarma si no toca; G38.3 = sin alarma

# --- límites de seguridad del intérprete ----------------------------------------------------------
# Máximo de líneas ejecutadas (contando cada repetición de subprograma). Protege contra un M98 que se
# llama a sí mismo o un L enorme: el G-code no es confiable. 5 millones alcanza para relieves 3D de
# varios cientos de miles de líneas con repeticiones. Más alto = un programa malicioso cuelga más
# tiempo la revisión (y en PCs viejas, la interfaz).
MAX_STEPS = 5_000_000            # líneas ejecutadas (con subprogramas y repeticiones): corta bucles infinitos
# Máximo de subprogramas anidados (M98 dentro de M98). Fanuc/Mach3 usan 4–10; 16 sobra.
MAX_CALL_DEPTH = 16

# --- expresiones regulares y tablas de apoyo ------------------------------------------------------
LABEL = re.compile(r"\s*[Oo](\d+)")      # «O1000» al principio de la línea: etiqueta de subprograma
KNOWN_G = set().union(*GROUPS) | set(UNSUPPORTED_G)
# Para cada plano de arcos: (eje a, eje b, eje normal, letra del centro en a, en b, en la normal).
# G18 es ZX (no XZ) para que el sentido horario/antihorario coincida con RS274/NGC mirando desde +Y.
PLANES = {"17": ("X", "Y", "Z", "I", "J", "K"), "18": ("Z", "X", "Y", "K", "I", "J"),
          "19": ("Y", "Z", "X", "J", "K", "I")}
WORD = re.compile(r"([A-Z])([-+]?(?:\d+\.?\d*|\.\d+))")  # una palabra: letra + número («X-1.5», «F.5»)
# Rendimiento: un bloque entero válido de una sola vez (si no calza, se busca el error con WORD)
BLOCK_OK = re.compile(r"(?:[A-Z][-+]?(?:\d+\.?\d*|\.\d+))*")


# ==================================================================================================
# ======================================= RESULTADOS =============================================
# ==================================================================================================

@dataclass
class Diag:
    """Un diagnóstico: error (el programa no se puede correr) o aviso (se corre, pero revisar)."""
    level: str          # "error" | "aviso"
    line: int           # número de línea del archivo (desde 1)
    msg: str

    def __str__(self):
        return f"línea {self.line}: {self.msg}"


@dataclass
class Arc:
    """Geometría de un arco G2/G3 ya resuelto (centro, radio y ángulo barrido).

    El punto inicial y final están en la :class:`Op`; el eje normal al plano (Z en G17) se interpola
    linealmente a lo largo del arco: eso da la hélice.
    """
    center: tuple[float, float]       # en el plano (coordenadas de máquina)
    plane: tuple[str, str, str]       # ejes (a, b, normal)
    radius: float                     # mm
    sweep: float                      # radianes; negativo = horario (G2)


@dataclass
class Op:
    """Una operación elemental, ya en coordenadas de máquina. Es lo único que ven el planificador,
    la revisión y la traducción.

    ``start``/``end`` son ``{eje: mm}`` con todos los ejes de la máquina. ``frame`` es el origen G54
    vigente: la traducción emite coordenadas relativas a G54 (resta ``frame``), así que G55..G59, G92,
    G52 y el largo de herramienta quedan «horneados» en los números y el controlador no los necesita.
    """
    kind: str                 # rapid | feed | arc | probe | dwell | spindle | coolant | tool | pause | wcs | end
    line: int
    start: dict[str, float] = field(default_factory=dict)
    end: dict[str, float] = field(default_factory=dict)
    feed: float = 0.0         # mm/min (0 en rapid)
    arc: Arc | None = None
    exact_stop: bool = False  # G61: parar al final del movimiento
    value: object = None      # segundos (dwell), (dir, rpm) (spindle), set (coolant), n (tool), texto (pause)
    source: str = ""          # p. ej. "G83" si viene de un ciclo fijo expandido
    frame: dict[str, float] | None = None   # origen G54 vigente (para traducir con los orígenes que cambian en marcha)


@dataclass
class Program:
    """Resultado de interpretar un programa completo."""
    ops: list[Op]
    diags: list[Diag]
    lines: int                # última línea procesada (número de líneas si llegó al final)
    tools: list[int]          # herramientas usadas con M6, en orden de aparición
    ended: bool               # se encontró M2/M30

    @property
    def errors(self) -> list[Diag]:
        """Solo los errores (con al menos uno, el programa no se debe correr)."""
        return [d for d in self.diags if d.level == "error"]

    @property
    def warnings(self) -> list[Diag]:
        """Solo los avisos."""
        return [d for d in self.diags if d.level == "aviso"]

    @property
    def ok(self) -> bool:
        """True si no hay errores (puede haber avisos)."""
        return not self.errors


# Error en una línea de G-code. Interpreter.run lo atrapa y lo anota como diagnóstico con el número de
# línea; afuera del intérprete solo lo levantan strip_line y parse_words. (Comentario y no docstring
# para dejar el «pass» original sin que ruff lo marque como sobrante.)
class GcodeError(ValueError):
    pass


# ==================================================================================================
# ===================================== LECTURA DE LÍNEAS ==========================================
# ==================================================================================================

_FAST_KEYS = frozenset("XYZABCF")    # palabras que admite el camino rápido de Interpreter.block
_GKEY_CACHE: dict[str, str] = {}


def _gkey(v: str) -> str:
    """Número de G/M normalizado: «01», «1.0» y «1» → «1»; «38.20» → «38.2».

    Rendimiento: un programa repite los mismos pocos códigos miles de veces; se guardan ya
    convertidos (la caché queda chica: solo los textos distintos que aparecen)."""
    k = _GKEY_CACHE.get(v)
    if k is None:
        k = _GKEY_CACHE[v] = f"{float(v):g}"     # «:g» saca ceros sobrantes: 1.0 → «1», 38.20 → «38.2»
    return k


def strip_line(raw: str) -> tuple[str, str]:
    """Quita comentarios; devuelve (código en mayúsculas sin espacios, mensaje MSG si hay).

    Comentarios admitidos: ``( … )`` (RS274/NGC; se aceptan anidados, como Mach3) y ``;`` hasta el fin
    de línea (grbl/LinuxCNC). ``(MSG, texto)`` guarda el texto: Mach3/LinuxCNC lo muestran al
    operador y acá se usa en la pausa M0/M1 siguiente. Los espacios se quitan porque en RS274/NGC no
    significan nada (``G 0 1 X 1 0`` es ``G01X10``).

    Raises:
        GcodeError: paréntesis sin cerrar.
    """
    if "(" not in raw and ";" not in raw and ")" not in raw:
        # Rendimiento: la gran mayoría de las líneas de un CAM no tienen comentarios
        return raw.upper().replace(" ", "").replace("\t", ""), ""
    msg = ""
    out, depth, cur = [], 0, []      # código que queda, nivel de paréntesis, texto del comentario actual
    for ch in raw.split(";", 1)[0]:
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
    """Separa un bloque limpio (sin comentarios ni espacios, en mayúsculas) en palabras.

    Args:
        code: p. ej. ``"G1X10.5Y-2F300"``.

    Returns:
        ``[("G", "1"), ("X", "10.5"), ("Y", "-2"), ("F", "300")]``; los números quedan como texto
        (``_gkey`` y ``float`` los convierten después según la letra).

    Raises:
        GcodeError: parámetros/expresiones (``#``, ``[ ]``) o texto que no es una palabra válida
            (indica cuál, para que el operador lo encuentre).
    """
    if "#" in code or "[" in code:
        raise GcodeError("parámetros y expresiones (#, [ ]) no soportados en v0")
    if BLOCK_OK.fullmatch(code):                      # rendimiento: bloque válido → todas las palabras juntas
        return WORD.findall(code)
    # Camino lento (solo con errores): recorrer palabra por palabra para señalar el texto que sobra.
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
# ========================================= INTÉRPRETE =============================================
# ==================================================================================================

class Interpreter:
    """Máquina de estados de RS274/NGC: recorre el programa bloque a bloque, mantiene el estado modal
    (plano, unidades, absoluto/incremental, origen, herramienta, husillo…) y genera :class:`Op`.

    Se usa una vez por programa (:func:`interpret`) o se mantiene viva para seguir el estado modal en
    la interfaz (``session.py``, :func:`modal_text`).

    Args:
        machine: configuración de la máquina.
        start: posición inicial en coordenadas de máquina (``{eje: mm}``); los ejes que faltan, en 0.
        block_delete: True = saltear las líneas que empiezan con ``/`` (botón «Block Delete» de Mach3).
        optional_stop: True = M1 pausa; False = M1 se ignora.
    """

    def __init__(self, machine: Machine, start: dict[str, float] | None = None, block_delete: bool = True,
                 optional_stop: bool = True):
        self.m = machine
        self.axes = machine.axis_names
        self.pos = {a: float((start or {}).get(a, 0.0)) for a in self.axes}   # posición actual (máquina, mm)
        self.wcs = "G54"
        self.offsets = {w: machine.offset(w) for w in WCS}                 # copia: G10 la cambia en marcha
        self.g92 = {a: 0.0 for a in self.axes}
        self.g52 = {a: 0.0 for a in self.axes}      # origen temporal de Mach3 (G52)
        self.polar: tuple[float, float] | None = None   # G16: centro (trabajo) de las coordenadas polares
        self.polar_ra = [0.0, 0.0]                  # último radio y ángulo (grados)
        self._jump = None                           # salto pendiente de M98/M99 (lo resuelve run)
        self.tlo = 0.0                              # compensación de largo de herramienta en Z (mm), G43
        self.scale = 1.0                            # mm por unidad del programa: 1 (G21) o 25.4 (G20)
        self.absolute = True                        # G90 / G91
        self.arc_absolute = False                   # G90.1 / G91.1 (IJK absolutos o relativos al inicio)
        self.plane = "17"
        self.motion = "0"                           # código de movimiento modal vigente
        self.feed: float | None = None              # mm/min (ya convertido de pulgadas)
        self.spindle = ("off", 0.0)                 # ("cw" | "ccw" | "off", rpm)
        self.rpm = 0.0
        self.coolant: set[str] = set()
        self.selected_tool: int | None = None       # T elegido (todavía no cargado)
        self.tool: int | None = None                # herramienta cargada con M6
        self.exact = False                          # G61 / G61.1 (parada exacta)
        self.retract_r = False                  # G99
        # Valores modales de los ciclos fijos (en unidades de trabajo, mm). Zinit = Z al entrar al ciclo,
        # adonde vuelve G98.
        self.cycle = {"Z": None, "R": None, "Q": None, "P": 0.0, "Zinit": None}
        self.block_delete = block_delete
        self.optional_stop = optional_stop      # M1 para solo si está activo (botón «M1 Optional Stop» de Mach3)
        self.ops: list[Op] = []
        self.diags: list[Diag] = []
        self.tools: list[int] = []
        self.line = 0
        self.message = ""                       # último (MSG, …) leído, para la próxima pausa
        self.ended = False
        # Posiciones de G28/G30 en coordenadas de máquina (de «origenes: G28/G30» en la configuración).
        self.home = {"28": machine.offset("G28"), "30": machine.offset("G30")}
        self._frame: dict[str, float] = {}         # última copia del origen G54 (compartida entre operaciones)

    # --- utilidades ----------------------------------------------------------------
    def _diag(self, level: str, msg: str):
        """Anota un diagnóstico en la línea actual."""
        self.diags.append(Diag(level, self.line, msg))

    def _off(self, a: str) -> float:
        """Desplazamiento total del eje ``a`` entre trabajo y máquina (mm): máquina = trabajo + _off.

        Suma el origen G54..G59 activo, G92, G52 y, en Z, el largo de herramienta (G43): con una
        herramienta más larga el mismo Z de trabajo queda más arriba en coordenadas de máquina, así
        que la punta llega exactamente al mismo lugar.
        """
        return self.offsets[self.wcs][a] + self.g92[a] + self.g52[a] + (self.tlo if a == "Z" else 0.0)

    def _s(self, a: str) -> float:
        """Factor de unidades del eje: los rotativos son grados y G20 no los convierte."""
        return 1.0 if a in ROTARY else self.scale

    def work(self, a: str, machine_value: float | None = None) -> float:
        """Coordenada de trabajo (mm) del eje ``a`` en la posición actual o en ``machine_value``."""
        return (self.pos[a] if machine_value is None else machine_value) - self._off(a)

    def _target(self, w: dict[str, float], machine_coords: bool = False) -> dict[str, float]:
        """Destino en coordenadas de máquina a partir de las palabras de ejes del programa.

        Args:
            w: ``{eje: valor}`` en unidades del programa (pulgadas con G20).
            machine_coords: True para G53 (los valores ya son de máquina, sin orígenes).

        Returns:
            Posición completa (los ejes no nombrados quedan donde están). En G91 el valor se suma a la
            posición actual; en G90 se le suma el desplazamiento del origen.
        """
        t = dict(self.pos)
        for a in self.axes:
            if a in w:
                v = w[a] * self._s(a)
                t[a] = v if machine_coords else (v + self._off(a) if self.absolute else self.pos[a] + v)
        return t

    def _move(self, kind: str, target: dict[str, float], source: str = "", feed: float | None = None):
        """Agrega un movimiento recto (``"rapid"`` o ``"feed"``) hasta ``target`` (máquina, mm).

        Un movimiento de largo cero no genera operación (los CAM repiten puntos y eso frenaría al
        planificador en cada uno). En ``feed`` aplica el avance del perfil (``job_feed``/``job_plunge``)
        solo a líneas del programa (``source`` vacío): las secuencias internas (ciclos, perforación)
        mantienen su propio avance.

        Raises:
            GcodeError: movimiento con avance sin F definido.
        """
        pos = self.pos
        for a in self.axes:                                # ¿se mueve algún eje? (bucle simple: más rápido)
            if abs(target[a] - pos[a]) >= 1e-9:
                break
        else:
            return
        if kind == "feed":
            feed = self.feed if feed is None else feed
            if not feed:
                raise GcodeError("movimiento con avance (G1/G2/G3) sin F")
            # Con movimiento en XY es corte (job_feed); si solo baja en Z es una entrada (job_plunge).
            if not source and any(abs(target[a] - self.pos[a]) > 1e-9 for a in ("X", "Y") if a in self.axes):
                feed = self._job_feed(feed)
            elif not source and "Z" in self.axes and target["Z"] < self.pos["Z"] - 1e-9:
                feed = self._job_feed(feed, self.m.job_plunge)
        # Rendimiento y memoria: self.pos nunca se modifica en el lugar (siempre se reemplaza), así que la
        # operación guarda el mismo diccionario sin copiarlo; y el origen G54 se comparte entre todas las
        # operaciones mientras no cambie (antes: una copia por línea, 100 000 copias en un relieve 3D).
        g54 = self.offsets["G54"]
        if self._frame != g54:
            self._frame = dict(g54)
        end = dict(target)
        self.ops.append(Op(kind, self.line, pos, end, feed or 0.0,
                           exact_stop=self.exact, source=source, frame=self._frame))
        self.pos = end

    def _job_feed(self, feed: float, value: float | None = None) -> float:
        """Avance de corte del perfil (solo con husillo/antorcha encendidos y movimiento en el plano).

        Args:
            feed: F del programa (mm/min).
            value: avance del perfil a usar (``job_plunge``); ``None`` = ``job_feed``.

        Returns:
            En modo «reemplazar», el del perfil; en «limitar», el menor de los dos. Sin perfil o con el
            husillo apagado (p. ej. recorridos de prueba), la F del programa tal cual.
        """
        jf = self.m.job_feed if value is None else value
        if not jf or self.spindle[0] == "off":
            return feed
        return jf if self.m.job_feed_mode == "reemplazar" else min(feed, jf)

    def _work_target(self, axis: str, work_value: float) -> dict[str, float]:
        """Posición actual con un solo eje llevado a ``work_value`` en coordenadas de trabajo (mm)."""
        return {**self.pos, axis: work_value + self._off(axis)}

    # --- interpretación ----------------------------------------------------------
    def run(self, text: str) -> Program:
        """Interpreta el programa completo y devuelve el :class:`Program`.

        Además de llamar a :meth:`block` en cada línea, maneja lo que es «de archivo»: ``%``, ``/``
        (borrado de bloque), etiquetas ``O…`` y los saltos de subprogramas M98/M99. Los errores de cada
        línea se anotan y se sigue (no lanza :class:`GcodeError`).
        """
        lines = text.splitlines()
        # Primera pasada: dónde está cada «O<n>» (la primera aparición gana), para que M98 pueda llamar
        # a subprogramas definidos más abajo, como en Fanuc/Mach3.
        labels: dict[int, int] = {}
        for i, raw in enumerate(lines):
            m = LABEL.match(raw)
            if m:
                labels.setdefault(int(m.group(1)), i)
        stack: list[list[int]] = []                   # [vuelta, repeticiones que faltan, inicio del O]
        i = steps = 0
        while i < len(lines) and not self.ended:
            steps += 1
            if steps > MAX_STEPS:
                self._diag("error", "demasiadas líneas ejecutadas: ¿subprograma que se llama sin fin?")
                break
            self.line, raw = i + 1, lines[i]
            i += 1
            s = raw.strip()
            if not s or s.startswith("%"):            # «%» marca inicio/fin de cinta (Fanuc): se ignora
                continue
            if s.startswith("/") and self.block_delete:
                continue
            m = LABEL.match(s)
            if m:                                     # «O1000 (nombre)»: etiqueta, no se ejecuta
                s = s[m.end():].strip()
            try:
                code, msg = strip_line(s.lstrip("/"))
                if msg:
                    self.message = msg
                if code:
                    self.block(parse_words(code))
            except GcodeError as e:
                self._diag("error", str(e))
                self._jump = None                     # una línea con error no salta
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
                elif stack:                           # M99: volver (o repetir si quedan)
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
        """Ejecuta un bloque (una línea ya separada en palabras por :func:`parse_words`).

        Sigue el orden de ejecución de RS274/NGC (sección 3.8 de la norma): primero unidades, F, S,
        T, M de husillo/refrigerante/cambio, G4, plano, largo de herramienta, origen, modos, después
        los no modales (G10, G28, G92…) y al final el movimiento y las M de parada/fin. Así, por
        ejemplo, ``G20 G0 X1`` mueve 1 pulgada y ``S1000 M3 G1 X10`` corta con el husillo ya encendido.
        Una excepción deliberada: G20/G21 se aplica antes que F para que ``G20 F10`` sean 10 pulg/min.

        Raises:
            GcodeError: letra repetida, eje no configurado, códigos incompatibles o datos faltantes.
        """
        gs, ms, w = [], [], {}            # códigos G, códigos M, resto de palabras {letra: valor}
        for letter, val in words:
            if letter == "G":
                gs.append(_gkey(val))
            elif letter == "M":
                ms.append(_gkey(val))
            elif letter == "N":           # número de línea: no hace nada
                continue
            elif letter in w:
                raise GcodeError(f"la letra {letter} aparece dos veces en el bloque")
            elif letter in "XYZABC" and letter not in self.axes:
                raise GcodeError(f"eje {letter} no configurado en la máquina")
            elif letter == "E":
                raise GcodeError("E es de impresora 3D (extrusor): este programa no es para una CNC")
            else:
                w[letter] = float(val)
        # --- camino rápido ---------------------------------------------------------------------
        # Rendimiento: la inmensa mayoría de las líneas de un CAM son «G1 X… Y… Z… F…» (o solo ejes,
        # con el G1 modal). Si el bloque tiene a lo sumo un G0/G1, solo ejes y F, sin M y sin
        # coordenadas polares, ninguna de las ~40 comprobaciones modales de abajo puede aplicar: se
        # mueve directo. El resultado es idéntico al camino completo (lo comprueban las pruebas).
        if not ms and len(gs) <= 1 and self.polar is None:
            g = gs[0] if gs else self.motion
            if g in ("0", "1") and _FAST_KEYS.issuperset(w):
                if "F" in w:
                    if w["F"] < 0:
                        raise GcodeError("F negativo")
                    self.feed = w["F"] * self.scale
                if gs:
                    if self.motion in CANNED:               # salir de un ciclo fijo borra sus valores
                        self.cycle = {"Z": None, "R": None, "Q": None, "P": 0.0, "Zinit": None}
                    self.motion = g
                axis_w = {a: v for a, v in w.items() if a != "F"}
                if axis_w:
                    self._move("rapid" if g == "0" else "feed", self._target(axis_w))
                return
        # --- camino completo -------------------------------------------------------------------
        for g in gs:
            if g in UNSUPPORTED_G:
                raise GcodeError(UNSUPPORTED_G[g])
            if g not in KNOWN_G:
                raise GcodeError(f"G{g} desconocido")
        for grp in GROUPS[:-1] if len(gs) > 1 else ():   # con un solo G no puede haber conflicto
            if len([g for g in gs if g in grp]) > 1:
                raise GcodeError("dos códigos G del mismo grupo modal en un bloque: " + ", ".join(
                    "G" + g for g in gs if g in grp))
        gset = set(gs)
        axis_w = {a: v for a, v in w.items() if a in self.axes}

        # orden de ejecución RS274/NGC
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
            # Cambiar S con el husillo girando cambia la velocidad en el acto (sin M3 nuevo), como Mach3.
            if self.spindle[0] != "off" and not ({"3", "4", "5"} & set(ms)):
                self._spindle(self.spindle[0])
        if "T" in w:
            self.selected_tool = int(w["T"])        # T solo elige; la carga es M6
        self._mcodes_pre(ms)
        if "4" in gset:
            p = w.get("P")
            if p is None or p < 0:
                raise GcodeError("G4 necesita P ≥ 0 (segundos)")
            if p > 1000:
                # Mach3 interpreta P en ms o en s según una casilla de configuración: un P enorme suele
                # ser un programa escrito para milisegundos.
                self._diag("aviso", f"G4 P{p:g}: en este intérprete P son segundos (Mach3 puede usar ms)")
            self.ops.append(Op("dwell", self.line, value=float(p)))
        for g in ("17", "18", "19"):
            if g in gset:
                self.plane = g
        if "43" in gset:
            # G43 sin H usa la herramienta cargada (LinuxCNC hace igual).
            n = int(w["H"]) if "H" in w else self.tool
            t = self.m.tool(n) if n is not None else None
            if t is None:
                raise GcodeError(f"G43: la herramienta {n} no está en la tabla (herramientas: en la configuración)")
            self.tlo = t.length
        if "43.1" in gset:
            # G43.1 Z…: largo dinámico (sin tabla). El Z se consume: no es un movimiento.
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

        # --- no modales (grupo 0) --------------------------------------------------------------
        # uses_axes: los que «consumen» las palabras X Y Z del bloque (no hay movimiento modal después).
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
            # G92: «la posición actual se llama v». Se guarda el desplazamiento que hace que
            # trabajo = máquina − (origen + G92 + G52 + tlo) dé v; por eso se descuenta también G52
            # (Mach3 suma G52 al origen: sin descontarlo, la coordenada quedaba corrida en G52).
            if not axis_w:
                raise GcodeError("G92 necesita al menos un eje")
            for a, v in axis_w.items():
                self.g92[a] = (self.pos[a] - self.offsets[self.wcs][a] - self.g52[a] - (self.tlo if a == "Z" else 0)
                               - v * self._s(a))
        elif "92.1" in gset:
            self.g92 = {a: 0.0 for a in self.axes}
        elif "52" in gset:
            # G52 de Mach3/Fanuc: corre el origen de trabajo en (X, Y, Z) respecto del G54..G59 activo,
            # sin tocar la tabla de orígenes. G52 X0 Y0 Z0 lo anula.
            if not axis_w:
                raise GcodeError("G52 necesita al menos un eje")
            for a, v in axis_w.items():
                self.g52[a] = v * self._s(a)
        elif "12" in gset or "13" in gset:
            self._pocket("12" if "12" in gset else "13", w)
        if "16" in gset:
            # G16 de Mach3: el centro de las polares es el punto (de trabajo) donde está la herramienta.
            self.polar = (self.work("X"), self.work("Y"))
            self.polar_ra = [0.0, 0.0]
        if "15" in gset:
            self.polar = None
        if self.polar is not None and ("X" in axis_w or "Y" in axis_w) and not uses_axes:
            axis_w = self._from_polar(axis_w, gs)

        # --- movimiento ------------------------------------------------------------------------
        motion = [g for g in gs if g in MOTION]
        if motion:
            self.motion = motion[0]
            if self.motion in CANNED and self.cycle["Zinit"] is None:
                self.cycle["Zinit"] = self.work("Z") if "Z" in self.axes else None   # altura para G98
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
                self._cycle(w, axis_w)       # con un ciclo modal, cada línea con X/Y es otro agujero
        # --- subprogramas y paradas (después del movimiento) -----------------------------------
        if "98" in ms:
            if "P" not in w:
                raise GcodeError("M98 necesita P (número del subprograma O…)")
            self._jump = ("call", int(w["P"]), int(w.get("L", 1)))
        elif "99" in ms:
            self._jump = ("ret",)
        self._mcodes_post(ms)

    # --- M ------------------------------------------------------------------------------
    def _spindle(self, d: str):
        """Enciende (``"cw"``/``"ccw"``) o apaga (``"off"``) el husillo o la antorcha.

        Avisa si S supera el máximo o es 0 (en fresadora/router/láser). En plasma con perforación
        automática, el primer M3 dispara la secuencia de :meth:`_pierce` y M5 sube a la altura de
        traslado (si el CAM no lo hizo) para no arrastrar la antorcha por la chapa.
        """
        pl = self.m.plasma if self.m.is_plasma else None
        if d != "off" and self.rpm <= 0 and self.m.job_rpm and pl is None:
            self.rpm = float(self.m.job_rpm)            # el perfil pone la S que falta
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
        """Secuencia de perforación de plasma: [sensado] → altura de perforación → antorcha → retardo → corte.

        Con sensado (IHS): si está por debajo de la altura de perforación sube a la de traslado, baja
        con G38.2 hasta ``probe_depth`` buscando la chapa y fija Z0 de G54 con ``G10 L20 Z-holgura``:
        cuando actúa el interruptor del cabezal flotante, la boquilla ya está ``float_offset`` mm por
        debajo del punto de contacto; declarar esa posición como Z = −holgura deja Z0 justo en la chapa.
        Corrige así chapas alabeadas o mesas desniveladas en cada perforación.
        """
        src = "perforación"
        if pl.probe:
            if self.wcs != "G54":
                raise GcodeError("el sensado automático de plasma trabaja en G54")
            if self.work("Z") < pl.pierce_height:
                self._move("rapid", self._work_target("Z", pl.travel_height), source=src)
            # El destino se pasa en unidades del programa (/ _s) porque _probe vuelve a multiplicar.
            self._probe("38.2", {"Z": (self.work("Z") - pl.probe_depth) / self._s("Z")}, feed=pl.probe_feed, source=src)
            self._set_wcs(20, 1, {"Z": -pl.float_offset}, source=src)
        self._move("rapid", self._work_target("Z", pl.pierce_height), source=src)
        self.spindle = ("cw", self.rpm)
        self.ops.append(Op("spindle", self.line, value=self.spindle, source=src))
        if pl.pierce_delay:
            self.ops.append(Op("dwell", self.line, value=float(pl.pierce_delay), source=src))
        self._move("feed", self._work_target("Z", pl.cut_height), source=src, feed=pl.plunge_feed)

    # --- sensado (G38.2 / G38.3 / G31 de Mach3) ---------------------------------------------
    def _probe(self, g: str, axis_w: dict, feed: float | None = None, source: str = ""):
        """El sensado termina donde toque. Para planificar se supone el contacto en la superficie esperada
        (Z0 del origen vigente, menos la holgura del cabezal flotante en plasma); para revisar límites se
        usa el recorrido completo (``value``), que es lo peor que puede pasar.

        Args:
            g: ``"38.2"``, ``"38.3"`` o ``"31"``.
            axis_w: destino del sensado (unidades del programa).
            feed: avance (mm/min); ``None`` = F modal.
            source: secuencia interna que lo pide (``"perforación"``); con ``source`` el destino es
                siempre absoluto en coordenadas de trabajo, aunque el programa esté en G91.

        Raises:
            GcodeError: sin ejes, sin F o sin recorrido.
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
            # Sensado vertical hacia abajo (el caso típico: buscar la superficie de la pieza).
            flo = self.m.plasma.float_offset if self.m.is_plasma and self.m.plasma else 0.0
            surface = self._off("Z") - flo                       # Z0 de trabajo en coordenadas de máquina
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
        """Cambia un origen de trabajo (lógica de G10 L2 / L20).

        Args:
            lv: 2 = ``vals`` es el origen en coordenadas de máquina; 20 = ``vals`` es la coordenada de
                trabajo que debe tener la posición actual (el origen se calcula: máquina − G92 − G52 − tlo − v).
            p: 0 = origen activo; 1–6 = G54–G59.
            vals: ``{eje: mm}`` ya convertidos a mm.
            source: secuencia interna que lo pide (para la traducción).

        Si cambia G54 se emite una :class:`Op` ``"wcs"``: la traducción trabaja relativa a G54, así que
        el controlador tiene que recibir el cambio en el mismo punto del programa (p. ej. tras un sensado).
        """
        wcs = self.wcs if p == 0 else WCS[p - 1]
        for a, v in vals.items():
            # L20: «la posición actual vale v» → origen = máquina − (G92 + G52 + tlo) − v, con los mismos
            # términos que suma _off() (sin G52 quedaba corrida en G52 si había uno activo).
            self.offsets[wcs][a] = v if lv == 2 else (self.pos[a] - self.g92[a] - self.g52[a]
                                                      - (self.tlo if a == "Z" else 0) - v)
        if wcs == "G54":                 # el controlador también tiene que enterarse (sensado en marcha)
            self.ops.append(Op("wcs", self.line, value={a: self.offsets["G54"][a] for a in vals},
                               source=source or f"G10 L{lv}", frame=dict(self.offsets["G54"]),
                               end={"L": lv, **{a: (self.pos[a] - self.offsets["G54"][a]) for a in vals}}))

    def _mcodes_pre(self, ms: list[str]):
        """M que actúan ANTES del movimiento del bloque: M6, M3/M4/M5, M7/M8/M9 (orden RS274/NGC).

        Las M de parada/fin y de subprogramas se dejan para :meth:`_mcodes_post`/:meth:`run`. Las M
        propias de Mach3 y las macros de usuario se ignoran con aviso: no se puede saber qué hacían.
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
                continue                                 # después del bloque (run)
            elif mc in ("0", "1", "2", "30"):
                continue                                 # después del movimiento
            elif mc in ("48", "49", "47"):
                # M47 (repetir desde el principio), M48/M49 (habilitar/inhibir overrides) de Mach3.
                self._diag("aviso", f"M{mc} (Mach3) se ignora")
            elif n >= 100:
                self._diag("aviso", f"M{mc}: macro de usuario de Mach3; se ignora (reescribirla para este control)")
            else:
                self._diag("aviso", f"M{mc} desconocido; se ignora")

    def _mcodes_post(self, ms: list[str]):
        """M que actúan DESPUÉS del movimiento: M0/M1 (pausa, con el último MSG) y M2/M30 (fin).

        Al terminar se apaga el husillo si quedó encendido (M2/M30 lo hacen en todo controlador), y se
        marca el programa como terminado: lo que sigue no se interpreta.
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

    # --- G no modales ---------------------------------------------------------------
    def _from_polar(self, axis_w: dict, gs: list) -> dict:
        """G16 de Mach3: X = radio, Y = ángulo (grados) alrededor del punto donde se activó.

        Convierte (radio, ángulo) en X/Y cartesianos de trabajo: X = cx + r·cos(θ), Y = cy + r·sin(θ),
        con θ en grados desde +X en sentido antihorario. Si el bloque trae solo uno de los dos, el otro
        es el último usado (modal), como en Mach3: «Y90» gira al mismo radio.

        Returns:
            ``axis_w`` con X/Y cartesianos en unidades del programa (``_target`` los vuelve a escalar,
            por eso se divide por ``scale``).

        Raises:
            GcodeError: fuera de G17/G90 o con arcos.
        """
        motion = next((g for g in gs if g in MOTION), self.motion)
        if self.plane != "17" or not self.absolute:
            raise GcodeError("coordenadas polares (G16) solo en G17 y G90")
        if motion in ("2", "3"):
            raise GcodeError("coordenadas polares (G16) solo con G0, G1 o ciclos de taladrado")
        r = axis_w.get("X", self.polar_ra[0] / self.scale) * self.scale     # radio en mm
        ang = axis_w.get("Y", self.polar_ra[1])                             # grados: no se escala
        self.polar_ra = [r, ang]
        cx, cy = self.polar
        out = dict(axis_w)
        out["X"] = (cx + r * math.cos(math.radians(ang))) / self.scale
        out["Y"] = (cy + r * math.sin(math.radians(ang))) / self.scale
        return out

    def _pocket(self, g: str, w: dict):
        """G12/G13 de Mach3: sale en X la distancia I, hace un círculo (horario o antihorario) con centro
        en el punto de partida y vuelve al centro.

        Se expande en: avance recto a (X+I), un arco de 360° (G12 horario = barrido −2π, G13 = +2π) y
        avance de vuelta al centro. Así lo entienden grbl y LinuxCNC, que no tienen G12/G13.

        Raises:
            GcodeError: fuera de G17, sin I (o I0), o sin F.
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
        """G10 L2/L20 P0..6: fija un origen de trabajo (ver :meth:`_set_wcs`).

        Raises:
            GcodeError: otro L (p. ej. L1 = tabla de herramientas, no soportado) o P fuera de 0–6.
        """
        lv, p = int(w.get("L", -1)), int(w.get("P", 0))
        if lv not in (2, 20):
            raise GcodeError("G10: solo L2 y L20 (orígenes de trabajo)")
        if not 0 <= p <= 6:
            raise GcodeError("G10: P entre 0 (actual) y 6 (G59)")
        self._set_wcs(lv, p, {a: v * self._s(a) for a, v in axis_w.items()})

    def _g28(self, which: str, axis_w):
        """G28/G30: ir a la posición de referencia guardada (``origenes: G28/G30`` o G28.1/G30.1).

        Con ejes (``G28 Z5``) primero va en rápido a ese punto intermedio (coordenadas de trabajo) y
        después lleva SOLO esos ejes a la referencia, como RS274/NGC; sin ejes lleva todos. El punto
        intermedio existe para subir Z antes de cruzar la pieza.
        """
        if axis_w:
            self._move("rapid", self._target(axis_w), source="G" + which)
            home = dict(self.pos)
            for a in axis_w:
                home[a] = self.home[which][a]
        else:
            home = dict(self.home[which])
        self._move("rapid", home, source="G" + which)

    # --- arcos ----------------------------------------------------------------------
    def _arc(self, w, axis_w):
        """G2 (horario) / G3 (antihorario) en el plano activo, con centro IJK o radio R.

        Los ejes que no son del plano (Z en G17) van linealmente del inicio al fin: arco helicoidal.
        ``P`` (LinuxCNC) agrega vueltas completas: P2 = una vuelta extra antes de llegar al final.

        Raises:
            GcodeError: sin F, sin centro ni R, R imposible, radio cero, o radio inicial y final
                distintos (centro mal calculado por el CAM).
        """
        a, b, n, ia, ib, _ = PLANES[self.plane]
        if a not in self.axes or b not in self.axes:
            raise GcodeError(f"arco en G{self.plane}: faltan los ejes {a}/{b}")
        if not self.feed:
            raise GcodeError("arco G2/G3 sin F")
        cw = self.motion == "2"
        end = self._target(axis_w)
        x, y = end[a] - self.pos[a], end[b] - self.pos[b]          # cuerda del inicio al fin, en el plano
        if "R" in w:
            # --- formato radio (igual que grbl, mc_arc / gc_execute_block) ---
            # El centro está sobre la mediatriz de la cuerda, a una distancia
            # d = sqrt(r² − (c/2)²) del punto medio (c = largo de la cuerda). Con
            # h = −2·d / c el centro relativo al inicio es (x − y·h, y + x·h) / 2: el punto medio más el
            # vector perpendicular (−y, x) escalado. El signo elige de qué lado: para G2 con R positivo,
            # el arco corto (≤ 180°); R negativo pide el arco largo (> 180°), y se invierte el lado.
            r = w["R"] * self.scale
            if abs(x) < 1e-9 and abs(y) < 1e-9:
                raise GcodeError("arco con R y el mismo punto inicial y final: usar I/J/K")
            h = 4.0 * r * r - x * x - y * y                       # (2d)² = (2r)² − c²
            if h < 0:
                # La cuerda es más larga que el diámetro. Si es por poco (redondeo del CAM, dentro de la
                # tolerancia de arco) se toma medio círculo exacto; si no, el R está mal.
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
            # --- formato centro (IJK) ---
            if ia not in w and ib not in w:
                raise GcodeError(f"arco sin {ia}/{ib} ni R")
            if self.arc_absolute:
                # G90.1: IJK son coordenadas de trabajo del centro (la letra que falta = posición actual).
                ca = w.get(ia, self.work(a)) * self.scale + self._off(a)
                cb = w.get(ib, self.work(b)) * self.scale + self._off(b)
            else:
                # G91.1 (por defecto, como Mach3 «IJ incremental»): IJK relativos al punto inicial.
                ca = self.pos[a] + w.get(ia, 0.0) * self.scale
                cb = self.pos[b] + w.get(ib, 0.0) * self.scale
            radius = math.hypot(self.pos[a] - ca, self.pos[b] - cb)
            r_end = math.hypot(end[a] - ca, end[b] - cb)
            if radius < 1e-9:
                raise GcodeError("arco de radio cero")
            # Mismo criterio que LinuxCNC: diferencia admitida 0.005 mm o 0.1 % del radio (lo que sea
            # mayor). Más que eso indica un centro mal calculado y el arco terminaría con un salto.
            if abs(radius - r_end) > max(0.005, 0.001 * radius):
                raise GcodeError(f"arco inválido: radio inicial {radius:.4f} y final {r_end:.4f} distintos")
        # Ángulo barrido: diferencia de ángulos polares de inicio y fin alrededor del centro, llevada al
        # sentido pedido (negativo = horario). Inicio = fin da un círculo completo (±2π), no un arco nulo.
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
        sweep += math.copysign(2 * math.pi * (turns - 1), sweep)  # vueltas extra, en el mismo sentido
        self.ops.append(Op("arc", self.line, dict(self.pos), dict(end), self._job_feed(self.feed), exact_stop=self.exact,
                           arc=Arc((ca, cb), (a, b, n), radius, sweep), frame=dict(self.offsets["G54"])))
        self.pos = dict(end)

    # --- ciclos fijos ---------------------------------------------------------------
    def _cycle(self, w, axis_w):
        """Expande un ciclo fijo de taladrado en movimientos simples (para controladores sin ciclos).

        Secuencia para cada agujero (Z, R y Q en coordenadas de trabajo, modales entre líneas):

        1. si la herramienta está por debajo de R, sube en rápido a R (no cruzar la pieza a ras);
        2. rápido a X/Y del agujero; 3. rápido a R;
        4. G81: avance hasta Z. G82: ídem + espera P segundos en el fondo (limpia el fondo plano).
           G83 (picado profundo): baja de a Q, sale a R después de cada picada para sacar la viruta y
           vuelve en rápido hasta :data:`PECK_CLEARANCE` sobre el fondo anterior.
           G73 (rompe viruta): baja de a Q y retrocede solo :data:`PECK_CLEARANCE` (más rápido, para
           materiales de viruta larga);
        5. retracción: G99 → a R; G98 → al mayor entre la Z inicial del ciclo y R.

        Raises:
            GcodeError: fuera de G17/G90, sin Z o R, Z por encima de R, Q ≤ 0 en G83/G73, o sin F.
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
            self.cycle["P"] = w["P"]              # segundos: no se escala
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
        mz = lambda v: v + off_z                                  # trabajo → máquina (Z)
        if self.work("Z") < r:
            self._move("rapid", {**self.pos, "Z": mz(r)}, src)
        xy = {k: v for k, v in axis_w.items() if k != "Z"}       # Z del bloque es el fondo, no un destino
        self._move("rapid", self._target(xy), src)
        self._move("rapid", {**self.pos, "Z": mz(r)}, src)
        if g in ("81", "82"):
            self._move("feed", {**self.pos, "Z": mz(z)}, src)
        else:
            depth = r                                             # fondo alcanzado hasta ahora (trabajo)
            while depth > z + 1e-9:
                nxt = max(depth - q, z)                           # la última picada puede ser más corta
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
# ===================================== FUNCIONES PÚBLICAS =========================================
# ==================================================================================================

def interpret(text: str, machine: Machine, start: dict[str, float] | None = None, block_delete: bool = True,
              optional_stop: bool = True) -> Program:
    """Interpreta un programa G-code completo (atajo de ``Interpreter(...).run(text)``).

    Args:
        text: el programa (texto).
        machine: la máquina (ver ``hope.cnc.machine``).
        start: posición inicial en coordenadas de máquina (mm); por defecto todo en 0.
        block_delete: saltear líneas con ``/``.
        optional_stop: si M1 pausa.

    Returns:
        :class:`Program` con operaciones y diagnósticos. No lanza por errores del G-code.
    """
    return Interpreter(machine, start, block_delete, optional_stop).run(text)


def modal_text(it: Interpreter) -> str:
    """Estado modal como la línea de G-codes activos de Mach3 (p. ej. «G0 G17 G21 G90 G94 G54 G49 G98 G64»).

    Agrega husillo (M3/M4/M5), refrigerante (M7/M8/M9), herramienta, F (en unidades del programa) y S.
    Lo muestra la interfaz durante la ejecución.
    """
    g = [f"G{it.motion}", f"G{it.plane}", "G20" if it.scale != 1.0 else "G21", "G90" if it.absolute else "G91",
         "G94", it.wcs, "G43" if it.tlo else "G49", "G99" if it.retract_r else "G98", "G61" if it.exact else "G64"]
    m = {"cw": "M3", "ccw": "M4"}.get(it.spindle[0], "M5")
    cool = " ".join(sorted("M7" if c == "niebla" else "M8" for c in it.coolant)) or "M9"
    return " ".join(g) + f" {m} {cool} T{it.tool or 0}" + (f" F{it.feed / it.scale:g}" if it.feed else "") + \
        (f" S{it.rpm:g}" if it.rpm else "")


def linearize(op: Op, tolerance: float) -> list[dict[str, float]]:
    """Puntos finales de las cuerdas de un arco (el error de cuerda no supera ``tolerance``).

    Geometría: una cuerda que abarca un ángulo θ se separa del arco como máximo la flecha
    ``r·(1 − cos(θ/2))``. Igualándola a la tolerancia, θ = 2·acos(1 − tol/r); se usan
    ``ceil(|barrido| / θ)`` cuerdas iguales. Con tolerancia ≥ radio (arcos diminutos) se usan cuartos
    de vuelta. Los ejes fuera del plano (hélice) se interpolan linealmente con el mismo parámetro t.

    Args:
        op: operación ``"arc"`` (con ``op.arc``).
        tolerance: error máximo (mm), normalmente ``machine.arc_tolerance``.

    Returns:
        Lista de posiciones completas ``{eje: mm}`` (sin el punto inicial; el último es exactamente
        ``op.end``, para no acumular error de redondeo entre operaciones).

    Lo usan el planificador (tiempos), la revisión de límites y la traducción a controladores que no
    tienen arcos o los manejan mal.
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
