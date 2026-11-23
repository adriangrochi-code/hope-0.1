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

No soportado (error claro): parámetros y expresiones (``#1``, ``[ ]``), subprogramas M98/M99,
compensación de radio G41/G42, tiempo inverso G93, y los códigos propios de Mach3 G12/G13
(cajeras circulares), G15/G16 (polares), G50/G51 (escala) y G52. Las macros de usuario de Mach3
(M100 en adelante) se ignoran con un aviso: hay que reescribirlas.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from hope.cnc.machine import ROTARY, WCS, Machine

INCH = 25.4
PECK_CLEARANCE = 0.254          # mm que el G83/G73 deja antes del fondo anterior (0.010" como LinuxCNC)

MOTION = {"0", "1", "2", "3", "80", "81", "82", "83", "73"}
CANNED = {"81", "82", "83", "73"}
GROUPS = [MOTION, {"17", "18", "19"}, {"90", "91"}, {"90.1", "91.1"}, {"93", "94"}, {"20", "21"},
          {"40", "41", "42"}, {"43", "43.1", "49"}, {"98", "99"}, {"54", "55", "56", "57", "58", "59"},
          {"61", "61.1", "64"}, {"4", "10", "28", "28.1", "30", "30.1", "31", "38.2", "38.3", "53", "92", "92.1"}]
UNSUPPORTED_G = {
    "12": "G12/G13 (cajera circular de Mach3) no es estándar: generar la trayectoria desde el CAM",
    "13": "G12/G13 (cajera circular de Mach3) no es estándar: generar la trayectoria desde el CAM",
    "15": "G15/G16 (coordenadas polares de Mach3) no están soportadas",
    "16": "G15/G16 (coordenadas polares de Mach3) no están soportadas",
    "50": "G50/G51 (escala de Mach3) no está soportada",
    "51": "G50/G51 (escala de Mach3) no está soportada",
    "52": "G52 (origen local de Mach3) no está soportado: usar G92 o G10 L20",
    "41": "compensación de radio G41/G42 no soportada en v0: que la haga el CAM",
    "42": "compensación de radio G41/G42 no soportada en v0: que la haga el CAM",
    "93": "modo de avance de tiempo inverso (G93) no soportado en v0",
    "38.4": "G38.4/G38.5 (sensado alejándose) no soportados: usar G38.2 o G31",
    "38.5": "G38.4/G38.5 (sensado alejándose) no soportados: usar G38.2 o G31",
}
PROBES = {"31", "38.2", "38.3"}
KNOWN_G = set().union(*GROUPS) | set(UNSUPPORTED_G)
PLANES = {"17": ("X", "Y", "Z", "I", "J", "K"), "18": ("Z", "X", "Y", "K", "I", "J"),
          "19": ("Y", "Z", "X", "J", "K", "I")}
WORD = re.compile(r"([A-Z])([-+]?(?:\d+\.?\d*|\.\d+))")


@dataclass
class Diag:
    level: str          # "error" | "aviso"
    line: int
    msg: str

    def __str__(self):
        return f"línea {self.line}: {self.msg}"


@dataclass
class Arc:
    center: tuple[float, float]       # en el plano (coordenadas de máquina)
    plane: tuple[str, str, str]       # ejes (a, b, normal)
    radius: float
    sweep: float                      # radianes; negativo = horario (G2)


@dataclass
class Op:
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
    ops: list[Op]
    diags: list[Diag]
    lines: int
    tools: list[int]
    ended: bool               # se encontró M2/M30

    @property
    def errors(self) -> list[Diag]:
        return [d for d in self.diags if d.level == "error"]

    @property
    def warnings(self) -> list[Diag]:
        return [d for d in self.diags if d.level == "aviso"]

    @property
    def ok(self) -> bool:
        return not self.errors


class GcodeError(ValueError):
    pass


def _gkey(v: str) -> str:
    return f"{float(v):g}"


def strip_line(raw: str) -> tuple[str, str]:
    """Quita comentarios; devuelve (código en mayúsculas sin espacios, mensaje MSG si hay)."""
    msg = ""
    out, depth, cur = [], 0, []
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
    if "#" in code or "[" in code:
        raise GcodeError("parámetros y expresiones (#, [ ]) no soportados en v0")
    words, pos = [], 0
    for m in WORD.finditer(code):
        if m.start() != pos:
            raise GcodeError(f"texto no reconocido «{code[pos:m.start()]}»")
        words.append((m.group(1), m.group(2)))
        pos = m.end()
    if pos != len(code):
        raise GcodeError(f"texto no reconocido «{code[pos:]}»")
    return words


class Interpreter:
    def __init__(self, machine: Machine, start: dict[str, float] | None = None, block_delete: bool = True,
                 optional_stop: bool = True):
        self.m = machine
        self.axes = machine.axis_names
        self.pos = {a: float((start or {}).get(a, 0.0)) for a in self.axes}
        self.wcs = "G54"
        self.offsets = {w: machine.offset(w) for w in WCS}
        self.g92 = {a: 0.0 for a in self.axes}
        self.tlo = 0.0
        self.scale = 1.0
        self.absolute = True
        self.arc_absolute = False
        self.plane = "17"
        self.motion = "0"
        self.feed: float | None = None
        self.spindle = ("off", 0.0)
        self.rpm = 0.0
        self.coolant: set[str] = set()
        self.selected_tool: int | None = None
        self.tool: int | None = None
        self.exact = False
        self.retract_r = False                  # G99
        self.cycle = {"Z": None, "R": None, "Q": None, "P": 0.0, "Zinit": None}
        self.block_delete = block_delete
        self.optional_stop = optional_stop      # M1 para solo si está activo (botón «M1 Optional Stop» de Mach3)
        self.ops: list[Op] = []
        self.diags: list[Diag] = []
        self.tools: list[int] = []
        self.line = 0
        self.message = ""
        self.ended = False
        self.home = {"28": machine.offset("G28"), "30": machine.offset("G30")}

    # --- utilidades ----------------------------------------------------------------
    def _diag(self, level: str, msg: str):
        self.diags.append(Diag(level, self.line, msg))

    def _off(self, a: str) -> float:
        return self.offsets[self.wcs][a] + self.g92[a] + (self.tlo if a == "Z" else 0.0)

    def _s(self, a: str) -> float:
        return 1.0 if a in ROTARY else self.scale

    def work(self, a: str, machine_value: float | None = None) -> float:
        return (self.pos[a] if machine_value is None else machine_value) - self._off(a)

    def _target(self, w: dict[str, float], machine_coords: bool = False) -> dict[str, float]:
        t = dict(self.pos)
        for a in self.axes:
            if a in w:
                v = w[a] * self._s(a)
                t[a] = v if machine_coords else (v + self._off(a) if self.absolute else self.pos[a] + v)
        return t

    def _move(self, kind: str, target: dict[str, float], source: str = "", feed: float | None = None):
        if all(abs(target[a] - self.pos[a]) < 1e-9 for a in self.axes):
            return
        if kind == "feed":
            feed = self.feed if feed is None else feed
            if not feed:
                raise GcodeError("movimiento con avance (G1/G2/G3) sin F")
        self.ops.append(Op(kind, self.line, dict(self.pos), dict(target), feed or 0.0,
                           exact_stop=self.exact, source=source, frame=dict(self.offsets["G54"])))
        self.pos = dict(target)

    def _work_target(self, axis: str, work_value: float) -> dict[str, float]:
        return {**self.pos, axis: work_value + self._off(axis)}

    # --- interpretación ----------------------------------------------------------
    def run(self, text: str) -> Program:
        for self.line, raw in enumerate(text.splitlines(), start=1):
            if self.ended:
                break
            s = raw.strip()
            if not s or s.startswith("%"):
                continue
            if s.startswith("/") and self.block_delete:
                continue
            try:
                code, msg = strip_line(s.lstrip("/"))
                if msg:
                    self.message = msg
                if code:
                    self.block(parse_words(code))
            except GcodeError as e:
                self._diag("error", str(e))
        return Program(self.ops, self.diags, self.line, self.tools, self.ended)

    def block(self, words: list[tuple[str, str]]):
        gs, ms, w = [], [], {}
        for letter, val in words:
            if letter == "G":
                gs.append(_gkey(val))
            elif letter == "M":
                ms.append(_gkey(val))
            elif letter == "N":
                continue
            elif letter in w:
                raise GcodeError(f"la letra {letter} aparece dos veces en el bloque")
            elif letter in "XYZABC" and letter not in self.axes:
                raise GcodeError(f"eje {letter} no configurado en la máquina")
            elif letter == "E":
                raise GcodeError("E es de impresora 3D (extrusor): este programa no es para una CNC")
            else:
                w[letter] = float(val)
        for g in gs:
            if g in UNSUPPORTED_G:
                raise GcodeError(UNSUPPORTED_G[g])
            if g not in KNOWN_G:
                raise GcodeError(f"G{g} desconocido")
        for grp in GROUPS[:-1]:
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
            if self.spindle[0] != "off" and not ({"3", "4", "5"} & set(ms)):
                self._spindle(self.spindle[0])
        if "T" in w:
            self.selected_tool = int(w["T"])
        self._mcodes_pre(ms)
        if "4" in gset:
            p = w.get("P")
            if p is None or p < 0:
                raise GcodeError("G4 necesita P ≥ 0 (segundos)")
            if p > 1000:
                self._diag("aviso", f"G4 P{p:g}: en este intérprete P son segundos (Mach3 puede usar ms)")
            self.ops.append(Op("dwell", self.line, value=float(p)))
        for g in ("17", "18", "19"):
            if g in gset:
                self.plane = g
        if "43" in gset:
            n = int(w["H"]) if "H" in w else self.tool
            t = self.m.tool(n) if n is not None else None
            if t is None:
                raise GcodeError(f"G43: la herramienta {n} no está en la tabla (herramientas: en la configuración)")
            self.tlo = t.length
        if "43.1" in gset:
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

        nonmodal = gset & ({"10", "28", "28.1", "30", "30.1", "53", "92", "92.1"} | PROBES)
        uses_axes = nonmodal & ({"10", "28", "30", "92"} | PROBES)
        if nonmodal & PROBES:
            self._probe(next(iter(nonmodal & PROBES)), axis_w)
        elif "10" in gset:
            self._g10(w, axis_w)
        elif "28" in gset or "30" in gset:
            self._g28("28" if "28" in gset else "30", axis_w)
        elif "28.1" in gset or "30.1" in gset:
            self.home["28" if "28.1" in gset else "30"] = dict(self.pos)
        elif "92" in gset:
            if not axis_w:
                raise GcodeError("G92 necesita al menos un eje")
            for a, v in axis_w.items():
                self.g92[a] = self.pos[a] - self.offsets[self.wcs][a] - (self.tlo if a == "Z" else 0) - v * self._s(a)
        elif "92.1" in gset:
            self.g92 = {a: 0.0 for a in self.axes}

        motion = [g for g in gs if g in MOTION]
        if motion:
            self.motion = motion[0]
            if self.motion in CANNED and self.cycle["Zinit"] is None:
                self.cycle["Zinit"] = self.work("Z") if "Z" in self.axes else None
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
                self._cycle(w, axis_w)
        self._mcodes_post(ms)

    # --- M ------------------------------------------------------------------------------
    def _spindle(self, d: str):
        pl = self.m.plasma if self.m.is_plasma else None
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
        """Secuencia de perforación de plasma: [sensado] → altura de perforación → antorcha → retardo → corte."""
        src = "perforación"
        if pl.probe:
            if self.wcs != "G54":
                raise GcodeError("el sensado automático de plasma trabaja en G54")
            if self.work("Z") < pl.pierce_height:
                self._move("rapid", self._work_target("Z", pl.travel_height), source=src)
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
        usa el recorrido completo (``value``), que es lo peor que puede pasar."""
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
            flo = self.m.plasma.float_offset if self.m.is_plasma and self.m.plasma else 0.0
            surface = self._off("Z") - flo
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
        wcs = self.wcs if p == 0 else WCS[p - 1]
        for a, v in vals.items():
            self.offsets[wcs][a] = v if lv == 2 else self.pos[a] - self.g92[a] - (self.tlo if a == "Z" else 0) - v
        if wcs == "G54":                 # el controlador también tiene que enterarse (sensado en marcha)
            self.ops.append(Op("wcs", self.line, value={a: self.offsets["G54"][a] for a in vals},
                               source=source or f"G10 L{lv}", frame=dict(self.offsets["G54"]),
                               end={"L": lv, **{a: (self.pos[a] - self.offsets["G54"][a]) for a in vals}}))

    def _mcodes_pre(self, ms: list[str]):
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
                raise GcodeError("subprogramas M98/M99 no soportados en v0")
            elif mc in ("0", "1", "2", "30"):
                continue                                 # después del movimiento
            elif mc in ("48", "49", "47"):
                self._diag("aviso", f"M{mc} (Mach3) se ignora")
            elif n >= 100:
                self._diag("aviso", f"M{mc}: macro de usuario de Mach3; se ignora (reescribirla para este control)")
            else:
                self._diag("aviso", f"M{mc} desconocido; se ignora")

    def _mcodes_post(self, ms: list[str]):
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
    def _g10(self, w, axis_w):
        lv, p = int(w.get("L", -1)), int(w.get("P", 0))
        if lv not in (2, 20):
            raise GcodeError("G10: solo L2 y L20 (orígenes de trabajo)")
        if not 0 <= p <= 6:
            raise GcodeError("G10: P entre 0 (actual) y 6 (G59)")
        self._set_wcs(lv, p, {a: v * self._s(a) for a, v in axis_w.items()})

    def _g28(self, which: str, axis_w):
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
        a, b, n, ia, ib, _ = PLANES[self.plane]
        if a not in self.axes or b not in self.axes:
            raise GcodeError(f"arco en G{self.plane}: faltan los ejes {a}/{b}")
        if not self.feed:
            raise GcodeError("arco G2/G3 sin F")
        cw = self.motion == "2"
        end = self._target(axis_w)
        x, y = end[a] - self.pos[a], end[b] - self.pos[b]
        if "R" in w:
            r = w["R"] * self.scale
            if abs(x) < 1e-9 and abs(y) < 1e-9:
                raise GcodeError("arco con R y el mismo punto inicial y final: usar I/J/K")
            h = 4.0 * r * r - x * x - y * y
            if h < 0:
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
            if ia not in w and ib not in w:
                raise GcodeError(f"arco sin {ia}/{ib} ni R")
            if self.arc_absolute:
                ca = w.get(ia, self.work(a)) * self.scale + self._off(a)
                cb = w.get(ib, self.work(b)) * self.scale + self._off(b)
            else:
                ca = self.pos[a] + w.get(ia, 0.0) * self.scale
                cb = self.pos[b] + w.get(ib, 0.0) * self.scale
            radius = math.hypot(self.pos[a] - ca, self.pos[b] - cb)
            r_end = math.hypot(end[a] - ca, end[b] - cb)
            if radius < 1e-9:
                raise GcodeError("arco de radio cero")
            if abs(radius - r_end) > max(0.005, 0.001 * radius):
                raise GcodeError(f"arco inválido: radio inicial {radius:.4f} y final {r_end:.4f} distintos")
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
        sweep += math.copysign(2 * math.pi * (turns - 1), sweep)
        self.ops.append(Op("arc", self.line, dict(self.pos), dict(end), self.feed, exact_stop=self.exact,
                           arc=Arc((ca, cb), (a, b, n), radius, sweep), frame=dict(self.offsets["G54"])))
        self.pos = dict(end)

    # --- ciclos fijos ---------------------------------------------------------------
    def _cycle(self, w, axis_w):
        g = self.motion
        if self.plane != "17":
            raise GcodeError("ciclos fijos solo en G17 en v0")
        if not self.absolute:
            raise GcodeError("ciclos fijos en G91 no soportados en v0")
        for k in ("Z", "R", "Q"):
            if k in w:
                self.cycle[k] = w[k] * self.scale
        if "P" in w:
            self.cycle["P"] = w["P"]
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
        xy = {k: v for k, v in axis_w.items() if k != "Z"}
        self._move("rapid", self._target(xy), src)
        self._move("rapid", {**self.pos, "Z": mz(r)}, src)
        if g in ("81", "82"):
            self._move("feed", {**self.pos, "Z": mz(z)}, src)
        else:
            depth = r
            while depth > z + 1e-9:
                nxt = max(depth - q, z)
                if g == "83" and depth < r:
                    self._move("rapid", {**self.pos, "Z": mz(depth + PECK_CLEARANCE)}, src)
                self._move("feed", {**self.pos, "Z": mz(nxt)}, src)
                depth = nxt
                if depth > z + 1e-9:
                    self._move("rapid", {**self.pos, "Z": mz(r if g == "83" else depth + PECK_CLEARANCE)}, src)
        if g == "82" and self.cycle["P"]:
            self.ops.append(Op("dwell", self.line, value=float(self.cycle["P"]), source=src))
        self._move("rapid", {**self.pos, "Z": mz(r if self.retract_r else max(zinit, r))}, src)


def interpret(text: str, machine: Machine, start: dict[str, float] | None = None, block_delete: bool = True,
              optional_stop: bool = True) -> Program:
    return Interpreter(machine, start, block_delete, optional_stop).run(text)


def modal_text(it: Interpreter) -> str:
    """Estado modal como la línea de G-codes activos de Mach3 (p. ej. «G0 G17 G21 G90 G94 G54 G49 G98 G64»)."""
    g = [f"G{it.motion}", f"G{it.plane}", "G20" if it.scale != 1.0 else "G21", "G90" if it.absolute else "G91",
         "G94", it.wcs, "G43" if it.tlo else "G49", "G99" if it.retract_r else "G98", "G61" if it.exact else "G64"]
    m = {"cw": "M3", "ccw": "M4"}.get(it.spindle[0], "M5")
    cool = " ".join(sorted("M7" if c == "niebla" else "M8" for c in it.coolant)) or "M9"
    return " ".join(g) + f" {m} {cool} T{it.tool or 0}" + (f" F{it.feed / it.scale:g}" if it.feed else "") + \
        (f" S{it.rpm:g}" if it.rpm else "")


def linearize(op: Op, tolerance: float) -> list[dict[str, float]]:
    """Puntos finales de las cuerdas de un arco (el error de cuerda no supera ``tolerance``)."""
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
