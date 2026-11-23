"""Configuración de la máquina CNC y sistemas de coordenadas.

Unidades internas: **mm** para longitudes (grados en ejes rotativos), **mm/min** para avances y
velocidades máximas (como en el G-code y en Mach3) y **mm/s²** para aceleraciones.

Las coordenadas de máquina son las del controlador tras el referenciado (homing). Las de trabajo
(G54–G59) se obtienen restando el origen de trabajo y el desplazamiento G92.

Formato YAML (ver ``configs/cnc_maquina.yaml``)::

    nombre: router 3018
    tipo: fresadora            # fresadora | router | laser | plasma
    ejes:
      X: {pasos_mm: 800, vel_max: 3000, acel: 200, min: 0, max: 300}
      ...
    desviacion_union: 0.01     # mm (lookahead estilo grbl)
    tolerancia_arco: 0.002     # mm (cuerda máxima al dividir arcos)
    husillo: {rpm_min: 0, rpm_max: 24000}
    origenes: {G54: {X: 0, Y: 0, Z: 0}}
    herramientas: {1: {largo: 0, diametro: 3.175}}
    controlador: {tipo: sim}   # sim | grbl | linuxcnc
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

LINEAR = ("X", "Y", "Z")
ROTARY = ("A", "B", "C")
ALL_AXES = LINEAR + ROTARY
WCS = ("G54", "G55", "G56", "G57", "G58", "G59")
KINDS = ("fresadora", "router", "laser", "plasma")


@dataclass
class Axis:
    name: str
    steps_per_mm: float = 800.0
    max_rate: float = 3000.0          # mm/min
    accel: float = 200.0              # mm/s²
    min: float | None = None          # límite de recorrido en coordenadas de máquina (None: sin límite)
    max: float | None = None

    @property
    def rotary(self) -> bool:
        return self.name in ROTARY

    @property
    def vmax(self) -> float:          # mm/s
        return self.max_rate / 60.0


@dataclass
class Tool:
    number: int
    length: float = 0.0
    diameter: float = 0.0
    name: str = ""


@dataclass
class Machine:
    name: str = "máquina genérica"
    kind: str = "fresadora"
    axes: dict[str, Axis] = field(default_factory=lambda: {
        "X": Axis("X", min=0.0, max=300.0), "Y": Axis("Y", min=0.0, max=200.0),
        "Z": Axis("Z", max_rate=1000.0, accel=100.0, min=-60.0, max=0.0)})
    junction_deviation: float = 0.01   # mm
    arc_tolerance: float = 0.002       # mm
    spindle_min: float = 0.0
    spindle_max: float = 24000.0
    offsets: dict[str, dict[str, float]] = field(default_factory=dict)   # G54..G59 → {eje: valor}
    tools: dict[int, Tool] = field(default_factory=dict)
    controller: dict = field(default_factory=lambda: {"tipo": "sim"})

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"tipo de máquina «{self.kind}» no válido (use {', '.join(KINDS)})")
        for n in self.axes:
            if n not in ALL_AXES:
                raise ValueError(f"eje «{n}» no válido (use {', '.join(ALL_AXES)})")
        for n, a in self.axes.items():
            if a.max_rate <= 0 or a.accel <= 0 or a.steps_per_mm <= 0:
                raise ValueError(f"eje {n}: pasos_mm, vel_max y acel deben ser > 0")
            if a.min is not None and a.max is not None and a.min > a.max:
                raise ValueError(f"eje {n}: min > max")
        if self.junction_deviation < 0 or self.arc_tolerance <= 0:
            raise ValueError("desviacion_union ≥ 0 y tolerancia_arco > 0")

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(a for a in ALL_AXES if a in self.axes)

    @property
    def needs_spindle(self) -> bool:
        """En fresadora o router cortar con el husillo apagado es un error típico; en láser no aplica igual."""
        return self.kind in ("fresadora", "router")

    def offset(self, wcs: str) -> dict[str, float]:
        o = self.offsets.get(wcs, {})
        return {a: float(o.get(a, 0.0)) for a in self.axis_names}

    def tool(self, n: int) -> Tool | None:
        return self.tools.get(int(n))


def _f(v, default=None):
    return default if v is None else float(v)


def machine_from_dict(d: dict) -> Machine:
    axes = {}
    for name, a in (d.get("ejes") or {}).items():
        name = str(name).upper()
        a = a or {}
        axes[name] = Axis(name, steps_per_mm=_f(a.get("pasos_mm"), 800.0), max_rate=_f(a.get("vel_max"), 3000.0),
                          accel=_f(a.get("acel"), 200.0), min=_f(a.get("min")), max=_f(a.get("max")))
    hus = d.get("husillo") or {}
    tools = {}
    for n, t in (d.get("herramientas") or {}).items():
        t = t or {}
        tools[int(n)] = Tool(int(n), length=_f(t.get("largo"), 0.0), diameter=_f(t.get("diametro"), 0.0),
                             name=str(t.get("nombre", "")))
    offsets = {str(k).upper(): {str(a).upper(): float(v) for a, v in (o or {}).items()}
               for k, o in (d.get("origenes") or {}).items()}
    kw = {"name": str(d.get("nombre", "máquina")), "kind": str(d.get("tipo", "fresadora")),
          "junction_deviation": _f(d.get("desviacion_union"), 0.01),
          "arc_tolerance": _f(d.get("tolerancia_arco"), 0.002),
          "spindle_min": _f(hus.get("rpm_min"), 0.0), "spindle_max": _f(hus.get("rpm_max"), 24000.0),
          "offsets": offsets, "tools": tools, "controller": dict(d.get("controlador") or {"tipo": "sim"})}
    if axes:
        kw["axes"] = axes
    return Machine(**kw)


def load_machine(path: str | Path | None) -> Machine:
    if path is None:
        return Machine()
    import yaml
    return machine_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
