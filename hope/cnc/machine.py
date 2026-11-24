"""CNC machine configuration and coordinate systems.

Internal units: **mm** for lengths (degrees on rotary axes), **mm/min** for feeds and maximum
rates (as in G-code and Mach3) and **mm/s²** for accelerations.

Machine coordinates are the controller's after referencing (homing). Work coordinates
(G54–G59) are obtained by subtracting the work origin and the G92 offset.

YAML format (see ``configs/cnc_maquina.yaml``)::

    nombre: router 3018
    tipo: fresadora            # fresadora | router | laser | plasma
    ejes:
      X: {pasos_mm: 800, vel_max: 3000, acel: 200, min: 0, max: 300}
      ...
    desviacion_union: 0.01     # mm (grbl-style lookahead)
    tolerancia_arco: 0.002     # mm (maximum chord error when splitting arcs)
    husillo: {rpm_min: 0, rpm_max: 24000}
    origenes: {G54: {X: 0, Y: 0, Z: 0}}
    herramientas: {1: {largo: 0, diametro: 3.175}}
    controlador: {tipo: sim}   # sim | grbl | linuxcnc
    plasma:                    # only with tipo: plasma (see configs/cnc_plasma.yaml)
      altura_perforacion: 3.8
      ...

What it is
----------
The equivalent of Marlin's ``Configuration.h`` or Mach3's «Config → Motor Tuning» screen: it
describes the physical machine (axes, travel, rates, accelerations, spindle, tools, work origins,
plasma) in simple data classes (:class:`Machine`, :class:`Axis`, :class:`Tool`, :class:`Plasma`)
and loads them from YAML (:func:`load_machine`).

How it fits
-----------
* It does not depend on any other ``hope/cnc`` module (it is the base of everything).
* Used by: ``gcode.py`` (origins, tools, machine type, plasma), ``planner.py`` (per-axis rates
  and accelerations, junction deviation), ``check.py`` (travel limits, spindle),
  ``dialect.py`` (translation), ``session.py``, ``controller.py``, ``links.py``, ``backends/sim.py``,
  ``profiles.py`` (material profiles that fill in ``job_*``) and ``tool.py`` (Hope's catalog).

Conventions
-----------
* **Machine** coordinates: the controller's after referencing. Each axis's ``min``/``max`` limits
  are expressed this way.
* **Work** coordinate = machine − (origin G54..G59 + G92 + G52 + tool length in Z).
  That subtraction is done by ``gcode.Interpreter._off``; here only the origins are stored.
* YAML keys are in Spanish (``pasos_mm``, ``vel_max``…); attributes are in English.
* Nothing depends on ``dict`` order (in Python 3.4, used on Windows XP, it is not preserved): to
  show axes in order, use :func:`axis_rank` or :attr:`Machine.axis_names`.

Usage example::

    from hope.cnc.machine import load_machine
    m = load_machine("configs/cnc_maquina.yaml")
    m.axes["X"].vmax           # maximum rate of X in mm/s
    m.offset("G54")            # {"X": 20.0, "Y": 20.0, "Z": -40.0}
    m.tool(2).length           # length of tool 2 (mm), for G43 H2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ==================================================================================================
# ======================================== AXES AND NAMES ==========================================
# ==================================================================================================

# Linear (mm) and rotary (degrees) axes, with RS274/NGC names. The order of the tuples is the order
# in which they are displayed and emitted (X Y Z A B C), same as in Mach3 and LinuxCNC.
LINEAR = ("X", "Y", "Z")
ROTARY = ("A", "B", "C")
ALL_AXES = LINEAR + ROTARY


def axis_rank(a: str) -> int:
    """Display order of the axes (X Y Z A B C), without depending on a dict's order.

    Args:
        a: axis name (``"X"``…``"C"``).

    Returns:
        Position 0–5; an unknown name goes last (6). Used as the ``key=`` of ``sorted``.
    """
    return ALL_AXES.index(a) if a in ALL_AXES else len(ALL_AXES)


# Work coordinate systems (origins) G54..G59. Index + 1 is the P of «G10 L2 Pn».
WCS = ("G54", "G55", "G56", "G57", "G58", "G59")
# Supported machine types. They change the check: on a mill/router, cutting with the spindle off is
# an error; on plasma, M3 turns the torch on and may trigger the pierce sequence.
KINDS = ("fresadora", "router", "laser", "plasma")


# ==================================================================================================
# ============================================== AXIS ==============================================
# ==================================================================================================

@dataclass
class Axis:
    """One machine axis (like a row of Mach3's «Motor Tuning» or grbl's ``$100..$132``).

    Attributes (YAML keys in parentheses):

    * ``name``: ``"X"``…``"C"``.
    * ``steps_per_mm`` (``pasos_mm``): motor steps per mm (per degree on rotary axes). Depends on the
      driver (microstepping) and the transmission: typical 80–3200. Default 800. Reported to the
      controller; the planner does not use it (it works in mm). If wrong, the machine travels a
      different distance.
    * ``max_rate`` (``vel_max``): maximum rate, mm/min. Typical 1000–20000 on a router; Z is usually
      slower. Default 3000. Too high = the motor loses steps (stepper) or faults (servo).
    * ``accel`` (``acel``): acceleration, mm/s². Typical 50–1000. Default 200. Too high = lost
      steps and knocks on the frame; too low = long machining times and slow corners.
    * ``min`` / ``max``: travel limits in machine coordinates (mm). ``None`` = no limit
      (normal on rotary axes). The check warns if the program goes beyond them.
    """
    name: str
    steps_per_mm: float = 800.0
    max_rate: float = 3000.0          # mm/min
    accel: float = 200.0              # mm/s²
    min: float | None = None          # travel limit in machine coordinates (None: no limit)
    max: float | None = None

    @property
    def rotary(self) -> bool:
        """True if it is A, B or C (units in degrees; G20 does not scale them)."""
        return self.name in ROTARY

    @property
    def vmax(self) -> float:          # mm/s
        """Maximum rate in mm/s (the planner works in seconds, G-code in minutes)."""
        return self.max_rate / 60.0


# ==================================================================================================
# ============================================== TOOL ==============================================
# ==================================================================================================

@dataclass
class Tool:
    """One entry of the tool table (``herramientas:`` in the YAML; Mach3's «Tool Table»).

    * ``number``: T number in the program (``T2 M6``) and H of ``G43 H2``.
    * ``length`` (``largo``): length offset, mm. With ``G43 Hn`` it is added to the Z origin: the
      tip of a cutter 5 mm longer sits 5 mm lower, so the control raises Z by 5 mm so that the work
      coordinate still points at the tip. Default 0.
    * ``diameter`` (``diametro``): mm. Informational/display only (there is no G41/G42). Default 0.
    * ``name`` (``nombre``): free description.
    """
    number: int
    length: float = 0.0
    diameter: float = 0.0
    name: str = ""


# ==================================================================================================
# ============================================= PLASMA =============================================
# ==================================================================================================

@dataclass
class Plasma:
    """Plasma cutting: heights in work coordinates (Z0 = sheet surface).

    The pierce sequence (built by ``gcode.Interpreter._pierce`` on every M3 if ``auto_pierce``):

    1. (if ``probe``) initial height sensing (IHS): moves down with G38.2 until it touches the sheet
       and redefines G54 Z0 compensating for ``float_offset``;
    2. rapid to ``pierce_height``; 3. torch on; 4. wait ``pierce_delay``;
    5. move down to ``cut_height`` at ``plunge_feed``; when turning off (M5) rise to ``travel_height``.

    Why pierce higher than the cut height: when going through the sheet, molten metal splashes
    upward; at cut height it destroys the nozzle within a few pierces.

    Raises:
        ValueError: if the heights do not satisfy ``0 < cut ≤ pierce ≤ travel`` or some feed,
            delay or depth is invalid.
    """
    # Pierce height (mm above the sheet). Typical 1.5–2× the cut height: 3–5 mm. Default 3.8.
    # Too low = spatter that damages the consumable; too high = the arc does not transfer and does not pierce.
    pierce_height: float = 3.8        # mm
    # Cut height (mm). Given by the manufacturer per amperage and consumable: typical 1–2 mm. Default 1.5.
    # Too high = beveled cut and dross; too low = the nozzle touches the sheet.
    cut_height: float = 1.5           # mm
    # Pierce time (s) before starting to move. Grows with thickness: 0–2 s. Default 0.5.
    # Short = does not go through and the cut starts without piercing; long = large entry hole and wear.
    pierce_delay: float = 0.5         # s
    # Travel height (mm) for moving between cuts without hitting tipped-up parts. Typical 5–25.
    travel_height: float = 10.0       # mm
    plunge_feed: float = 1000.0       # mm/min (from pierce height to cut height)
    auto_pierce: bool = False         # insert the sequence on every M3 (if the CAM does not include it)
    probe: bool = False               # initial height sensing (IHS) before each pierce
    # Probing feed (mm/min). Slow so as not to bend the sheet or overshoot: typical 100–500.
    probe_feed: float = 300.0         # mm/min
    # Maximum probing travel (mm). If it does not touch within that distance, the controller alarms.
    # Must exceed travel_height with margin; too large = if the sensor fails, the torch crashes harder.
    probe_depth: float = 20.0         # mm it may move down looking for the sheet
    # Floating head slack (mm): how far the Z axis still moves down after the torch touched the
    # sheet until the switch trips. Measured once with a feeler gauge. Typical 0.5–3.
    float_offset: float = 1.5         # mm the floating head moves down from the touch until the switch trips

    def __post_init__(self):
        """Validates the configuration as soon as it is created (fails when loading the YAML, not mid-cut)."""
        if not 0 < self.cut_height <= self.pierce_height <= self.travel_height:
            raise ValueError("plasma: 0 < altura_corte ≤ altura_perforacion ≤ altura_traslado")
        if self.pierce_delay < 0 or self.plunge_feed <= 0 or self.probe_feed <= 0 or self.probe_depth <= 0:
            raise ValueError("plasma: retardo ≥ 0 y avances y profundidad de sensado > 0")


# ==================================================================================================
# ============================================ MACHINE =============================================
# ==================================================================================================

@dataclass
class Machine:
    """The complete machine. The default values describe a 300×200 mm hobby router.

    Raises (on construction):
        ValueError: invalid machine type or axis name, rates/accelerations/steps ≤ 0,
            ``min > max`` on an axis, unknown ``job_feed_mode`` or tolerances out of range.
    """
    name: str = "máquina genérica"
    kind: str = "fresadora"           # one of KINDS
    # Default axes: X 0–300, Y 0–200 and Z from −60 to 0 (Z0 at the very top, as homing leaves most
    # routers: all the work happens at negative machine Z).
    axes: dict[str, Axis] = field(default_factory=lambda: {
        "X": Axis("X", min=0.0, max=300.0), "Y": Axis("Y", min=0.0, max=200.0),
        "Z": Axis("Z", max_rate=1000.0, accel=100.0, min=-60.0, max=0.0)})
    # Junction deviation (mm), like grbl's $11. How much the planner may «round» a corner without
    # stopping completely. Higher = faster on curves made of short segments, less precise on
    # corners. Typical 0.005–0.05. 0 = stop at every corner (slow, like G61). Default 0.01 (grbl).
    junction_deviation: float = 0.01   # mm
    # Arc tolerance (mm), like grbl's $12: maximum error between the ideal arc and the straight chords
    # it is split into (gcode.linearize). Smaller = more segments (more CPU and more lines to send);
    # larger = faceted arcs. Typical 0.001–0.01. Default 0.002. It is also the tolerance used to
    # accept an R arc that «almost» reaches the end point.
    arc_tolerance: float = 0.002       # mm
    # Spindle range (rpm). An S above spindle_max produces a warning. On a laser, the range is
    # usually 0–1000 (power) like grbl's $30.
    spindle_min: float = 0.0
    spindle_max: float = 24000.0
    offsets: dict[str, dict[str, float]] = field(default_factory=dict)   # G54..G59 → {axis: value}
    tools: dict[int, Tool] = field(default_factory=dict)
    # Controller the program is sent to: {"tipo": "sim" | "grbl" | "linuxcnc", port, baud rate…}.
    # «sim» is an internal simulator: it never moves anything real.
    controller: dict = field(default_factory=lambda: {"tipo": "sim"})
    plasma: Plasma | None = None       # only with kind == "plasma" (if missing, it is created with defaults)
    # Job settings coming from the profiles (material + consumable + process), as in a slicer:
    job_feed: float | None = None     # cutting mm/min (torch/spindle on, with motion in XY)
    job_feed_mode: str = "reemplazar"  # «reemplazar» (replace) the program's F or «limitar» (cap) to this maximum
    job_rpm: float | None = None      # S when the program turns on the spindle without S
    job_plunge: float | None = None   # mm/min of Z plunges (milling), with the same mode

    def __post_init__(self):
        """Fills in the default plasma and validates everything (clear errors at load time, not while running)."""
        if self.kind == "plasma" and self.plasma is None:
            self.plasma = Plasma()
        if self.kind not in KINDS:
            raise ValueError(f"tipo de máquina «{self.kind}» no válido (use {', '.join(KINDS)})")
        for n in self.axes:
            if n not in ALL_AXES:
                raise ValueError(f"eje «{n}» no válido (use {', '.join(ALL_AXES)})")
        for n, a in self.axes.items():
            # Values ≤ 0 would divide by zero in the planner (time = distance / velocity).
            if a.max_rate <= 0 or a.accel <= 0 or a.steps_per_mm <= 0:
                raise ValueError(f"eje {n}: pasos_mm, vel_max y acel deben ser > 0")
            if a.min is not None and a.max is not None and a.min > a.max:
                raise ValueError(f"eje {n}: min > max")
        if self.job_feed_mode not in ("reemplazar", "limitar"):
            raise ValueError("job_feed_mode: reemplazar o limitar")
        if self.junction_deviation < 0 or self.arc_tolerance <= 0:
            raise ValueError("desviacion_union ≥ 0 y tolerancia_arco > 0")

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Configured axes in the canonical order X Y Z A B C (not in the YAML/dict order)."""
        return tuple(a for a in ALL_AXES if a in self.axes)

    @property
    def is_plasma(self) -> bool:
        """True if the machine is a plasma (M3 = torch, heights and piercing)."""
        return self.kind == "plasma"

    @property
    def needs_spindle(self) -> bool:
        """On a mill or router, cutting with the spindle off is a typical error; not so on a laser."""
        return self.kind in ("fresadora", "router")

    def offset(self, wcs: str) -> dict[str, float]:
        """Work origin ``wcs`` (``"G54"``…, also ``"G28"``/``"G30"``) in machine coordinates.

        Returns:
            ``{axis: mm}`` with ALL the machine's axes; those missing from the configuration are 0
            (so the interpreter never hits a missing key). Returns a new dict: modifying it does
            not change the configuration.
        """
        o = self.offsets.get(wcs, {})
        return {a: float(o.get(a, 0.0)) for a in self.axis_names}

    def tool(self, n: int) -> Tool | None:
        """Tool number ``n`` from the table, or ``None`` if it is not there (``n`` may come as a float)."""
        return self.tools.get(int(n))


# ==================================================================================================
# ============================================ LOADING =============================================
# ==================================================================================================

def as_controller(m: Machine) -> Machine:
    """The machine as seen from the controller: the translated program already carries the pierce
    sequence, so when it is interpreted again (simulator, timings) it is not added a second time.

    Args:
        m: the configured machine.

    Returns:
        ``m`` itself if there is no automatic piercing; otherwise a copy with ``auto_pierce=False``
        (``dataclasses.replace``: the original is not touched, it is still used for translating).
    """
    import dataclasses
    if m.plasma is None or not m.plasma.auto_pierce:
        return m
    return dataclasses.replace(m, plasma=dataclasses.replace(m.plasma, auto_pierce=False))


def _f(v, default=None):
    """YAML number to float, or ``default`` if missing (``None``). Accepts int, float or numeric text."""
    return default if v is None else float(v)


def machine_from_dict(d: dict) -> Machine:
    """Builds a :class:`Machine` from the YAML dict (keys in Spanish, see the header).

    Missing keys take the default values documented in each class. Axis and origin names are
    upper-cased (``x`` → ``X``, ``g54`` → ``G54``), as Mach3 accepts.

    Args:
        d: the already-read YAML (``yaml.safe_load``).

    Returns:
        The validated machine.

    Raises:
        ValueError: non-numeric values or invalid configuration (see :class:`Machine`, :class:`Plasma`).
    """
    axes = {}
    for name, a in (d.get("ejes") or {}).items():
        name = str(name).upper()
        a = a or {}                                   # «X:» with no values in the YAML → all defaults
        # «activo: false»: the axis is described but the machine does not have it (so the profile editor
        # can keep the data of a dismounted 4th axis without it showing up). No key = active.
        if a.get("activo") is False:
            continue
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
    pl = d.get("plasma")
    if pl:
        sen = pl.get("sensado") or {}                 # «sensado:» (IHS) subsection of the plasma block
        kw["plasma"] = Plasma(
            pierce_height=_f(pl.get("altura_perforacion"), 3.8), cut_height=_f(pl.get("altura_corte"), 1.5),
            pierce_delay=_f(pl.get("retardo_perforacion"), 0.5), travel_height=_f(pl.get("altura_traslado"), 10.0),
            plunge_feed=_f(pl.get("avance_bajada"), 1000.0), auto_pierce=bool(pl.get("perforacion_auto", False)),
            probe=bool(sen.get("activo", False)), probe_feed=_f(sen.get("avance"), 300.0),
            probe_depth=_f(sen.get("profundidad_max"), 20.0), float_offset=_f(sen.get("holgura_flotante"), 1.5))
    if axes:                                          # without «ejes:» the default machine's axes are used
        kw["axes"] = axes
    return Machine(**kw)


def load_machine(path: str | Path | None) -> Machine:
    """Reads the machine from a YAML file (UTF-8). ``None`` returns the default generic machine.

    Raises:
        OSError: the file does not exist or cannot be read.
        yaml.YAMLError: malformed YAML.
        ValueError: invalid configuration (see :func:`machine_from_dict`).
    """
    if path is None:
        return Machine()
    import yaml  # late import: only needed if there is a file
    return machine_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
