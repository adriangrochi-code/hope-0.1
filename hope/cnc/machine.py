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
    plasma:                    # solo con tipo: plasma (ver configs/cnc_plasma.yaml)
      altura_perforacion: 3.8
      ...

Qué es
------
El equivalente a ``Configuration.h`` de Marlin o a la pantalla «Config → Motor Tuning» de Mach3:
describe la máquina física (ejes, recorridos, velocidades, aceleraciones, husillo, herramientas,
orígenes de trabajo, plasma) en clases de datos simples (:class:`Machine`, :class:`Axis`,
:class:`Tool`, :class:`Plasma`) y las carga desde YAML (:func:`load_machine`).

Cómo encaja
-----------
* No depende de ningún otro módulo de ``hope/cnc`` (es la base de todo).
* Lo usan: ``gcode.py`` (orígenes, herramientas, tipo de máquina, plasma), ``planner.py`` (velocidades
  y aceleraciones por eje, desviación de unión), ``check.py`` (límites de recorrido, husillo),
  ``dialect.py`` (traducción), ``session.py``, ``controller.py``, ``links.py``, ``backends/sim.py``,
  ``profiles.py`` (perfiles de material que completan ``job_*``) y ``tool.py`` (catálogo de Hope).

Convenciones
------------
* Coordenadas de **máquina**: las del controlador después de referenciar. Los límites ``min``/``max``
  de cada eje se expresan así.
* Coordenada de **trabajo** = máquina − (origen G54..G59 + G92 + G52 + largo de herramienta en Z).
  Esa resta la hace ``gcode.Interpreter._off``; acá solo se guardan los orígenes.
* Las claves del YAML están en español (``pasos_mm``, ``vel_max``…); los atributos en inglés.
* Nada depende del orden de los ``dict`` (en Python 3.4, que usa Windows XP, no se conserva): para
  mostrar ejes en orden se usa :func:`axis_rank` o :attr:`Machine.axis_names`.

Ejemplo de uso::

    from hope.cnc.machine import load_machine
    m = load_machine("configs/cnc_maquina.yaml")
    m.axes["X"].vmax           # velocidad máxima de X en mm/s
    m.offset("G54")            # {"X": 20.0, "Y": 20.0, "Z": -40.0}
    m.tool(2).length           # largo de la herramienta 2 (mm), para G43 H2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ==================================================================================================
# ========================================= EJES Y NOMBRES =========================================
# ==================================================================================================

# Ejes lineales (mm) y rotativos (grados), con los nombres de RS274/NGC. El orden de las tuplas es el
# orden en que se muestran y se emiten (X Y Z A B C), igual que en Mach3 y LinuxCNC.
LINEAR = ("X", "Y", "Z")
ROTARY = ("A", "B", "C")
ALL_AXES = LINEAR + ROTARY


def axis_rank(a: str) -> int:
    """Orden de los ejes para mostrar (X Y Z A B C), sin depender del orden de un dict.

    Args:
        a: nombre del eje (``"X"``…``"C"``).

    Returns:
        Posición 0–5; un nombre desconocido va al final (6). Se usa como ``key=`` de ``sorted``.
    """
    return ALL_AXES.index(a) if a in ALL_AXES else len(ALL_AXES)


# Sistemas de coordenadas de trabajo (orígenes) G54..G59. El índice + 1 es el P de «G10 L2 Pn».
WCS = ("G54", "G55", "G56", "G57", "G58", "G59")
# Tipos de máquina admitidos. Cambian la revisión: en fresadora/router cortar con husillo apagado es
# un error; en plasma, M3 enciende la antorcha y puede disparar la secuencia de perforación.
KINDS = ("fresadora", "router", "laser", "plasma")


# ==================================================================================================
# ============================================== EJE ===============================================
# ==================================================================================================

@dataclass
class Axis:
    """Un eje de la máquina (como una fila de «Motor Tuning» de Mach3 o los ``$100..$132`` de grbl).

    Atributos (claves YAML entre paréntesis):

    * ``name``: ``"X"``…``"C"``.
    * ``steps_per_mm`` (``pasos_mm``): pasos del motor por mm (por grado en rotativos). Depende del
      driver (micropasos) y la transmisión: típico 80–3200. Default 800. Se informa al controlador;
      el planificador no lo usa (trabaja en mm). Si está mal, la máquina recorre otra distancia.
    * ``max_rate`` (``vel_max``): velocidad máxima, mm/min. Típico 1000–20000 en router; Z suele ser
      más lento. Default 3000. Muy alto = el motor pierde pasos (paso a paso) o da error (servo).
    * ``accel`` (``acel``): aceleración, mm/s². Típico 50–1000. Default 200. Muy alto = pérdida de
      pasos y golpes en la estructura; muy bajo = tiempos de mecanizado largos y esquinas lentas.
    * ``min`` / ``max``: límites de recorrido en coordenadas de máquina (mm). ``None`` = sin límite
      (lo normal en ejes rotativos). La revisión avisa si el programa se sale de ellos.
    """
    name: str
    steps_per_mm: float = 800.0
    max_rate: float = 3000.0          # mm/min
    accel: float = 200.0              # mm/s²
    min: float | None = None          # límite de recorrido en coordenadas de máquina (None: sin límite)
    max: float | None = None

    @property
    def rotary(self) -> bool:
        """True si es A, B o C (unidades en grados; G20 no los escala)."""
        return self.name in ROTARY

    @property
    def vmax(self) -> float:          # mm/s
        """Velocidad máxima en mm/s (el planificador trabaja en segundos, el G-code en minutos)."""
        return self.max_rate / 60.0


# ==================================================================================================
# =========================================== HERRAMIENTA ==========================================
# ==================================================================================================

@dataclass
class Tool:
    """Una entrada de la tabla de herramientas (``herramientas:`` en el YAML; «Tool Table» de Mach3).

    * ``number``: número T del programa (``T2 M6``) y H de ``G43 H2``.
    * ``length`` (``largo``): compensación de largo, mm. Con ``G43 Hn`` se suma al origen en Z: la
      punta de una fresa 5 mm más larga queda 5 mm más abajo, así que el control sube Z en 5 mm
      para que la coordenada de trabajo siga apuntando a la punta. Default 0.
    * ``diameter`` (``diametro``): mm. Solo informativo/visualización (no hay G41/G42). Default 0.
    * ``name`` (``nombre``): descripción libre.
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
    """Corte por plasma: alturas en coordenadas de trabajo (Z0 = superficie de la chapa).

    La secuencia de perforación (la arma ``gcode.Interpreter._pierce`` en cada M3 si ``auto_pierce``):

    1. (si ``probe``) sensado inicial de altura (IHS): baja con G38.2 hasta tocar la chapa y redefine
       Z0 de G54 compensando ``float_offset``;
    2. rápido a ``pierce_height``; 3. antorcha encendida; 4. espera ``pierce_delay``;
    5. baja a ``cut_height`` a ``plunge_feed``; al apagar (M5) sube a ``travel_height``.

    Por qué perforar más alto que la altura de corte: al atravesar la chapa el metal fundido
    salpica hacia arriba; a la altura de corte destruye la boquilla en pocas perforaciones.

    Raises:
        ValueError: si las alturas no cumplen ``0 < corte ≤ perforación ≤ traslado`` o algún avance,
            retardo o profundidad es inválido.
    """
    # Altura de perforación (mm sobre la chapa). Típico 1.5–2× la altura de corte: 3–5 mm. Default 3.8.
    # Muy baja = salpicaduras que dañan el consumible; muy alta = el arco no transfiere y no perfora.
    pierce_height: float = 3.8        # mm
    # Altura de corte (mm). La da el fabricante según amperaje y consumible: típico 1–2 mm. Default 1.5.
    # Muy alta = corte en bisel y escoria; muy baja = la boquilla toca la chapa.
    cut_height: float = 1.5           # mm
    # Tiempo de perforación (s) antes de empezar a moverse. Crece con el espesor: 0–2 s. Default 0.5.
    # Corto = no atraviesa y el corte arranca sin pasar; largo = agujero de entrada grande y desgaste.
    pierce_delay: float = 0.5         # s
    # Altura de traslado (mm) para moverse entre cortes sin chocar con piezas levantadas. Típico 5–25.
    travel_height: float = 10.0       # mm
    plunge_feed: float = 1000.0       # mm/min (de perforación a altura de corte)
    auto_pierce: bool = False         # insertar la secuencia en cada M3 (si el CAM no la trae)
    probe: bool = False               # sensado inicial de altura (IHS) antes de cada perforación
    # Avance del sensado (mm/min). Lento para no doblar la chapa ni pasarse: típico 100–500.
    probe_feed: float = 300.0         # mm/min
    # Recorrido máximo del sensado (mm). Si no toca en esa distancia, el controlador da alarma.
    # Debe superar travel_height con margen; muy grande = si falla el sensor, la antorcha choca más.
    probe_depth: float = 20.0         # mm que puede bajar buscando la chapa
    # Holgura del cabezal flotante (mm): cuánto baja todavía el eje Z después de que la antorcha tocó la
    # chapa hasta que actúa el interruptor. Se mide una vez con una galga. Típico 0.5–3.
    float_offset: float = 1.5         # mm que baja el cabezal flotante desde el toque hasta que actúa el interruptor

    def __post_init__(self):
        """Valida la configuración apenas se crea (falla al cargar el YAML, no a mitad del corte)."""
        if not 0 < self.cut_height <= self.pierce_height <= self.travel_height:
            raise ValueError("plasma: 0 < altura_corte ≤ altura_perforacion ≤ altura_traslado")
        if self.pierce_delay < 0 or self.plunge_feed <= 0 or self.probe_feed <= 0 or self.probe_depth <= 0:
            raise ValueError("plasma: retardo ≥ 0 y avances y profundidad de sensado > 0")


# ==================================================================================================
# ============================================= MÁQUINA ============================================
# ==================================================================================================

@dataclass
class Machine:
    """La máquina completa. Los valores por defecto describen un router de hobby de 300×200 mm.

    Raises (al construirla):
        ValueError: tipo de máquina o nombre de eje inválido, velocidades/aceleraciones/pasos ≤ 0,
            ``min > max`` en un eje, ``job_feed_mode`` desconocido o tolerancias fuera de rango.
    """
    name: str = "máquina genérica"
    kind: str = "fresadora"           # uno de KINDS
    # Ejes por defecto: X 0–300, Y 0–200 y Z de −60 a 0 (Z0 arriba del todo, como deja el homing de
    # la mayoría de los routers: todo el trabajo está en Z negativo de máquina).
    axes: dict[str, Axis] = field(default_factory=lambda: {
        "X": Axis("X", min=0.0, max=300.0), "Y": Axis("Y", min=0.0, max=200.0),
        "Z": Axis("Z", max_rate=1000.0, accel=100.0, min=-60.0, max=0.0)})
    # Desviación de unión (mm), como $11 de grbl. Cuánto puede «redondear» el planificador una esquina
    # sin frenar del todo. Más alto = más rápido en curvas hechas de tramos cortos, menos preciso en
    # esquinas. Típico 0.005–0.05. 0 = parada en cada esquina (lento, como G61). Default 0.01 (grbl).
    junction_deviation: float = 0.01   # mm
    # Tolerancia de arco (mm), como $12 de grbl: error máximo entre el arco ideal y las cuerdas rectas
    # en que se divide (gcode.linearize). Más chico = más tramos (más CPU y más líneas a enviar); más
    # grande = arcos facetados. Típico 0.001–0.01. Default 0.002. También es la tolerancia con que se
    # acepta un arco con R que «casi» llega al punto final.
    arc_tolerance: float = 0.002       # mm
    # Rango del husillo (rpm). Un S por encima de spindle_max genera un aviso. En láser, el rango
    # suele ser 0–1000 (potencia) como $30 de grbl.
    spindle_min: float = 0.0
    spindle_max: float = 24000.0
    offsets: dict[str, dict[str, float]] = field(default_factory=dict)   # G54..G59 → {eje: valor}
    tools: dict[int, Tool] = field(default_factory=dict)
    # Controlador al que se envía el programa: {"tipo": "sim" | "grbl" | "linuxcnc", puerto, baudios…}.
    # «sim» es un simulador interno: nunca mueve nada real.
    controller: dict = field(default_factory=lambda: {"tipo": "sim"})
    plasma: Plasma | None = None       # solo con kind == "plasma" (si falta, se crea con valores por defecto)
    # Ajustes del trabajo que salen de los perfiles (material + consumible + proceso), como en un slicer:
    job_feed: float | None = None     # mm/min de corte (antorcha/husillo encendidos, con movimiento en XY)
    job_feed_mode: str = "reemplazar"  # «reemplazar» la F del programa o «limitar» a este máximo
    job_rpm: float | None = None      # S cuando el programa enciende el husillo sin S
    job_plunge: float | None = None   # mm/min de las bajadas en Z (fresado), con el mismo modo

    def __post_init__(self):
        """Completa el plasma por defecto y valida todo (errores claros al cargar, no en marcha)."""
        if self.kind == "plasma" and self.plasma is None:
            self.plasma = Plasma()
        if self.kind not in KINDS:
            raise ValueError(f"tipo de máquina «{self.kind}» no válido (use {', '.join(KINDS)})")
        for n in self.axes:
            if n not in ALL_AXES:
                raise ValueError(f"eje «{n}» no válido (use {', '.join(ALL_AXES)})")
        for n, a in self.axes.items():
            # Valores ≤ 0 harían dividir por cero en el planificador (tiempo = distancia / velocidad).
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
        """Ejes configurados en el orden canónico X Y Z A B C (no en el orden del YAML/dict)."""
        return tuple(a for a in ALL_AXES if a in self.axes)

    @property
    def is_plasma(self) -> bool:
        """True si la máquina es de plasma (M3 = antorcha, alturas y perforación)."""
        return self.kind == "plasma"

    @property
    def needs_spindle(self) -> bool:
        """En fresadora o router cortar con el husillo apagado es un error típico; en láser no aplica igual."""
        return self.kind in ("fresadora", "router")

    def offset(self, wcs: str) -> dict[str, float]:
        """Origen de trabajo ``wcs`` (``"G54"``…, también ``"G28"``/``"G30"``) en coordenadas de máquina.

        Returns:
            ``{eje: mm}`` con TODOS los ejes de la máquina; los que faltan en la configuración valen 0
            (así el intérprete nunca encuentra una clave faltante). Devuelve un dict nuevo: modificarlo
            no cambia la configuración.
        """
        o = self.offsets.get(wcs, {})
        return {a: float(o.get(a, 0.0)) for a in self.axis_names}

    def tool(self, n: int) -> Tool | None:
        """Herramienta número ``n`` de la tabla, o ``None`` si no está (``n`` puede venir como float)."""
        return self.tools.get(int(n))


# ==================================================================================================
# ============================================ CARGA ===============================================
# ==================================================================================================

def as_controller(m: Machine) -> Machine:
    """La máquina vista desde el controlador: el programa traducido ya trae la secuencia de perforación,
    así que al volver a interpretarlo (simulador, tiempos) no se agrega otra vez.

    Args:
        m: la máquina configurada.

    Returns:
        ``m`` mismo si no hay perforación automática; si no, una copia con ``auto_pierce=False``
        (``dataclasses.replace``: no se toca la original, que sigue usándose para traducir).
    """
    import dataclasses
    if m.plasma is None or not m.plasma.auto_pierce:
        return m
    return dataclasses.replace(m, plasma=dataclasses.replace(m.plasma, auto_pierce=False))


def _f(v, default=None):
    """Número del YAML a float, o ``default`` si falta (``None``). Acepta int, float o texto numérico."""
    return default if v is None else float(v)


def machine_from_dict(d: dict) -> Machine:
    """Construye una :class:`Machine` desde el dict del YAML (claves en español, ver encabezado).

    Las claves que faltan toman los valores por defecto documentados en cada clase. Los nombres de
    ejes y orígenes se pasan a mayúsculas (``x`` → ``X``, ``g54`` → ``G54``), como acepta Mach3.

    Args:
        d: el YAML ya leído (``yaml.safe_load``).

    Returns:
        La máquina validada.

    Raises:
        ValueError: valores no numéricos o configuración inválida (ver :class:`Machine`, :class:`Plasma`).
    """
    axes = {}
    for name, a in (d.get("ejes") or {}).items():
        name = str(name).upper()
        a = a or {}                                   # «X:» sin valores en el YAML → todo por defecto
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
        sen = pl.get("sensado") or {}                 # subsección «sensado:» (IHS) del bloque plasma
        kw["plasma"] = Plasma(
            pierce_height=_f(pl.get("altura_perforacion"), 3.8), cut_height=_f(pl.get("altura_corte"), 1.5),
            pierce_delay=_f(pl.get("retardo_perforacion"), 0.5), travel_height=_f(pl.get("altura_traslado"), 10.0),
            plunge_feed=_f(pl.get("avance_bajada"), 1000.0), auto_pierce=bool(pl.get("perforacion_auto", False)),
            probe=bool(sen.get("activo", False)), probe_feed=_f(sen.get("avance"), 300.0),
            probe_depth=_f(sen.get("profundidad_max"), 20.0), float_offset=_f(sen.get("holgura_flotante"), 1.5))
    if axes:                                          # sin «ejes:» se usan los de la máquina por defecto
        kw["axes"] = axes
    return Machine(**kw)


def load_machine(path: str | Path | None) -> Machine:
    """Lee la máquina de un archivo YAML (UTF-8). ``None`` devuelve la máquina genérica por defecto.

    Raises:
        OSError: el archivo no existe o no se puede leer.
        yaml.YAMLError: YAML mal formado.
        ValueError: configuración inválida (ver :func:`machine_from_dict`).
    """
    if path is None:
        return Machine()
    import yaml  # import tardío: solo hace falta si hay archivo
    return machine_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
