"""3D-slicer-style profiles (PrusaSlicer, OrcaSlicer, TISMA Engine) for TISMA CNC.

As in a slicer, four profiles are chosen and the app computes the rest:

* **Machine** (the «printer»): travel, speeds, controller, plasma hardware.
* **Material** (the «filament»): plasma cut table by thickness and amperage, or chip load and RPM
  for milling.
* **Consumable / tool** (the «nozzle»): torch amperage, or the end mill's diameter and flutes.
* **Process** (the «print profile»): how that data is used (feed from the profile or the program,
  percentage, automatic pierce, height sensing, travel height…).

Format: one JSON file per profile, with ``"hereda": "<parent name>"`` (inherits) to store only what
changes (dicts are merged, lists are replaced). **System** profiles ship with the app
(``hope/cnc/perfiles/``) and are never modified: changes are saved as **user** profiles
(«Guardar como…», Save As), which inherit from them. Only JSON and the standard library: works on
Python 3.8.

HOW IT FITS
    * Used by ``hope.cnc.app.server`` and ``hope.cnc.__main__`` (they create the ``ProfileStore`` with
      the user folder ``<data>/perfiles``) and ``hope.cnc.session`` (lists, edits, saves and deletes
      profiles for the interface, and recomputes the machine with ``effective`` whenever the selection
      changes).
    * Uses ``hope.cnc.machine`` (``machine_from_dict``) to validate machine profiles and build the
      effective ``Machine``. What this module computes travels in ``Machine.job_feed``,
      ``job_feed_mode``, ``job_plunge`` and ``job_rpm``, and ``hope.cnc.gcode`` applies it when
      interpreting.
    * The interface's editor is drawn solely from ``SCHEMA`` (groups, labels, units, limits): adding
      a field here is enough for it to appear on screen and be validated on save.

INHERITANCE («hereda»)
    A profile may say ``"hereda": "Mesa de plasma 1250×1250"``. When resolving it (``resolve``) the
    chain is followed up to the root and merged from the root toward the child: the child wins. When
    saving a user profile that inherits, **only what differs** from the parent is stored (``_diff``):
    that way, if an app update fixes a value in the system profile, user profiles that did not touch
    that value get the fix (same as in PrusaSlicer).
    ``predeterminado`` (default; a list of machine types) marks the profile chosen by default and is
    **not** inherited: otherwise every child of a default profile would be default too.

FEED CALCULATION (``effective``)
    * Plasma: the material's cut table row with the **nearest thickness** among those for the torch's
      amperage is taken (no interpolation: the rows are points tested by the manufacturer and a clean
      cut is not guaranteed between two of them). From it come speed, pierce and cut heights, delay,
      kerf and arc voltage.
    * Milling: feed (mm/min) = chip load (mm/tooth) × flutes × rpm. The chip load comes from the
      material's table, linearly interpolated by end mill diameter.
    In both cases the feed is multiplied by the process percentage and applied according to
    ``avance_modo`` (feed mode): «perfil» (replaces the program's F), «limitar» (the program's F,
    capped at the computed one) or «programa» (the program's F, like Mach3; the computed one is only
    shown).

UNITS
    Lengths in mm, feeds in mm/min, accelerations in mm/s², times in s, current in A, voltage in V,
    spindle in rpm, chip load in mm/tooth. Key names are in Spanish without accents (they are also
    the JSON and old YAML names).

FILES
    ``hope/cnc/perfiles/<kind>/<name>.json`` (system) and ``<data>/perfiles/<kind>/<name>.json``
    (user), where ``<kind>`` is ``maquina``, ``material``, ``consumible`` or ``proceso``. The profile
    name lives inside the JSON (``"nombre"``); the file name is just an ASCII ``slug``.

EXAMPLE
    >>> store = ProfileStore(user_dir="datos/perfiles")
    >>> sel = default_selection(store)          # {"maquina": …, "material": …, "consumible": …, …}
    >>> m, resumen = effective(store, sel)
    >>> resumen["texto"]                        # e.g. «3800 mm/min · 3 mm · perf. 3.8 / corte 1.5 mm · …»
    >>> store.save("proceso", "Mi plasma", {"avance_pct": 90}, inherits="Plasma: tabla de corte")
    {'nombre': 'Mi plasma', 'hereda': 'Plasma: tabla de corte', 'avance_pct': 90}
"""

from __future__ import annotations

import copy
import json
import re
from collections import OrderedDict
from pathlib import Path

from hope.cnc.machine import Machine, machine_from_dict

# ==================================================================================================
# ========================================= CONFIGURATION ==========================================
# ==================================================================================================

# The four profile kinds. They are also the names of the subfolders (system and user) and the keys
# of a selection. The order is that of the interface's tabs.
KINDS = ("maquina", "material", "consumible", "proceso")
# Readable name of each profile kind, for the interface and the error messages.
KIND_NAMES = {"maquina": "Máquina", "material": "Material", "consumible": "Consumible / herramienta",
              "proceso": "Proceso"}
# Machine types. A material/consumable/process declares in «tipos» which ones it is for (e.g. a
# plasma cut table makes no sense on a milling machine); the editor hides the SCHEMA fields whose
# «tipos» does not include the selected machine's type.
MACHINE_KINDS = ("fresadora", "router", "laser", "plasma")
# System profiles: they ship with the app and are read-only.
SYSTEM_DIR = Path(__file__).parent / "perfiles"
# Maximum number of inheritance hops («hereda») when resolving a profile. Protects against absurd
# chains or hand-edited files that inherit in a circle. In practice 1–3 levels are used; 10 is plenty.
MAX_DEPTH = 10


# Profile error (unreadable, nonexistent, invalid, system profile, circular inheritance…). The
# message is shown as-is in the interface. Inherits from ValueError so that whoever validates data
# catches it together with ``machine_from_dict``'s ValueErrors.
class ProfileError(ValueError):
    pass


# ==================================================================================================
# ====================================== DICTIONARY UTILITIES ======================================
# ==================================================================================================
# Profiles are nested dicts (JSON). These functions merge them and access them by «dotted key»
# («ejes.X.vel_max»), which is how SCHEMA names each editor field.

def slug(name: str) -> str:
    """ASCII-only file name (old Windows, FAT USB sticks, ZIP).

    Example: «Mesa de plasma 1250×1250» → ``mesa_de_plasma_12501250``. Accents are dropped
    («Máquina» → ``maquina``), anything that is not a letter, digit or hyphen becomes «_», and it is
    cut at 80 characters (far from the Windows XP path limit). If nothing is left, ``perfil``.
    Similar names may collide; that is why the real name lives inside the JSON.
    """
    import unicodedata
    # NFKD splits the letter from its accent («á» → «a» + «´»); converting to ASCII drops the accent.
    s = unicodedata.normalize("NFKD", name.strip().lower()).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\-]+", "_", s).strip("_")
    return s[:80] or "perfil"


def deep_merge(base: dict, over: dict) -> dict:
    """Merge ``over`` onto ``base`` and return a new dict (neither input is modified).

    Dicts are merged key by key, recursively; any other value (numbers, strings **and lists**) in
    ``over`` replaces the one in ``base`` entirely. Lists are replaced on purpose: a child's cut
    table or tool table is the complete table, not «extra rows».
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_path(d: dict, key: str, default=None):
    """Read a dotted key: ``get_path(perfil, "ejes.X.vel_max")``. ``default`` if any part is missing."""
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(d: dict, key: str, value) -> None:
    """Write a dotted key into ``d`` (modifies it), creating any missing intermediate dicts."""
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def del_path(d: dict, key: str) -> None:
    """Delete a dotted key from ``d`` (modifies it). Does nothing if it does not exist.

    Useful to «go back to the parent's value» in an inherited profile: without the key, the parent's wins.
    """
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            return
        cur = cur[part]
    cur.pop(parts[-1], None)


def flat_keys(d: dict, prefix: str = "") -> list[str]:
    """All «leaf» keys of ``d``, dotted: ``{"ejes": {"X": {"acel": 500}}}`` → ``["ejes.X.acel"]``.

    Lists count as leaves (a table is a single editor field). So does an empty dict.
    """
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:                  # empty dict = leaf (otherwise the key would vanish)
            out += flat_keys(v, key + ".")
        else:
            out.append(key)
    return out


# ==================================================================================================
# ============================================= SCHEMA =============================================
# ==================================================================================================
# Schema: what the editor shows (groups, labels, units and limits), like a slicer's tabs.
# The interface (JavaScript) receives SCHEMA as is and builds the form; ``ProfileStore._validate``
# uses the same «min»/«max» when saving. A field = a dict with the keys ``_f`` builds.

def _f(key, label, unit="", group="General", kind="num", mn=None, mx=None, help="", tipos=None, options=None,
       columns=None):
    """Build the description of an editor field (short name so SCHEMA reads like a table).

    Parameters:
        key:     dotted key inside the profile («ejes.X.acel»).
        label:   visible label. unit: unit shown next to it («mm/min»; empty = no unit).
        group:   editor group/tab where it appears («Eje X», «Plasma»…).
        kind:    control type: «num» (number, the only one validated with min/max), «texto» (text),
                 «bool» (checkbox), «opcion» (drop-down list), «multiple» (several checkboxes) or
                 «tabla» (table).
        mn, mx:  limits accepted when saving (None = no limit).
        help:    short help shown next to it or on mouse hover.
        tipos:   machine types for which the field is shown (None = all).
        options: possible values for «opcion»/«multiple».
        columns: columns of a «tabla» (list of {clave, etiqueta, unidad[, texto]}).

    Returns the dict with the Spanish keys the interface expects. Optional keys are only included
    when they have a value, so the JSON sent to the browser stays short.
    """
    d = {"clave": key, "etiqueta": label, "unidad": unit, "grupo": group, "tipo": kind, "ayuda": help}
    if mn is not None:
        d["min"] = mn
    if mx is not None:
        d["max"] = mx
    if tipos:
        d["tipos"] = list(tipos)
    if options:
        d["opciones"] = list(options)
    if columns:
        d["columnas"] = columns
    return d


def _axis_fields(a: str) -> list[dict]:
    """The five fields of an axis (``a`` = «X», «Y» or «Z»), in their own «Eje X» group.

    The help texts name the equivalent Mach3 field (Config → Motor Tuning) so someone migrating can
    copy the values as is:
        pasos_mm: motor steps per mm of travel (≥ 0.001). Wrong value = parts of a different size.
        vel_max:  axis maximum speed (mm/min, ≥ 1). Used for G0 and caps the computed feeds.
                  Too high = the motor loses steps.
        acel:     acceleration (mm/s², ≥ 0.1). Typical 100–1000 on home-built routers. High = loses
                  steps or shakes the machine; low = slow corners and short segments.
        min/max:  travel in machine coordinates (mm). Empty = no limit (not checked).
    """
    g = f"Eje {a}"
    return [_f(f"ejes.{a}.pasos_mm", "Pasos por mm", "pasos/mm", g, mn=0.001,
               help="«Steps per» de Config → Motor Tuning en Mach3"),
            _f(f"ejes.{a}.vel_max", "Velocidad máxima", "mm/min", g, mn=1, help="«Velocity» en Mach3"),
            _f(f"ejes.{a}.acel", "Aceleración", "mm/s²", g, mn=0.1, help="«Acceleration» en Mach3"),
            _f(f"ejes.{a}.min", "Recorrido mínimo", "mm", g, help="coordenada de máquina; vacío = sin límite"),
            _f(f"ejes.{a}.max", "Recorrido máximo", "mm", g)]


def _rotary_fields(a: str) -> list[dict]:
    """The fields of a rotary axis (``a`` = «A», «B» or «C»), in their own group.

    RS274/NGC convention (the same as Mach3, LinuxCNC and grbl): A rotates around X, B around Y and
    C around Z. Units are degrees instead of mm, and G20 (inches) does not affect them.
        activo:   checkbox «the machine has this axis». Unchecked, the axis does not exist for the
                  program (G-code with «A…» is an error) even though its values stay stored in the profile.
        pasos_mm: motor steps per DEGREE (same key name as the linear axes). Example: a 200-step motor
                  × 16 microsteps with 1:4 reduction → 200 × 16 × 4 / 360 ≈ 35.56 steps/°.
        vel_max:  °/min. Typical 1800–7200 (5 to 20 turns per minute) on a belt-driven indexer.
        acel:     °/s². Typical 100–1000; a heavy part in the chuck calls for less.
        min/max:  limits in machine degrees. Empty = free rotation (normal for a 4th axis: it can turn
                  endlessly). With limits (5-axis trunnion table), the check enforces them.
    """
    eje = {"A": "X", "B": "Y", "C": "Z"}[a]
    g = f"Eje {a} (rotativo)"
    return [_f(f"ejes.{a}.activo", f"Tiene eje {a}", "", g, "bool",
               help=f"gira alrededor de {eje}; «Rotational» en Config → Motor Tuning de Mach3"),
            _f(f"ejes.{a}.pasos_mm", "Pasos por grado", "pasos/°", g, mn=0.001,
               help="pasos del motor × micropasos × reducción / 360"),
            _f(f"ejes.{a}.vel_max", "Velocidad máxima", "°/min", g, mn=1, help="«Velocity» en Mach3 (grados)"),
            _f(f"ejes.{a}.acel", "Aceleración", "°/s²", g, mn=0.1, help="«Acceleration» en Mach3 (grados)"),
            _f(f"ejes.{a}.min", "Límite mínimo", "°", g, help="vacío = giro libre (sin límite)"),
            _f(f"ejes.{a}.max", "Límite máximo", "°", g)]


# --- table columns -------------------------------------------------------------------------------

# Plasma cut table (one row per thickness and amperage), copied from the power source manufacturer's
# table. Each row is a tested point; ``effective`` picks the one with the nearest thickness.
PLASMA_COLUMNS = [
    {"clave": "espesor", "etiqueta": "Espesor", "unidad": "mm"},              # sheet; > 0
    {"clave": "amperaje", "etiqueta": "Amperaje", "unidad": "A"},             # must match the torch
    {"clave": "velocidad", "etiqueta": "Velocidad", "unidad": "mm/min"},      # cutting feed; > 0
    # Pierce height: higher than the cut height so the dross thrown up while piercing does not
    # damage the nozzle. Typical 3–4 mm. Must be ≥ cut height (``_validate`` enforces it).
    {"clave": "altura_perforacion", "etiqueta": "Alt. perforación", "unidad": "mm"},
    # Cut height (torch to sheet). Typical 1–2 mm. Too low = the nozzle touches and wears; too high
    # = bevelled cut with dross.
    {"clave": "altura_corte", "etiqueta": "Alt. corte", "unidad": "mm"},
    # Pierce delay: time standing still with the arc on until it goes through the sheet. Short = does
    # not pierce; long = large hole and wear. If missing, 0.5 s.
    {"clave": "retardo", "etiqueta": "Retardo", "unidad": "s"},
    {"clave": "kerf", "etiqueta": "Sangría (kerf)", "unidad": "mm"},          # cut width (informational)
    # Reference arc voltage for torch height control (THC). Informational for now.
    {"clave": "tension", "etiqueta": "Tensión de arco", "unidad": "V"},
]
# Recommended chip load by end mill diameter (what end mill manufacturers publish). Between two
# diameters it is linearly interpolated (``_chipload``).
CHIP_COLUMNS = [{"clave": "diametro", "etiqueta": "Diámetro de fresa", "unidad": "mm"},
                {"clave": "carga", "etiqueta": "Carga de viruta", "unidad": "mm/diente"}]
# Machine profile tool table (T, length H for G43, diameter, name), like Mach3's Tool Table. In the
# profile it is a list; ``machine_dict`` converts it to the {number: data} format.
TOOL_COLUMNS = [{"clave": "numero", "etiqueta": "T", "unidad": ""},
                {"clave": "largo", "etiqueta": "Largo (H)", "unidad": "mm"},
                {"clave": "diametro", "etiqueta": "Diámetro", "unidad": "mm"},
                {"clave": "nombre", "etiqueta": "Nombre", "unidad": "", "texto": True}]   # free text

# --- editor fields, per profile kind -------------------------------------------------------------
# The order of each list is the on-screen order. Only fields that make sense to edit from the editor
# are listed; a profile may have other keys (e.g. «nota») that are preserved anyway.
SCHEMA = {
    # --- Machine: what is fixed in the equipment (mechanics, spindle, plasma hardware, controller) ---
    "maquina": [
        # Type: decides which fields are shown in the four profiles and which materials/processes apply.
        _f("tipo", "Tipo de máquina", "", "General", "opcion", options=MACHINE_KINDS),
        *_axis_fields("X"), *_axis_fields("Y"), *_axis_fields("Z"),
        # 4th, 5th and 6th axis (rotary): indexer, table-top lathe, trunnion table, head.
        *_rotary_fields("A"), *_rotary_fields("B"), *_rotary_fields("C"),
        # Motion. Junction deviation (mm, 0–1): how much the planner may «round» a corner without
        # fully stopping. Typical 0.01–0.05 (grbl uses 0.01). Arc tolerance (mm, 0.0001–1): maximum
        # error when splitting G2/G3 into straight segments; smaller = more segments.
        _f("desviacion_union", "Desviación en esquinas", "mm", "Movimiento", mn=0, mx=1,
           help="más alto = esquinas más rápidas y menos exactas (junction deviation de grbl)"),
        _f("tolerancia_arco", "Tolerancia de arcos", "mm", "Movimiento", mn=0.0001, mx=1),
        # Spindle (milling machine/router only). rpm_max also caps the material's RPM in ``effective``
        # (24000 if not given): the feed is computed with the rpm the machine can actually reach.
        _f("husillo.rpm_min", "RPM mínimas", "rpm", "Husillo", mn=0, tipos=("fresadora", "router")),
        _f("husillo.rpm_max", "RPM máximas", "rpm", "Husillo", mn=0, tipos=("fresadora", "router")),
        # Reference G54 offset (mm, machine coordinates). Used to check the program with no controller
        # connected; with a real one its offsets are used (``Backend.work_offsets``).
        _f("origenes.G54.X", "G54 X", "mm", "Origen de trabajo de referencia"),
        _f("origenes.G54.Y", "G54 Y", "mm", "Origen de trabajo de referencia"),
        _f("origenes.G54.Z", "G54 Z", "mm", "Origen de trabajo de referencia",
           help="antes de enviar se leen los orígenes reales del controlador"),
        # Plasma hardware: initial height sensing (IHS). The torch moves down (G38.2) until it touches
        # the sheet at «avance_sensado» (mm/min; slow = more accurate, typical 200–500) without going
        # beyond «profundidad_max» (mm), and is corrected by «holgura_flotante» (mm the floating head
        # travels before tripping the switch; typical 1–3). Whether the machine *has* it goes here;
        # whether it is *used* is decided by the process.
        _f("plasma_hw.sensado", "Tiene sensado de altura (IHS)", "", "Plasma", "bool", tipos=("plasma",),
           help="cabezal flotante o sensor óhmico conectado como sonda"),
        _f("plasma_hw.holgura_flotante", "Holgura del cabezal flotante", "mm", "Plasma", mn=0, tipos=("plasma",)),
        _f("plasma_hw.profundidad_max", "Profundidad máxima de sensado", "mm", "Plasma", mn=1, tipos=("plasma",)),
        _f("plasma_hw.avance_sensado", "Avance de sensado", "mm/min", "Plasma", mn=1, tipos=("plasma",)),
        _f("herramientas", "Tabla de herramientas", "", "Herramientas", "tabla", columns=TOOL_COLUMNS,
           tipos=("fresadora", "router")),
        # Controller: who the program is sent to (see ``hope.cnc.backends``). «bufer» (bytes, ≥ 64)
        # must be the real receive buffer size: larger overflows it.
        _f("controlador.tipo", "Controlador", "", "Controlador", "opcion", options=("sim", "grbl", "ok", "linuxcnc"),
           help="sim = simulador; grbl = grbl/grblHAL/FluidNC; ok = Marlin/Smoothieware/RepRapFirmware"),
        _f("controlador.puerto", "Puerto", "", "Controlador", "texto",
           help="COM3, /dev/ttyUSB0 o tcp://192.168.1.50:23"),
        _f("controlador.baudios", "Baudios", "", "Controlador", mn=1200),
        _f("controlador.bufer", "Búfer del controlador", "bytes", "Controlador", mn=64,
           help="128 en grbl 1.1; 1024 en grblHAL"),
    ],
    # --- Material: how that material is cut (plasma table or milling data) --------------------------
    "material": [
        # «Sirve para» (Suitable for): which machine types this profile is offered for (empty = all).
        # Same in consumable and process.
        _f("tipos", "Sirve para", "", "General", "multiple", options=MACHINE_KINDS),
        _f("plasma.tabla", "Tabla de corte", "", "Plasma", "tabla", columns=PLASMA_COLUMNS, tipos=("plasma",),
           help="valores de partida; ajustalos con un corte de prueba"),
        # Milling. Recommended RPM for the material (capped at the machine's maximum rpm). The chip
        # load by diameter gives the feed; «bajada_pct» (1–100 %) is the Z plunge feed as a fraction
        # of the cutting feed: plunging straight down loads the end mill heavily, typical 30–50 %.
        _f("fresado.rpm", "RPM", "rpm", "Fresado", mn=0, tipos=("fresadora", "router")),
        _f("fresado.carga_viruta", "Carga de viruta por diámetro", "", "Fresado", "tabla", columns=CHIP_COLUMNS,
           tipos=("fresadora", "router")),
        _f("fresado.bajada_pct", "Avance de bajada", "% del de corte", "Fresado", mn=1, mx=100,
           tipos=("fresadora", "router")),
    ],
    # --- Consumable / tool: what gets changed often (plasma nozzle, end mill) -----------------------
    "consumible": [
        _f("tipos", "Sirve para", "", "General", "multiple", options=MACHINE_KINDS),
        # Nozzle/torch amperage (A): filters the rows of the material's cut table.
        _f("amperaje", "Amperaje", "A", "Plasma", mn=1, tipos=("plasma",)),
        # End mill: diameter (mm) to look up the chip load, flutes (1–12) to multiply it.
        # The shape is informational (it does not change the calculation yet).
        _f("diametro", "Diámetro", "mm", "Fresa", mn=0.01, tipos=("fresadora", "router")),
        _f("filos", "Filos", "", "Fresa", mn=1, mx=12, tipos=("fresadora", "router")),
        _f("forma", "Forma", "", "Fresa", "opcion", options=("plana", "bola", "V", "compresión"),
           tipos=("fresadora", "router")),
    ],
    # --- Process: how the data above is used in this job -----------------------------------------
    "proceso": [
        _f("tipos", "Sirve para", "", "General", "multiple", options=MACHINE_KINDS),
        # Feed mode (see the module header). «programa» is what Mach3 would do and is the default if
        # the process does not say: the F is never changed unless the user asks for it.
        _f("avance_modo", "Avance de corte", "", "Avance", "opcion", options=("perfil", "limitar", "programa"),
           help="perfil = usar el del material; limitar = la F del programa sin pasarse del perfil; "
                "programa = la F del programa (como Mach3)"),
        # Percentage of the computed feed (10–200 %, default 100). Less = better finish (e.g. plasma
        # «calidad» (quality) at 85 %); over 100 only if tests allow it.
        _f("avance_pct", "Porcentaje del avance del perfil", "%", "Avance", mn=10, mx=200),
        _f("perforacion_auto", "Perforación automática en cada M3", "", "Plasma", "bool", tipos=("plasma",),
           help="apagar si el CAM ya genera la secuencia (SheetCam y otros)"),
        # Sensing: only takes effect if the machine also has sensing (plasma_hw.sensado).
        _f("sensado", "Sensar la altura antes de perforar", "", "Plasma", "bool", tipos=("plasma",)),
        # Travel height (mm, ≥ 0.5; 10 if missing): torch height between cuts. Never below the pierce
        # height. Low = fast travels but risk of hitting tipped-up sheet pieces.
        _f("altura_traslado", "Altura de traslado", "mm", "Plasma", mn=0.5, tipos=("plasma",)),
        # Added to the table's delay (s, ≥ 0): for dirty or rusty sheet or a weak power source.
        _f("retardo_extra", "Retardo extra de perforación", "s", "Plasma", mn=0, tipos=("plasma",)),
        _f("rpm_si_falta", "Poner la S del material si el programa no trae", "", "Fresado", "bool",
           tipos=("fresadora", "router")),
    ],
}


# ==================================================================================================
# ========================================= PROFILE STORE ==========================================
# ==================================================================================================

class ProfileStore:
    """System and user profiles, read from disk on every query.

    Nothing is cached in memory on purpose: if the user copies a profile into the folder by hand (or
    another window imports it), it shows up without restarting. They are few small files, so it is cheap.

    Rules:
        * System profiles are never modified or deleted (``save``/``delete`` refuse it).
        * A user profile with the same name as a system one is ignored (it does not «shadow» it).
        * A profile with children cannot be deleted (it would leave them orphaned).
    """

    def __init__(self, user_dir: str | Path | None = None, system_dir: str | Path = SYSTEM_DIR):
        """``user_dir``: user profile folder (None = read-only, no «Guardar» (Save)).
        ``system_dir``: system profile folder (tests pass a different one)."""
        self.system_dir = Path(system_dir)
        self.user_dir = Path(user_dir) if user_dir else None

    # --- reading -----------------------------------------------------------------
    def _files(self, kind: str, origin: str) -> list[Path]:
        """.json files of a kind («maquina»…) and origin («sistema»/«usuario»), sorted by name
        (stable order on any file system). Empty list if the folder does not exist."""
        root = self.system_dir if origin == "sistema" else self.user_dir
        if root is None or not (root / kind).is_dir():
            return []
        return sorted((root / kind).glob("*.json"))

    def _load_all(self, kind: str) -> dict[str, tuple[dict, str, Path]]:
        """All profiles of a kind: ``{name: (data as is from the JSON, origin, file)}``.

        System ones first and then user ones, each group sorted by file. The name is the JSON's
        ``"nombre"`` (or the file's if missing). Raises ``ProfileError`` if the kind does not exist or
        if any file is not readable JSON (better to warn than to silently lose a profile).
        """
        if kind not in KINDS:
            raise ProfileError(f"tipo de perfil desconocido: {kind}")
        # OrderedDict: in Python 3.4 (Windows XP) a dict does not preserve insertion order
        out: dict[str, tuple[dict, str, Path]] = OrderedDict()
        for origin in ("sistema", "usuario"):
            for f in self._files(kind, origin):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    raise ProfileError(f"perfil ilegible {f.name}: {e}") from None
                name = str(data.get("nombre") or f.stem)
                if origin == "usuario" and name in out and out[name][1] == "sistema":
                    continue                                   # a user profile does not shadow a system one
                out[name] = (data, origin, f)
        return out

    def names(self, kind: str) -> list[str]:
        """Names of all profiles of a kind (system first, then user)."""
        return list(self._load_all(kind))

    def list(self, kind: str, machine_kind: str | None = None) -> list[dict]:
        """Profiles of a kind for the interface's selectors.

        ``machine_kind`` («plasma»…) filters out the material/consumable/process profiles that do not
        apply to that machine (it does not filter machines). «tipos» is read from the **resolved**
        profile (it may come from the parent); without «tipos», a machine counts for its «tipo» and the
        other profiles apply to all.

        Returns ``[{"nombre", "origen", "hereda", "tipos"}, …]``.
        """
        out = []
        for name, (data, origin, _) in self._load_all(kind).items():
            res = self.resolve(kind, name)
            tipos = res.get("tipos") or ([res.get("tipo")] if kind == "maquina" else list(MACHINE_KINDS))
            if machine_kind and kind != "maquina" and machine_kind not in tipos:
                continue
            out.append({"nombre": name, "origen": origin, "hereda": data.get("hereda"), "tipos": tipos})
        return out

    def raw(self, kind: str, name: str) -> tuple[dict, str]:
        """The profile exactly as in its file (inheritance not resolved) and its origin.

        Raises ``ProfileError`` if it does not exist.
        """
        allp = self._load_all(kind)
        if name not in allp:
            raise ProfileError(f"no existe el perfil de {KIND_NAMES[kind].lower()} «{name}»")
        data, origin, _ = allp[name]
        return data, origin

    def resolve(self, kind: str, name: str) -> dict:
        """The complete profile, with everything inherited: what actually applies.

        Follows «hereda» up to the root and merges from the root toward ``name`` with ``deep_merge``
        (the child wins). «hereda» and «predeterminado» are not carried into the result (they are not
        inherited); «nombre» is ``name``'s. Raises ``ProfileError`` if any profile in the chain is
        missing or if the inheritance is circular or deeper than ``MAX_DEPTH`` levels.
        """
        chain, seen = [], set()                            # chain: [child, parent, grandparent, …]
        cur = name
        while cur:
            if cur in seen or len(chain) > MAX_DEPTH:
                raise ProfileError(f"herencia circular o demasiado profunda en «{name}»")
            seen.add(cur)
            data, _ = self.raw(kind, cur)
            chain.append(data)
            cur = data.get("hereda")
        out: dict = {}
        for data in reversed(chain):
            out = deep_merge(out, {k: v for k, v in data.items() if k not in ("hereda", "predeterminado")})
        out["nombre"] = name
        return out

    def own_keys(self, kind: str, name: str) -> list[str]:
        """Keys the profile defines itself (the rest come from the parent): they are shown highlighted."""
        data, _ = self.raw(kind, name)
        if not data.get("hereda"):
            return flat_keys({k: v for k, v in data.items() if k not in ("nombre", "hereda")})
        own = {k: v for k, v in data.items() if k not in ("nombre", "hereda")}
        return flat_keys(own)

    # --- writing (user profiles only) ---------------------------------------------------
    def save(self, kind: str, name: str, data: dict, inherits: str | None = None) -> dict:
        """Save (create or replace) a user profile. It is the editor's «Guardar» / «Guardar como…»
        (Save / Save As).

        Parameters:
            kind:     profile kind. name: visible name (cannot be that of a system profile).
            data:     the **complete** profile as the editor shows it («nombre» and «hereda» are ignored).
            inherits: parent name, or None for a standalone profile.

        With a parent, only what differs from it is stored (``_diff``). Why: the user profile stays
        «hooked» to the system one, and when a new app version improves a value the user did not
        touch, the change reaches them; also, the file shows at a glance what was changed.

        Before writing, the result already merged with the parent is validated (``_validate``). If the
        profile already existed its same file is rewritten; otherwise a new one named ``slug``.

        Returns what was written to the JSON. Raises ``ProfileError`` (no user folder, no name, system
        name, inheriting from itself, nonexistent parent, invalid values) or ``machine_from_dict``'s
        ``ValueError`` for machine profiles.
        """
        if self.user_dir is None:
            raise ProfileError("no hay carpeta de perfiles de usuario")
        name = str(name).strip()
        if not name:
            raise ProfileError("el perfil necesita un nombre")
        allp = self._load_all(kind)
        if name in allp and allp[name][1] == "sistema":
            raise ProfileError(f"«{name}» es un perfil del sistema: guardalo con otro nombre")
        if inherits:
            if inherits == name:
                raise ProfileError("un perfil no puede heredar de sí mismo")
            self.raw(kind, inherits)                       # only so it fails if the parent does not exist
        body = {k: v for k, v in data.items() if k not in ("nombre", "hereda")}
        if inherits:                               # store only what changes relative to the parent
            parent = self.resolve(kind, inherits)
            body = _diff(parent, body)
        out = {"nombre": name, **({"hereda": inherits} if inherits else {}), **body}
        self._validate(kind, self._preview(kind, out, inherits))
        d = self.user_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        old = allp.get(name)
        path = old[2] if old else d / f"{slug(name)}.json"     # keep the file even if the slug changes
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def _preview(self, kind, out, inherits):
        """How profile ``out`` would look once resolved (parent + differences) before writing it."""
        base = self.resolve(kind, inherits) if inherits else {}
        return deep_merge(base, {k: v for k, v in out.items() if k != "hereda"})

    def delete(self, kind: str, name: str) -> None:
        """Delete a user profile (its file).

        Raises ``ProfileError`` if it does not exist, if it is a system profile or if other profiles
        inherit from it (they are named in the message: they must be deleted or re-parented first).
        """
        allp = self._load_all(kind)
        if name not in allp:
            raise ProfileError(f"no existe «{name}»")
        if allp[name][1] == "sistema":
            raise ProfileError("los perfiles del sistema no se borran")
        children = [n for n, (d, _, _) in allp.items() if d.get("hereda") == name]
        if children:
            raise ProfileError(f"«{name}» es el padre de: {', '.join(children)}")
        allp[name][2].unlink()

    def import_text(self, kind: str, text: str, name: str | None = None) -> dict:
        """Import a profile from JSON text (a file shared by another user) as a user profile.

        ``name`` replaces the name in the JSON (by default that one, or «importado»). If the parent it
        declares does not exist in this installation, it is imported as a standalone profile: the data
        it carries are only the differences, so the rest is filled in by the default values.
        Returns what was saved (see ``save``). Raises ``ProfileError`` if it is not a valid JSON object.
        """
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ProfileError(f"no es un perfil JSON válido: {e}") from None
        if not isinstance(data, dict):
            raise ProfileError("el perfil tiene que ser un objeto JSON")
        nm = name or str(data.get("nombre") or "importado")
        inherits = data.get("hereda")
        if inherits and inherits not in self.names(kind):
            inherits = None
        return self.save(kind, nm, data, inherits)

    # --- validation -----------------------------------------------------------------
    def _validate(self, kind: str, data: dict) -> None:
        """Check an already resolved profile before saving it; raises ``ProfileError`` with the reason.

        * Machine: ``machine_from_dict`` must accept it (axes, travel, controller…).
        * All: every numeric SCHEMA field present must be a number and lie between min and max.
        * Material: every cut table row with thickness, speed and heights > 0, and the cut height
          ≤ the pierce height (the other way round, the torch would pierce closer than it cuts and the
          dross would spatter onto it).
        """
        if kind == "maquina":
            machine_from_dict(machine_dict(data))     # ValueError with the reason
        for f in SCHEMA[kind]:
            v = get_path(data, f["clave"])
            if v is None or f["tipo"] != "num":
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ProfileError(f"{f['etiqueta']}: tiene que ser un número") from None
            if "min" in f and v < f["min"] or "max" in f and v > f["max"]:
                raise ProfileError(f"{f['etiqueta']}: fuera de rango ({f.get('min', '−∞')} a {f.get('max', '∞')})")
        if kind == "material":
            for row in get_path(data, "plasma.tabla", []) or []:
                for c in ("espesor", "velocidad", "altura_perforacion", "altura_corte"):
                    if not isinstance(row.get(c), (int, float)) or row[c] <= 0:
                        raise ProfileError(f"tabla de corte: «{c}» tiene que ser un número > 0")
                if row["altura_corte"] > row["altura_perforacion"]:
                    raise ProfileError("tabla de corte: la altura de corte no puede superar la de perforación")


def _diff(parent: dict, child: dict) -> dict:
    """What in ``child`` differs from ``parent`` (recursive in dicts; lists as a whole).

    It is the inverse of ``deep_merge``: ``deep_merge(parent, _diff(parent, child))`` gives ``child``
    for every key ``child`` has. What the child lacks cannot be «deleted» by difference: it keeps
    coming from the parent.
    """
    out = {}
    for k, v in child.items():
        pv = parent.get(k)
        if isinstance(v, dict) and isinstance(pv, dict):
            d = _diff(pv, v)
            if d:
                out[k] = d
        elif v != pv:
            out[k] = v
    return out


# ==================================================================================================
# =============================== FROM PROFILES TO EFFECTIVE MACHINE ===============================
# ==================================================================================================

# --- conversion between profile and configuration ------------------------------------------------

def machine_dict(m: dict) -> dict:
    """The machine profile in configuration format (the same as the YAML).

    Differences it resolves: the tool table goes from a list (handy for the editor) to
    ``{number: {largo, diametro, nombre}}``, and ``plasma_hw`` becomes ``plasma.sensado`` with the
    default values (play 1.5 mm, depth 20 mm, feed 300 mm/min). The result goes to
    ``machine_from_dict``.
    """
    d = {k: v for k, v in m.items() if k not in ("plasma_hw", "herramientas", "hereda")}
    tools = m.get("herramientas") or []
    if isinstance(tools, list):
        d["herramientas"] = {int(t["numero"]): {k: t[k] for k in ("largo", "diametro", "nombre") if k in t}
                             for t in tools if "numero" in t}
    elif isinstance(tools, dict):
        d["herramientas"] = tools
    if m.get("tipo") == "plasma":
        hw = m.get("plasma_hw") or {}
        d["plasma"] = {"sensado": {"activo": bool(hw.get("sensado", False)),
                                   "holgura_flotante": hw.get("holgura_flotante", 1.5),
                                   "profundidad_max": hw.get("profundidad_max", 20),
                                   "avance": hw.get("avance_sensado", 300)}}
    return d


def machine_profile_from_config(cfg: dict) -> dict:
    """Convert an old YAML configuration (configs/cnc_*.yaml) into a machine profile.

    It is the inverse of ``machine_dict``. The app uses it when starting with an old YAML
    configuration, to import the previous configuration as a user profile.
    """
    out = {k: copy.deepcopy(v) for k, v in cfg.items() if k not in ("plasma", "herramientas")}
    tools = cfg.get("herramientas") or {}
    out["herramientas"] = [{"numero": int(n), **(t or {})} for n, t in tools.items()]
    pl = cfg.get("plasma")
    if pl:
        sen = pl.get("sensado") or {}
        out["plasma_hw"] = {"sensado": bool(sen.get("activo", False)),
                            "holgura_flotante": sen.get("holgura_flotante", 1.5),
                            "profundidad_max": sen.get("profundidad_max", 20), "avance_sensado": sen.get("avance", 300)}
    return out


# --- cut tables ----------------------------------------------------------------------------------

def _nearest(rows: list[dict], key: str, value: float) -> dict | None:
    """The row whose ``key`` is closest to ``value`` (the first one on ties); None if there are no rows.

    For the plasma table: thicknesses are not interpolated because each row is a tested combination
    (speed, heights, delay); a mix of two rows is not guaranteed.
    """
    return min(rows, key=lambda r: abs(float(r.get(key, 0)) - value)) if rows else None


def _chipload(table: list[dict], dia: float) -> float | None:
    """Chip load (mm/tooth) for an end mill of ``dia`` mm according to the material's table.

    * Between two diameters in the table: linear interpolation.
    * Smaller than the smallest: proportional to the diameter (the line goes through 0), because a
      thinner end mill is more fragile and the recommended load drops roughly in proportion. That way
      a 1 mm one does not inherit a 3 mm one's load and does not break.
    * Larger than the largest: the largest one's load (no upward extrapolation, for safety).

    Returns None if the table is empty or its rows lack «diametro» and «carga».
    """
    rows = sorted((float(r["diametro"]), float(r["carga"])) for r in table or [] if "diametro" in r and "carga" in r)
    if not rows:
        return None
    if dia <= rows[0][0]:
        return rows[0][1] * dia / rows[0][0]
    for (d0, c0), (d1, c1) in zip(rows, rows[1:]):
        if d0 <= dia <= d1:
            return c0 + (c1 - c0) * (dia - d0) / (d1 - d0)
    return rows[-1][1]


def thickness_options(material: dict, consumable: dict | None) -> list[float]:
    """Thicknesses (mm) offered by the plasma selector: those in the material's table with the
    consumable's amperage, sorted and without duplicates.

    If the consumable has no amperage, all of the table's. If no row matches its amperage, also all
    of them (``effective`` warns and uses the nearest row). A row without amperage applies to any torch.
    """
    rows = get_path(material, "plasma.tabla", []) or []
    amp = (consumable or {}).get("amperaje")
    sel = [r for r in rows if amp is None or float(r.get("amperaje", amp)) == float(amp)] or rows
    return sorted({float(r["espesor"]) for r in sel})


def effective(store: ProfileStore, sel: dict) -> tuple[Machine, dict]:
    """Effective machine and on-screen summary, from the profile selection.

    It is this slicer's «slicing»: combines machine + material + consumable + process and computes
    what the interpreter will apply to the program.

    Parameters:
        store: the profile store.
        sel:   the selection ``{"maquina", "material", "consumible", "proceso", "espesor"}`` (names;
               «espesor» (thickness) in mm, plasma only). Material/consumable/process may be missing.

    Calculations:
        Plasma:  cut table row (consumable amperage, nearest thickness) →
                 feed = speed × avance_pct; pierce and cut heights; delay = the table's +
                 the process' retardo_extra; travel height = max(the process', the pierce height);
                 sensing only if the machine has it **and** the process asks for it.
        Milling: rpm = min(material rpm, machine rpm_max);
                 feed = chip load (mm/tooth) × flutes × rpm (mm/min), because each flute removes a
                 chip of that thickness per revolution; it is capped at the lower of X/Y maximum speeds
                 and multiplied by avance_pct. Plunge = feed × the material's bajada_pct.

    Returns ``(Machine, summary)``. The ``Machine`` gets ``job_feed``/``job_feed_mode``/
    ``job_plunge`` (only if avance_modo is «perfil» or «limitar») and ``job_rpm`` (if the process asks
    to add the missing S). The summary carries the values used, ``avisos`` (warnings: incompatibilities,
    thickness or amperage not in the table, feed the machine cannot reach) and ``texto`` (one line for
    the status bar).

    Raises: ``ProfileError`` if any profile does not exist; ``ValueError`` if the machine is invalid.
    """
    mprof = store.resolve("maquina", sel["maquina"])
    kind = mprof.get("tipo", "fresadora")
    mat = store.resolve("material", sel["material"]) if sel.get("material") else {}
    cons = store.resolve("consumible", sel["consumible"]) if sel.get("consumible") else {}
    proc = store.resolve("proceso", sel["proceso"]) if sel.get("proceso") else {}
    warnings = []
    for k, prof in (("material", mat), ("consumible", cons), ("proceso", proc)):
        if prof and kind not in (prof.get("tipos") or MACHINE_KINDS):
            warnings.append(f"el {KIND_NAMES[k].lower()} «{prof['nombre']}» no es para {kind}")
    cfg = machine_dict(mprof)
    mode = proc.get("avance_modo", "programa")          # no process: like Mach3, the program's F is respected
    pct = float(proc.get("avance_pct", 100)) / 100.0
    summary: dict = {"tipo": kind, "avance_modo": mode}
    job_feed = None                                     # computed feed (mm/min); None = no data

    # --- plasma: cut table row ---
    if kind == "plasma":
        espesores = thickness_options(mat, cons)
        summary["espesores"] = espesores
        rows = get_path(mat, "plasma.tabla", []) or []
        amp = cons.get("amperaje")
        # Rows for the torch's amperage (a row without amperage applies to any).
        cand = [r for r in rows if amp is None or float(r.get("amperaje", amp)) == float(amp)]
        if rows and not cand:
            warnings.append(f"la tabla de «{mat.get('nombre')}» no tiene filas para {amp:g} A: se usa la más cercana")
            cand = rows
        esp = float(sel.get("espesor") or (espesores[0] if espesores else 0))
        row = _nearest(cand, "espesor", esp)
        pl = cfg.setdefault("plasma", {})
        if row:
            if float(row["espesor"]) != esp and sel.get("espesor"):
                warnings.append(f"no hay {esp:g} mm en la tabla: se usa {float(row['espesor']):g} mm")
            pl.update(altura_perforacion=row["altura_perforacion"], altura_corte=row["altura_corte"],
                      retardo_perforacion=float(row.get("retardo", 0.5)) + float(proc.get("retardo_extra", 0)))
            job_feed = float(row["velocidad"]) * pct
            summary.update(espesor=float(row["espesor"]), amperaje=row.get("amperaje"), velocidad=row["velocidad"],
                           kerf=row.get("kerf"), tension=row.get("tension"),
                           altura_perforacion=row["altura_perforacion"], altura_corte=row["altura_corte"],
                           retardo=pl["retardo_perforacion"])
        trav = float(proc.get("altura_traslado", 10))
        # Never travel lower than the pierce height: the torch would go down on arrival and back up again.
        pl["altura_traslado"] = max(trav, float(pl.get("altura_perforacion", 3.8)))
        pl["perforacion_auto"] = bool(proc.get("perforacion_auto", False))
        sen = pl.setdefault("sensado", {})
        want_probe = bool(proc.get("sensado", False))
        if want_probe and not sen.get("activo"):
            warnings.append("el proceso pide sensado de altura pero la máquina no lo tiene")
        sen["activo"] = bool(sen.get("activo")) and want_probe     # needs both the hardware and the request
        summary.update(perforacion_auto=pl["perforacion_auto"], sensado=sen["activo"])

    # --- milling: feed from chip load ---
    elif kind in ("fresadora", "router"):
        rpm_max = float(get_path(cfg, "husillo.rpm_max", 24000) or 24000)
        rpm = min(float(get_path(mat, "fresado.rpm", 0) or 0), rpm_max) or None   # 0 → None: no data
        dia, flutes = cons.get("diametro"), cons.get("filos")
        cl = _chipload(get_path(mat, "fresado.carga_viruta", []), float(dia)) if dia else None
        if rpm and cl and flutes:
            # Cutting happens in the XY plane: the feed cannot exceed the slower of the two axes.
            vmax = min(float(get_path(cfg, f"ejes.{a}.vel_max", 1e9) or 1e9) for a in ("X", "Y"))
            # mm/tooth × teeth/rev × rev/min = mm/min
            raw_feed = cl * float(flutes) * rpm
            job_feed = min(raw_feed, vmax) * pct
            if raw_feed > vmax:
                warnings.append(f"la máquina no llega a {raw_feed:.0f} mm/min: se usa {vmax:.0f}")
            summary.update(rpm=rpm, carga_viruta=round(cl, 4), diametro=dia, filos=flutes)
            bp = get_path(mat, "fresado.bajada_pct")
            if bp:
                cfg["_job_plunge"] = job_feed * float(bp) / 100.0
        if rpm and proc.get("rpm_si_falta"):
            cfg["_job_rpm"] = rpm

    # --- build the Machine and apply the feed mode ---
    # The «_job_*» keys are temporary: they are not machine configuration and do not go to machine_from_dict.
    m = machine_from_dict({k: v for k, v in cfg.items() if not k.startswith("_")})
    if mode in ("perfil", "limitar") and job_feed:
        m.job_feed = round(job_feed, 1)
        m.job_feed_mode = "reemplazar" if mode == "perfil" else "limitar"
    if cfg.get("_job_rpm"):
        m.job_rpm = cfg["_job_rpm"]
    if mode in ("perfil", "limitar") and cfg.get("_job_plunge"):
        m.job_plunge = round(cfg["_job_plunge"], 1)
        summary["bajada"] = m.job_plunge
    summary.update(avance=m.job_feed, avance_calculado=round(job_feed, 1) if job_feed else None, avisos=warnings)
    summary["texto"] = _summary_text(kind, summary, m)
    return m, summary


def _summary_text(kind: str, s: dict, m: Machine) -> str:
    """One readable line with what will be used, e.g. «máx. 2600 mm/min · 3 mm · perf. 3.8 / corte
    1.5 mm · 0.40 s · kerf 1.5». Without computed data: «la F y la S del programa»."""
    parts = []
    if s.get("avance_calculado"):
        how = {"perfil": "", "limitar": "máx. ", "programa": "(programa) "}[s["avance_modo"]]
        parts.append(f"{how}{s['avance_calculado']:.0f} mm/min")
    if kind == "plasma" and "espesor" in s:
        parts.append(f"{s['espesor']:g} mm")
        parts.append(f"perf. {s['altura_perforacion']:g} / corte {s['altura_corte']:g} mm · {s['retardo']:.2f} s")
        if s.get("kerf"):
            parts.append(f"kerf {s['kerf']:g}")
    if s.get("rpm"):
        parts.append(f"{s['rpm']:.0f} rpm · {s['carga_viruta']:g} mm/diente")
    if s.get("bajada"):
        parts.append(f"bajada {s['bajada']:.0f} mm/min")
    return " · ".join(parts) or "la F y la S del programa"


# ==================================================================================================
# =========================================== SELECTION ============================================
# ==================================================================================================

def default_selection(store: ProfileStore, machine: str | None = None) -> dict:
    """Initial selection: the requested machine (if it exists), otherwise the one marked
    «predeterminado» (default), otherwise the first; and for it the preferred material, consumable
    and process (``_preferred``).

    On plasma, 3 mm thickness is chosen if the table has it (the most common sheet), otherwise the
    smallest. Raises ``ProfileError`` if there is no machine profile at all.
    """
    names = store.names("maquina")
    if not names:
        raise ProfileError("no hay perfiles de máquina")
    marked = [n for n in names if store.raw("maquina", n)[0].get("predeterminado")]
    mname = machine if machine in names else (marked[0] if marked else names[0])
    kind = store.resolve("maquina", mname).get("tipo", "fresadora")
    sel = {"maquina": mname}
    for k in ("material", "consumible", "proceso"):
        sel[k] = _preferred(store, k, kind)
    if kind == "plasma" and sel.get("material"):
        esp = thickness_options(store.resolve("material", sel["material"]),
                                store.resolve("consumible", sel["consumible"]) if sel.get("consumible") else None)
        sel["espesor"] = 3.0 if 3.0 in esp else (esp[0] if esp else None)
    return sel


def _preferred(store: ProfileStore, kind: str, machine_kind: str) -> str | None:
    """The profile marked «predeterminado» (default) for that machine type, or the first suitable one."""
    opts = store.list(kind, machine_kind)
    for o in opts:
        pre = store.raw(kind, o["nombre"])[0].get("predeterminado") or []     # not inherited
        if machine_kind in pre:
            return o["nombre"]
    return opts[0]["nombre"] if opts else None


def fix_selection(store: ProfileStore, sel: dict) -> dict:
    """Complete or fix a selection (deleted profiles, incompatible with the machine, thickness).

    Called with the saved selection at startup and every time the user changes something: if the
    machine no longer exists it falls back to ``default_selection``; every profile that does not exist
    or does not fit that machine is replaced by the preferred one; on plasma, a thickness not in the
    table becomes 3 mm (or the smallest). Returns a new selection (does not modify ``sel``).
    """
    names = store.names("maquina")
    if sel.get("maquina") not in names:
        return default_selection(store)
    kind = store.resolve("maquina", sel["maquina"]).get("tipo", "fresadora")
    out = {"maquina": sel["maquina"]}
    for k in ("material", "consumible", "proceso"):
        ok = [o["nombre"] for o in store.list(k, kind)]
        out[k] = sel.get(k) if sel.get(k) in ok else _preferred(store, k, kind)
    if kind == "plasma" and out.get("material"):
        esp = thickness_options(store.resolve("material", out["material"]),
                                store.resolve("consumible", out["consumible"]) if out.get("consumible") else None)
        want = sel.get("espesor")
        out["espesor"] = float(want) if want is not None and float(want) in esp else \
            (3.0 if 3.0 in esp else (esp[0] if esp else None))
    return out
