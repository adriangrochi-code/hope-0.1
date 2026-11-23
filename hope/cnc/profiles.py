"""Perfiles al estilo de un slicer 3D (PrusaSlicer, OrcaSlicer, TISMA Engine) para TISMA CNC.

Como en un slicer, se eligen cuatro perfiles y la app calcula el resto:

* **Máquina** (la «impresora»): recorridos, velocidades, controlador, hardware de plasma.
* **Material** (el «filamento»): tabla de corte de plasma por espesor y amperaje, o carga de viruta y
  RPM para fresar.
* **Consumible / herramienta** (la «boquilla»): amperaje de la antorcha, o diámetro y filos de la fresa.
* **Proceso** (el «perfil de impresión»): cómo se usan esos datos (avance del perfil o del programa,
  porcentaje, perforación automática, sensado de altura, altura de traslado…).

Formato: un archivo JSON por perfil, con ``"hereda": "<nombre del padre>"`` para guardar solo lo que
cambia (los diccionarios se combinan, las listas se reemplazan). Los perfiles del **sistema** vienen
con la app (``hope/cnc/perfiles/``) y no se modifican: se guardan como perfiles de **usuario**
(«Guardar como…»), que los heredan. Solo JSON y la biblioteca estándar: funciona en Python 3.8.

CÓMO ENCAJA
    * Lo usan ``hope.cnc.app.server`` y ``hope.cnc.__main__`` (crean el ``ProfileStore`` con la carpeta
      de usuario ``<datos>/perfiles``) y ``hope.cnc.session`` (lista, edita, guarda y borra perfiles
      para la interfaz, y recalcula la máquina con ``effective`` cada vez que cambia la selección).
    * Usa ``hope.cnc.machine`` (``machine_from_dict``) para validar perfiles de máquina y armar la
      ``Machine`` efectiva. Lo que este módulo calcula viaja en ``Machine.job_feed``,
      ``job_feed_mode``, ``job_plunge`` y ``job_rpm``, y ``hope.cnc.gcode`` lo aplica al interpretar.
    * El editor de la interfaz se dibuja solo con ``SCHEMA`` (grupos, etiquetas, unidades, límites):
      agregar un campo acá alcanza para que aparezca en pantalla y se valide al guardar.

HERENCIA («hereda»)
    Un perfil puede decir ``"hereda": "Mesa de plasma 1250×1250"``. Al resolverlo (``resolve``) se
    sigue la cadena hasta la raíz y se combinan de la raíz hacia el hijo: el hijo gana. Al guardar
    un perfil de usuario que hereda, se guarda **solo lo que difiere** del padre (``_diff``): así, si
    una actualización de la app corrige un valor del perfil del sistema, los perfiles de usuario que
    no tocaron ese valor reciben la corrección (igual que en PrusaSlicer).
    ``predeterminado`` (lista de tipos de máquina) marca el perfil que se elige por defecto y **no**
    se hereda: si no, todos los hijos de un perfil predeterminado también lo serían.

CÁLCULO DEL AVANCE (``effective``)
    * Plasma: se toma la fila de la tabla de corte del material con el **espesor más cercano** entre
      las del amperaje de la antorcha (no se interpola: las filas son puntos probados del fabricante
      y entre dos de ellas no se garantiza un corte limpio). De ahí salen velocidad, alturas de
      perforación y corte, retardo, kerf y tensión de arco.
    * Fresado: avance (mm/min) = carga de viruta (mm/diente) × filos × rpm. La carga de viruta sale de
      la tabla del material, interpolada linealmente por diámetro de fresa.
    En ambos casos el avance se multiplica por el porcentaje del proceso y se aplica según
    ``avance_modo``: «perfil» (reemplaza la F del programa), «limitar» (la F del programa, sin pasarse
    del calculado) o «programa» (la F del programa, como Mach3; el calculado solo se muestra).

UNIDADES
    Longitudes en mm, avances en mm/min, aceleraciones en mm/s², tiempos en s, corriente en A,
    tensión en V, husillo en rpm, carga de viruta en mm/diente. Los nombres de claves están en
    español sin tildes (son también los del JSON y del YAML viejo).

ARCHIVOS
    ``hope/cnc/perfiles/<tipo>/<nombre>.json`` (sistema) y ``<datos>/perfiles/<tipo>/<nombre>.json``
    (usuario), donde ``<tipo>`` es ``maquina``, ``material``, ``consumible`` o ``proceso``. El nombre
    del perfil está dentro del JSON (``"nombre"``); el del archivo es solo un ``slug`` ASCII.

EJEMPLO
    >>> store = ProfileStore(user_dir="datos/perfiles")
    >>> sel = default_selection(store)          # {"maquina": …, "material": …, "consumible": …, …}
    >>> m, resumen = effective(store, sel)
    >>> resumen["texto"]                        # p. ej. «3800 mm/min · 3 mm · perf. 3.8 / corte 1.5 mm · …»
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
# ========================================== CONFIGURACIÓN =========================================
# ==================================================================================================

# Los cuatro tipos de perfil. También son los nombres de las subcarpetas (sistema y usuario) y las
# claves de una selección. El orden es el de las pestañas de la interfaz.
KINDS = ("maquina", "material", "consumible", "proceso")
# Nombre legible de cada tipo de perfil, para la interfaz y los mensajes de error.
KIND_NAMES = {"maquina": "Máquina", "material": "Material", "consumible": "Consumible / herramienta",
              "proceso": "Proceso"}
# Tipos de máquina. Un material/consumible/proceso declara en «tipos» para cuáles sirve (p. ej. una
# tabla de corte de plasma no tiene sentido en una fresadora); el editor oculta los campos de SCHEMA
# cuyo «tipos» no incluye el de la máquina elegida.
MACHINE_KINDS = ("fresadora", "router", "laser", "plasma")
# Perfiles del sistema: vienen con la app y son de solo lectura.
SYSTEM_DIR = Path(__file__).parent / "perfiles"
# Máximo de saltos de herencia («hereda») al resolver un perfil. Protege contra cadenas absurdas o
# archivos editados a mano que se hereden en círculo. En la práctica se usan 1–3 niveles; 10 sobra.
MAX_DEPTH = 10


# Error de perfiles (ilegible, inexistente, inválido, del sistema, herencia circular…). El mensaje se
# muestra tal cual en la interfaz. Hereda de ValueError para que quien valide datos lo atrape junto
# con los ValueError de ``machine_from_dict``.
class ProfileError(ValueError):
    pass


# ==================================================================================================
# ===================================== UTILIDADES DE DICCIONARIOS =================================
# ==================================================================================================
# Los perfiles son diccionarios anidados (JSON). Estas funciones los combinan y acceden por «clave
# con puntos» («ejes.X.vel_max»), que es como SCHEMA nombra cada campo del editor.

def slug(name: str) -> str:
    """Nombre de archivo solo ASCII (Windows viejos, pendrives FAT, ZIP).

    Ejemplo: «Mesa de plasma 1250×1250» → ``mesa_de_plasma_12501250``. Las tildes se pierden
    («Máquina» → ``maquina``), todo lo que no sea letra, número o guion pasa a «_» y se corta en 80
    caracteres (lejos del límite de rutas de Windows XP). Si no queda nada, ``perfil``.
    Puede haber choques entre nombres parecidos; por eso el nombre real vive dentro del JSON.
    """
    import unicodedata
    # NFKD separa la letra de su tilde («á» → «a» + «´»); al pasar a ASCII la tilde se descarta.
    s = unicodedata.normalize("NFKD", name.strip().lower()).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\-]+", "_", s).strip("_")
    return s[:80] or "perfil"


def deep_merge(base: dict, over: dict) -> dict:
    """Combina ``over`` sobre ``base`` y devuelve un diccionario nuevo (no toca ninguno de los dos).

    Los diccionarios se combinan clave por clave, recursivamente; cualquier otro valor (números,
    textos **y listas**) de ``over`` reemplaza entero al de ``base``. Las listas se reemplazan a
    propósito: una tabla de corte o de herramientas del hijo es la tabla completa, no «filas extra».
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_path(d: dict, key: str, default=None):
    """Lee una clave con puntos: ``get_path(perfil, "ejes.X.vel_max")``. ``default`` si falta algún tramo."""
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(d: dict, key: str, value) -> None:
    """Escribe una clave con puntos en ``d`` (lo modifica), creando los diccionarios intermedios que falten."""
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def del_path(d: dict, key: str) -> None:
    """Borra una clave con puntos de ``d`` (lo modifica). Si no existe, no hace nada.

    Útil para «volver al valor del padre» en un perfil heredado: sin la clave, gana la del padre.
    """
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            return
        cur = cur[part]
    cur.pop(parts[-1], None)


def flat_keys(d: dict, prefix: str = "") -> list[str]:
    """Todas las claves «hoja» de ``d`` con puntos: ``{"ejes": {"X": {"acel": 500}}}`` → ``["ejes.X.acel"]``.

    Las listas cuentan como hoja (una tabla es un solo campo del editor). Un diccionario vacío también.
    """
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:                  # dict vacío = hoja (si no, desaparecería la clave)
            out += flat_keys(v, key + ".")
        else:
            out.append(key)
    return out


# ==================================================================================================
# ============================================= ESQUEMA ============================================
# ==================================================================================================
# Esquema: lo que muestra el editor (grupos, etiquetas, unidades y límites), como las pestañas de un slicer.
# La interfaz (JavaScript) recibe SCHEMA tal cual y arma el formulario; ``ProfileStore._validate``
# usa los mismos «min»/«max» al guardar. Un campo = un diccionario con las claves que arma ``_f``.

def _f(key, label, unit="", group="General", kind="num", mn=None, mx=None, help="", tipos=None, options=None,
       columns=None):
    """Arma la descripción de un campo del editor (abreviado para que SCHEMA se lea como una tabla).

    Parámetros:
        key:     clave con puntos dentro del perfil («ejes.X.acel»).
        label:   etiqueta visible. unit: unidad que se muestra al lado («mm/min»; vacío = sin unidad).
        group:   grupo/pestaña del editor donde aparece («Eje X», «Plasma»…).
        kind:    tipo de control: «num» (número, el único que se valida con min/max), «texto», «bool»
                 (casilla), «opcion» (lista desplegable), «multiple» (varias casillas) o «tabla».
        mn, mx:  límites aceptados al guardar (None = sin límite).
        help:    ayuda corta que aparece al lado o al pasar el mouse.
        tipos:   tipos de máquina para los que se muestra el campo (None = todos).
        options: valores posibles de «opcion»/«multiple».
        columns: columnas de una «tabla» (lista de {clave, etiqueta, unidad[, texto]}).

    Devuelve el diccionario con las claves en español que espera la interfaz. Las opcionales solo
    se incluyen si tienen valor, para que el JSON que viaja al navegador sea corto.
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
    """Los cinco campos de un eje (``a`` = «X», «Y» o «Z»), en un grupo «Eje X» propio.

    Las ayudas nombran el campo equivalente de Mach3 (Config → Motor Tuning) para que quien migra
    pueda copiar los valores tal cual:
        pasos_mm: pasos del motor por mm de recorrido (≥ 0.001). Mal puesto = piezas de otra medida.
        vel_max:  velocidad máxima del eje (mm/min, ≥ 1). Se usa en G0 y limita los avances calculados.
                  Exagerada = el motor pierde pasos.
        acel:     aceleración (mm/s², ≥ 0.1). Típico 100–1000 en routers caseros. Alta = pierde
                  pasos o sacude la máquina; baja = esquinas y tramos cortos lentos.
        min/max:  recorrido en coordenadas de máquina (mm). Vacío = sin límite (no se revisa).
    """
    g = f"Eje {a}"
    return [_f(f"ejes.{a}.pasos_mm", "Pasos por mm", "pasos/mm", g, mn=0.001,
               help="«Steps per» de Config → Motor Tuning en Mach3"),
            _f(f"ejes.{a}.vel_max", "Velocidad máxima", "mm/min", g, mn=1, help="«Velocity» en Mach3"),
            _f(f"ejes.{a}.acel", "Aceleración", "mm/s²", g, mn=0.1, help="«Acceleration» en Mach3"),
            _f(f"ejes.{a}.min", "Recorrido mínimo", "mm", g, help="coordenada de máquina; vacío = sin límite"),
            _f(f"ejes.{a}.max", "Recorrido máximo", "mm", g)]


# --- columnas de las tablas ----------------------------------------------------------------------

# Tabla de corte de plasma (una fila por espesor y amperaje), copiada de la tabla del fabricante de
# la fuente. Cada fila es un punto probado; ``effective`` elige la de espesor más cercano.
PLASMA_COLUMNS = [
    {"clave": "espesor", "etiqueta": "Espesor", "unidad": "mm"},              # chapa; > 0
    {"clave": "amperaje", "etiqueta": "Amperaje", "unidad": "A"},             # debe coincidir con la antorcha
    {"clave": "velocidad", "etiqueta": "Velocidad", "unidad": "mm/min"},      # avance de corte; > 0
    # Altura de perforación: más alta que la de corte para que la escoria que salta al perforar no
    # dañe la boquilla. Típico 3–4 mm. Tiene que ser ≥ altura de corte (lo exige ``_validate``).
    {"clave": "altura_perforacion", "etiqueta": "Alt. perforación", "unidad": "mm"},
    # Altura de corte (antorcha–chapa). Típico 1–2 mm. Muy baja = la boquilla toca y se gasta; muy alta
    # = corte biselado y con escoria.
    {"clave": "altura_corte", "etiqueta": "Alt. corte", "unidad": "mm"},
    # Retardo de perforación: tiempo quieto con el arco encendido hasta atravesar la chapa. Corto = no
    # perfora; largo = agujero grande y desgaste. Si falta, 0.5 s.
    {"clave": "retardo", "etiqueta": "Retardo", "unidad": "s"},
    {"clave": "kerf", "etiqueta": "Sangría (kerf)", "unidad": "mm"},          # ancho del corte (informativo)
    # Tensión de arco de referencia para el control de altura (THC). Informativa por ahora.
    {"clave": "tension", "etiqueta": "Tensión de arco", "unidad": "V"},
]
# Carga de viruta recomendada según el diámetro de la fresa (lo que publican los fabricantes de
# fresas). Entre dos diámetros se interpola linealmente (``_chipload``).
CHIP_COLUMNS = [{"clave": "diametro", "etiqueta": "Diámetro de fresa", "unidad": "mm"},
                {"clave": "carga", "etiqueta": "Carga de viruta", "unidad": "mm/diente"}]
# Tabla de herramientas del perfil de máquina (T, largo H para G43, diámetro, nombre), como la Tool
# Table de Mach3. En el perfil es una lista; ``machine_dict`` la pasa al formato {número: datos}.
TOOL_COLUMNS = [{"clave": "numero", "etiqueta": "T", "unidad": ""},
                {"clave": "largo", "etiqueta": "Largo (H)", "unidad": "mm"},
                {"clave": "diametro", "etiqueta": "Diámetro", "unidad": "mm"},
                {"clave": "nombre", "etiqueta": "Nombre", "unidad": "", "texto": True}]   # texto libre

# --- campos del editor, por tipo de perfil -------------------------------------------------------
# El orden de cada lista es el orden en pantalla. Solo se listan los campos que tiene sentido tocar
# desde el editor; un perfil puede tener otras claves (p. ej. «nota») que se conservan igual.
SCHEMA = {
    # --- Máquina: lo que es fijo del equipo (mecánica, husillo, hardware de plasma, controlador) ---
    "maquina": [
        # Tipo: decide qué campos se muestran en los cuatro perfiles y qué materiales/procesos sirven.
        _f("tipo", "Tipo de máquina", "", "General", "opcion", options=MACHINE_KINDS),
        *_axis_fields("X"), *_axis_fields("Y"), *_axis_fields("Z"),
        # Movimiento. Desviación de unión (mm, 0–1): cuánto puede «redondear» el planificador una
        # esquina sin frenar del todo. Típico 0.01–0.05 (grbl usa 0.01). Tolerancia de arcos (mm,
        # 0.0001–1): error máximo al partir G2/G3 en tramos rectos; más chica = más tramos.
        _f("desviacion_union", "Desviación en esquinas", "mm", "Movimiento", mn=0, mx=1,
           help="más alto = esquinas más rápidas y menos exactas (junction deviation de grbl)"),
        _f("tolerancia_arco", "Tolerancia de arcos", "mm", "Movimiento", mn=0.0001, mx=1),
        # Husillo (solo fresadora/router). rpm_max también recorta las RPM del material en
        # ``effective`` (24000 si no se indica): el avance se calcula con las rpm que la máquina da.
        _f("husillo.rpm_min", "RPM mínimas", "rpm", "Husillo", mn=0, tipos=("fresadora", "router")),
        _f("husillo.rpm_max", "RPM máximas", "rpm", "Husillo", mn=0, tipos=("fresadora", "router")),
        # Origen G54 de referencia (mm, coordenadas de máquina). Sirve para revisar el programa sin
        # controlador conectado; con uno real se usan sus orígenes (``Backend.work_offsets``).
        _f("origenes.G54.X", "G54 X", "mm", "Origen de trabajo de referencia"),
        _f("origenes.G54.Y", "G54 Y", "mm", "Origen de trabajo de referencia"),
        _f("origenes.G54.Z", "G54 Z", "mm", "Origen de trabajo de referencia",
           help="antes de enviar se leen los orígenes reales del controlador"),
        # Hardware de plasma: sensado inicial de altura (IHS). La antorcha baja (G38.2) hasta tocar la
        # chapa a «avance_sensado» (mm/min; lento = más exacto, típico 200–500) sin pasar de
        # «profundidad_max» (mm), y se corrige por la «holgura_flotante» (mm que se mueve el cabezal
        # flotante antes de accionar el interruptor; típico 1–3). Que la máquina lo *tenga* va acá;
        # que se *use* lo decide el proceso.
        _f("plasma_hw.sensado", "Tiene sensado de altura (IHS)", "", "Plasma", "bool", tipos=("plasma",),
           help="cabezal flotante o sensor óhmico conectado como sonda"),
        _f("plasma_hw.holgura_flotante", "Holgura del cabezal flotante", "mm", "Plasma", mn=0, tipos=("plasma",)),
        _f("plasma_hw.profundidad_max", "Profundidad máxima de sensado", "mm", "Plasma", mn=1, tipos=("plasma",)),
        _f("plasma_hw.avance_sensado", "Avance de sensado", "mm/min", "Plasma", mn=1, tipos=("plasma",)),
        _f("herramientas", "Tabla de herramientas", "", "Herramientas", "tabla", columns=TOOL_COLUMNS,
           tipos=("fresadora", "router")),
        # Controlador: a quién se le envía el programa (ver ``hope.cnc.backends``). «bufer» (bytes,
        # ≥ 64) tiene que ser el tamaño real del búfer de recepción: más grande lo desborda.
        _f("controlador.tipo", "Controlador", "", "Controlador", "opcion", options=("sim", "grbl", "ok", "linuxcnc"),
           help="sim = simulador; grbl = grbl/grblHAL/FluidNC; ok = Marlin/Smoothieware/RepRapFirmware"),
        _f("controlador.puerto", "Puerto", "", "Controlador", "texto",
           help="COM3, /dev/ttyUSB0 o tcp://192.168.1.50:23"),
        _f("controlador.baudios", "Baudios", "", "Controlador", mn=1200),
        _f("controlador.bufer", "Búfer del controlador", "bytes", "Controlador", mn=64,
           help="128 en grbl 1.1; 1024 en grblHAL"),
    ],
    # --- Material: cómo se corta ese material (tabla de plasma o datos de fresado) ------------------
    "material": [
        # «Sirve para»: con qué tipos de máquina se ofrece este perfil (vacío = con todos). Igual en
        # consumible y proceso.
        _f("tipos", "Sirve para", "", "General", "multiple", options=MACHINE_KINDS),
        _f("plasma.tabla", "Tabla de corte", "", "Plasma", "tabla", columns=PLASMA_COLUMNS, tipos=("plasma",),
           help="valores de partida; ajustalos con un corte de prueba"),
        # Fresado. RPM recomendadas para el material (se recortan a las rpm máximas de la máquina).
        # La carga de viruta por diámetro da el avance; «bajada_pct» (1–100 %) es el avance de las
        # bajadas en Z como fracción del de corte: entrar de punta carga mucho la fresa, típico 30–50 %.
        _f("fresado.rpm", "RPM", "rpm", "Fresado", mn=0, tipos=("fresadora", "router")),
        _f("fresado.carga_viruta", "Carga de viruta por diámetro", "", "Fresado", "tabla", columns=CHIP_COLUMNS,
           tipos=("fresadora", "router")),
        _f("fresado.bajada_pct", "Avance de bajada", "% del de corte", "Fresado", mn=1, mx=100,
           tipos=("fresadora", "router")),
    ],
    # --- Consumible / herramienta: lo que se cambia seguido (boquilla de plasma, fresa) -------------
    "consumible": [
        _f("tipos", "Sirve para", "", "General", "multiple", options=MACHINE_KINDS),
        # Amperaje de la boquilla/antorcha (A): filtra las filas de la tabla de corte del material.
        _f("amperaje", "Amperaje", "A", "Plasma", mn=1, tipos=("plasma",)),
        # Fresa: diámetro (mm) para buscar la carga de viruta, filos (1–12) para multiplicarla.
        # La forma es informativa (todavía no cambia el cálculo).
        _f("diametro", "Diámetro", "mm", "Fresa", mn=0.01, tipos=("fresadora", "router")),
        _f("filos", "Filos", "", "Fresa", mn=1, mx=12, tipos=("fresadora", "router")),
        _f("forma", "Forma", "", "Fresa", "opcion", options=("plana", "bola", "V", "compresión"),
           tipos=("fresadora", "router")),
    ],
    # --- Proceso: cómo se usan los datos anteriores en este trabajo ------------------------------
    "proceso": [
        _f("tipos", "Sirve para", "", "General", "multiple", options=MACHINE_KINDS),
        # Modo de avance (ver el encabezado). «programa» es lo que haría Mach3 y es el valor por
        # defecto si el proceso no lo dice: nunca cambia la F sin que el usuario lo pida.
        _f("avance_modo", "Avance de corte", "", "Avance", "opcion", options=("perfil", "limitar", "programa"),
           help="perfil = usar el del material; limitar = la F del programa sin pasarse del perfil; "
                "programa = la F del programa (como Mach3)"),
        # Porcentaje sobre el avance calculado (10–200 %, por defecto 100). Menos = mejor terminación
        # (p. ej. plasma «calidad» al 85 %); más de 100 solo si las pruebas lo permiten.
        _f("avance_pct", "Porcentaje del avance del perfil", "%", "Avance", mn=10, mx=200),
        _f("perforacion_auto", "Perforación automática en cada M3", "", "Plasma", "bool", tipos=("plasma",),
           help="apagar si el CAM ya genera la secuencia (SheetCam y otros)"),
        # Sensar: solo tiene efecto si además la máquina tiene sensado (plasma_hw.sensado).
        _f("sensado", "Sensar la altura antes de perforar", "", "Plasma", "bool", tipos=("plasma",)),
        # Altura de traslado (mm, ≥ 0.5; 10 si falta): altura de la antorcha entre cortes. Nunca queda
        # por debajo de la de perforación. Baja = traslados rápidos pero riesgo de chocar chapas levantadas.
        _f("altura_traslado", "Altura de traslado", "mm", "Plasma", mn=0.5, tipos=("plasma",)),
        # Se suma al retardo de la tabla (s, ≥ 0): para chapa sucia, oxidada o una fuente floja.
        _f("retardo_extra", "Retardo extra de perforación", "s", "Plasma", mn=0, tipos=("plasma",)),
        _f("rpm_si_falta", "Poner la S del material si el programa no trae", "", "Fresado", "bool",
           tipos=("fresadora", "router")),
    ],
}


# ==================================================================================================
# ======================================= ALMACÉN DE PERFILES ======================================
# ==================================================================================================

class ProfileStore:
    """Los perfiles del sistema y del usuario, leídos de disco en cada consulta.

    No guarda nada en memoria a propósito: si el usuario copia un perfil a la carpeta a mano (o lo
    importa otra ventana), aparece sin reiniciar. Son pocos archivos chicos, así que es barato.

    Reglas:
        * Los del sistema no se modifican ni se borran (``save``/``delete`` lo rechazan).
        * Un perfil de usuario con el mismo nombre que uno del sistema se ignora (no lo «tapa»).
        * Un perfil con hijos no se puede borrar (los dejaría huérfanos).
    """

    def __init__(self, user_dir: str | Path | None = None, system_dir: str | Path = SYSTEM_DIR):
        """``user_dir``: carpeta de perfiles del usuario (None = solo lectura, sin «Guardar»).
        ``system_dir``: carpeta de los del sistema (las pruebas pasan otra)."""
        self.system_dir = Path(system_dir)
        self.user_dir = Path(user_dir) if user_dir else None

    # --- lectura -----------------------------------------------------------------
    def _files(self, kind: str, origin: str) -> list[Path]:
        """Archivos .json de un tipo («maquina»…) y origen («sistema»/«usuario»), ordenados por nombre
        (orden estable en cualquier sistema de archivos). Lista vacía si la carpeta no existe."""
        root = self.system_dir if origin == "sistema" else self.user_dir
        if root is None or not (root / kind).is_dir():
            return []
        return sorted((root / kind).glob("*.json"))

    def _load_all(self, kind: str) -> dict[str, tuple[dict, str, Path]]:
        """Todos los perfiles de un tipo: ``{nombre: (datos tal cual del JSON, origen, archivo)}``.

        Primero los del sistema y después los del usuario, cada grupo ordenado por archivo. El nombre
        es el ``"nombre"`` del JSON (o el del archivo si falta). Lanza ``ProfileError`` si el tipo no
        existe o si algún archivo no es JSON legible (mejor avisar que perder un perfil en silencio).
        """
        if kind not in KINDS:
            raise ProfileError(f"tipo de perfil desconocido: {kind}")
        # OrderedDict: en Python 3.4 (Windows XP) un dict no conserva el orden de la lista
        out: dict[str, tuple[dict, str, Path]] = OrderedDict()
        for origin in ("sistema", "usuario"):
            for f in self._files(kind, origin):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    raise ProfileError(f"perfil ilegible {f.name}: {e}") from None
                name = str(data.get("nombre") or f.stem)
                if origin == "usuario" and name in out and out[name][1] == "sistema":
                    continue                                   # un perfil de usuario no tapa uno del sistema
                out[name] = (data, origin, f)
        return out

    def names(self, kind: str) -> list[str]:
        """Nombres de todos los perfiles de un tipo (sistema primero, después usuario)."""
        return list(self._load_all(kind))

    def list(self, kind: str, machine_kind: str | None = None) -> list[dict]:
        """Perfiles de un tipo para los selectores de la interfaz.

        ``machine_kind`` («plasma»…) filtra los material/consumible/proceso que no sirven para esa
        máquina (no filtra máquinas). «tipos» se lee del perfil **resuelto** (puede venir del padre);
        sin «tipos», una máquina vale por su «tipo» y los demás perfiles sirven para todas.

        Devuelve ``[{"nombre", "origen", "hereda", "tipos"}, …]``.
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
        """El perfil tal cual está en su archivo (sin resolver la herencia) y su origen.

        Lanza ``ProfileError`` si no existe.
        """
        allp = self._load_all(kind)
        if name not in allp:
            raise ProfileError(f"no existe el perfil de {KIND_NAMES[kind].lower()} «{name}»")
        data, origin, _ = allp[name]
        return data, origin

    def resolve(self, kind: str, name: str) -> dict:
        """El perfil completo, con todo lo heredado: lo que realmente vale.

        Sigue «hereda» hasta la raíz y combina de la raíz hacia ``name`` con ``deep_merge`` (el hijo
        gana). «hereda» y «predeterminado» no pasan al resultado (no se heredan); «nombre» queda el
        de ``name``. Lanza ``ProfileError`` si falta algún perfil de la cadena o si la herencia es
        circular o tiene más de ``MAX_DEPTH`` niveles.
        """
        chain, seen = [], set()                            # chain: [hijo, padre, abuelo, …]
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
        """Claves que el perfil define él mismo (las demás vienen del padre): se muestran resaltadas."""
        data, _ = self.raw(kind, name)
        if not data.get("hereda"):
            return flat_keys({k: v for k, v in data.items() if k not in ("nombre", "hereda")})
        own = {k: v for k, v in data.items() if k not in ("nombre", "hereda")}
        return flat_keys(own)

    # --- escritura (solo perfiles de usuario) -------------------------------------------
    def save(self, kind: str, name: str, data: dict, inherits: str | None = None) -> dict:
        """Guarda (crea o reemplaza) un perfil de usuario. Es el «Guardar» / «Guardar como…» del editor.

        Parámetros:
            kind:     tipo de perfil. name: nombre visible (no puede ser el de uno del sistema).
            data:     el perfil **completo** como lo muestra el editor («nombre» y «hereda» se ignoran).
            inherits: nombre del padre, o None para un perfil independiente.

        Con padre se guarda solo lo que difiere de él (``_diff``). Por qué: el perfil de usuario sigue
        «enganchado» al del sistema, y cuando una versión nueva de la app mejora un valor que el
        usuario no tocó, el cambio le llega; además, el archivo muestra de un vistazo qué se cambió.

        Antes de escribir se valida el resultado ya combinado con el padre (``_validate``). Si el
        perfil ya existía se reescribe su mismo archivo; si no, uno nuevo con nombre ``slug``.

        Devuelve lo que quedó escrito en el JSON. Lanza ``ProfileError`` (sin carpeta de usuario, sin
        nombre, nombre del sistema, herencia de sí mismo, padre inexistente, valores inválidos) o
        ``ValueError`` de ``machine_from_dict`` en perfiles de máquina.
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
            self.raw(kind, inherits)                       # solo para que falle si el padre no existe
        body = {k: v for k, v in data.items() if k not in ("nombre", "hereda")}
        if inherits:                               # guardar solo lo que cambia respecto del padre
            parent = self.resolve(kind, inherits)
            body = _diff(parent, body)
        out = {"nombre": name, **({"hereda": inherits} if inherits else {}), **body}
        self._validate(kind, self._preview(kind, out, inherits))
        d = self.user_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        old = allp.get(name)
        path = old[2] if old else d / f"{slug(name)}.json"     # mantener el archivo aunque cambie el slug
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def _preview(self, kind, out, inherits):
        """Cómo quedaría resuelto el perfil ``out`` (padre + diferencias) antes de escribirlo."""
        base = self.resolve(kind, inherits) if inherits else {}
        return deep_merge(base, {k: v for k, v in out.items() if k != "hereda"})

    def delete(self, kind: str, name: str) -> None:
        """Borra un perfil de usuario (su archivo).

        Lanza ``ProfileError`` si no existe, si es del sistema o si otros perfiles lo heredan (se
        nombran en el mensaje: hay que borrarlos o cambiarles el padre primero).
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
        """Importa un perfil desde texto JSON (archivo que pasó otro usuario) como perfil de usuario.

        ``name`` reemplaza al nombre que trae el JSON (por defecto ese, o «importado»). Si el padre
        que declara no existe en esta instalación, se importa como perfil independiente: los datos
        que trae son solo las diferencias, así que al resto lo completan los valores por defecto.
        Devuelve lo guardado (ver ``save``). Lanza ``ProfileError`` si no es un objeto JSON válido.
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

    # --- validación -----------------------------------------------------------------
    def _validate(self, kind: str, data: dict) -> None:
        """Revisa un perfil ya resuelto antes de guardarlo; lanza ``ProfileError`` con el motivo.

        * Máquina: que ``machine_from_dict`` la acepte (ejes, recorridos, controlador…).
        * Todos: cada campo numérico de SCHEMA presente tiene que ser número y estar entre min y max.
        * Material: cada fila de la tabla de corte con espesor, velocidad y alturas > 0, y la altura
          de corte ≤ la de perforación (al revés, la antorcha perforaría más cerca que al cortar y la
          escoria le saltaría encima).
        """
        if kind == "maquina":
            machine_from_dict(machine_dict(data))     # ValueError con el motivo
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
    """Lo de ``child`` que difiere de ``parent`` (recursivo en diccionarios; las listas, enteras).

    Es la inversa de ``deep_merge``: ``deep_merge(parent, _diff(parent, child))`` da ``child`` para
    toda clave que ``child`` tenga. Lo que el hijo no tiene no se puede «borrar» por diferencia: sigue
    viniendo del padre.
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
# ================================ DE LOS PERFILES A LA MÁQUINA EFECTIVA ===========================
# ==================================================================================================

# --- conversión entre perfil y configuración -----------------------------------------------------

def machine_dict(m: dict) -> dict:
    """El perfil de máquina en el formato de configuración (el mismo del YAML).

    Diferencias que resuelve: la tabla de herramientas pasa de lista (cómoda para el editor) a
    ``{número: {largo, diametro, nombre}}``, y ``plasma_hw`` pasa a ``plasma.sensado`` con los
    valores por defecto (holgura 1.5 mm, profundidad 20 mm, avance 300 mm/min). El resultado va a
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
    """Convierte una configuración YAML vieja (configs/cnc_*.yaml) en perfil de máquina.

    Es la inversa de ``machine_dict``. La usa la app al arrancar con una configuración YAML vieja,
    para importar la configuración de antes como perfil de usuario.
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


# --- tablas de corte -------------------------------------------------------------------------------

def _nearest(rows: list[dict], key: str, value: float) -> dict | None:
    """La fila cuyo ``key`` está más cerca de ``value`` (la primera si empatan); None si no hay filas.

    Para la tabla de plasma: no se interpola entre espesores porque cada fila es una combinación
    probada (velocidad, alturas, retardo); una mezcla de dos filas no está garantizada.
    """
    return min(rows, key=lambda r: abs(float(r.get(key, 0)) - value)) if rows else None


def _chipload(table: list[dict], dia: float) -> float | None:
    """Carga de viruta (mm/diente) para una fresa de ``dia`` mm según la tabla del material.

    * Entre dos diámetros de la tabla: interpolación lineal.
    * Más chica que la menor: proporcional al diámetro (la recta pasa por 0), porque una fresa más
      fina es más frágil y la carga recomendada baja más o menos en proporción. Así una de 1 mm no
      hereda la carga de una de 3 mm y no se rompe.
    * Más grande que la mayor: la carga de la mayor (no se extrapola hacia arriba, por seguridad).

    Devuelve None si la tabla está vacía o sus filas no tienen «diametro» y «carga».
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
    """Espesores (mm) que ofrece el selector de plasma: los de la tabla del material con el amperaje
    del consumible, ordenados y sin repetir.

    Si el consumible no tiene amperaje, todos los de la tabla. Si ninguna fila coincide con su
    amperaje, también todos (``effective`` avisa y usa la fila más cercana). Una fila sin amperaje
    sirve para cualquier antorcha.
    """
    rows = get_path(material, "plasma.tabla", []) or []
    amp = (consumable or {}).get("amperaje")
    sel = [r for r in rows if amp is None or float(r.get("amperaje", amp)) == float(amp)] or rows
    return sorted({float(r["espesor"]) for r in sel})


def effective(store: ProfileStore, sel: dict) -> tuple[Machine, dict]:
    """Máquina efectiva y resumen para la pantalla, a partir de la selección de perfiles.

    Es el «slicear» de este slicer: junta máquina + material + consumible + proceso y calcula lo que
    el intérprete aplicará al programa.

    Parámetros:
        store: el almacén de perfiles.
        sel:   la selección ``{"maquina", "material", "consumible", "proceso", "espesor"}`` (nombres;
               «espesor» en mm, solo plasma). Material/consumible/proceso pueden faltar.

    Cálculos:
        Plasma:  fila de la tabla de corte (amperaje del consumible, espesor más cercano) →
                 avance = velocidad × avance_pct; alturas de perforación y corte; retardo = el de la
                 tabla + retardo_extra del proceso; altura de traslado = máx(la del proceso, la de
                 perforación); sensado solo si la máquina lo tiene **y** el proceso lo pide.
        Fresado: rpm = mín(rpm del material, rpm_max de la máquina);
                 avance = carga de viruta (mm/diente) × filos × rpm (mm/min), porque cada filo saca
                 una viruta de ese espesor por vuelta; se recorta a la menor velocidad máxima de X/Y
                 y se multiplica por avance_pct. Bajada = avance × bajada_pct del material.

    Devuelve ``(Machine, resumen)``. En la ``Machine`` quedan ``job_feed``/``job_feed_mode``/
    ``job_plunge`` (solo si avance_modo es «perfil» o «limitar») y ``job_rpm`` (si el proceso pide
    poner la S que falte). El resumen trae los valores usados, ``avisos`` (incompatibilidades,
    espesor o amperaje que no estaban en la tabla, avance que la máquina no alcanza) y ``texto`` (una
    línea para la barra de estado).

    Errores: ``ProfileError`` si algún perfil no existe; ``ValueError`` si la máquina es inválida.
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
    mode = proc.get("avance_modo", "programa")          # sin proceso: como Mach3, se respeta la F del programa
    pct = float(proc.get("avance_pct", 100)) / 100.0
    summary: dict = {"tipo": kind, "avance_modo": mode}
    job_feed = None                                     # avance calculado (mm/min); None = no hay datos

    # --- plasma: fila de la tabla de corte ---
    if kind == "plasma":
        espesores = thickness_options(mat, cons)
        summary["espesores"] = espesores
        rows = get_path(mat, "plasma.tabla", []) or []
        amp = cons.get("amperaje")
        # Filas del amperaje de la antorcha (una fila sin amperaje sirve para cualquiera).
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
        # Nunca trasladar más bajo de lo que se perfora: la antorcha bajaría al llegar y volvería a subir.
        pl["altura_traslado"] = max(trav, float(pl.get("altura_perforacion", 3.8)))
        pl["perforacion_auto"] = bool(proc.get("perforacion_auto", False))
        sen = pl.setdefault("sensado", {})
        want_probe = bool(proc.get("sensado", False))
        if want_probe and not sen.get("activo"):
            warnings.append("el proceso pide sensado de altura pero la máquina no lo tiene")
        sen["activo"] = bool(sen.get("activo")) and want_probe     # hace falta el hardware y el pedido
        summary.update(perforacion_auto=pl["perforacion_auto"], sensado=sen["activo"])

    # --- fresado: avance por carga de viruta ---
    elif kind in ("fresadora", "router"):
        rpm_max = float(get_path(cfg, "husillo.rpm_max", 24000) or 24000)
        rpm = min(float(get_path(mat, "fresado.rpm", 0) or 0), rpm_max) or None   # 0 → None: sin dato
        dia, flutes = cons.get("diametro"), cons.get("filos")
        cl = _chipload(get_path(mat, "fresado.carga_viruta", []), float(dia)) if dia else None
        if rpm and cl and flutes:
            # El corte es en el plano XY: el avance no puede superar al eje más lento de los dos.
            vmax = min(float(get_path(cfg, f"ejes.{a}.vel_max", 1e9) or 1e9) for a in ("X", "Y"))
            # mm/diente × dientes/vuelta × vueltas/min = mm/min
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

    # --- armar la Machine y aplicar el modo de avance ---
    # Las claves «_job_*» son temporales: no son configuración de máquina y no pasan a machine_from_dict.
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
    """Una línea legible con lo que se va a usar, p. ej. «máx. 2600 mm/min · 3 mm · perf. 3.8 / corte
    1.5 mm · 0.40 s · kerf 1.5». Sin datos calculados: «la F y la S del programa»."""
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
# ============================================ SELECCIÓN ===========================================
# ==================================================================================================

def default_selection(store: ProfileStore, machine: str | None = None) -> dict:
    """Selección inicial: la máquina pedida (si existe), si no la marcada «predeterminado», si no la
    primera; y para ella el material, consumible y proceso preferidos (``_preferred``).

    En plasma se elige 3 mm de espesor si la tabla lo tiene (la chapa más común), si no el menor.
    Lanza ``ProfileError`` si no hay ningún perfil de máquina.
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
    """El perfil marcado «predeterminado» para ese tipo de máquina, o el primero que sirva."""
    opts = store.list(kind, machine_kind)
    for o in opts:
        pre = store.raw(kind, o["nombre"])[0].get("predeterminado") or []     # no se hereda
        if machine_kind in pre:
            return o["nombre"]
    return opts[0]["nombre"] if opts else None


def fix_selection(store: ProfileStore, sel: dict) -> dict:
    """Completa o corrige una selección (perfiles borrados, incompatibles con la máquina, espesor).

    Se llama con la selección guardada al arrancar y cada vez que el usuario cambia algo: si la
    máquina ya no existe se vuelve a ``default_selection``; cada perfil que no existe o no sirve para
    esa máquina se cambia por el preferido; en plasma, un espesor que no está en la tabla pasa a 3 mm
    (o al menor). Devuelve una selección nueva (no modifica ``sel``).
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
