"""Hope CNC como herramienta interna del catálogo: ⟦cnc: pieza.nc | maquina=configs/cnc_maquina.yaml⟧.

Solo **revisa y simula** (en el sandbox, porque el G-code no es confiable): errores, límites de
recorrido, tiempo con aceleración, envolvente y herramientas; con ``dialecto=grbl|linuxcnc``
también devuelve las primeras líneas traducidas. **No mueve la máquina**: eso es
``python -m hope.cnc enviar`` con aprobación humana del guardián.

Qué es
------
El «enchufe» que conecta el intérprete de G-code de TISMA CNC (``hope/cnc``) con el catálogo de
herramientas de Hope (``hope/tools/catalog.py``). Así un asistente o un usuario puede pedir «revisá
este programa» y recibir un informe sin tocar el hardware.

Cómo encaja
-----------
* Lo usa ``hope/tools/extended.py``: registra :data:`SPEC` en el catálogo, ejecuta :func:`review`
  dentro del sandbox con ``call_function("hope.cnc.tool", "review", ...)`` (límite :data:`TIMEOUT`)
  y formatea el resultado con :func:`review_text`.
* Usa ``hope.cnc.machine`` (configuración), ``hope.cnc.gcode`` (intérprete), ``hope.cnc.check``
  (revisión: límites, husillo, tiempos) y ``hope.cnc.dialect`` (traducción a grbl/LinuxCNC).

Unidades y convenciones
-----------------------
Todo en mm, mm/min (F), mm/s² (aceleración) y rpm (S), como el resto de ``hope/cnc``. El
resultado es un ``dict`` serializable a JSON (requisito del sandbox), con claves en español.

Ejemplo de uso::

    from hope.cnc.tool import review, review_text
    r = review(open("pieza.nc").read(), open("configs/cnc_maquina.yaml").read(), dialect="grbl")
    print(review_text(r))
"""

from __future__ import annotations

from hope.tools.catalog import ToolSpec

# ==================================================================================================
# ======================================== CONFIGURACIÓN ===========================================
# ==================================================================================================

# Tiempo máximo (s) que el sandbox deja correr una revisión antes de cortarla. Un programa de
# relieve 3D de cientos de miles de líneas tarda unos segundos en una PC moderna y bastante más en
# una vieja; 60 s deja margen. Típico 30–120. Muy bajo = programas grandes fallan por tiempo; muy
# alto = un G-code malicioso o un bucle de subprogramas tiene más tiempo para consumir CPU.
TIMEOUT = 60.0
# Cantidad máxima de líneas traducidas que se devuelven como muestra en ``traduccion.inicio``.
# Solo es una vista previa (el programa completo lo traduce «enviar»). Típico 20–100. Más alto =
# respuestas JSON más grandes para el asistente, sin ganar mucho.
MAX_PREVIEW = 40


# ==================================================================================================
# ========================================== REVISIÓN ==============================================
# ==================================================================================================

def review(text: str, machine_yaml: str | None = None, dialect: str | None = None) -> dict:
    """Función pura que corre dentro del sandbox (argumentos y resultado en JSON).

    Interpreta el programa, lo revisa contra la máquina y, si se pide, lo traduce a un dialecto.

    Args:
        text: contenido del programa G-code (texto, no una ruta).
        machine_yaml: texto YAML de la máquina (formato de ``configs/cnc_maquina.yaml``). ``None``:
            máquina genérica por defecto (:class:`hope.cnc.machine.Machine`).
        dialect: ``"grbl"`` o ``"linuxcnc"`` para agregar la traducción; ``None`` para no traducir.

    Returns:
        El informe de :func:`hope.cnc.check.check_program` (``runnable``, ``errores``, ``avisos``,
        ``tiempo_s``, ``envolvente_g54``…) y, si hubo dialecto y el programa no tiene errores,
        ``traduccion = {"dialecto", "lineas", "inicio"}`` con las primeras :data:`MAX_PREVIEW` líneas.

    Raises:
        ValueError: si la configuración de la máquina es inválida (la levanta ``machine_from_dict``) o
            el dialecto no existe (``emit``). Los errores del G-code NO se lanzan: van en ``errores``.

    Por qué los imports van adentro: la función se importa y ejecuta en un proceso aislado (sandbox);
    importar ``yaml`` y el intérprete solo ahí evita cargarlos en el proceso principal de Hope cuando
    solo se consulta el catálogo.
    """
    import yaml

    from hope.cnc.check import check_program
    from hope.cnc.dialect import emit
    from hope.cnc.gcode import interpret
    from hope.cnc.machine import Machine, machine_from_dict
    # safe_load: el YAML viene de afuera, nunca construir objetos arbitrarios. «or {}» cubre un YAML vacío.
    m = machine_from_dict(yaml.safe_load(machine_yaml) or {}) if machine_yaml else Machine()
    prog = interpret(text, m)
    r = check_program(prog, m)
    if dialect and prog.ok:              # con errores no tiene sentido traducir: el resultado sería inválido
        lines = emit(prog, m, dialect)   # lista de (texto, línea del programa original)
        r["traduccion"] = {"dialecto": dialect, "lineas": len(lines), "inicio": [t for t, _ in lines[:MAX_PREVIEW]]}
    return r


def review_text(r: dict) -> str:
    """Informe legible (texto plano) a partir del resultado de :func:`review`.

    Args:
        r: el ``dict`` que devolvió :func:`review`.

    Returns:
        El texto de :func:`hope.cnc.check.report_text` más, si hay traducción, un resumen con las
        primeras 12 líneas (menos que :data:`MAX_PREVIEW` para que el mensaje quede corto en el chat).
    """
    from hope.cnc.check import report_text
    s = report_text(r)
    if "traduccion" in r:
        t = r["traduccion"]
        s += f"\nTraducción a {t['dialecto']} ({t['lineas']} líneas), inicio:\n" + "\n".join(t["inicio"][:12])
    return s


# ==================================================================================================
# ================================== FICHA PARA EL CATÁLOGO ========================================
# ==================================================================================================

# Ficha de la herramienta (ver ``hope/tools/catalog.py``): identificador, nombre, versión, dominio,
# palabras clave para encontrarla, descripción, entradas/salidas, unidades, límites, riesgos y
# permisos. El catálogo la muestra al asistente para que sepa cuándo usarla y cuándo no.
SPEC = ToolSpec(
    "hte_cnc_v1", "Revisar programa CNC", "1.0", "mecanizado CNC",
    "cnc fresadora router láser plasma mecanizado revisar programa límites recorrido husillo mach3 grbl linuxcnc",
    "Intérprete propio de G-code (RS274/NGC con hábitos de Mach3): errores por línea, límites de recorrido "
    "de la máquina (arcos incluidos), corte con husillo apagado, tiempo con aceleración y esquinas "
    "(planificador tipo grbl), envolvente G54, herramientas y traducción opcional a grbl o LinuxCNC.",
    {"source": "str (ruta o contenido)", "machine": "str? (YAML de la máquina)", "dialect": "str? grbl|linuxcnc"},
    {"runnable": "bool", "errores": "list", "avisos": "list", "tiempo_s": "float", "envolvente_g54": "dict"},
    units_and_formats="mm, mm/min (F), mm/s² (aceleración), rpm (S)", dependencies=["pyyaml"],
    execution="sandbox", cost="medio", expected_precision="tiempo según el modelo de grbl (±5 % en la máquina real)",
    limitations=["sin parámetros (#) ni G41/G42", "ciclos fijos solo en G17/G90",
                 "límites con los orígenes de la configuración: «enviar» vuelve a revisar con los reales"],
    risks=["no mueve la máquina: ejecutar es hardware.control (aprobación humana)"],
    permissions=["tool.execute", "file.read (fuera de la carpeta de trabajo: aprobación humana)"],
    tests=["tests/test_cnc.py"],
    usage_examples=["⟦cnc: pieza.nc⟧", "⟦cnc: pieza.nc | maquina=configs/cnc_maquina.yaml dialecto=grbl⟧"],
    do_not_use_when=["G-code de impresora 3D (usar gcode)"],
    provenance="Hope CNC (docs/CNC_HOPE.md)")
