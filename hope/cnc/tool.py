"""Hope CNC como herramienta interna del catálogo: ⟦cnc: pieza.nc | maquina=configs/cnc_maquina.yaml⟧.

Solo **revisa y simula** (en el sandbox, porque el G-code no es confiable): errores, límites de
recorrido, tiempo con aceleración, envolvente y herramientas; con ``dialecto=grbl|linuxcnc``
también devuelve las primeras líneas traducidas. **No mueve la máquina**: eso es
``python -m hope.cnc enviar`` con aprobación humana del guardián.
"""

from __future__ import annotations

from hope.tools.catalog import ToolSpec

TIMEOUT = 60.0
MAX_PREVIEW = 40


def review(text: str, machine_yaml: str | None = None, dialect: str | None = None) -> dict:
    """Función pura que corre dentro del sandbox (argumentos y resultado en JSON)."""
    import yaml

    from hope.cnc.check import check_program
    from hope.cnc.dialect import emit
    from hope.cnc.gcode import interpret
    from hope.cnc.machine import Machine, machine_from_dict
    m = machine_from_dict(yaml.safe_load(machine_yaml) or {}) if machine_yaml else Machine()
    prog = interpret(text, m)
    r = check_program(prog, m)
    if dialect and prog.ok:
        lines = emit(prog, m, dialect)
        r["traduccion"] = {"dialecto": dialect, "lineas": len(lines), "inicio": [t for t, _ in lines[:MAX_PREVIEW]]}
    return r


def review_text(r: dict) -> str:
    from hope.cnc.check import report_text
    s = report_text(r)
    if "traduccion" in r:
        t = r["traduccion"]
        s += f"\nTraducción a {t['dialecto']} ({t['lineas']} líneas), inicio:\n" + "\n".join(t["inicio"][:12])
    return s


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
    limitations=["sin parámetros (#), subprogramas ni G41/G42", "ciclos fijos solo en G17/G90",
                 "límites con los orígenes de la configuración: «enviar» vuelve a revisar con los reales"],
    risks=["no mueve la máquina: ejecutar es hardware.control (aprobación humana)"],
    permissions=["tool.execute", "file.read (fuera de la carpeta de trabajo: aprobación humana)"],
    tests=["tests/test_cnc.py"],
    usage_examples=["⟦cnc: pieza.nc⟧", "⟦cnc: pieza.nc | maquina=configs/cnc_maquina.yaml dialecto=grbl⟧"],
    do_not_use_when=["G-code de impresora 3D (usar gcode)"],
    provenance="Hope CNC (docs/CNC_HOPE.md)")
