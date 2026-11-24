"""Hope CNC as an internal catalog tool: ⟦cnc: pieza.nc | maquina=configs/cnc_maquina.yaml⟧.

It only **checks and simulates** (in the sandbox, because G-code is not trusted): errors, travel
limits, time with acceleration, envelope and tools; with ``dialecto=grbl|linuxcnc`` it also
returns the first translated lines. **It does not move the machine**: that is
``python -m hope.cnc enviar`` with human approval from the guardian.

What it is
----------
The "plug" that connects the TISMA CNC G-code interpreter (``hope/cnc``) to Hope's tool catalog
(``hope/tools/catalog.py``). This way an assistant or a user can ask "check this program" and get
a report without touching the hardware.

How it fits
-----------
* Used by ``hope/tools/extended.py``: it registers :data:`SPEC` in the catalog, runs :func:`review`
  inside the sandbox with ``call_function("hope.cnc.tool", "review", ...)`` (limit :data:`TIMEOUT`)
  and formats the result with :func:`review_text`.
* Uses ``hope.cnc.machine`` (configuration), ``hope.cnc.gcode`` (interpreter), ``hope.cnc.check``
  (checks: limits, spindle, timings) and ``hope.cnc.dialect`` (translation to grbl/LinuxCNC).

Units and conventions
---------------------
Everything in mm, mm/min (F), mm/s² (acceleration) and rpm (S), like the rest of ``hope/cnc``. The
result is a JSON-serializable ``dict`` (a sandbox requirement), with keys in Spanish.

Usage example::

    from hope.cnc.tool import review, review_text
    r = review(open("pieza.nc").read(), open("configs/cnc_maquina.yaml").read(), dialect="grbl")
    print(review_text(r))
"""

from __future__ import annotations

from hope.tools.catalog import ToolSpec

# ==================================================================================================
# ======================================== CONFIGURATION ===========================================
# ==================================================================================================

# Maximum time (s) the sandbox lets a review run before cutting it off. A 3D relief program with
# hundreds of thousands of lines takes a few seconds on a modern PC and quite a bit longer on an
# old one; 60 s leaves margin. Typical 30–120. Too low = large programs fail by timeout; too
# high = a malicious G-code or a subprogram loop has more time to burn CPU.
TIMEOUT = 60.0
# Maximum number of translated lines returned as a sample in ``traduccion.inicio``.
# It is only a preview (the full program is translated by «enviar»). Typical 20–100. Higher =
# bigger JSON responses for the assistant, without gaining much.
MAX_PREVIEW = 40


# ==================================================================================================
# =========================================== REVIEW ===============================================
# ==================================================================================================

def review(text: str, machine_yaml: str | None = None, dialect: str | None = None) -> dict:
    """Pure function that runs inside the sandbox (arguments and result in JSON).

    Interprets the program, checks it against the machine and, if requested, translates it to a dialect.

    Args:
        text: contents of the G-code program (text, not a path).
        machine_yaml: YAML text of the machine (format of ``configs/cnc_maquina.yaml``). ``None``:
            default generic machine (:class:`hope.cnc.machine.Machine`).
        dialect: ``"grbl"`` or ``"linuxcnc"`` to add the translation; ``None`` to not translate.

    Returns:
        The report from :func:`hope.cnc.check.check_program` (``runnable``, ``errores``, ``avisos``,
        ``tiempo_s``, ``envolvente_g54``…) and, if a dialect was given and the program has no errors,
        ``traduccion = {"dialecto", "lineas", "inicio"}`` with the first :data:`MAX_PREVIEW` lines.

    Raises:
        ValueError: if the machine configuration is invalid (raised by ``machine_from_dict``) or
            the dialect does not exist (``emit``). G-code errors are NOT raised: they go in ``errores``.

    Why the imports are inside: the function is imported and run in an isolated process (sandbox);
    importing ``yaml`` and the interpreter only there avoids loading them in Hope's main process when
    the catalog is merely being queried.
    """
    import yaml

    from hope.cnc.check import check_program
    from hope.cnc.dialect import emit
    from hope.cnc.gcode import interpret
    from hope.cnc.machine import Machine, machine_from_dict
    # safe_load: the YAML comes from outside, never build arbitrary objects. «or {}» covers an empty YAML.
    m = machine_from_dict(yaml.safe_load(machine_yaml) or {}) if machine_yaml else Machine()
    prog = interpret(text, m)
    r = check_program(prog, m)
    if dialect and prog.ok:              # with errors translating makes no sense: the result would be invalid
        lines = emit(prog, m, dialect)   # list of (text, line of the original program)
        r["traduccion"] = {"dialecto": dialect, "lineas": len(lines), "inicio": [t for t, _ in lines[:MAX_PREVIEW]]}
    return r


def review_text(r: dict) -> str:
    """Readable report (plain text) built from the result of :func:`review`.

    Args:
        r: the ``dict`` returned by :func:`review`.

    Returns:
        The text from :func:`hope.cnc.check.report_text` plus, if there is a translation, a summary with
        the first 12 lines (fewer than :data:`MAX_PREVIEW` so the message stays short in the chat).
    """
    from hope.cnc.check import report_text
    s = report_text(r)
    if "traduccion" in r:
        t = r["traduccion"]
        s += f"\nTraducción a {t['dialecto']} ({t['lineas']} líneas), inicio:\n" + "\n".join(t["inicio"][:12])
    return s


# ==================================================================================================
# ======================================= CATALOG ENTRY ============================================
# ==================================================================================================

# Tool entry (see ``hope/tools/catalog.py``): identifier, name, version, domain, keywords to find
# it, description, inputs/outputs, units, limits, risks and permissions. The catalog shows it to the
# assistant so it knows when to use it and when not to.
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
