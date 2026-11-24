"""TISMA CNC performance measurement on THIS computer.

    python -m hope.cnc medir              # a few seconds; moves nothing and does not touch the machine
    python -m hope.cnc medir --lineas 50000

What it does
------------
It generates in memory a test program similar to what a CAM produces for a 3D relief (the worst
case: lots and lots of short «G1 X Y Z F» segments), and times each stage of loading a program in
the app:

* **interpreter**: read the G-code and turn it into moves (gcode.py);
* **planner**: speeds at each corner and estimated time (planner.py);
* **check**: travel limits, spindle, plasma (check.py);
* **translation**: the G-code that is sent to the controller (dialect.py).

From that it estimates how long a 100,000-line program takes to load and recommends the screen's
light mode if the PC is slow. Useful for comparing old PCs and for seeing whether an optimization
helped.

How it fits
-----------
* **Uses:** ``gcode``, ``planner``, ``check`` and ``dialect`` (exactly the same functions the app
  uses when opening a program) and ``machine`` to build a test machine.
* **Used by:** the console (``python -m hope.cnc medir``, from ``__main__``) and the performance
  tests (``tests/test_cnc_rendimiento.py``).

Units and conventions
---------------------
Times in wall-clock seconds (``time.perf_counter``); the test program in mm. Results depend on the
PC and on whatever else is running at the same time: it is worth measuring two or three times.

Sample output
-------------
::

    TISMA CNC — rendimiento en esta PC (Python 3.7.9, Windows-Vista-…)
    Programa de prueba: 19889 líneas de relieve 3D.
      intérprete      0.900 s
      …
    Un programa de 100 000 líneas tardaría unos 6.2 s en cargarse.
"""

from __future__ import annotations

import math
import platform
import sys
import time

from hope.cnc.check import check_program
from hope.cnc.dialect import emit
from hope.cnc.gcode import interpret
from hope.cnc.machine import Machine, machine_from_dict
from hope.cnc.planner import plan

# ==================================================================================================
# ========================================= CONFIGURATION ==========================================
# ==================================================================================================

# A "slow" PC for TISMA CNC: if loading 100,000 lines takes longer than this, light mode is
# recommended (refreshes the screen less often and draws fewer points). Seconds.
# Default 15 s (what an operator tolerates waiting when opening a large program). Reasonable 5–30:
# lower recommends light mode on PCs that do not need it; higher stops recommending it on PCs
# where the screen will already feel sluggish.
UMBRAL_LENTA_S = 15.0

# ==================================================================================================
# =========================================== TEST DATA ============================================
# ==================================================================================================


def programa_de_prueba(lineas: int) -> str:
    """G-code for a 150 × 100 mm zigzag 3D relief with about ``lineas`` lines of «G1 X Y Z F»
    (Z follows a wavy surface, like a CAM finishing pass).

    It is a grid of √lineas × √lineas points (minimum 2 × 2), so the actual number of lines is
    approximate. Deterministic: always the same text for the same ``lineas``."""
    filas = max(2, int(math.sqrt(lineas)))
    out = ["G21 G90 G17 G54", "S18000 M3", "G0 Z5", "G0 X0 Y0", "G1 Z-2 F600"]
    for j in range(filas):
        y = j * 100.0 / filas
        columnas = range(filas) if j % 2 == 0 else range(filas - 1, -1, -1)   # zigzag
        for i in columnas:
            x = i * 150.0 / filas
            z = -2 - 1.5 * math.sin(x / 15) * math.cos(y / 12)   # between −3.5 and −0.5 mm: Z always changes
            out.append(f"G1 X{x:.4f} Y{y:.4f} Z{z:.4f} F1800")
    out += ["G0 Z5", "M5", "M30"]
    return "\n".join(out)


def maquina_de_prueba() -> Machine:
    """300 × 200 × 80 mm router with the work zero at the corner of the table (no travel errors for
    the test program)."""
    return machine_from_dict({
        "nombre": "máquina de prueba", "tipo": "router",
        "ejes": {"X": {"vel_max": 5000, "acel": 300, "min": 0, "max": 300},
                 "Y": {"vel_max": 5000, "acel": 300, "min": 0, "max": 200},
                 "Z": {"vel_max": 1500, "acel": 150, "min": -80, "max": 0}},
        "origenes": {"G54": {"X": 50, "Y": 50, "Z": -40}},
    })


# ==================================================================================================
# ========================================= MEASUREMENT ============================================
# ==================================================================================================


def medir(lineas: int = 20000) -> dict:
    """Times each stage with a program of ``lineas`` lines. Returns seconds per stage, the total and
    the estimate for 100,000 lines (time grows in proportion to the number of lines).

    Result keys: ``lineas`` (the lines the program actually had), ``etapas_s`` (list of
    ``(name, s)`` in the order they run), ``total_s``, ``estimado_100k_s``, ``listo`` (whether the
    check accepted it: it must be True, otherwise the test is not representative), ``python``
    and ``sistema``. Touches no machine and no file."""
    m = maquina_de_prueba()
    texto = programa_de_prueba(lineas)
    reloj = time.perf_counter
    etapas: list[tuple[str, float]] = []           # a list, not a dict: in Python 3.4 the order would be lost
    t0 = reloj()
    prog = interpret(texto, m)
    etapas.append(("intérprete", reloj() - t0))
    t0 = reloj()
    pl = plan(prog, m)
    etapas.append(("planificador", reloj() - t0))
    t0 = reloj()
    informe = check_program(prog, m, pl)
    etapas.append(("revisión", reloj() - t0))
    t0 = reloj()
    emit(prog, m, "grbl")
    etapas.append(("traducción", reloj() - t0))
    total = sum(s for _, s in etapas)
    n = informe["lineas"]
    return {"lineas": n, "etapas_s": etapas, "total_s": total, "estimado_100k_s": total * 100000.0 / max(1, n),
            "listo": informe["runnable"], "python": platform.python_version(), "sistema": platform.platform()}


def informe_texto(r: dict) -> str:
    """Result of ``medir`` as text for the console, with the light-mode recommendation based on
    ``UMBRAL_LENTA_S``."""
    est = r["estimado_100k_s"]
    out = [f"TISMA CNC — rendimiento en esta PC (Python {r['python']}, {r['sistema']})",
           f"Programa de prueba: {r['lineas']} líneas de relieve 3D."]
    for etapa, s in r["etapas_s"]:
        out.append(f"  {etapa:<13} {s:7.3f} s")
    out.append(f"  {'total':<13} {r['total_s']:7.3f} s")
    out.append(f"Un programa de 100 000 líneas tardaría unos {est:.1f} s en cargarse.")
    if est > UMBRAL_LENTA_S:
        out.append("PC lenta para programas grandes: conviene el modo liviano (Configuración → Modo liviano).")
    else:
        out.append("Rendimiento suficiente: no hace falta el modo liviano.")
    return "\n".join(out)


def main(lineas: int = 20000) -> int:
    """Measures and prints the report. Returns 0 (exit code for the console)."""
    print(informe_texto(medir(lineas)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000))
