"""Medición de rendimiento de TISMA CNC en ESTA computadora.

    python -m hope.cnc medir              # unos segundos; no mueve nada ni toca la máquina
    python -m hope.cnc medir --lineas 50000

Qué hace
--------
Genera en memoria un programa de prueba parecido a lo que sale de un CAM para un relieve 3D (el peor
caso: muchísimos tramos cortos de «G1 X Y Z F»), y mide cada etapa de la carga de un programa en la
app:

* **intérprete**: leer el G-code y convertirlo en movimientos (gcode.py);
* **planificador**: velocidades en cada esquina y tiempo estimado (planner.py);
* **revisión**: límites de recorrido, husillo, plasma (check.py);
* **traducción**: el G-code que se envía al controlador (dialect.py).

Con eso estima cuánto tarda en cargar un programa de 100 000 líneas y recomienda el modo liviano de
la pantalla si la PC es lenta. Sirve para comparar PCs viejas y para ver si una optimización ayudó.
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

# Una PC «lenta» para TISMA CNC: si cargar 100 000 líneas lleva más que esto, se recomienda el modo
# liviano (refresca menos la pantalla y dibuja menos puntos). Segundos.
UMBRAL_LENTA_S = 15.0


def programa_de_prueba(lineas: int) -> str:
    """G-code de un relieve 3D en zigzag de 150 × 100 mm con unas ``lineas`` líneas de «G1 X Y Z F»
    (Z sigue una superficie ondulada, como el acabado de un CAM)."""
    filas = max(2, int(math.sqrt(lineas)))
    out = ["G21 G90 G17 G54", "S18000 M3", "G0 Z5", "G0 X0 Y0", "G1 Z-2 F600"]
    for j in range(filas):
        y = j * 100.0 / filas
        columnas = range(filas) if j % 2 == 0 else range(filas - 1, -1, -1)   # zigzag
        for i in columnas:
            x = i * 150.0 / filas
            z = -2 - 1.5 * math.sin(x / 15) * math.cos(y / 12)
            out.append(f"G1 X{x:.4f} Y{y:.4f} Z{z:.4f} F1800")
    out += ["G0 Z5", "M5", "M30"]
    return "\n".join(out)


def maquina_de_prueba() -> Machine:
    """Router de 300 × 200 × 80 mm con el cero de pieza en la esquina de la mesa (sin errores de
    recorrido para el programa de prueba)."""
    return machine_from_dict({
        "nombre": "máquina de prueba", "tipo": "router",
        "ejes": {"X": {"vel_max": 5000, "acel": 300, "min": 0, "max": 300},
                 "Y": {"vel_max": 5000, "acel": 300, "min": 0, "max": 200},
                 "Z": {"vel_max": 1500, "acel": 150, "min": -80, "max": 0}},
        "origenes": {"G54": {"X": 50, "Y": 50, "Z": -40}},
    })


def medir(lineas: int = 20000) -> dict:
    """Mide cada etapa con un programa de ``lineas`` líneas. Devuelve segundos por etapa, el total y
    la estimación para 100 000 líneas (el tiempo crece en proporción a las líneas)."""
    m = maquina_de_prueba()
    texto = programa_de_prueba(lineas)
    reloj = time.perf_counter
    etapas: list[tuple[str, float]] = []           # lista y no dict: en Python 3.4 el orden se perdería
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
    print(informe_texto(medir(lineas)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000))
