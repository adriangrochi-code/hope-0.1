"""Hope CNC — control de máquinas CNC moderno, como herramienta interna de Hope (docs/CNC_HOPE.md).

Reemplazo de Mach3 pensado en dos capas:

* **PC (este paquete, Python):** interpreta el G-code (incluidos los hábitos de Mach3), planifica
  el movimiento con anticipación (lookahead) y límites de velocidad y aceleración por eje, revisa
  el programa contra la máquina (límites de recorrido, husillo, herramientas) y lo traduce al
  dialecto del controlador.
* **Tiempo real (fuera de Python):** los pulsos de paso los genera un controlador dedicado
  (grblHAL/FluidNC por USB o Ethernet) o LinuxCNC (puerto paralelo en PCs viejas, tarjetas Mesa en
  las modernas). Python no genera pulsos: no es de tiempo real.

Módulos:

* ``machine`` — configuración de la máquina (YAML) y sistemas de coordenadas.
* ``gcode`` — intérprete: G-code → operaciones (movimientos, arcos, pausas, husillo…).
* ``planner`` — planificador con lookahead (desviación de unión estilo grbl) y tiempos exactos.
* ``check`` — revisión del programa contra la máquina; informe en texto.
* ``dialect`` — traducción a grbl/grblHAL o LinuxCNC (ciclos fijos expandidos, M6 manual, G43).
* ``backends`` — simulador, grbl/grblHAL (serie) y LinuxCNC.
* ``controller`` — une todo; **el movimiento real pasa por el guardián** (``hardware.control``,
  nivel humano): Hope puede revisar y simular, pero nunca mover la máquina por su cuenta.
"""
