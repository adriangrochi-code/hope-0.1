"""Simulador: «ejecuta» el programa sin hardware y lleva la posición y el tiempo virtual.

QUÉ ES
    Un controlador de mentira que cumple la interfaz ``Backend``. No mueve nada (``physical = False``,
    así que el ``Controller`` no pide permiso al guardián) y responde al instante.

PARA QUÉ SIRVE
    * Probar un programa y la cadena completa (perfiles → revisión → traducción → envío) sin máquina.
    * Las pruebas automáticas (``tests/test_cnc.py``).
    * Es el controlador por defecto: un perfil sin ``controlador.tipo`` arranca en simulación.

CÓMO FUNCIONA
    Vuelve a interpretar el programa traducido con ``hope.cnc.gcode.interpret`` (posición final de
    cada movimiento) y lo pasa por ``hope.cnc.planner.plan`` para sumar el tiempo que tardaría la
    máquina real con sus aceleraciones. No hay tiempo real: ``stream`` termina de una.

UNIDADES
    Posiciones en mm (coordenadas de máquina), ``clock`` en segundos de tiempo virtual acumulado.

EJEMPLO
    >>> sim = SimBackend()
    >>> sim.stream(["G21 G90", "G1 X50 F1000"]).ok, sim.pos["X"], round(sim.clock, 1) > 0
    (True, 50.0, True)
"""

from __future__ import annotations

from hope.cnc.backends.base import Backend, Status, StreamResult
from hope.cnc.gcode import interpret
from hope.cnc.machine import Machine, as_controller
from hope.cnc.planner import plan

# ==================================================================================================
# =========================================== SIMULADOR ============================================
# ==================================================================================================

class SimBackend(Backend):
    """Controlador simulado: guarda posición, reloj virtual y todo lo enviado.

    Atributos:
        machine: la máquina «vista desde el controlador» (ver ``as_controller``).
        pos:     posición de máquina actual por eje (mm). Arranca en 0 en todos los ejes.
        clock:   tiempo virtual acumulado de todos los programas enviados (s).
        log:     todas las líneas recibidas, en orden (para que las pruebas vean qué se envió).
        state:   «Idle» o «Run» (al estilo grbl).
    """

    name = "sim"
    physical = False                                   # no mueve nada: no hace falta aprobación del guardián

    def __init__(self, machine: Machine | None = None):
        """``machine``: la máquina efectiva del perfil; None = ``Machine()`` genérica."""
        self.machine = as_controller(machine or Machine())   # recibe el programa ya traducido
        self.pos = {a: 0.0 for a in self.machine.axis_names}
        self.clock = 0.0
        self.log: list[str] = []
        self.state = "Idle"

    def status(self) -> Status:
        """Estado actual. La posición de trabajo se calcula siempre respecto de G54 (el simulador no
        sigue qué sistema de coordenadas dejó activo el programa)."""
        g54 = self.machine.offset("G54")
        return Status(self.state, dict(self.pos), {a: v - g54[a] for a, v in self.pos.items()})

    def work_offsets(self):
        """Los orígenes G54–G59 del perfil de máquina (en el simulador son los «reales»)."""
        from hope.cnc.machine import WCS
        return {w: self.machine.offset(w) for w in WCS}

    def stream(self, lines, progress=None, stop_on_error=True) -> StreamResult:
        """«Ejecuta» el programa: actualiza ``pos``, suma su duración a ``clock`` y lo anota en ``log``.

        Parámetros: como ``Backend.stream``. Si el intérprete encuentra errores y ``stop_on_error`` es
        True, no se ejecuta nada (0 enviadas, abortado); si es False se ejecuta igual y los errores
        vuelven en el resultado. ``progress`` se llama una sola vez, al final (no hay avance gradual).
        """
        prog = interpret("\n".join(lines), self.machine, start=self.pos)   # sigue desde donde quedó
        errors = [str(d) for d in prog.errors]
        if errors and stop_on_error:
            return StreamResult(0, 0, errors, aborted=True)
        self.state = "Run"
        pl = plan(prog, self.machine)                  # mismo planificador que estima los tiempos en pantalla
        self.clock += pl.total_time
        for op in prog.ops:
            if op.end:                                 # solo las operaciones que mueven tienen posición final
                self.pos = dict(op.end)
        self.log.extend(lines)
        self.state = "Idle"
        if progress:
            progress(len(lines), len(lines), len(lines))
        return StreamResult(len(lines), len(lines), errors)
