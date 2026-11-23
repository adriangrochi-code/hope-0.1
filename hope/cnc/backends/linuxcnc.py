"""LinuxCNC por su API de Python (módulo ``linuxcnc``, viene con LinuxCNC).

LinuxCNC hace el tiempo real: en PCs viejas genera los pulsos por el **puerto paralelo** (la misma
placa de conexión que usaba Mach3, configurada con StepConf); en PCs modernas, con tarjetas **Mesa**
(7i96S, 7i76E) por Ethernet o con una Raspberry Pi 4/5. Hope CNC le entrega el programa traducido
(dialecto ``linuxcnc``) como archivo y lo ejecuta en modo automático.

CÓMO ENCAJA
    ``make_backend({"tipo": "linuxcnc"})`` crea un ``LinuxCNCBackend``. LinuxCNC tiene que estar
    instalado **y en marcha** (con su configuración .ini cargada): este módulo solo le habla por la
    API de Python, no lo arranca. A diferencia de grbl, acá no hay que dosificar el envío: LinuxCNC
    lee el archivo entero y planifica él solo.

    API usada (ver «Python Interface» en la documentación de LinuxCNC):
        ``linuxcnc.stat()``          lectura del estado (hay que llamar a ``poll()`` antes de leer).
        ``linuxcnc.command()``       órdenes: modo, abrir programa, ejecutar, pausar, abortar.
        ``linuxcnc.error_channel()`` errores que reporta LinuxCNC (intérprete, límites, etc.).

UNIDADES
    LinuxCNC informa la velocidad en unidades de máquina **por segundo**; acá se pasa a mm/min (× 60)
    para que coincida con grbl y con la interfaz. Se asume una configuración en mm.

EJEMPLO
    >>> b = make_backend({"tipo": "linuxcnc"})
    >>> b.stream(["G21 G90", "G0 X10", "M2"]).ok
    True
"""

from __future__ import annotations

import tempfile
import time

from hope.cnc.backends.base import Backend, BackendError, Status, StreamResult

# Orden de los ejes en los vectores de posición de LinuxCNC (``actual_position``, ``g5x_offset``):
# siempre 9 valores X Y Z A B C U V W; acá se usan los 6 primeros.
AXES = ("X", "Y", "Z", "A", "B", "C")


# ==================================================================================================
# ========================================== CONTROLADOR ===========================================
# ==================================================================================================

class LinuxCNCBackend(Backend):
    """Controlador LinuxCNC (puerto paralelo, Mesa, Raspberry Pi) a través de su API de Python."""

    name = "linuxcnc"
    physical = True

    def __init__(self, ini: str | None = None, module=None):
        """Se engancha al LinuxCNC que está corriendo.

        Parámetros:
            ini:    ruta del .ini de la máquina (solo se guarda como referencia: el LinuxCNC en marcha
                    ya tiene el suyo cargado).
            module: el módulo ``linuxcnc`` a usar; None = importar el real. Las pruebas pasan uno falso.

        Lanza ``BackendError`` si el módulo ``linuxcnc`` no está instalado.
        """
        if module is None:
            try:
                import linuxcnc as module
            except ImportError:
                raise BackendError("el módulo «linuxcnc» no está: este controlador necesita LinuxCNC instalado "
                                   "y en marcha (linuxcnc.org)") from None
        self.lc = module
        self.ini = ini
        self.stat = module.stat()
        self.cmd = module.command()

    def status(self) -> Status:
        """Estado traducido a los nombres de grbl (Idle, Run, Hold, Alarm; «Error» si es otro).

        Prioridad: parada de emergencia (Alarm) > pausa (Hold) > estado de la tarea. La posición de
        trabajo se calcula restando el origen G5x activo (no incluye G92 ni la compensación de largo
        de herramienta). El avance viene en mm/s y se pasa a mm/min.
        """
        self.stat.poll()
        state = {self.lc.RCS_DONE: "Idle", self.lc.RCS_EXEC: "Run"}.get(self.stat.state, "Error")
        if self.stat.paused:
            state = "Hold"
        if self.stat.estop:
            state = "Alarm"
        pos = {a: float(v) for a, v in zip(AXES, self.stat.actual_position)}
        off = {a: float(v) for a, v in zip(AXES, self.stat.g5x_offset)}
        return Status(state, pos, {a: pos[a] - off[a] for a in pos}, feed=float(self.stat.current_vel) * 60.0)

    def stream(self, lines, progress=None, stop_on_error=True) -> StreamResult:
        """Escribe el programa en un .ngc temporal, lo abre en LinuxCNC y lo ejecuta hasta el final.

        Parámetros: como ``Backend.stream``. ``progress`` se llama cada 0.1 s con la línea que está
        ejecutando LinuxCNC (``motion_line``) como «confirmadas». Con ``stop_on_error`` el primer
        mensaje del canal de errores aborta el programa.

        Bloquea hasta que el intérprete queda inactivo y la tarea terminada. El archivo temporal no se
        borra (``delete=False``): LinuxCNC lo sigue leyendo mientras ejecuta y queda para revisar.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".ngc", delete=False, encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")
            path = f.name
        self.cmd.mode(self.lc.MODE_AUTO)                # solo en modo automático se puede correr un programa
        self.cmd.wait_complete()
        self.cmd.program_open(path)
        self.cmd.auto(self.lc.AUTO_RUN, 0)              # 0 = empezar desde la primera línea
        errors = []
        err = self.lc.error_channel()
        while True:
            time.sleep(0.1)                             # sondeo a 10 Hz: alcanza para la barra y no carga la CPU
            self.stat.poll()
            e = err.poll()                              # None o (tipo, texto)
            if e:
                errors.append(str(e[1]))
                if stop_on_error:
                    self.cmd.abort()
                    return StreamResult(len(lines), self.stat.motion_line, errors, aborted=True)
            if progress:
                progress(len(lines), self.stat.motion_line, len(lines))
            if self.stat.interp_state == self.lc.INTERP_IDLE and self.stat.state == self.lc.RCS_DONE:
                return StreamResult(len(lines), len(lines), errors)

    def hold(self) -> None:
        """Pausa el programa (como el botón de pausa de AXIS)."""
        self.cmd.auto(self.lc.AUTO_PAUSE)

    def resume(self) -> None:
        """Reanuda un programa pausado."""
        self.cmd.auto(self.lc.AUTO_RESUME)

    def reset(self) -> None:
        """Aborta el programa y detiene los movimientos. No reemplaza la parada de emergencia física."""
        self.cmd.abort()
