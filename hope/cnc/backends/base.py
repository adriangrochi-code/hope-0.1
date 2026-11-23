"""Interfaz común de los controladores de tiempo real (el «contrato» que cumple cada backend).

QUÉ ES
    Hope CNC / TISMA CNC no genera pulsos de motor: revisa, traduce y **entrega** el programa a un
    controlador que sí hace el tiempo real (grbl, LinuxCNC) o a un simulador. Este archivo define lo
    mínimo que la app necesita de cualquiera de ellos, para que el resto del código no sepa (ni le
    importe) con cuál está hablando.

CÓMO ENCAJA
    * Lo implementan ``sim.SimBackend``, ``grbl.GrblBackend`` y ``linuxcnc.LinuxCNCBackend``.
    * Lo crea ``hope.cnc.backends.make_backend`` a partir de la sección ``controlador`` del perfil.
    * Lo usan ``hope.cnc.controller.Controller`` (envío con guardián), ``hope.cnc.__main__`` (consola)
      y la app (``hope.cnc.session`` / ``hope.cnc.app.server``, que atrapan ``BackendError``).

CONVENCIONES
    * Posiciones en **mm**, avance en **mm/min**, husillo en **rpm** (las del controlador).
    * Los nombres de estado siguen a grbl (``Idle``, ``Run``, ``Hold``, ``Alarm``, ``Home``, ``Door``)
      aunque el backend sea otro: así la interfaz muestra y decide lo mismo para todos.
    * ``physical = True`` significa «mueve una máquina real»: con eso el ``Controller`` exige la
      aprobación del guardián antes de enviar (ver CLAUDE.md, «Seguridad»).

EJEMPLO
    >>> from hope.cnc.backends import make_backend
    >>> b = make_backend({"tipo": "sim"})
    >>> b.connect(); r = b.stream(["G0 X10 Y5"]); r.ok, b.status().mpos["X"]
    (True, 10.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# ==================================================================================================
# ============================================= ERRORES ============================================
# ==================================================================================================

# Falla del controlador o de la comunicación (puerto, protocolo, módulo que falta, alarma al pedir).
# El mensaje está pensado para mostrarse tal cual al operador (en español y con la solución cuando se
# conoce, p. ej. «falta pyserial…: pip install pyserial»). La app lo atrapa y lo muestra.
class BackendError(RuntimeError):
    pass


# ==================================================================================================
# ===================================== ESTADO Y RESULTADO =========================================
# ==================================================================================================

@dataclass
class Status:
    """Foto del estado del controlador en un instante (respuesta a «¿cómo estás?»).

    Campos:
        state:   estado al estilo grbl (ver comentario en línea). ``"desconocido"`` si no se pudo leer.
        mpos:    posición de **máquina** por eje (mm), relativa al cero de referenciado ($H).
        wpos:    posición de **trabajo** por eje (mm) = mpos − origen activo (G54…G59 + G92).
        feed:    avance real en ese momento (mm/min); 0 si el controlador no lo informa.
        spindle: velocidad del husillo (rpm); 0 si está apagado o no se informa.
        raw:     el texto original de la respuesta (para depurar y para el registro).
    """

    state: str = "desconocido"                         # Idle, Run, Hold, Alarm, Home, Door…
    mpos: dict[str, float] = field(default_factory=dict)
    wpos: dict[str, float] = field(default_factory=dict)
    feed: float = 0.0
    spindle: float = 0.0
    raw: str = ""


@dataclass
class StreamResult:
    """Cómo terminó el envío de un programa (lo que devuelve ``Backend.stream``).

    Campos:
        sent:         líneas enviadas al controlador.
        acknowledged: líneas que el controlador confirmó (``ok``/``error`` en grbl; línea en ejecución
                      en LinuxCNC). Si se abortó, dice hasta dónde llegó la máquina (para retomar).
        errors:       mensajes de error ya traducidos al español, con la línea que los causó.
        aborted:      True si el envío se cortó antes de terminar (error, alarma, operador, tiempo).
        alarm:        texto de la alarma si el controlador entró en alarma (vacío si no).
    """

    sent: int
    acknowledged: int
    errors: list[str]
    aborted: bool = False
    alarm: str = ""

    @property
    def ok(self) -> bool:
        """True si el programa se envió entero y sin errores (lo que la app muestra como «completado»)."""
        return not self.errors and not self.aborted


if TYPE_CHECKING:                                     # solo para las anotaciones (no existe en Python 3.4)
    from typing import Callable
    Progress = Callable[[int, int, int], None]          # (enviadas, confirmadas, total)


# ==================================================================================================
# ============================================ INTERFAZ =============================================
# ==================================================================================================

class Backend:
    """Lo que Hope CNC necesita de un controlador. ``physical`` indica si mueve una máquina real.

    Es una clase base «a mano» (sin ``abc``, por compatibilidad con el traductor a Python 3.4): los
    métodos que no aplican a un controlador quedan como no-op (``...``); ``status`` y ``stream`` son
    obligatorios y lanzan ``NotImplementedError`` si una subclase se olvida de definirlos.

    Contrato:
        connect()       abre/despierta la conexión. No-op si no hace falta (simulador, LinuxCNC).
        close()         libera el puerto.
        status()        devuelve un ``Status``; lanza ``BackendError`` si el controlador no responde.
        work_offsets()  orígenes G54–G59 reales del controlador, para revisar el programa con ellos.
        stream(lines, progress, stop_on_error)
                        envía el programa ya traducido al dialecto del controlador y espera a que
                        termine; devuelve un ``StreamResult``. ``progress(enviadas, confirmadas,
                        total)`` se llama a medida que avanza (para la barra de la interfaz).
        hold()/resume() retención de avance (feed hold) y reanudación: la máquina frena con su
                        aceleración, sin perder posición.
        reset()         aborta el programa (en grbl, Ctrl-X). NO reemplaza la parada de emergencia
                        física: esa siempre corta la energía por hardware.
    """

    name = "base"                                      # identifica el tipo («sim», «grbl», «linuxcnc»)
    physical = True                                    # por las dudas, se asume máquina real (pide guardián)

    # Abre o despierta la conexión con el controlador. La base no hace nada.
    def connect(self) -> None: ...

    # Cierra el puerto o la conexión. La base no hace nada.
    def close(self) -> None: ...

    def status(self) -> Status:
        """Estado actual del controlador (posición, estado, avance). Lo implementa cada backend."""
        raise NotImplementedError

    def work_offsets(self) -> dict[str, dict[str, float]]:
        """Orígenes G54–G59 que tiene el controlador (para revisar con los valores reales).

        Devuelve ``{"G54": {"X": mm, ...}, ...}``. Vacío (lo que hace la base) = «no se sabe»: entonces
        se revisa con los orígenes guardados en el perfil de máquina.
        """
        return {}

    def stream(self, lines: list[str], progress: Progress | None = None, stop_on_error: bool = True) -> StreamResult:
        """Envía ``lines`` (G-code ya traducido, una orden por línea) y espera a que termine.

        ``stop_on_error``: si es True, el primer error detiene la máquina y el envío (lo seguro);
        si es False se anota y se sigue. Lo implementa cada backend.
        """
        raise NotImplementedError

    # Retención de avance (feed hold): la máquina frena sin perder posición. La base no hace nada.
    def hold(self) -> None: ...

    # Reanuda después de una retención. La base no hace nada.
    def resume(self) -> None: ...

    # Aborta el programa en curso (reinicio por software). La base no hace nada.
    def reset(self) -> None: ...
