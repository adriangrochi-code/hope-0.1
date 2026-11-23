"""Conexiones en vivo para la app (hope/cnc/app): simulador en tiempo real, grbl y firmwares «ok».

QUÉ ES
------
Un «enlace» (*link*) es el objeto que habla con el controlador de la máquina mientras la app está
abierta. A diferencia de ``backends`` (envío por lotes desde la consola: se manda el programa y se
espera a que termine), una conexión en vivo atiende **a la vez** el envío del programa, el estado
cada 200 ms, el jog, MDI, el referenciado y los ajustes de avance/husillo en marcha (overrides),
como hace Mach3. Todos los enlaces comparten la interfaz :class:`Link`, así la app no sabe ni le
importa qué controlador hay del otro lado.

* :class:`SimLink` — simulador: ejecuta los bloques del planificador en tiempo real (aceleración,
  esquinas, overrides, pausas M0, retención, jog, referenciado). Sirve para aprender la app y
  probar programas sin máquina. Un hilo propio avanza el reloj de la simulación cada ``tick``.
* :class:`GrblLink` — grbl / grblHAL / FluidNC: un hilo lector despacha ``ok``/``error``/estado y
  otro pide ``?``; el programa se envía con **conteo de caracteres** (se llena el búfer de recepción
  del controlador sin desbordarlo). Los ajustes en marcha son bytes «en tiempo real» (0x90…0x9D),
  el jog es ``$J=``, el referenciado ``$H``.
* :class:`OkLink` — Marlin / Smoothieware / RepRapFirmware (MPCNC, Lowrider, placas de impresora
  3D): protocolo **línea por línea**, cada línea espera su ``ok``. La posición se pide con ``M114``,
  el avance se ajusta con ``M220`` y la parada usa ``M410``/``M108`` (EMERGENCY_PARSER).

CÓMO ENCAJA
-----------
* Lo usa: ``hope/cnc/session.py`` (:class:`CncSession`), que a su vez sirve a la interfaz web
  (``hope/cnc/app/server.py``). La sesión consulta :meth:`Link.snapshot` varias veces por segundo y
  llama a ``start``/``hold``/``resume``/``stop``/``jog``/… cuando el operador toca un botón.
* Usa: ``hope.cnc.gcode`` (intérprete, para el simulador), ``hope.cnc.planner`` (perfiles de
  velocidad del simulador) y ``hope.cnc.backends.grbl`` (``open_port`` abre el puerto serie/TCP,
  ``describe`` traduce ``error:N``/``ALARM:N`` a texto).
* :func:`make_link` elige el enlace según la sección ``controlador`` del perfil de la máquina.

CONVENCIONES
------------
* Unidades: milímetros (mm), mm/s dentro del simulador, mm/min en lo que se muestra (``avance``) y en
  la palabra F, segundos para tiempos, porcentaje entero (100 = sin ajuste) para overrides.
* Estados (``snapshot()["estado"]``): los nombres de grbl — ``Idle`` (reposo), ``Run`` (ejecutando),
  ``Hold`` (retenido), ``Jog``, ``Home`` (referenciando), ``Alarm``, más ``Desconectado``.
  Los otros enlaces imitan esos nombres para que la interfaz trate a todos igual.
* Diccionario del trabajo (``_job``): ``activo``, ``enviadas`` (líneas mandadas), ``confirmadas``
  (líneas con ``ok``), ``total``, ``error`` (texto o None) y ``terminado``.
* Concurrencia: cada enlace físico tiene **dos candados**. ``lock`` (RLock) protege el estado
  compartido entre el hilo lector, el hilo de consulta y los pedidos de la interfaz; ``wlock``
  serializa las escrituras al puerto para que dos hilos no mezclen bytes de líneas distintas.
  Están separados para que el hilo de consulta pueda escribir ``?`` sin esperar al lector.
* Los textos para el operador van en español; los comandos al firmware, tal como los espera.

EJEMPLO
-------
::

    from hope.cnc.links import make_link
    link = make_link(machine)                     # según machine.controller: {"tipo": "grbl", ...}
    link.start(["G21 G90", "G0 X10 Y10", "G1 X50 F600", "M30"])
    while link.snapshot()["trabajo"]["activo"]:
        time.sleep(0.2)                           # la interfaz muestra estado/posición mientras tanto
    link.close()
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque

from hope.cnc.backends.base import BackendError
from hope.cnc.backends.grbl import describe, open_port
from hope.cnc.gcode import Interpreter, Op
from hope.cnc.machine import Machine, as_controller
from hope.cnc.planner import distance_at, plan

# ==================================================================================================
# ================================ COMANDOS EN TIEMPO REAL DE GRBL =================================
# ==================================================================================================

# Overrides en tiempo real de grbl 1.1: (tipo, acción) -> byte a enviar.
#
# grbl separa estos bytes del flujo de G-code apenas llegan por el puerto (en la interrupción de
# recepción): NO ocupan lugar en el búfer de 128 bytes, NO llevan salto de línea y NO responden «ok».
# Por eso se pueden mandar en cualquier momento, aun con el búfer lleno, y no se cuentan en ``pending``.
#
#   avance  (F de G1/G2/G3):  0x90 = 100 %, 0x91 = +10 %, 0x92 = −10 %, 0x93 = +1 %, 0x94 = −1 %.
#                             Rango que acepta grbl: 10 %–200 % (lo que exceda se recorta).
#   rápido  (G0):             0x95 = 100 %, 0x96 = 50 %, 0x97 = 25 %. Solo esos tres valores.
#   husillo (S):              0x99 = 100 %, 0x9A = +10 %, 0x9B = −10 %, 0x9C = +1 %, 0x9D = −1 %.
#                             Rango 10 %–200 %. (0x9E, parar husillo en retención, no se usa.)
#
# El simulador usa esta misma tabla para validar (tipo, acción) e imita esos rangos.
OVERRIDES = {
    ("avance", "100"): 0x90, ("avance", "+10"): 0x91, ("avance", "-10"): 0x92, ("avance", "+1"): 0x93,
    ("avance", "-1"): 0x94, ("rapido", "100"): 0x95, ("rapido", "50"): 0x96, ("rapido", "25"): 0x97,
    ("husillo", "100"): 0x99, ("husillo", "+10"): 0x9A, ("husillo", "-10"): 0x9B, ("husillo", "+1"): 0x9C,
    ("husillo", "-1"): 0x9D,
}
# Cancelar jog (grbl 1.1). Frena con la aceleración configurada y descarta el resto del movimiento
# ``$J=`` que haya en el búfer; la máquina vuelve a Idle sin perder posición (a diferencia de Ctrl-X).
JOG_CANCEL = 0x85
# Alternar refrigerante por inundación (M8/M9) en tiempo real, sin esperar a que se vacíe el búfer.
# (0xA1 sería el de niebla, M7; la app solo maneja inundación.)
FLOOD_TOGGLE = 0xA0


# --- trabajo --------------------------------------------------------------------------------------
def _job(total=0):
    """Diccionario nuevo con el avance de un programa (ver «Convenciones» en el encabezado).

    total: cantidad de líneas del programa (0 = no hay programa).
    Devuelve un trabajo inactivo: ``activo`` lo enciende quien arranca el ciclo.
    """
    return {"activo": False, "enviadas": 0, "confirmadas": 0, "total": total, "error": None, "terminado": False}


# ==================================================================================================
# ===================================== INTERFAZ COMÚN (Link) ======================================
# ==================================================================================================

class Link:
    """Interfaz común de las conexiones en vivo. La app solo habla con estos métodos.

    Las subclases redefinen lo que su controlador sabe hacer; lo que no redefinen queda como «no hace
    nada» (``...``) o lanza ``NotImplementedError`` si es imprescindible. Todos los métodos se llaman
    desde los hilos de la interfaz y deben devolver rápido (salvo los que esperan una respuesta del
    controlador, como ``home``, que documentan su tiempo máximo).

    Atributos de clase:
        physical: True si mueve una máquina real (la sesión pide aprobación del guardián).
        name:     nombre corto que se muestra como «conexión» si el controlador no se presentó.
        single:   modo bloque a bloque (Single BLK de Mach3) encendido o no.

    Errores: los pedidos inválidos (máquina ocupada, comando rechazado, sin respuesta) lanzan
    :class:`BackendError` con un texto en español listo para mostrar al operador.
    """

    physical = True
    name = "base"

    # Abre la conexión y arranca los hilos de fondo. Idempotente: llamarla dos veces no duplica hilos.
    def open(self) -> None: ...

    # Detiene los hilos de fondo y cierra el puerto. No para la máquina (para eso está ``stop``).
    def close(self) -> None: ...

    def snapshot(self) -> dict:
        """Foto del estado para la interfaz. Se llama varias veces por segundo: no debe bloquear.

        Devuelve un diccionario nuevo (la interfaz puede guardarlo sin que cambie por debajo) con:
            estado (str), mpos (posición de máquina por eje, mm), wco (origen de trabajo activo, mm),
            avance (mm/min real), husillo (rpm), husillo_dir ("cw"/"ccw"/"off"), refrigerante (bool),
            ov ({"avance", "rapido", "husillo"} en %), referenciado (bool), trabajo (ver ``_job``),
            mensaje, alarma, wcs ("G54"…), conexion (nombre/banner), pausa_g4 (bool),
            bloque_a_bloque (bool) y, en el simulador, avance_prog (mm/min programado).
        """
        raise NotImplementedError

    def offsets(self) -> dict[str, dict[str, float]]:
        """Orígenes de trabajo del controlador: ``{"G54": {"X": mm, ...}, ..., "G92": {...}}``.

        Devuelve {} si el controlador no los informa (la app usa entonces los suyos).
        """
        return {}

    def start(self, lines: list[str]) -> None:
        """Arranca un programa (Cycle Start). ``lines``: líneas de G-code ya traducidas, una por elemento.

        Vuelve enseguida; el envío sigue en segundo plano y se sigue con ``snapshot()["trabajo"]``.
        Lanza BackendError si ya hay un programa en marcha o la máquina no está en reposo.
        """
        raise NotImplementedError

    # Feed Hold: frenar sin perder posición. Se sigue con ``resume``.
    def hold(self) -> None: ...

    # Cycle Start durante una retención o una pausa M0/M1: seguir.
    def resume(self) -> None: ...

    # Stop: abortar el programa, apagar husillo y dejar la máquina en reposo.
    def stop(self) -> None: ...

    def jog(self, moves: dict[str, float], feed: float) -> None:
        """Movimiento manual incremental.

        moves: desplazamiento por eje en mm, p. ej. ``{"X": 1.0, "Z": -0.1}``.
        feed:  velocidad en mm/min.
        Lanza BackendError si la máquina está ocupada o el movimiento sale del recorrido.
        """
        raise NotImplementedError

    # Corta un jog en curso (al soltar la tecla de jog continuo).
    def jog_cancel(self) -> None: ...

    def home(self) -> None:
        """Referenciado (Ref All Home de Mach3). Puede esperar hasta que termine el ciclo."""
        raise NotImplementedError

    # Quitar el bloqueo de alarma sin referenciar ($X en grbl, M999 en Marlin).
    def unlock(self) -> None: ...

    def mdi(self, line: str) -> None:
        """Ejecuta una línea de G-code suelta (MDI). Lanza BackendError si hay programa en marcha."""
        raise NotImplementedError

    # Ajuste en marcha: kind = "avance" | "rapido" | "husillo"; action = clave de OVERRIDES ("+10", "100"…).
    def override(self, kind: str, action: str) -> None: ...

    # Prende/apaga el refrigerante (botón Flood de Mach3), aun con un programa corriendo.
    def coolant_toggle(self) -> None: ...

    # Bloque a bloque apagado por defecto: el programa corre de corrido.
    single = False

    def set_single(self, on: bool) -> None:
        """Bloque a bloque (Single BLK de Mach3): cada Cycle Start ejecuta una línea y queda en pausa."""
        self.single = bool(on)

    def modal(self) -> str:
        """Texto con los códigos modales activos (G90 G21 G54…), o "" si el enlace no los conoce."""
        return ""

    def tool(self) -> int | None:
        """Número de herramienta activa (T), o None si el enlace no lo sabe."""
        return None

    def clear_job(self) -> None:
        """Rebobinar (Mach3): olvida el avance del último programa si no hay uno en marcha."""
        job = getattr(self, "job", None)
        if job is not None and not job["activo"]:
            self.job = _job()


# ==================================================================================================
# ====================================== SIMULADOR (SimLink) =======================================
# ==================================================================================================

class SimLink(Link):
    """Simulador en tiempo real: una «máquina» que vive en memoria y se mueve como la de verdad.

    Interpreta el G-code con el mismo intérprete de la app (``Interpreter``), lo pasa por el mismo
    planificador (``plan``: aceleración, velocidad en esquinas) y después avanza los bloques planeados
    al ritmo del reloj, de modo que la posición, la velocidad y los tiempos que se ven en la pantalla
    son los que tendría la máquina real. Imita a grbl: mismos estados, mismos rangos de override,
    retención, M0, jog y referenciado.

    Parámetros:
        machine:   perfil de la máquina (ejes, recorridos, velocidades, aceleraciones).
        speed:     multiplicador del reloj. 1.0 = tiempo real; 10 = diez veces más rápido (útil para
                   probar programas largos). Muy alto (>100) hace saltos grandes por tick y la vista
                   se ve «a los saltos», pero el resultado final es el mismo.
        tick:      cada cuánto avanza la simulación, en segundos. 0.02 s = 50 Hz, más fino que la
                   consulta de la interfaz (200 ms). Más chico gasta más CPU (ojo en PCs viejas);
                   más grande hace que la posición avance a saltos.
        autostart: arrancar el hilo de simulación ya en el constructor (las pruebas lo apagan y llaman
                   a ``step`` a mano para controlar el tiempo).

    Todo el estado se toca con ``self.lock`` tomado: el hilo de simulación y los pedidos de la
    interfaz corren en paralelo.
    """

    physical = False
    name = "simulador"

    def __init__(self, machine: Machine, speed: float = 1.0, tick: float = 0.02, autostart: bool = True):
        self.m = as_controller(machine)               # recibe el programa ya traducido
        self.speed = speed
        self.tick = tick
        self.lock = threading.RLock()
        self.interp = Interpreter(self.m)             # estado modal y orígenes, como el del controlador
        self._pos = dict(self.interp.pos)             # posición física simulada (el intérprete va por delante)
        self.state = "Idle"
        self.homed = False
        self.ov = {"avance": 100, "rapido": 100, "husillo": 100}   # overrides en %
        self.spindle = ("off", 0.0)                   # (sentido "cw"/"ccw"/"off", rpm programadas)
        self.coolant = False
        # Cola de lo que falta ejecutar, en orden: ("block", bloque planeado) para movimientos y
        # ("op", Op) para eventos que no mueven (husillo, refrigerante, G4, M0, cambio de herramienta, fin).
        self.items: deque = deque()
        self.cur = None                               # elemento en ejecución (mismo formato que items) o None
        self.t_in = 0.0                               # segundos transcurridos dentro de ``cur``
        self.v = 0.0                                  # velocidad actual sobre la trayectoria, mm/s
        self.job = _job()
        self.jogging = False
        self.message = ""                             # texto para el operador (p. ej. el de un M0)
        self.alarm = ""
        self._home_left = 0.0                         # segundos que faltan del referenciado simulado
        self._last_line = None                        # línea del programa del último elemento empezado
        self._single_go = False                       # bloque a bloque: permiso para empezar una línea más
        self._run = True                              # False detiene el hilo de simulación
        self._thread = None
        if autostart:
            self.open()

    @property
    def pos(self) -> dict[str, float]:
        """Posición de máquina simulada, mm por eje (la real, no la que ya interpretó el intérprete)."""
        return self._pos

    def open(self) -> None:
        """Arranca el hilo de simulación (daemon: no impide cerrar el programa)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="hope-cnc-sim")
            self._thread.start()

    def close(self) -> None:
        """Pide al hilo de simulación que termine en el próximo tick."""
        self._run = False

    def _loop(self):
        """Hilo de simulación: cada ``tick`` avanza el tiempo real transcurrido (× ``speed``).

        Se mide el tiempo con ``time.monotonic`` y no se supone que el sleep duró exactamente ``tick``:
        en una PC cargada el sleep se alarga y, así, la simulación no se atrasa respecto del reloj.
        """
        last = time.monotonic()
        while self._run:
            time.sleep(self.tick)
            now = time.monotonic()
            self.step((now - last) * self.speed)
            last = now

    # --- ejecución --------------------------------------------------------------------------------
    def _queue(self, ops: list[Op], line_base: int = 0):
        """Planifica ``ops`` y agrega a ``self.items`` los bloques intercalados con sus eventos.

        ``plan`` devuelve los bloques de movimiento y, aparte, la lista ``events`` con pares
        (índice del bloque antes del cual ocurre, Op). Acá se vuelven a mezclar en el orden del
        programa: un M3 entre dos G1 se ejecuta cuando termina el primero, como en la máquina.
        ``line_base`` no se usa (queda por compatibilidad de la firma).
        """
        events: list = []
        pl = plan(ops, self.m, events=events)
        ev = deque(events)
        for i, b in enumerate(pl.blocks):
            while ev and ev[0][0] == i:
                self.items.append(("op", ev.popleft()[1]))
            self.items.append(("block", b))
        while ev:                                     # eventos después del último movimiento (M5, M30…)
            self.items.append(("op", ev.popleft()[1]))

    def step(self, dt: float) -> None:
        """Avanza la simulación ``dt`` segundos de máquina.

        En un mismo paso pueden terminar varios bloques cortos: el tiempo que sobra al terminar uno
        se usa en el siguiente (bucle ``while dt``), así la duración total no depende de ``tick``.
        """
        with self.lock:
            if self.state == "Home":
                self._home_left -= dt
                if self._home_left <= 0:
                    # Referenciado simulado: Z arriba (máximo, como grbl que sube Z primero) y el resto
                    # en el mínimo del recorrido; 0 si el eje no tiene límite configurado.
                    for a, ax in self.m.axes.items():
                        self.pos[a] = (ax.max if ax.max is not None else 0.0) if a == "Z" else \
                            (ax.min if ax.min is not None else 0.0)
                    self.homed, self.state = True, "Idle"
                return
            while dt > 1e-12 and self.state in ("Run", "Jog"):
                if self.cur is None:
                    if not self.items:
                        self._finish()
                        return
                    nxt = self.items[0][1]
                    # Bloque a bloque: al pasar a una línea nueva del programa se retiene, salvo que un
                    # Cycle Start (resume) haya dado permiso para una línea más (_single_go).
                    if self.single and self.job["activo"] and self._last_line is not None \
                            and nxt.line != self._last_line:
                        if not self._single_go:
                            self.state, self.v = "Hold", 0.0     # bloque a bloque: espera Cycle Start
                            return
                        self._single_go = False
                    self._last_line = nxt.line
                    self.cur, self.t_in = self.items.popleft(), 0.0
                kind, it = self.cur
                if self.job["activo"]:
                    # Todo lo anterior a la línea en ejecución ya «se confirmó» (como los ok de grbl).
                    self.job["confirmadas"] = max(self.job["confirmadas"], it.line - 1)
                if kind == "block":
                    # El override estira o acorta el tiempo del bloque: al 50 % cada segundo real avanza
                    # medio segundo del perfil planeado. G0 usa el de rápidos y el resto el de avance.
                    scale = (self.ov["rapido"] if it.kind == "rapid" else self.ov["avance"]) / 100.0
                    self.t_in += dt * scale
                    if self.t_in >= it.time:
                        # Terminó: queda en el punto final exacto (sin error acumulado) y el tiempo
                        # sobrante, pasado de nuevo a segundos reales, sigue con el próximo elemento.
                        dt = (self.t_in - it.time) / scale
                        for i, a in enumerate(self.m.axis_names):
                            self.pos[a] = it.end[i]
                        self.cur = None
                    else:
                        d = distance_at(it, self.t_in)   # mm recorridos dentro del bloque según su perfil
                        # Velocidad instantánea por diferencia finita de 1 ms (para el avance en pantalla).
                        self.v = (distance_at(it, self.t_in + 1e-3) - d) / 1e-3 * scale
                        for i, a in enumerate(self.m.axis_names):
                            self.pos[a] = it.start[i] + it.unit[i] * d   # unit = vector dirección unitario
                        dt = 0.0
                else:
                    dt = self._event(it, dt)

    def _event(self, op: Op, dt: float) -> float:
        """Ejecuta un evento sin movimiento y devuelve el tiempo (s) que sobra para lo siguiente.

        * dwell (G4): consume ``op.value`` segundos; no lo afecta el override, igual que en grbl.
        * spindle / coolant: cambian el estado al instante.
        * pause (M0/M1) y tool (M6): retienen con un mensaje hasta el Cycle Start.
        * end (M2/M30): descarta lo que quede en la cola.
        """
        if op.kind == "dwell":
            self.t_in += dt
            if self.t_in < float(op.value):
                return 0.0
            dt, self.cur = self.t_in - float(op.value), None
            return dt
        self.cur = None
        if op.kind == "spindle":
            self.spindle = op.value
        elif op.kind == "coolant":
            self.coolant = bool(op.value)
        elif op.kind in ("pause", "tool"):
            self.message = str(op.value) if op.kind == "pause" else f"cambiar a la herramienta T{op.value}"
            self.state, self.v = "Hold", 0.0
            return 0.0
        elif op.kind == "end":
            self.items.clear()
        return dt

    def _finish(self):
        """Se vació la cola: el programa (o el jog) terminó y la máquina vuelve a reposo."""
        self.cur, self.v = None, 0.0
        if self.job["activo"]:
            self.job.update(activo=False, terminado=True, confirmadas=self.job["total"])
        self.state = "Idle"
        self.jogging = False

    # --- interfaz ---------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Ver :meth:`Link.snapshot`. ``avance`` se pasa de mm/s a mm/min; el husillo incluye el override."""
        with self.lock:
            return {"estado": self.state, "mpos": dict(self.pos), "wco": {a: self.interp._off(a) for a in self.pos},
                    "avance": round(self.v * 60.0, 1), "husillo": self.spindle[1] * self.ov["husillo"] / 100.0
                    if self.spindle[0] != "off" else 0.0, "husillo_dir": self.spindle[0],
                    "refrigerante": self.coolant, "ov": dict(self.ov), "referenciado": self.homed,
                    "trabajo": dict(self.job), "mensaje": self.message, "alarma": self.alarm,
                    "wcs": self.interp.wcs, "conexion": self.name,
                    "pausa_g4": bool(self.cur and self.cur[0] == "op" and self.cur[1].kind == "dwell"),
                    "avance_prog": round(self.cur[1].v_nominal * 60.0, 1)
                    if self.cur and self.cur[0] == "block" and self.cur[1].kind == "feed" else None,
                    "bloque_a_bloque": self.single}

    def set_machine(self, machine: Machine) -> None:
        """Otra máquina (perfil): nuevos límites, velocidades y orígenes; la posición se conserva."""
        with self.lock:
            self.m = as_controller(machine)
            pos = dict(self._pos)
            self.interp = Interpreter(self.m)
            self._pos = {a: pos.get(a, 0.0) for a in self.m.axis_names}   # ejes nuevos arrancan en 0
            self.interp.pos = dict(self._pos)

    def modal(self) -> str:
        """Códigos modales activos del intérprete del simulador (G90 G21 G54 G17…)."""
        from hope.cnc.gcode import modal_text
        with self.lock:
            return modal_text(self.interp)

    def tool(self) -> int | None:
        """Herramienta activa según el intérprete del simulador."""
        return self.interp.tool

    def offsets(self) -> dict[str, dict[str, float]]:
        """Copia de los orígenes G54…G59 del intérprete más el desplazamiento G92."""
        with self.lock:
            out = {w: dict(v) for w, v in self.interp.offsets.items()}
            out["G92"] = dict(self.interp.g92)
            return out

    def _idle(self, what: str):
        """Lanza BackendError («<what>: la máquina no está en reposo») si el estado no es Idle."""
        if self.state != "Idle":
            raise BackendError(f"{what}: la máquina no está en reposo ({self.state})")

    def _run_text(self, text: str) -> list[Op]:
        """Interpreta ``text`` desde la posición física actual y devuelve las operaciones resultantes.

        Reutiliza el intérprete (conserva estado modal y orígenes, como un controlador real entre un
        MDI y otro) pero vacía su salida anterior. Lanza BackendError con todos los errores juntos.
        """
        it = self.interp
        it.ops, it.diags, it.ended = [], [], False
        it.pos = dict(self._pos)
        it.run(text)
        if any(d.level == "error" for d in it.diags):
            raise BackendError("; ".join(str(d) for d in it.diags if d.level == "error"))
        return it.ops

    def start(self, lines: list[str]) -> None:
        """Ver :meth:`Link.start`. Interpreta y planifica todo el programa de una vez antes de arrancar."""
        with self.lock:
            self._idle("ciclo")
            ops = self._run_text("\n".join(lines))
            self.items.clear()
            self._queue(ops)
            self.job = _job(len(lines))
            self.job.update(activo=True, enviadas=len(lines))   # en el simulador «enviar» es instantáneo
            self.message = ""
            self._last_line, self._single_go = None, False
            self.state = "Run"

    def hold(self) -> None:
        """Retención inmediata (el simulador frena en seco; grbl lo haría con su aceleración)."""
        with self.lock:
            if self.state == "Run":
                self.state, self.v = "Hold", 0.0

    def resume(self) -> None:
        """Sigue desde una retención o un M0; en bloque a bloque habilita una línea más."""
        with self.lock:
            if self.state == "Hold":
                self.message = ""
                self._single_go = self.single
                self.state = "Run"

    def stop(self) -> None:
        """Aborta: vacía la cola, apaga husillo (conserva las rpm programadas) y refrigerante."""
        with self.lock:
            self.items.clear()
            self.cur, self.v = None, 0.0
            if self.job["activo"]:
                self.job.update(activo=False, error="detenido por el operador")
            self.spindle = ("off", self.spindle[1])
            self.coolant = False
            self.state = "Idle"

    def jog(self, moves: dict[str, float], feed: float) -> None:
        """Ver :meth:`Link.jog`. Un jog nuevo reemplaza al que esté en curso (jog continuo).

        Comprueba los recorridos antes de mover (como los soft limits de grbl con ``$J=``: error 15).
        ``feed`` llega en mm/min y se planifica como un G1.
        """
        with self.lock:
            if self.state == "Jog":
                self.jog_cancel()
            self._idle("jog")
            target = dict(self.pos)
            for a, d in moves.items():
                target[a] += d
            ax_bad = [a for a in moves if not self._inside(a, target[a])]
            if ax_bad:
                raise BackendError(f"el jog sale del recorrido en {', '.join(ax_bad)}")
            self.items.clear()
            self._queue([Op("feed", 0, dict(self.pos), target, feed)])
            self.state, self.jogging = "Jog", True

    def _inside(self, a: str, v: float) -> bool:
        """True si la coordenada ``v`` (mm) está dentro del recorrido del eje ``a`` (1 µm de tolerancia)."""
        ax = self.m.axes[a]
        return (ax.min is None or v >= ax.min - 1e-6) and (ax.max is None or v <= ax.max + 1e-6)

    def jog_cancel(self) -> None:
        """Corta el jog donde esté (sin rampa de frenado) y vuelve a Idle."""
        with self.lock:
            if self.state == "Jog":
                self.items.clear()
                self.cur, self.v = None, 0.0
                self.state, self.jogging = "Idle", False

    def home(self) -> None:
        """Referenciado simulado: 1.5 s en estado Home y después cada eje a su final (ver ``step``).

        No espera a que termine (a diferencia de grbl): la interfaz ve el estado Home en el snapshot.
        """
        with self.lock:
            if self.state not in ("Idle", "Alarm"):
                raise BackendError("referenciar: la máquina no está en reposo")
            self.state, self._home_left, self.alarm = "Home", 1.5, ""

    def unlock(self) -> None:
        """Como ``$X``: sale de Alarm sin referenciar."""
        with self.lock:
            if self.state == "Alarm":
                self.state, self.alarm = "Idle", ""

    def mdi(self, line: str) -> None:
        """Ver :meth:`Link.mdi`. Si la línea no mueve nada (M3, G54, M8…) se aplica sin pasar por Run."""
        with self.lock:
            self._idle("MDI")
            ops = self._run_text(line)
            self.items.clear()
            self._queue(ops)
            if self.items and all(k == "op" and it.kind != "dwell" for k, it in self.items):
                while self.items:                          # sin movimiento: se aplica en el acto
                    self._event(self.items.popleft()[1], 0.0)
            if self.items:
                self.state = "Run"

    def override(self, kind: str, action: str) -> None:
        """Ver :meth:`Link.override`. Imita los rangos de grbl: avance/husillo 10–200 %, rápido 25–100 %."""
        with self.lock:
            if (kind, action) not in OVERRIDES:
                raise BackendError(f"ajuste desconocido: {kind} {action}")
            lo, hi = (10, 200) if kind != "rapido" else (25, 100)
            if action in ("100", "50", "25"):              # valores absolutos
                self.ov[kind] = int(action)
            else:                                          # pasos relativos (+10, −1…), recortados al rango
                self.ov[kind] = max(lo, min(hi, self.ov[kind] + int(action)))

    def coolant_toggle(self) -> None:
        """Alterna el refrigerante al instante."""
        with self.lock:
            self.coolant = not self.coolant


# ==================================================================================================
# ============================== GRBL / GRBLHAL / FLUIDNC (GrblLink) ===============================
# ==================================================================================================

# Informe de estado de grbl 1.1 (respuesta a «?»), p. ej.:
#     <Run|MPos:10.000,5.000,-1.000|FS:600,12000|Ov:100,100,100|A:SF|WCO:0.000,0.000,0.000>
# Grupo 1: estado (puede traer subestado: «Hold:0», «Door:1»). Grupo 2: campos «nombre:valor» con «|».
STATUS = re.compile(r"<([^|>]+)\|(.*)>")


class GrblLink(Link):
    """grbl 1.1 / grblHAL / FluidNC por puerto serie o TCP, con dos hilos de fondo.

    PROTOCOLO (por qué se hace así):

    * **Conteo de caracteres.** grbl tiene un búfer de recepción serie de 128 bytes (en el Arduino
      Uno; grblHAL y FluidNC suelen tener 1024 o más). Si se espera el ``ok`` de cada línea antes de
      mandar la siguiente, el planificador del controlador se queda sin bloques en los tramos cortos
      (curvas hechas de segmentos) y la máquina tartamudea. Si se manda sin control, el búfer se
      desborda y se pierden caracteres (líneas cortadas = movimientos equivocados). La solución que
      recomienda grbl es contar: se anotan los bytes de cada línea enviada (``pending``) y se manda
      otra solo si la suma cabe en el búfer; cada ``ok``/``error`` libera la línea más vieja, porque
      grbl responde en orden.
    * **Comandos en tiempo real** (``?``, ``!``, ``~``, Ctrl-X, overrides 0x90…): grbl los saca del
      flujo al recibirlos, no ocupan búfer ni responden ``ok``; se mandan en cualquier momento.
    * **Estado**: el hilo de consulta manda ``?`` cada ``poll`` segundos; la respuesta ``<…>`` la
      procesa el hilo lector.
    * **Comandos sueltos** (``$H``, ``$J=``, ``$#``, MDI): solo con la máquina sin programa; el ``ok``
      se entrega al que espera (``waiter``), junto con las líneas que lleguen antes.

    Parámetros:
        port:      objeto puerto (``write``/``readline``/``close``), ver ``backends.grbl.open_port``.
        machine:   perfil de la máquina; de él se toman los nombres de los ejes, en el orden en que
                   grbl informa las coordenadas.
        rx_buffer: tamaño del búfer de recepción del controlador, en bytes. 128 para grbl 1.1 en Uno
                   (valor por defecto, seguro para todos). grblHAL/FluidNC aceptan 1024 o más y con
                   eso alimentan mejor programas de muchos tramos cortos. Si se pone MÁS grande que el
                   real, el controlador pierde caracteres y ejecuta líneas corruptas: no exagerar.
                   Más chico solo resta fluidez.
        poll:      período de consulta de estado ``?``, en segundos. 0.2 s (5 Hz) es lo que recomienda
                   grbl para la interfaz. Bajarlo a menos de ~0.05 s carga el puerto y el procesador
                   del controlador; subirlo hace que la posición en pantalla se actualice a saltos.
        autostart: abrir (arrancar hilos) ya en el constructor.
    """

    name = "grbl"

    def __init__(self, port, machine: Machine, rx_buffer: int = 128, poll: float = 0.2, autostart: bool = True):
        self.port = port
        self.m = machine
        self.axes = machine.axis_names
        self.rx = int(rx_buffer)
        self.poll = poll
        self.wlock = threading.Lock()                 # una escritura al puerto por vez (ver encabezado)
        self.lock = threading.RLock()                 # estado compartido entre lector, interfaz y comandos
        # Último estado informado por grbl. mpos siempre en coordenadas de máquina (mm).
        self.st = {"estado": "Desconectado", "mpos": {a: 0.0 for a in self.axes}, "avance": 0.0, "husillo": 0.0,
                   "husillo_dir": "off", "refrigerante": False, "ov": {"avance": 100, "rapido": 100, "husillo": 100},
                   "pines": ""}
        self.wco = {a: 0.0 for a in self.axes}       # origen de trabajo activo (WCO = MPos − WPos), mm
        self.homed = False
        self.banner = ""                              # «Grbl 1.1h ['$' for help]» o similar, al conectar/reiniciar
        self.message = ""
        self.alarm = ""
        self.lines: list[str] = []                    # programa en curso, ya limpio (sin comentarios ni espacios)
        self.pending: deque[int] = deque()            # bytes de cada línea enviada que todavía no tuvo ok/error
        self.job = _job()
        # Comando suelto esperando respuesta: {"event": Event, "lines": [respuestas], "error": texto|None}.
        self.waiter: dict | None = None
        self._tokens = 0                              # bloque a bloque: líneas que se pueden mandar todavía
        self.log: deque[str] = deque(maxlen=200)      # últimas respuestas (sin los informes de estado)
        self._run = False                             # False detiene los hilos lector y de consulta
        if autostart:
            self.open()

    @classmethod
    def from_config(cls, cfg: dict, machine: Machine) -> GrblLink:
        """Crea el enlace desde la sección ``controlador`` del perfil.

        Claves: ``puerto`` (obligatorio: ``/dev/ttyUSB0``, ``COM3`` o ``tcp://host:puerto``),
        ``baudios`` (115200, el de grbl 1.1) y ``bufer`` (bytes del búfer de recepción, ver la clase).
        Lanza BackendError si falta el puerto o no se puede abrir.
        """
        if not cfg.get("puerto"):
            raise BackendError("controlador grbl sin «puerto» (p. ej. /dev/ttyUSB0, COM3 o tcp://192.168.1.50:23)")
        return cls(open_port(str(cfg["puerto"]), int(cfg.get("baudios", 115200))), machine,
                   rx_buffer=int(cfg.get("bufer", 128)))

    # --- hilos ------------------------------------------------------------------------------------
    def open(self) -> None:
        """Despierta a grbl y arranca los hilos lector y de consulta (daemon). Idempotente."""
        if self._run:
            return
        self._run = True
        # Dos saltos de línea: cierran cualquier línea a medias que haya quedado en el búfer de grbl
        # (p. ej. de una sesión anterior cortada) y lo «despiertan», como hace el stream.py de grbl.
        self._write(b"\r\n\r\n")
        threading.Thread(target=self._reader, daemon=True, name="hope-cnc-grbl-rx").start()
        threading.Thread(target=self._poller, daemon=True, name="hope-cnc-grbl-poll").start()

    def close(self) -> None:
        """Detiene los hilos y cierra el puerto (un error al cerrar solo se anota en el log)."""
        self._run = False
        try:
            self.port.close()
        except Exception as e:                             # noqa: BLE001
            self.log.append(f"al cerrar: {e}")

    def _write(self, data: bytes) -> None:
        """Escribe bytes crudos al puerto con ``wlock``: líneas y bytes de tiempo real no se mezclan."""
        with self.wlock:
            self.port.write(data)

    def _poller(self):
        """Hilo de consulta: manda ``?`` cada ``poll`` s. Si el puerto falla, marca Desconectado y sale."""
        while self._run:
            try:
                self._write(b"?")
            except Exception as e:                          # noqa: BLE001
                self.alarm = f"conexión perdida: {e}"
                self.st["estado"] = "Desconectado"
                return
            time.sleep(self.poll)

    def _reader(self):
        """Hilo lector: lee línea por línea y se las pasa a :meth:`handle`.

        ``readline`` vuelve vacío al vencer su tiempo de espera (0.2 s en ``open_port``); entonces se
        duerme 2 ms para no girar en vacío y se vuelve a intentar.
        """
        while self._run:
            try:
                raw = self.port.readline()
            except Exception as e:                          # noqa: BLE001
                self.alarm = f"conexión perdida: {e}"
                self.st["estado"] = "Desconectado"
                return
            if not raw:
                time.sleep(0.002)
                continue
            self.handle(raw.decode("ascii", errors="replace").strip())

    # --- protocolo --------------------------------------------------------------------------------
    def handle(self, line: str) -> None:
        """Procesa una línea recibida de grbl (llamado desde el hilo lector; público para las pruebas).

        Tipos de línea:
            ``<…>``         informe de estado → :meth:`_status` (no se guarda en el log: llegan 5/s).
            ``ok``/``error:N``  confirma la línea más vieja: del programa si hay uno, si no del comando.
            ``ALARM:N``     alarma (final de carrera, soft limit, reinicio en movimiento…): aborta todo.
            ``Grbl …``      banner de arranque: el controlador se reinició y vació su búfer.
            ``[MSG:…]``     mensaje para el operador (p. ej. «Caution: Unlocked»).
            otras           ``[G54:…]``, ``$N=…``… son parte de la respuesta a un comando suelto.
        """
        if not line:
            return
        with self.lock:
            if line.startswith("<"):
                self._status(line)
                return
            self.log.append(line)
            if line == "ok" or line.startswith("error:"):
                if self.job["activo"] and self.pending:
                    self.pending.popleft()                 # grbl responde en orden: se libera la más vieja
                    self.job["confirmadas"] += 1
                    if line != "ok":
                        # Una línea rechazada: grbl sigue con las que ya tiene en el búfer, que pueden
                        # depender de ella (p. ej. un G1 sin el G90 que falló). Por seguridad se aborta
                        # con reinicio (Ctrl-X), que descarta el búfer y el planificador.
                        n = self.job["confirmadas"]
                        self.job.update(activo=False, error=f"línea {n} «{self.lines[n - 1]}»: {describe(line)}")
                        self._write(b"\x18")
                        return
                    if self.job["confirmadas"] == self.job["total"]:
                        self.job.update(activo=False, terminado=True)
                    self._fill()                           # se liberó lugar en el búfer: mandar más
                elif self.waiter is not None:
                    self.waiter["error"] = None if line == "ok" else describe(line)
                    self.waiter["event"].set()
            elif line.startswith("ALARM:"):
                self.alarm = describe(line)
                self.st["estado"] = "Alarm"
                if self.job["activo"]:
                    self.job.update(activo=False, error=self.alarm)
                if self.waiter is not None:
                    self.waiter["error"] = self.alarm
                    self.waiter["event"].set()
            elif line.lower().startswith(("grbl", "fluidnc")) or "grbl" in line.lower():
                # Banner: el controlador arrancó o se reinició (Ctrl-X, cable, watchdog). Todo lo que
                # estaba en su búfer se perdió, así que no se esperan más ok de esas líneas.
                self.banner = line
                self.pending.clear()
                if self.job["activo"]:
                    self.job.update(activo=False, error="el controlador se reinició")
            elif line.startswith("[MSG:"):
                self.message = line[5:].rstrip("]").strip()
            elif self.waiter is not None:
                self.waiter["lines"].append(line)

    def _status(self, line: str) -> None:
        """Actualiza ``self.st`` con un informe ``<estado|campo:valor|…>`` (ver STATUS).

        Campos de grbl 1.1 que se usan:
            MPos / WPos  posición de máquina o de trabajo (según ``$10``); se guarda siempre la de
                         máquina: MPos = WPos + WCO.
            WCO          origen de trabajo activo. grbl lo manda solo cada 10–30 informes (para ahorrar
                         ancho de banda), por eso se recuerda el último.
            FS / F       avance real (mm/min) y rpm del husillo; «F» solo si no hay husillo variable.
            Ov           overrides en % (avance, rápido, husillo); también llega solo de vez en cuando.
            A            accesorios encendidos: S = husillo horario, C = antihorario, F = inundación,
                         M = niebla. grbl lo omite si no hay nada prendido, pero siempre lo manda junto
                         con Ov, así que «Ov sin A» significa «todo apagado».
            Pn           entradas activas (X/Y/Z finales de carrera, P sonda, D puerta, H retención…).
        """
        m = STATUS.match(line)
        if not m:
            return
        self.st["estado"] = m.group(1).split(":")[0]       # «Hold:0» → «Hold»
        fields = dict(f.split(":", 1) for f in m.group(2).split("|") if ":" in f)
        vec = lambda s: {a: float(v) for a, v in zip(self.axes, s.split(","))}
        if "WCO" in fields:
            self.wco = vec(fields["WCO"])
        if "MPos" in fields:
            self.st["mpos"] = vec(fields["MPos"])
        elif "WPos" in fields:
            w = vec(fields["WPos"])
            self.st["mpos"] = {a: w[a] + self.wco.get(a, 0.0) for a in w}
        fs = fields.get("FS") or fields.get("F")
        if fs:
            p = fs.split(",")
            self.st["avance"] = float(p[0])
            self.st["husillo"] = float(p[1]) if len(p) > 1 else 0.0
        if "Ov" in fields:
            f, r, s = (int(x) for x in fields["Ov"].split(","))
            self.st["ov"] = {"avance": f, "rapido": r, "husillo": s}
        if "Ov" in fields or "A" in fields:
            acc = fields.get("A", "")
            self.st["husillo_dir"] = "cw" if "S" in acc else "ccw" if "C" in acc else "off"
            self.st["refrigerante"] = "F" in acc or "M" in acc
        self.st["pines"] = fields.get("Pn", "")

    def _fill(self) -> None:
        """Manda líneas del programa mientras quepan en el búfer de grbl (conteo de caracteres).

        Se llama con ``lock`` tomado: al arrancar, tras cada ``ok`` y al reanudar. Cada línea ocupa
        ``len + 1`` bytes (el salto de línea cuenta). En bloque a bloque, además, solo se manda una
        línea por «ficha» (``_tokens``, una por Cycle Start) y solo con el búfer vacío, para que la
        máquina se detenga de verdad entre línea y línea.
        """
        while self.job["enviadas"] < self.job["total"]:
            if self.single:                        # bloque a bloque: una línea por Cycle Start
                if self.pending or self._tokens <= 0:
                    break
                self._tokens -= 1
            s = self.lines[self.job["enviadas"]]
            if sum(self.pending) + len(s) + 1 > self.rx:   # no cabe: esperar el próximo ok
                break
            self._write((s + "\n").encode("ascii"))
            self.pending.append(len(s) + 1)
            self.job["enviadas"] += 1

    def _command(self, line: str, timeout: float = 5.0) -> list[str]:
        """Manda un comando suelto y espera su ``ok``. Devuelve las líneas que llegaron antes del ok.

        line:    comando sin salto de línea (``$H``, ``$#``, ``G0 X0``…).
        timeout: segundos máximos de espera.
        Lanza BackendError si hay un programa en marcha (su ``ok`` se confundiría con los del
        programa), si ya hay otro comando esperando, si no responde a tiempo o si responde con
        ``error:N``/``ALARM:N``. La espera se hace sin ``lock`` para que el lector pueda responder.
        """
        with self.lock:
            if self.job["activo"]:
                raise BackendError("hay un programa en marcha")
            if self.waiter is not None:
                raise BackendError("el controlador está atendiendo otro comando")
            self.waiter = {"event": threading.Event(), "lines": [], "error": None}
            w = self.waiter
            self._write((line + "\n").encode("ascii"))
        try:
            if not w["event"].wait(timeout):
                raise BackendError(f"sin respuesta a «{line}»")
            if w["error"]:
                raise BackendError(w["error"])
            return w["lines"]
        finally:
            with self.lock:
                self.waiter = None

    # --- interfaz ---------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Ver :meth:`Link.snapshot`. Copia ``self.st`` (los dict internos también) más el trabajo.

        ``wcs`` se informa siempre como G54: grbl no dice qué sistema está activo en el informe de
        estado (habría que pedir ``$G``), pero sí su desplazamiento (``wco``), que es lo que importa.
        """
        with self.lock:
            return {**{k: (dict(v) if isinstance(v, dict) else v) for k, v in self.st.items()},
                    "wco": dict(self.wco), "referenciado": self.homed, "trabajo": dict(self.job),
                    "mensaje": self.message, "alarma": self.alarm, "wcs": "G54", "conexion": self.banner or self.name,
                    "pausa_g4": False, "bloque_a_bloque": self.single}

    def offsets(self) -> dict[str, dict[str, float]]:
        """Pide ``$#`` y devuelve G54…G59, G28, G30 y G92 (mm). Las líneas tienen la forma
        ``[G54:0.000,0.000,0.000]``; las demás (``[TLO:…]``, ``[PRB:…]``) se ignoran.
        Lanza BackendError si hay programa en marcha o no responde (ver :meth:`_command`)."""
        out = {}
        for line in self._command("$#"):
            m = re.match(r"\[(G5[4-9]|G28|G30|G92):([^\]]+)\]", line)
            if m:
                out[m.group(1)] = {a: float(v) for a, v in zip(self.axes, m.group(2).split(","))}
        return out

    def start(self, lines: list[str]) -> None:
        """Ver :meth:`Link.start`. Limpia las líneas y arranca el envío con conteo de caracteres.

        Limpieza: se quitan comentarios ``( … )`` y ``; …``, espacios (grbl los ignora y así entran más
        líneas en el búfer) y se pasa a mayúsculas. Una línea que por sí sola no entra en el búfer no
        se podría mandar nunca, así que se rechaza el programa entero antes de mover nada.
        Lanza BackendError si hay otro programa o la máquina no está en Idle (Alarm, Hold, Jog…).
        """
        clean = []
        for ln in lines:
            s = re.sub(r"\(.*?\)|;.*", "", ln).strip().upper().replace(" ", "")
            if s:
                if len(s) + 1 > self.rx:
                    raise BackendError(f"línea más larga que el búfer del controlador: {s[:40]}…")
                clean.append(s)
        with self.lock:
            if self.job["activo"]:
                raise BackendError("ya hay un programa en marcha")
            if self.st["estado"] not in ("Idle",):
                raise BackendError(f"la máquina no está en reposo ({self.st['estado']})")
            self.lines = clean
            self.pending.clear()
            self.job = _job(len(clean))
            self.job["activo"] = bool(clean)
            self.job["terminado"] = not clean
            self.message = ""
            self._tokens = 1                       # bloque a bloque: el Cycle Start que arrancó vale una línea
            self._fill()

    def hold(self) -> None:
        """Feed Hold: byte de tiempo real ``!``. grbl frena con su aceleración y no pierde posición."""
        self._write(b"!")

    def set_machine(self, machine: Machine) -> None:
        """Cambia el perfil (los ejes informados siguen siendo los del perfil con que se abrió)."""
        self.m = machine

    def set_single(self, on: bool) -> None:
        """Ver :meth:`Link.set_single`."""
        with self.lock:
            self.single = bool(on)
            if not self.single and self.job["activo"]:
                self._fill()                       # al apagar bloque a bloque sigue enviando de corrido

    def resume(self) -> None:
        """Cycle Start: byte ``~`` (sale de Hold). En bloque a bloque, además, habilita la línea siguiente."""
        self.message = ""
        self._write(b"~")
        with self.lock:
            if self.single and self.job["activo"]:
                self._tokens += 1
                self._fill()

    def stop(self) -> None:
        """Stop: retención (``!``), espera a que frene y reinicia grbl (Ctrl-X = 0x18).

        El reinicio descarta el búfer y el planificador, que es lo que corta el programa. Si se
        reiniciara en movimiento, grbl no sabría dónde quedó la máquina (ALARM:3, hay que referenciar);
        por eso primero la retención y 0.25 s de espera. Con avances muy altos y poca aceleración la
        máquina puede tardar más en frenar y entonces sí aparece la alarma 3.
        """
        self._write(b"!")
        time.sleep(0.25)                                   # frenar antes de reiniciar: no se pierde la posición
        with self.lock:
            if self.job["activo"]:
                self.job.update(activo=False, error="detenido por el operador")
            self.pending.clear()
        self._write(b"\x18")

    def jog(self, moves: dict[str, float], feed: float) -> None:
        """Jog con ``$J=G91 G21 X… F…`` (incremental, en mm, sin tocar el estado modal del programa).

        grbl comprueba los soft limits y responde ``error:15`` si el jog sale del recorrido (llega
        como BackendError). El ``ok`` llega al aceptar el movimiento, no al terminarlo, así que no
        bloquea durante el jog. Ejes con desplazamiento casi nulo se omiten.
        """
        axes = " ".join(f"{a}{v:.4f}" for a, v in moves.items() if abs(v) > 1e-9)
        if axes:
            self._command(f"$J=G91 G21 {axes} F{feed:.0f}")

    def jog_cancel(self) -> None:
        """Corta el jog con el byte de tiempo real 0x85 (frena con rampa, sin alarma)."""
        self._write(bytes([JOG_CANCEL]))

    def home(self) -> None:
        """Referenciado ``$H``. grbl responde ``ok`` recién al terminar el ciclo: se espera hasta 180 s
        (máquinas grandes y lentas). Lanza BackendError si falla (ALARM:6…9) o no responde."""
        self._command("$H", timeout=180.0)
        self.homed = True

    def unlock(self) -> None:
        """``$X``: quita el bloqueo de alarma sin referenciar (la posición puede no ser confiable)."""
        self._command("$X")
        self.alarm = ""

    def mdi(self, line: str) -> None:
        """Ver :meth:`Link.mdi`. Quita comentarios; espera el ``ok`` hasta 30 s."""
        s = re.sub(r"\(.*?\)|;.*", "", line).strip()
        if s:
            self._command(s, timeout=30.0)

    def override(self, kind: str, action: str) -> None:
        """Manda el byte de tiempo real de OVERRIDES. El valor nuevo se ve en el próximo informe (Ov)."""
        b = OVERRIDES.get((kind, action))
        if b is None:
            raise BackendError(f"ajuste desconocido: {kind} {action}")
        self._write(bytes([b]))

    def coolant_toggle(self) -> None:
        """Alterna el refrigerante por inundación con el byte de tiempo real 0xA0."""
        self._write(bytes([FLOOD_TOGGLE]))


# ==================================================================================================
# ======================== MARLIN / SMOOTHIEWARE / REPRAPFIRMWARE (OkLink) =========================
# ==================================================================================================

# Posición en la respuesta a M114, p. ej. (Marlin):
#     X:10.00 Y:5.00 Z:-1.00 E:0.00 Count X:800 Y:400 Z:-400
# Captura (eje, valor en mm). Lo que viene después de «Count» son pasos de motor y se descarta antes.
POS = re.compile(r"\b([XYZABC]):\s*(-?\d+(?:\.\d+)?)")


class OkLink(Link):
    """Marlin, Smoothieware, RepRapFirmware y cualquier controlador que responda «ok» por línea (MPCNC,
    Lowrider, routers con placas de impresora 3D). Envía línea por línea y pide M114 para la posición.

    Diferencias con grbl (limitaciones del firmware, no de la app): no hay retención inmediata (Feed Hold
    deja de enviar y la máquina termina lo que tiene en su búfer), Stop usa M410 (necesita
    EMERGENCY_PARSER en Marlin), M0 espera M108 para seguir, y el avance se ajusta con M220.

    PROTOCOLO (por qué se hace así):

    * **Una línea por vez.** Estos firmwares no ofrecen un conteo de caracteres confiable: el tamaño
      del búfer serie varía según la placa y la compilación, y Marlin no separa comandos de tiempo
      real. Lo seguro es lo que hacen los programas de impresión 3D: mandar una línea y esperar su
      ``ok``. El planificador del firmware sigue lleno porque el ``ok`` de un movimiento llega cuando
      entra en la cola, no cuando termina.
    * **Un solo «turno».** Como la respuesta ``ok`` no dice a qué línea corresponde, nunca hay más de
      una línea en vuelo; ``inflight`` recuerda de quién es: ``"job"`` (programa), ``"poll"`` (M114 del
      hilo de consulta), ``"cmd"`` (comando suelto con ``waiter``) o ``"extra"`` (M220/M8/M9 pedidos en
      marcha, encolados en ``extra`` e intercalados entre líneas del programa).
    * **M108 / M410 fuera de turno.** El firmware está bloqueado ejecutando M0 o con la cola llena, y
      un comando normal no se leería hasta que se libere. Marlin compilado con EMERGENCY_PARSER los
      reconoce apenas llegan por el puerto (como los bytes de tiempo real de grbl); sin esa opción
      M108 no despierta un M0 y M410 recién actúa cuando se vacía la cola.

    Parámetros:
        port:      objeto puerto (``write``/``readline``/``close``), ver ``backends.grbl.open_port``.
        machine:   perfil de la máquina (nombres de ejes).
        poll:      período de consulta de posición (M114), en segundos. 0.5 s por defecto: cada M114
                   ocupa el turno y demora la línea siguiente del programa, así que más seguido hace
                   más lento el programa (y tartamudeos en tramos cortos); más espaciado hace que la
                   posición en pantalla se actualice a saltos.
        autostart: abrir (arrancar hilos) ya en el constructor.
    """

    name = "ok"

    def __init__(self, port, machine: Machine, poll: float = 0.5, autostart: bool = True):
        self.port = port
        self.m = machine
        self.axes = machine.axis_names
        self.poll = poll
        self.wlock = threading.Lock()                 # una escritura al puerto por vez
        self.lock = threading.RLock()                 # estado compartido entre lector, consulta e interfaz
        self.pos = {a: 0.0 for a in self.axes}        # última posición de M114, mm (ver nota en handle)
        self.state = "Desconectado"                   # pasa a Idle con la primera línea que llegue
        self.homed = False
        self.ov = {"avance": 100, "rapido": 100, "husillo": 100}   # solo «avance» se manda (M220)
        self.spindle_on = False                       # deducido de los M3/M4/M5 mandados por MDI
        self.coolant = False
        self.message = ""
        self.alarm = ""
        self.banner = ""
        self.lines: list[str] = []
        self.job = _job()
        self.inflight: str | None = None          # "job" | "poll" | "cmd" | "extra": qué espera el próximo ok
        self.extra: deque[str] = deque()          # M220, M8/M9…: van en el próximo hueco, sin romper el turno
        self.waiter: dict | None = None               # comando suelto: {"event": Event, "error": texto|None}
        self.held = False                             # retenido: no se mandan más líneas del programa
        self.log: deque[str] = deque(maxlen=200)      # últimas respuestas (sin las de M114)
        self._run = False
        if autostart:
            self.open()

    @classmethod
    def from_config(cls, cfg: dict, machine: Machine) -> OkLink:
        """Crea el enlace desde la sección ``controlador`` del perfil: ``puerto`` (obligatorio) y
        ``baudios`` (115200 por defecto; Marlin suele usar 115200 o 250000). Lanza BackendError si
        falta el puerto o no se puede abrir."""
        if not cfg.get("puerto"):
            raise BackendError("controlador sin «puerto» (p. ej. /dev/ttyUSB0, COM3 o tcp://192.168.1.50:23)")
        return cls(open_port(str(cfg["puerto"]), int(cfg.get("baudios", 115200))), machine)

    # --- hilos ------------------------------------------------------------------------------------
    def open(self) -> None:
        """Arranca los hilos lector y de consulta (daemon). Idempotente."""
        if self._run:
            return
        self._run = True
        threading.Thread(target=self._reader, daemon=True, name="hope-cnc-ok-rx").start()
        threading.Thread(target=self._poller, daemon=True, name="hope-cnc-ok-poll").start()

    def close(self) -> None:
        """Detiene los hilos y cierra el puerto (un error al cerrar solo se anota en el log)."""
        self._run = False
        try:
            self.port.close()
        except Exception as e:                             # noqa: BLE001
            self.log.append(f"al cerrar: {e}")

    def set_machine(self, machine: Machine) -> None:
        """Cambia el perfil (los ejes informados siguen siendo los del perfil con que se abrió)."""
        self.m = machine

    def _write(self, line: str) -> None:
        """Escribe una línea (agrega el salto) con ``wlock``. Caracteres no ASCII se cambian por «?»."""
        with self.wlock:
            self.port.write((line + "\n").encode("ascii", errors="replace"))

    def _poller(self):
        """Hilo de consulta: cada ``poll`` s, si el turno está libre, manda lo encolado o un M114.

        Sirve también de «bomba»: si un ``extra`` quedó esperando porque el turno estaba ocupado, acá
        se manda. No pide M114 mientras un comando suelto espera turno (le daría prioridad al M114).
        """
        while self._run:
            time.sleep(self.poll)
            with self.lock:
                if self.inflight is None and self.extra:
                    self._pump()
                elif self.inflight is None and self.waiter is None:
                    self.inflight = "poll"
                    try:
                        self._write("M114")
                    except Exception as e:                 # noqa: BLE001
                        self.alarm, self.state = f"conexión perdida: {e}", "Desconectado"
                        return

    def _reader(self):
        """Hilo lector: lee línea por línea y se las pasa a :meth:`handle` (igual que en GrblLink)."""
        while self._run:
            try:
                raw = self.port.readline()
            except Exception as e:                         # noqa: BLE001
                self.alarm, self.state = f"conexión perdida: {e}", "Desconectado"
                return
            if not raw:
                time.sleep(0.002)
                continue
            self.handle(raw.decode("ascii", errors="replace").strip())

    # --- protocolo --------------------------------------------------------------------------------
    def handle(self, line: str) -> None:
        """Procesa una línea recibida (llamado desde el hilo lector; público para las pruebas).

        Tipos de línea:
            ``X:… Y:… Z:…``   respuesta a M114 → ``self.pos``. Ojo: Marlin informa la posición de
                              destino de lo último encolado, que puede ir adelante de la real.
            ``ok…``           cierra el turno (``inflight``) y manda lo siguiente (``_pump``).
            ``Error…``/``!!`` línea rechazada o firmware detenido (Smoothie «!!», Marlin «kill()»).
            ``start``/banner  el firmware arrancó o se reinició: el programa en curso se perdió.
            ``echo:…``        mensajes para el operador; ``echo:busy: processing`` (aviso de Marlin
                              de que sigue trabajando en un comando largo) se ignora.
        """
        if not line:
            return
        low = line.lower()
        with self.lock:
            if self.state == "Desconectado":
                self.state = "Idle"                        # llegó algo: el controlador está vivo
            if "x:" in low and "y:" in low and not low.startswith("ok"):
                for a, v in POS.findall(re.split(r"count", line, flags=re.IGNORECASE)[0]):   # «Count X:…» son pasos
                    if a in self.pos:
                        self.pos[a] = float(v)
                return
            self.log.append(line)
            if low.startswith("ok"):                       # «ok», «ok T:…», «ok N12» según firmware
                kind, self.inflight = self.inflight, None
                if kind == "job":
                    self.job["confirmadas"] += 1
                    if self.job["confirmadas"] >= self.job["total"]:
                        self.job.update(activo=False, terminado=True)
                        self.state = "Idle"
                    self._next()
                elif kind == "cmd" and self.waiter is not None:
                    self.waiter["event"].set()
                self._pump()                               # tras un M114 o un M220 sigue el programa
            elif low.startswith(("error", "!!")):
                msg = line
                if self.inflight == "job":
                    # Error en una línea del programa: se corta el envío (el firmware no sigue solo).
                    self.inflight = None
                    self.job.update(activo=False, error=f"línea {self.job['confirmadas'] + 1}: {msg}")
                    self.state = "Idle"
                elif self.inflight in ("poll", "extra"):
                    # Falló un M114 o un M220/M8: no es grave, se avisa y el programa sigue.
                    self.inflight = None
                    self.message = msg
                    self._pump()
                elif self.waiter is not None:
                    self.waiter["error"] = msg
                    self.waiter["event"].set()
                    self.inflight = None
                # «!!» o «kill»: el firmware quedó detenido y pide M999 (ver unlock); se muestra como alarma.
                self.alarm = msg if low.startswith("!!") or "kill" in low else self.alarm
            elif low.startswith(("start", "marlin", "smoothie", "reprapfirmware", "fw_name")):
                self.banner = line
                if self.job["activo"]:
                    self.job.update(activo=False, error="el controlador se reinició")
            elif low.startswith("echo:") and "busy" not in low:
                self.message = line[5:].strip()

    def _pump(self) -> None:
        """Con el candado tomado: si nada espera un ok, manda primero los comandos encolados y después el programa."""
        if self.inflight is not None:
            return
        if self.extra and not self.held:
            self.inflight = "extra"
            self._write(self.extra.popleft())
            return
        self._next()

    def _send_soon(self, line: str) -> None:
        """Encola ``line`` en ``extra`` para el próximo turno libre (lo manda ya si el turno está libre)."""
        with self.lock:
            self.extra.append(line)
            self._pump()

    def _next(self) -> None:
        """Envía la próxima línea del programa (una por vez: el controlador confirma cada una).

        Se llama con ``lock`` tomado. Si la línea es M0/M1 (pausa del programa), se marca la retención
        antes de mandarla: el firmware no responde ``ok`` hasta que el operador siga (M108, ver
        :meth:`resume`), y el texto que sigue al M0 se muestra como mensaje.
        """
        if not self.job["activo"] or self.held or self.inflight is not None:
            return
        i = self.job["enviadas"]
        if i >= self.job["total"]:
            return
        s = self.lines[i]
        self.job["enviadas"] += 1
        self.inflight = "job"
        # «M1 » con espacio para no confundir M1 con M104, M106… (M1 a secas al final no se detecta).
        if s.upper().startswith("M0") or s.upper().startswith("M1 "):
            self.held, self.state, self.message = True, "Hold", s[2:].strip() or "pausa"
        self._write(s)

    def _command(self, line: str, timeout: float = 10.0) -> None:
        """Manda un comando suelto en el próximo turno libre y espera su ``ok``.

        line:    comando sin salto de línea.
        timeout: segundos máximos de espera del ``ok`` (una vez enviado).
        Primero espera hasta 2 s a que se libere el turno (normalmente lo ocupa un M114 del hilo de
        consulta, que responde enseguida). Lanza BackendError si hay programa en marcha, si el turno
        no se libera («el controlador no responde»), si no llega el ok a tiempo o si llega un error.
        """
        with self.lock:
            if self.job["activo"]:
                raise BackendError("hay un programa en marcha")
            self.waiter = {"event": threading.Event(), "error": None}
            w = self.waiter
        deadline = time.monotonic() + 2.0
        while True:                                        # esperar a que se libere el turno (poll)
            with self.lock:
                if self.inflight is None:
                    self.inflight = "cmd"
                    self._write(line)
                    break
            if time.monotonic() > deadline:
                with self.lock:
                    self.waiter = None
                raise BackendError("el controlador no responde")
            time.sleep(0.01)
        try:
            if not w["event"].wait(timeout):
                raise BackendError(f"sin respuesta a «{line}»")
            if w["error"]:
                raise BackendError(w["error"])
        finally:
            with self.lock:
                self.waiter = None

    # --- interfaz ---------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Ver :meth:`Link.snapshot`. Estos firmwares no informan avance real, rpm ni origen de trabajo
        en la consulta de posición: esos campos van en 0 y la posición es la de M114."""
        with self.lock:
            return {"estado": self.state, "mpos": dict(self.pos), "wco": {a: 0.0 for a in self.pos}, "avance": 0.0,
                    "husillo": 0.0, "husillo_dir": "cw" if self.spindle_on else "off", "refrigerante": self.coolant,
                    "ov": dict(self.ov), "referenciado": self.homed, "trabajo": dict(self.job),
                    "mensaje": self.message, "alarma": self.alarm, "wcs": "G54", "conexion": self.banner or self.name,
                    "pausa_g4": False, "bloque_a_bloque": self.single}

    def start(self, lines: list[str]) -> None:
        """Ver :meth:`Link.start`. Quita comentarios (si la línea era solo comentario se manda tal cual
        y vacía se descarta) y manda la primera línea; el resto sale de a una con cada ``ok``.
        Lanza BackendError si ya hay un programa en marcha."""
        clean = [re.sub(r"\(.*?\)|;.*", "", ln).strip() or ln.strip() for ln in lines]
        clean = [c for c in clean if c]
        with self.lock:
            if self.job["activo"]:
                raise BackendError("ya hay un programa en marcha")
            self.lines, self.job, self.held = clean, _job(len(clean)), False
            self.job["activo"] = bool(clean)
            self.job["terminado"] = not clean
            self.state = "Run" if clean else "Idle"
            self.message = ""
            self._next()

    def hold(self) -> None:
        """Feed Hold «blando»: deja de mandar líneas. La máquina termina lo que ya tiene en su cola
        (Marlin no tiene retención inmediata por serie), así que puede seguir moviéndose un rato."""
        with self.lock:
            if self.job["activo"]:
                self.held, self.state = True, "Hold"

    def resume(self) -> None:
        """Cycle Start: sigue mandando líneas. Si la retención fue un M0/M1 (hay mensaje y la línea M0
        sigue esperando su ok), manda M108 fuera de turno para que el firmware salga de la pausa."""
        with self.lock:
            was_m0 = self.message and self.held
            self.held, self.message = False, ""
            if self.job["activo"]:
                self.state = "Run"
                if was_m0 and self.inflight == "job":
                    self._write("M108")                    # Marlin: seguir después de M0
                if self.single:
                    self.held = False
                self._next()

    def stop(self) -> None:
        """Stop: corta el envío y manda, fuera de turno, M410 (parada rápida: vacía la cola de
        movimientos) y M5 (husillo apagado). Sin EMERGENCY_PARSER, Marlin recién lee M410 cuando se
        libera lugar en su búfer, y la máquina puede perder la posición (conviene referenciar)."""
        with self.lock:
            if self.job["activo"]:
                self.job.update(activo=False, error="detenido por el operador")
            self.held, self.state = False, "Idle"
            self._write("M410")                            # parada rápida (Marlin con EMERGENCY_PARSER)
            self._write("M5")

    def jog(self, moves: dict[str, float], feed: float) -> None:
        """Jog con ``G91`` + ``G1 X… F…`` + ``G90`` (relativo y vuelta a absoluto, aunque falle el G1).

        No hay ``$J`` ni cancelación de jog en estos firmwares: cada paso es un G1 completo. El firmware
        no controla los recorridos salvo que tenga soft endstops activados.
        """
        axes = " ".join(f"{a}{v:.4f}" for a, v in moves.items() if abs(v) > 1e-9)
        if axes:
            self._command("G91")
            try:
                self._command(f"G1 {axes} F{feed:.0f}", timeout=60.0)
            finally:
                self._command("G90")

    def home(self) -> None:
        """Referenciado ``G28``. El firmware responde ``ok`` al terminar: se espera hasta 180 s."""
        self._command("G28", timeout=180.0)
        self.homed = True

    def unlock(self) -> None:
        """``M999``: saca al firmware del estado detenido (tras un «!!» o un kill) sin reiniciarlo."""
        self._command("M999")
        self.alarm = ""

    def mdi(self, line: str) -> None:
        """Ver :meth:`Link.mdi`. Espera el ok hasta 60 s y recuerda si el husillo quedó prendido
        (M3/M4/M5), porque el firmware no lo informa."""
        s = re.sub(r"\(.*?\)|;.*", "", line).strip()
        if s:
            self._command(s, timeout=60.0)
            up = s.upper()
            if up.startswith(("M3", "M4")):
                self.spindle_on = True
            elif up.startswith("M5"):
                self.spindle_on = False

    def override(self, kind: str, action: str) -> None:
        """Ajuste del avance con ``M220 S<porcentaje>`` (10–300 %, el rango que acepta Marlin sin
        sorpresas). Rápidos y husillo no tienen comando equivalente: lanza BackendError."""
        if kind != "avance":
            raise BackendError("este controlador solo ajusta el avance (M220)")
        pct = 100 if action == "100" else max(10, min(300, self.ov["avance"] + int(action)))
        self.ov["avance"] = pct
        self._send_soon(f"M220 S{pct}")                    # en el próximo hueco: nunca dos líneas en vuelo

    def coolant_toggle(self) -> None:
        """Alterna M8/M9 (en Marlin requiere COOLANT_FLOOD); se manda en el próximo hueco del programa."""
        self.coolant = not self.coolant
        self._send_soon("M8" if self.coolant else "M9")


# ==================================================================================================
# ============================================ FÁBRICA =============================================
# ==================================================================================================

def make_link(machine: Machine, cfg: dict | None = None) -> Link:
    """Crea el enlace en vivo que corresponde al controlador del perfil.

    machine: perfil de la máquina.
    cfg:     sección ``controlador`` (si es None se usa ``machine.controller``). Clave ``tipo``:
             ``sim`` (por defecto; ``velocidad`` = multiplicador del reloj), ``grbl``/``grblhal``/
             ``fluidnc`` (ver GrblLink.from_config) u ``ok``/``marlin``/``smoothie``/``reprap``/``rrf``
             (ver OkLink.from_config).
    Lanza BackendError si el tipo no se maneja en vivo (p. ej. LinuxCNC, que va por lotes) o si no se
    puede abrir el puerto.
    """
    cfg = cfg if cfg is not None else machine.controller
    kind = str(cfg.get("tipo", "sim")).lower()
    if kind == "sim":
        return SimLink(machine, speed=float(cfg.get("velocidad", 1.0)))
    if kind in ("grbl", "grblhal", "fluidnc"):
        return GrblLink.from_config(cfg, machine)
    if kind in ("ok", "marlin", "smoothie", "smoothieware", "reprap", "rrf"):
        return OkLink.from_config(cfg, machine)
    raise BackendError(f"la app todavía no maneja el controlador «{kind}» en vivo (sim, grbl u ok); "
                       "para LinuxCNC usar «python -m hope.cnc enviar»")
