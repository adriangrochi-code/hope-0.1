"""Sesión de la app Hope CNC: el «tablero» de Mach3 por detrás de la pantalla.

Une la máquina, la conexión en vivo (:mod:`hope.cnc.links`), el programa cargado (revisado y
traducido) y el guardián. Flujo de uso, como en Mach3:

1. Al abrir, la máquina está en **RESET** (deshabilitada): el botón parpadea. Pulsarlo habilita la
   máquina para el operador (jog, MDI, referenciar, ceros). En una máquina real eso es
   ``hardware.control`` con modo manual: el guardián pide aprobación y el operador la da en la
   misma pantalla (queda en la auditoría).
2. **Cargar G-code** → se revisa (errores, límites, tiempo) y se dibuja la trayectoria.
3. **Cycle Start** → se vuelve a revisar con los orígenes reales del controlador y, en una máquina
   real, el guardián pide aprobación de ESE programa (hash + envolvente). Durante la marcha:
   Feed Hold, Stop y los ajustes de avance/husillo.

Volver a pulsar RESET (o Esc) detiene todo y deshabilita la máquina.

Qué es
------
:class:`CncSession` es el estado de UNA pantalla de TISMA CNC (el reemplazo moderno de Mach3): qué
máquina está elegida, si está habilitada, qué programa hay cargado, qué aprobación espera al
operador, el registro de mensajes, el historial de MDI, los botones con LED (Single BLK, Block
Delete, M1), «Set Next Line» / «Run From Here», los archivos recientes y los perfiles tipo slicer
(máquina + material + consumible + proceso). Cada botón de la pantalla es un método de esta clase.

Cómo encaja
-----------
* **Lo usa**: :mod:`hope.cnc.app.server` (servidor HTTP de la pantalla). Cada pedido del
  navegador llama a un método (``reset``, ``load``, ``cycle_start``, ``jog``, ``mdi``…) y
  devuelve su ``dict`` como JSON; la pantalla pide ``state()`` varias veces por segundo.
  Los errores para el operador salen como :class:`SessionError` (el servidor los muestra tal cual).
* **Usa**:

  - :mod:`hope.cnc.links` (``Link``): la conexión con el controlador (simulador, grbl, Marlin…):
    ``start``, ``hold``, ``resume``, ``stop``, ``jog``, ``mdi``, ``snapshot`` (estado en vivo).
  - :mod:`hope.cnc.gcode`: interpreta el G-code (``interpret``) y lleva el estado modal (``Interpreter``).
  - :mod:`hope.cnc.planner` y :mod:`hope.cnc.check`: plan de movimiento (tiempos) y revisión
    (errores, límites, envolvente).
  - :mod:`hope.cnc.dialect`: traduce el programa al dialecto del controlador (``emit``).
  - :mod:`hope.cnc.profiles`: perfiles tipo slicer (importado tarde, solo si están activos).
  - ``guardian`` (opcional): pide aprobación humana antes de mover una máquina real.

Convenciones
------------
* Unidades: mm, mm/min, segundos, RPM (como el resto de :mod:`hope.cnc`).
* «Líneas originales» = líneas del archivo G-code del usuario (1, 2, 3…; 0 = antes de empezar).
  «Líneas emitidas» = lo que se envía al controlador, ya traducido (``emitidas[0]`` es el
  encabezado ``G21 G90 …``). Cada emitida guarda de qué línea original salió; un arco o un círculo
  pueden salir en varias emitidas. «Líneas enviadas» = lo que realmente se mandó en este Cycle
  Start (puede tener preparación «Run From Here» al principio); ``run_map`` las vuelve a llevar a
  las emitidas para mostrar progreso y tiempo restante.
* Seguridad: con una conexión física (``link.physical``) NADA se mueve sin el guardián, y el
  guardián pide que una persona apruebe en la pantalla (modo manual al pulsar RESET; cada programa
  —hash del texto + envolvente— al pulsar Cycle Start). En el simulador no se pide aprobación.
* Concurrencia: el servidor HTTP atiende pedidos en hilos; los métodos que cambian varias cosas
  juntas (cargar, arrancar, aprobar, perfiles) toman ``self.lock`` (``RLock``, reentrante: por eso
  ``cycle_start`` puede llamar a ``load`` adentro del lock). Los botones de un solo paso (Feed Hold,
  Stop, jog) NO lo toman a propósito: un Feed Hold no debe esperar a que termine una carga.
* Textos para el usuario y claves de los ``dict`` devueltos en español (los lee la pantalla).

Ejemplo de uso
--------------
::

    from hope.cnc.links import SimLink
    from hope.cnc.session import CncSession

    s = CncSession(maquina, SimLink(maquina))        # simulador: no pide aprobación
    s.reset()                                        # habilita (en una real devuelve {"aprobar": …})
    s.load("tapa.nc", open("tapa.nc").read())        # revisa, planifica y traduce
    s.set_next_line(120, from_here=True)             # opcional: «Run From Here» desde la línea 120
    s.cycle_start()                                  # arranca
    s.state()["programa"]["tiempo_restante"]         # segundos que faltan
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from hope.cnc.backends.base import BackendError
from hope.cnc.check import check_program, report_text
from hope.cnc.dialect import HEADER, HEADERS, dialect_for, emit
from hope.cnc.gcode import Interpreter, Program, interpret, linearize, modal_text
from hope.cnc.links import Link
from hope.cnc.machine import WCS, Machine, as_controller
from hope.cnc.planner import plan

# ==================================================================================================
# ========================================= CONFIGURACIÓN ==========================================
# ==================================================================================================

# Máximo de puntos de la trayectoria que se mandan a la vista 3D (puntos, no líneas: un arco se
# divide en varios). Pasado ese número se corta el dibujo y se avisa con «recortado». Más alto = se
# ve el programa entero aunque sea enorme, pero el JSON crece (~40 bytes por punto) y un navegador
# viejo (Firefox 52 en una PC con XP) se traba al dibujarlo. 250 000 ≈ 10 MB, todavía manejable.
MAX_PATH_POINTS = 250_000

# Nombre de la marca que se muestra en la pantalla y en los mensajes.
BRAND = "TISMA CNC"

# Pasos de jog incremental que ofrece la pantalla (mm), como el «Step» de Mach3. El paso elegido
# llega a ``CncSession.jog`` como ``step``; sin paso (None) el jog es continuo mientras se mantiene
# apretado el botón. Agregar valores solo agrega botones; 10 mm ya es mucho para tocar sin mirar.
JOG_STEPS = (0.001, 0.01, 0.1, 1.0, 10.0)


# Error para el operador: el mensaje (en español) se muestra tal cual en la pantalla. Se lanza cuando
# un botón no se puede ejecutar ahora (máquina en RESET, programa en marcha, sin programa, línea MDI
# con errores, aprobación denegada…). No indica un fallo del programa sino una acción que no
# corresponde en el estado actual. Hereda de ValueError para que el servidor la trate como pedido
# inválido (no como fallo interno).
class SessionError(ValueError):
    pass


# ==================================================================================================
# ======================================= TIEMPOS ACUMULADOS =======================================
# ==================================================================================================

def _cumulative_times(prog, pl, emitted: list, origins: list[int]) -> list[float]:
    """Tiempo acumulado (s) por líneas confirmadas: ``cum[k]`` = segundos que llevan las primeras k
    líneas emitidas (``cum[0]`` = 0, ``len(cum) = len(emitted) + 1``). Da el tiempo restante y la barra de progreso en vivo.

    El tiempo de movimiento de cada operación (del plan) se reparte entre las líneas de movimiento que
    emitió (un círculo completo sale en varios G1); las pausas G4 suman su duración en su línea.

    Parámetros:
        prog: ``Program`` interpretado (``prog.ops`` = operaciones).
        pl: plan de :func:`hope.cnc.planner.plan`; ``pl.time_by_op[k]`` = segundos de movimiento de la
            operación k (con aceleraciones).
        emitted: líneas emitidas ``[(texto, línea_original), …]`` (la 0 es el encabezado).
        origins: para cada emitida, índice de la operación que la produjo (-1 = encabezado), como lo
            anota :func:`hope.cnc.dialect.emit`.

    Devuelve la lista ``cum`` (s). Por qué así: el controlador confirma LÍNEAS («ok» de grbl), no
    operaciones; con esta tabla el progreso es una simple consulta ``cum[confirmadas]`` en cada
    ``state()``, sin recorrer el programa (con 100 000 líneas importa en una PC vieja)."""
    cum = [0.0]
    if not emitted:
        return cum
    # Primera pasada: cuántas líneas de MOVIMIENTO emitió cada operación, para repartir su tiempo.
    moves_per_op: dict[int, int] = {}
    for (t, _), k in zip(emitted, origins):
        if t[:2] in ("G0", "G1", "G2", "G3") or t.startswith(("G18 G", "G19 G")):    # G38.2 entra por «G3»
            moves_per_op[k] = moves_per_op.get(k, 0) + 1
    ops = prog.ops
    acc = 0.0
    # Segunda pasada: sumar. Las líneas que no son movimiento ni G4 (M3, F, G10…) no suman tiempo.
    for (t, _), k in zip(emitted, origins):       # cum[k + 1] = hasta la línea k inclusive (0 = encabezado)
        n = moves_per_op.get(k)
        if n:
            acc += pl.time_by_op.get(k, 0.0) / n
        elif k >= 0 and ops[k].kind == "dwell" and t.startswith("G4"):
            acc += float(ops[k].value)
        cum.append(acc)
    return cum


# ==================================================================================================
# ============================================ SESIÓN ==============================================
# ==================================================================================================

class CncSession:
    """Estado de una pantalla de TISMA CNC y la lógica de cada botón (ver el encabezado del módulo).

    Atributos principales (los demás se explican en ``__init__``):
        m: máquina EFECTIVA en uso (con perfiles: máquina + material + consumible + proceso).
        link: conexión con el controlador (``Link``).
        enabled: False = RESET (no se puede mover nada a mano ni arrancar programas).
        program: programa cargado (dict, ver :meth:`load`) o None.
        pending: aprobación del guardián que espera al operador, o None.
        lock: ``RLock`` que serializa las operaciones compuestas (ver «Concurrencia» en el módulo).
    """

    def __init__(self, machine: Machine, link: Link, guardian=None, machine_path: str = "",
                 recent_dir: str | None = None, profiles=None, selection: dict | None = None, link_factory=None):
        """``profiles`` (ProfileStore) y ``selection`` activan los perfiles tipo slicer: la máquina efectiva
        sale de máquina + material + consumible + proceso. ``link_factory(machine)`` crea la conexión
        cuando se elige otra máquina con otro controlador.

        Parámetros:
            machine: máquina configurada (si hay perfiles, se reemplaza por la efectiva).
            link: conexión ya abierta con el controlador (o el simulador).
            guardian: guardián de seguridad (``guardian.core``). Obligatorio si ``link.physical``: sin
                él, :meth:`_guard` se niega a mover la máquina.
            machine_path: archivo de configuración de la máquina (solo se muestra en :meth:`config`).
            recent_dir: carpeta de «recientes» (lista + copia de cada archivo). None = solo en memoria.
            profiles: ``ProfileStore`` de :mod:`hope.cnc.profiles`, o None (perfiles desactivados).
            selection: selección guardada ``{"maquina", "material", "consumible", "proceso", "espesor"}``;
                se corrige con ``fix_selection`` (perfiles borrados o incompatibles).
            link_factory: ``f(machine) -> Link`` para cambiar de controlador al elegir otra máquina.
        """
        self.m = machine
        self.link = link
        self.profiles = profiles
        self.link_factory = link_factory
        self.selection: dict | None = None
        self.profile_summary: dict | None = None
        if profiles is not None:
            # Importación tardía: los perfiles son opcionales y no se cargan si no se usan.
            from hope.cnc.profiles import effective, fix_selection
            self.selection = fix_selection(profiles, selection or {})
            self.m, self.profile_summary = effective(profiles, self.selection)
        self.guardian = guardian
        self.machine_path = machine_path
        self.id = secrets.token_hex(6)    # identifica esta sesión en las solicitudes al guardián (auditoría)
        self.enabled = False              # arranca en RESET, como Mach3: habilitar es una decisión del operador
        self.manual_grant = None          # aprobación del modo manual (máquina real)
        self.program = None               # dict con el programa cargado
        self.pending = None               # aprobación que espera al operador {"id", "tipo", "resumen"}
        self.started_at = None            # time.monotonic() del último Cycle Start (para «transcurrido»)
        self.paused_time = 0.0
        self.log: deque[tuple[float, str]] = deque(maxlen=300)    # (hora epoch, mensaje); la pantalla ve 40
        self.mdi_history: deque[str] = deque(maxlen=30)
        self.lock = threading.RLock()
        self.mdi_interp = Interpreter(as_controller(self.m))    # estado modal de MDI (G90/G91, unidades…)
        # botones de Mach3 con LED: Single BLK, Block Delete, M1 Optional Stop
        self.opts = {"bloque_a_bloque": False, "borrado_bloque": False, "parada_opcional": False}
        self.next_line = 0                # «Set Next Line» / «Run From Here» (0 = desde el principio)
        self.from_here = False            # True = «Run From Here»: preparar (subir Z, ir, husillo, bajar)
        self.run_map: list[int] | None = None   # línea enviada → índice en las emitidas del programa
        self.recent_dir = Path(recent_dir) if recent_dir else None
        self.recent: list[dict] = self._read_recent()
        self._recent_mem: dict[str, str] = {}   # sha → texto, cuando no hay carpeta de recientes
        self.say(f"{BRAND} listo. Pulsá RESET para habilitar la máquina.")

    # --- utilidades -------------------------------------------------------------------------------
    def say(self, msg: str) -> None:
        """Agrega un mensaje al registro que muestra la pantalla (con la hora actual, epoch en s)."""
        self.log.append((time.time(), msg))

    def _require_enabled(self, what: str):
        """Lanza :class:`SessionError` si la máquina está en RESET. ``what`` = nombre del botón (para el
        mensaje). Es la primera línea de todo lo que mueve la máquina o cambia su estado."""
        if not self.enabled:
            raise SessionError(f"{what}: la máquina está en RESET (pulsá RESET para habilitarla)")

    def _machine_real(self) -> Machine:
        """Copia de la máquina con los orígenes (G54…G59, G92) que tiene AHORA el controlador.

        Los orígenes pueden cambiar en el controlador (G10 por MDI, «Zero X», sensado de plasma) sin
        pasar por la configuración: revisar con los guardados daría límites y envolvente falsos. Si el
        controlador no responde o no informa orígenes, se usa la máquina tal cual."""
        try:
            real = self.link.offsets()
        except BackendError:
            real = {}
        if not real:
            return self.m
        return dataclasses.replace(self.m, offsets={**self.m.offsets, **{k: dict(v) for k, v in real.items()}})

    def _guard(self, action):
        """None si está permitido; si no, deja la aprobación pendiente para la pantalla.

        Parámetros:
            action: ``guardian.core.Action`` que describe lo que se quiere hacer.

        Devuelve None (se puede seguir) o el id de la solicitud que el operador tiene que aprobar.
        Lanza :class:`SessionError` si no hay guardián con una máquina real, o si el guardián lo
        niega sin opción de aprobar.

        Por qué: mover una máquina real pide aprobación de una persona (regla del proyecto). En el
        simulador (``link.physical`` False) no hay riesgo y no se pregunta."""
        if not self.link.physical:
            return None
        if self.guardian is None:
            raise SessionError("sin guardián no se mueve una máquina real (configs/guardian.yaml)")
        d = self.guardian.decide(action)
        if d.allowed:
            return None
        if not d.pending_request:
            raise SessionError("; ".join(d.reasons))
        return d.pending_request

    # ==============================================================================================
    # ============================== RESET Y APROBACIONES DEL GUARDIÁN =============================
    # ==============================================================================================

    # --- RESET (habilitar / deshabilitar) ---------------------------------------------------------
    def reset(self) -> dict:
        """Botón RESET de Mach3: alterna entre máquina deshabilitada y habilitada.

        * Habilitada → para todo (``link.stop``), olvida la aprobación manual y queda en RESET.
        * En RESET → la habilita. Con una máquina real pide al guardián ``hardware.control`` en modo
          manual; si hace falta aprobación devuelve ``{"ok": True, "aprobar": {...}}`` y la pantalla
          muestra el cuadro para que el operador apruebe (:meth:`approve`) o cancele (:meth:`cancel`).

        Devuelve ``{"ok": True, "habilitada": bool}`` o ``{"ok": True, "aprobar": …}``.
        Lanza :class:`SessionError` si el guardián falta o lo niega."""
        with self.lock:
            if self.enabled:
                self.link.stop()
                self.enabled = False
                self.manual_grant = None
                self.say("RESET: máquina deshabilitada.")
                return {"ok": True, "habilitada": False}
            if self.link.physical:
                # Importación tardía: solo con máquina real (el simulador no necesita el guardián).
                from guardian.core import Action
                act = Action("hardware.control", {"dispositivo": "cnc", "modo": "manual", "maquina": self.m.name,
                                                  "controlador": self.link.name, "sesion": self.id},
                             requested_by="hope.cnc.app", justification="habilitar la máquina para el operador")
                req = self._guard(act)
                if req:
                    self.pending = {"id": req, "tipo": "manual", "accion": act,
                                    "resumen": f"Habilitar «{self.m.name}» para jog, MDI y referenciado"}
                    return {"ok": True, "aprobar": self._pending_public()}
            self.enabled = True
            self.say("Máquina habilitada.")
            return {"ok": True, "habilitada": True}

    def _pending_public(self) -> dict | None:
        """La aprobación pendiente sin los datos internos (la ``Action`` y las líneas a enviar), para la
        pantalla: solo id, tipo («manual» o «programa»), resumen e informe de revisión."""
        p = self.pending
        return None if p is None else {k: v for k, v in p.items() if k in ("id", "tipo", "resumen", "informe")}

    def approve(self, request_id: str) -> dict:
        """El operador aprueba en la pantalla la solicitud que creó el guardián.

        Parámetros:
            request_id: id de la solicitud (lo trae la pantalla de ``pendiente``/``aprobar``).

        Registra la aprobación en el guardián (queda en su auditoría), vuelve a preguntarle con ese id
        y, si ahora lo permite: tipo «manual» → habilita la máquina; tipo «programa» → arranca las
        líneas EXACTAS que se mostraron al pedir la aprobación (:meth:`_launch` comprueba el hash).

        Lanza :class:`SessionError` si no hay solicitud con ese id o el guardián igual lo niega."""
        with self.lock:
            p = self.pending
            if p is None or p["id"] != request_id:
                raise SessionError("no hay ninguna aprobación pendiente con ese número")
            self.guardian.approve(request_id, "operador (app Hope CNC)")
            d = self.guardian.decide(p["accion"], approval_id=request_id)
            self.pending = None
            if not d.allowed:
                raise SessionError("; ".join(d.reasons))
            if p["tipo"] == "manual":
                self.enabled = True
                self.say("Máquina habilitada (aprobado por el operador).")
                return {"ok": True, "habilitada": True}
            return self._launch(p["lineas"], p["hash"], p.get("mapa"))

    def cancel(self) -> dict:
        """El operador rechaza la aprobación pendiente: se informa al guardián como denegada (queda
        en la auditoría) y se descarta. Devuelve ``{"ok": True}``."""
        with self.lock:
            if self.pending and self.guardian is not None:
                self.guardian.deny(self.pending["id"], "operador (app Hope CNC)")
            self.pending = None
            return {"ok": True}

    # ==============================================================================================
    # ========================================= PROGRAMA ===========================================
    # ==============================================================================================

    # --- carga y trayectoria ----------------------------------------------------------------------
    def load(self, name: str, text: str) -> dict:
        """Carga un programa G-code: lo interpreta, planifica, revisa y traduce al dialecto.

        Parámetros:
            name: nombre del archivo (para mostrar y para recientes).
            text: texto G-code completo.

        Guarda en ``self.program`` un dict con: ``nombre``, ``texto``, ``lineas`` (originales),
        ``informe`` (revisión de :func:`check_program` + ``sha256`` del texto), ``programa``
        (``Program`` interpretado), ``emitidas`` (``[(texto, línea_original)]`` en el dialecto del
        controlador), ``acumulado`` (tiempos, ver :func:`_cumulative_times`), ``maquina`` (con los
        orígenes reales), ``perforaciones`` (índices de las emitidas que encienden la antorcha),
        ``referencia`` (las mismas emitidas en sintaxis grbl) y ``dialecto``.

        Devuelve ``{"ok", "informe", "texto_informe"}``. Un programa con errores se carga igual (para
        verlos), pero no se puede arrancar. Lanza :class:`SessionError` si hay uno en marcha.

        Se vuelve a llamar desde Cycle Start, los botones Block Delete / M1 y los perfiles, porque
        todos cambian lo que se ejecuta."""
        with self.lock:
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("no se puede cargar otro programa con uno en marcha")
            m = self._machine_real()
            prog = interpret(text, m, block_delete=self.opts["borrado_bloque"],
                             optional_stop=self.opts["parada_opcional"])
            # Un solo plan para todo (antes se re-interpretaba y re-planificaba lo emitido: el doble de
            # tiempo al cargar, que en una PC vieja con 100 000 líneas son decenas de segundos).
            pl = plan(prog, m)
            report = check_program(prog, m, pl)
            report["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()    # lo que aprueba el guardián
            if self.profile_summary:
                report["perfil"] = self._profile_line()
            dialect = dialect_for(m)
            origins: list[int] = []        # emit() lo llena: emitida i → índice de la operación que la produjo
            emitted = emit(prog, m, dialect, origins) if prog.ok else []
            # versión grbl de lo mismo (igual cantidad de líneas): «correr desde aquí» la reinterpreta
            reference = emit(prog, m, "grbl") if prog.ok and dialect != "grbl" else emitted
            cum = _cumulative_times(prog, pl, emitted, origins)
            # Emitidas que encienden la antorcha (plasma): el contador «perforaciones hechas» de state().
            pierce_idx = [i for i, (t, _) in enumerate(emitted) if t.startswith(("M3", "M4"))]
            self.program = {"nombre": name, "texto": text, "lineas": text.splitlines(), "informe": report,
                            "programa": prog, "emitidas": emitted, "acumulado": cum, "maquina": m,
                            "perforaciones": pierce_idx, "referencia": reference, "dialecto": dialect}
            self.started_at = None
            if self.next_line > len(self.program["lineas"]):    # «Set Next Line» de un programa más largo
                self.next_line = 0
            self._remember(name, text, report["sha256"])
            self.say(f"Cargado «{name}»: " + report_text(report).splitlines()[1])
            return {"ok": True, "informe": report, "texto_informe": report_text(report)}

    def unload(self) -> dict:
        """Descarga el programa (botón «Cerrar G-code»). Lanza :class:`SessionError` si está en marcha."""
        with self.lock:
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("hay un programa en marcha")
            self.program = None
            return {"ok": True}

    def toolpath(self) -> dict:
        """Trayectoria en coordenadas G54 para la vista: [[x, y, z, tipo, línea], …] (tipo 0 rápido, 1 corte).

        Tipos: 0 rápido (G0), 1 corte (husillo/antorcha encendidos), 3 avance sin cortar (sensado,
        bajadas en vacío), -1 «salto» (empieza un trazo nuevo: no se une con el punto anterior).
        ``línea`` = línea original, para resaltar en la vista la línea en curso. Los arcos se dividen
        en segmentos (``linearize``, tolerancia mínima 0,01 mm para no generar millones de puntos).

        Devuelve ``{"puntos", "perforaciones" ([x, y, z] donde se enciende la antorcha en plasma),
        "recortado" (True si se pasó de MAX_PATH_POINTS), "limites" (ver _limits_work)}``. No toma el
        lock: solo lee ``self.program``, que se reemplaza entero al cargar (nunca se modifica a medias)."""
        p = self.program
        if p is None:
            return {"puntos": [], "limites": self._limits_work()}
        m = p["maquina"]
        g54 = m.offset("G54")
        pts, pierces = [], []
        on = False                         # husillo/antorcha encendidos en este punto del programa
        op_pos: dict[str, float] = {}      # posición (máquina) al final del último movimiento
        for op in p["programa"].ops:
            if op.kind == "spindle":
                if op.value[0] != "off" and not on and m.is_plasma:
                    pierces.append([round(op_pos.get(a, 0.0) - g54.get(a, 0.0), 4) for a in "XYZ"])
                on = op.value[0] != "off"
                continue
            if op.kind not in ("rapid", "feed", "arc", "probe"):
                continue
            # 0 rápido, 1 corte (husillo/antorcha encendidos), 3 avance sin cortar (sensado, bajadas en vacío)
            kind = 0 if op.kind == "rapid" else 1 if on and op.kind != "probe" else 3
            # Si el movimiento no empieza donde terminó el anterior (cambio de origen, G92, G28…),
            # se agrega un punto de salto (-1) para que la vista no dibuje una línea que no existe.
            if not pts or any(abs(pts[-1][i] - (op.start.get(a, 0.0) - g54.get(a, 0.0))) > 1e-9
                              for i, a in enumerate("XYZ")):
                pts.append([*(round(op.start.get(a, 0.0) - g54.get(a, 0.0), 4) for a in "XYZ"), -1, op.line])
            seq = linearize(op, max(m.arc_tolerance, 0.01)) if op.kind == "arc" else [op.end]
            for q in seq:
                pts.append([*(round(q.get(a, 0.0) - g54.get(a, 0.0), 4) for a in "XYZ"), kind, op.line])
            op_pos = op.end
            if len(pts) > MAX_PATH_POINTS:
                break
        return {"puntos": pts, "perforaciones": pierces, "recortado": len(pts) > MAX_PATH_POINTS,
                "limites": self._limits_work()}

    def _limits_work(self) -> dict:
        """Recorrido de cada eje en coordenadas G54 (mm): ``{"X": [min, max], …}`` (None = sin límite).
        La vista lo dibuja como la caja de la mesa."""
        g54 = self.m.offset("G54")
        return {a: [None if ax.min is None else ax.min - g54[a], None if ax.max is None else ax.max - g54[a]]
                for a, ax in self.m.axes.items()}

    # ==============================================================================================
    # ========================================== CICLO =============================================
    # ==============================================================================================

    # --- Cycle Start ------------------------------------------------------------------------------
    def cycle_start(self) -> dict:
        """Botón Cycle Start de Mach3 (Alt+R): arranca, reanuda o avanza un bloque.

        * En Feed Hold → reanuda. En bloque a bloque con el programa en marcha → ejecuta la siguiente.
        * Programa ya en marcha → no hace nada.
        * Si no, vuelve a cargar el programa con los orígenes reales (el G54 pudo cambiar desde que se
          cargó), arma las líneas a enviar (:meth:`_lines_to_run`, con «Set Next Line» / «Run From
          Here») y, en una máquina real, pide al guardián la aprobación de ESE programa: hash del
          texto, envolvente en coordenadas de máquina y, si es parcial, desde qué línea y el hash de
          las líneas EXACTAS que se van a enviar (incluida la preparación).

        Devuelve ``{"ok", "en_marcha"}`` o ``{"ok", "aprobar"}`` si espera al operador.
        Lanza :class:`SessionError` en RESET, sin programa o con errores."""
        with self.lock:
            self._require_enabled("Cycle Start")
            st = self.link.snapshot()
            if st["estado"] == "Hold" or (self.link.single and st["trabajo"]["activo"]):
                self.link.resume()
                self.say("Reanudado." if not self.link.single else "Bloque siguiente.")
                return {"ok": True}
            if st["trabajo"]["activo"]:
                return {"ok": True}
            if self.program is None:
                raise SessionError("no hay programa cargado (Cargar G-code)")
            # revisar otra vez con los orígenes reales: el G54 pudo cambiar desde que se cargó
            r = self.load(self.program["nombre"], self.program["texto"])["informe"]    # RLock: reentrante
            if not r["runnable"]:
                raise SessionError("el programa tiene errores: " + "; ".join(r["errores"][:3]))
            lines, run_map = self._lines_to_run()
            partial = self.next_line > 0
            if self.link.physical:
                from guardian.core import Action
                params = {"dispositivo": "cnc", "maquina": self.m.name, "controlador": self.link.name,
                          "programa_sha256": r["sha256"], "envolvente_maquina": r["envolvente_maquina"]}
                if partial:
                    # Lo enviado no es el archivo entero: el guardián aprueba también las líneas reales.
                    params.update(desde_linea=self.next_line, preparacion=self.from_here,
                                  lineas_sha256=hashlib.sha256("\n".join(lines).encode()).hexdigest())
                act = Action("hardware.control", params, requested_by="hope.cnc.app",
                             justification=f"ejecutar «{self.program['nombre']}» ({r['lineas']} líneas)"
                                           + (f" desde la línea {self.next_line}" if partial else ""))
                req = self._guard(act)
                if req:
                    # Se guardan las líneas YA armadas: lo que se aprueba es exactamente lo que se envía.
                    self.pending = {"id": req, "tipo": "programa", "accion": act, "lineas": lines, "mapa": run_map,
                                    "hash": r["sha256"], "informe": r,
                                    "resumen": f"Ejecutar «{self.program['nombre']}»" +
                                               (f" desde la línea {self.next_line}" +
                                                (" (Run From Here: sube Z, va al punto y baja)" if self.from_here
                                                 else "") if partial else "")}
                    return {"ok": True, "aprobar": self._pending_public()}
            return self._launch(lines, r["sha256"], run_map)

    # --- «Set Next Line» / «Run From Here» --------------------------------------------------------
    def _lines_to_run(self) -> tuple[list[str], list[int]]:
        """Programa a enviar según «Set Next Line» / «Run From Here» y el mapa a las líneas emitidas.

        Devuelve ``(lineas, mapa)``: ``lineas`` = textos a enviar al controlador; ``mapa[j]`` = índice
        en ``program["emitidas"]`` de la enviada j (las de preparación apuntan a la primera que se
        retoma). Con ``next_line == 0`` es el programa entero y el mapa es la identidad.

        Cómo retoma desde la línea original N:

        1. Busca la primera emitida que sale de una línea original >= N.
        2. Reinterpreta las emitidas anteriores (en sintaxis grbl, ``referencia``) para saber el
           estado en ese punto: posición, orígenes, husillo, refrigerante, avance.
        3. Plasma: si ese punto queda a mitad de un corte (antorcha encendida), retrocede hasta la
           perforación (M3) de ese corte, porque no se puede seguir cortando sin encender el arco.
        4. «Set Next Line» envía encabezado + desde ahí, tal cual (como Mach3: el operador se hace
           cargo de que la máquina esté en un lugar seguro).
           «Run From Here» antepone una preparación segura: sube Z al tope (G53), va en XY al punto,
           enciende husillo (no en plasma: la antorcha la enciende el propio programa) y refrigerante,
           baja Z en G1 lenta y restablece el avance F.

        Lanza :class:`SessionError` si desde esa línea no queda nada que ejecutar."""
        p = self.program
        emitted = p["emitidas"]
        texts = [t for t, _ in emitted]
        ref = [t for t, _ in p.get("referencia") or emitted]      # mismo largo; en sintaxis grbl
        header = HEADERS.get(p.get("dialecto"), HEADER)
        n = self.next_line
        if n <= 0:
            return texts, list(range(len(texts)))
        # Desde 1: la emitida 0 es el encabezado (línea original 0), nunca es el punto de retoma.
        i = next((k for k in range(1, len(emitted)) if emitted[k][1] >= n), None)
        if i is None:
            raise SessionError(f"desde la línea {n} no queda nada que ejecutar")
        mc = as_controller(p["maquina"])

        def replay(upto: int) -> Interpreter:
            """Interpreta las primeras ``upto`` emitidas (versión grbl) y devuelve el intérprete: su
            estado es el de la máquina justo antes de la emitida ``upto``."""
            it = Interpreter(mc)
            it.run("\n".join(ref[:upto]))
            return it

        it = replay(i)
        if self.m.is_plasma and it.spindle[0] != "off":
            # a mitad de un corte: volver a la perforación de ese corte (no se puede seguir con el arco apagado)
            k = max(j for j in range(1, i) if ref[j].startswith(("M3", "M4")))
            # Retroceder también las emitidas anteriores de la MISMA línea original que el M3 (p. ej. el
            # posicionamiento y la bajada que el dialecto agrega antes de perforar).
            while k - 1 >= 1 and emitted[k - 1][1] == emitted[k][1]:
                k -= 1
            i = k
            it = replay(i)
        if not self.from_here:
            # Encabezado + resto; el encabezado se mapea a la emitida i para que el progreso arranque ahí.
            return [header, *texts[i:]], [i] + list(range(i, len(texts)))
        g54 = it.offsets["G54"]
        work = {a: it.pos[a] - g54[a] for a in it.axes}    # destino en coordenadas de trabajo G54 (mm)
        prep = [header]
        zax = self.m.axes.get("Z")
        if zax is not None and zax.max is not None:
            # Primero subir Z al tope: el traslado en XY no debe chocar con la pieza ni las grapas.
            if p.get("dialecto") == "marlin":                # sin G53: el tope en coordenadas de trabajo
                prep.append(f"G0 Z{zax.max - g54['Z']:.4f}")
            else:
                prep.append(f"G53 G0 Z{zax.max:g}")
        prep.append("G0 " + " ".join(f"{a}{work[a]:.4f}" for a in it.axes if a != "Z"))
        if it.spindle[0] != "off" and not self.m.is_plasma:
            prep.append(("M3" if it.spindle[0] == "cw" else "M4") + (f" S{it.rpm:g}" if it.rpm else ""))
        for c in sorted(it.coolant):
            prep.append("M7" if c == "niebla" else "M8")
        if "Z" in work:
            # Bajar en G1 (no G0): al avance del programa pero nunca más rápido que el máximo del eje Z;
            # 300 mm/min si el programa todavía no fijó F.
            plunge = min(it.feed or 300.0, zax.max_rate if zax else 300.0)
            prep.append(f"G1 Z{work['Z']:.4f} F{plunge:g}")
        if it.feed:
            prep.append(f"F{it.feed:g}")
        # Toda la preparación se mapea a la emitida i (en la pantalla se ve «en la línea N» desde el inicio).
        return prep + texts[i:], [i] * len(prep) + list(range(i, len(texts)))

    def _launch(self, lines: list[str], sha: str, run_map: list[int] | None = None) -> dict:
        """Envía las líneas al controlador y arranca el reloj. Lo llaman :meth:`cycle_start` (sin
        aprobación necesaria) y :meth:`approve` (después de aprobar).

        Parámetros:
            lines: líneas a enviar (de :meth:`_lines_to_run`).
            sha: hash del texto del programa con el que se armaron las líneas.
            run_map: mapa enviada → emitida (ver ``self.run_map``).

        Lanza :class:`SessionError` si el programa cargado ya no es ese (se cargó otro mientras la
        aprobación esperaba): nunca se envía algo distinto de lo aprobado. Después de arrancar,
        «Set Next Line» se borra: el próximo Cycle Start vuelve a empezar desde el principio."""
        if self.program is None or self.program["informe"]["sha256"] != sha:
            raise SessionError("el programa cambió: volvé a pulsar Cycle Start")
        self.link.start(lines)
        self.run_map = run_map
        self.started_at = time.monotonic()    # monotónico: un cambio de hora del sistema no altera «transcurrido»
        self.say(f"Cycle Start: «{self.program['nombre']}»" +
                 (f" desde la línea {self.next_line}." if self.next_line else "."))
        self.next_line, self.from_here = 0, False
        return {"ok": True, "en_marcha": True}

    # --- botones de Mach3: Single BLK, Block Delete, M1, Set Next Line, Run From Here -------------
    def toggle(self, name: str) -> dict:
        """Alterna un botón con LED de Mach3.

        Parámetros:
            name: ``"bloque_a_bloque"`` (Single BLK: cada Cycle Start ejecuta una línea),
                ``"borrado_bloque"`` (Block Delete: saltea las líneas que empiezan con «/») o
                ``"parada_opcional"`` (M1 Optional Stop: M1 pausa como M0).

        Block Delete y M1 cambian lo que se ejecuta, así que se recarga el programa (si no está en
        marcha). Devuelve ``{"ok", "opciones"}``. Lanza :class:`SessionError` si el nombre no existe."""
        if name not in self.opts:
            raise SessionError(f"opción desconocida: {name}")
        with self.lock:
            self.opts[name] = not self.opts[name]
            if name == "bloque_a_bloque":
                self.link.set_single(self.opts[name])
            elif self.program is not None and not self.link.snapshot()["trabajo"]["activo"]:
                self.load(self.program["nombre"], self.program["texto"])     # cambia lo que se ejecuta
            return {"ok": True, "opciones": dict(self.opts)}

    def set_next_line(self, line: int, from_here: bool = False) -> dict:
        """«Fijar línea siguiente» (Set Next Line) o «Correr desde aquí» (Run From Here) de Mach3.

        Parámetros:
            line: línea ORIGINAL del archivo (1 = primera; 0 = desde el principio, anula lo anterior).
            from_here: True = «Run From Here» (el próximo Cycle Start prepara la máquina antes: sube Z,
                va al punto, husillo, baja; ver :meth:`_lines_to_run`). Con línea 0 no tiene efecto.

        Solo marca desde dónde empieza el PRÓXIMO Cycle Start (no mueve nada). Borra el progreso del
        trabajo anterior (como el «Rewind» de Mach3). Lanza :class:`SessionError` sin programa, con
        uno en marcha o con la línea fuera del archivo."""
        with self.lock:
            if self.program is None:
                raise SessionError("no hay programa cargado")
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("hay un programa en marcha")
            n = int(line)
            if not 0 <= n <= len(self.program["lineas"]):
                raise SessionError(f"línea fuera del programa (1 a {len(self.program['lineas'])})")
            self.next_line, self.from_here = n, bool(from_here) and n > 0
            self.link.clear_job()
            if n:
                self.say(f"{'Run From Here' if self.from_here else 'Set Next Line'}: el próximo Cycle Start "
                         f"empieza en la línea {n}.")
            return {"ok": True, "linea": n, "desde_aqui": self.from_here}

    # ==============================================================================================
    # ===================================== ARCHIVOS RECIENTES =====================================
    # ==============================================================================================
    # Se guardan los últimos 10 programas cargados: ``recientes.json`` (lista con nombre, sha256,
    # fecha y cantidad de líneas) y una copia de cada texto en ``<sha256[:16]>.nc``. Así «Recientes»
    # vuelve a abrir el MISMO texto aunque el archivo original se haya movido o cambiado.

    # --- archivos recientes -----------------------------------------------------------------------
    def _read_recent(self) -> list[dict]:
        """Lee ``recientes.json`` (máximo 10). Archivo faltante o dañado → lista vacía (no es grave)."""
        if self.recent_dir is None:
            return []
        try:
            return json.loads((self.recent_dir / "recientes.json").read_text(encoding="utf-8"))[:10]
        except (OSError, ValueError):
            return []

    def _remember(self, name: str, text: str, sha: str) -> None:
        """Pone el programa primero en recientes (sin duplicar por sha256), guarda su texto y borra los
        textos que salieron de la lista. Un error de disco solo se avisa en el registro: no impide
        cargar el programa."""
        self.recent = [r for r in self.recent if r["sha"] != sha]
        self.recent.insert(0, {"nombre": name, "sha": sha, "fecha": time.time(), "lineas": text.count("\n") + 1})
        self.recent = self.recent[:10]
        keep = {r["sha"] for r in self.recent}
        if self.recent_dir is None:
            self._recent_mem[sha] = text
            self._recent_mem = {k: v for k, v in self._recent_mem.items() if k in keep}
            return
        try:
            self.recent_dir.mkdir(parents=True, exist_ok=True)
            (self.recent_dir / f"{sha[:16]}.nc").write_text(text, encoding="utf-8")
            (self.recent_dir / "recientes.json").write_text(json.dumps(self.recent, ensure_ascii=False), encoding="utf-8")
            for f in self.recent_dir.glob("*.nc"):
                if not any(f.stem == k[:16] for k in keep):
                    f.unlink()
        except OSError as e:
            self.say(f"no se pudo guardar en recientes: {e}")

    def recent_files(self) -> dict:
        """Lista de recientes para el menú: ``{"recientes": [{"nombre", "sha", "fecha", "lineas"}, …]}``."""
        return {"recientes": self.recent}

    def load_recent(self, sha: str) -> dict:
        """Vuelve a cargar un reciente por su sha256 (como :meth:`load`). Lanza :class:`SessionError` si
        ya no está en la lista o su copia se perdió."""
        r = next((x for x in self.recent if x["sha"] == sha), None)
        if r is None:
            raise SessionError("ese archivo ya no está en recientes")
        if self.recent_dir is None:
            text = self._recent_mem.get(sha)
        else:
            f = self.recent_dir / f"{sha[:16]}.nc"
            text = f.read_text(encoding="utf-8") if f.is_file() else None
        if text is None:
            raise SessionError("no se encontró el texto del archivo reciente")
        return self.load(r["nombre"], text)

    # ==============================================================================================
    # =============================== MARCHA: REWIND, FEED HOLD, STOP ==============================
    # ==============================================================================================
    # Sin lock y sin pedir que la máquina esté habilitada: parar o pausar tiene que responder SIEMPRE
    # y al instante, aunque otro hilo esté cargando un programa. (La parada de emergencia de verdad
    # es la física: ningún programa la reemplaza.)

    def rewind(self) -> dict:
        """«Rewind» de Mach3: olvida el avance del último trabajo (la barra vuelve a 0)."""
        self.link.clear_job()
        self.started_at = None
        return {"ok": True}

    def feed_hold(self) -> dict:
        """«Feed Hold» de Mach3 (barra espaciadora): frena con rampa y queda en pausa; Cycle Start reanuda."""
        self.link.hold()
        self.say("Feed Hold.")
        return {"ok": True}

    def stop(self) -> dict:
        """«Stop» de Mach3 (Alt+S): detiene el programa y descarta lo que quedaba por enviar."""
        self.link.stop()
        self.say("Stop.")
        return {"ok": True}

    # ==============================================================================================
    # ===================================== MANUAL: JOG Y MDI ======================================
    # ==============================================================================================

    # --- jog, referenciado, desbloqueo ------------------------------------------------------------
    def jog(self, axis: str, direction: int, step: float | None, fast: bool, percent: float = 30.0) -> dict:
        """Mueve un eje a mano (jog), como las flechas del teclado en Mach3.

        Parámetros:
            axis: eje («X», «y»… se pasa a mayúscula).
            direction: > 0 positivo, si no negativo.
            step: paso en mm (incremental, ver JOG_STEPS) o None/0 = continuo hasta soltar.
            fast: True = a la velocidad máxima del eje (Shift en Mach3); False = ``percent``.
            percent: % de la velocidad máxima del eje (1–100; «Slow Jog Rate» de Mach3).

        El jog continuo pide un movimiento hasta el límite blando del eje (o 1000 mm si no tiene) y se
        corta con :meth:`jog_stop` al soltar: si se pierde la conexión con la pantalla, no sale más allá
        del límite. Lanza :class:`SessionError` en RESET, con eje desconocido o ya en el límite."""
        self._require_enabled("jog")
        axis = axis.upper()
        if axis not in self.m.axes:
            raise SessionError(f"eje {axis} no configurado")
        ax = self.m.axes[axis]
        feed = ax.max_rate * (1.0 if fast else max(1.0, min(100.0, percent)) / 100.0)    # mm/min
        pos = self.link.snapshot()["mpos"][axis]
        sign = 1 if direction > 0 else -1
        if step:
            d = sign * float(step)
        else:                                               # continuo: hasta el límite (se corta al soltar)
            lim = ax.max if sign > 0 else ax.min
            d = (lim - pos) if lim is not None else sign * 1000.0
            if sign * d <= 1e-6:
                raise SessionError(f"{axis} ya está en el límite")
        self.link.jog({axis: d}, feed)
        return {"ok": True}

    def jog_stop(self) -> dict:
        """Corta el jog en curso (al soltar el botón o la tecla)."""
        self.link.jog_cancel()
        return {"ok": True}

    def home(self) -> dict:
        """«Ref All Home» de Mach3: referencia todos los ejes. Corre en un hilo aparte porque puede
        tardar un minuto y el pedido HTTP no debe quedar esperando; el resultado sale en el registro."""
        self._require_enabled("referenciar")
        self.say("Referenciando todos los ejes…")
        threading.Thread(target=self._safe, args=(self.link.home, "referenciado"), daemon=True).start()
        return {"ok": True}

    def _safe(self, fn, what):
        """Ejecuta ``fn()`` (en un hilo de fondo) y avisa en el registro si terminó o falló. ``what`` =
        nombre de la tarea para el mensaje. Captura :class:`BackendError` (un hilo no tiene a quién
        lanzarle la excepción)."""
        try:
            fn()
            self.say(f"{what.capitalize()} terminado.")
        except BackendError as e:
            self.say(f"Error en {what}: {e}")

    def unlock(self) -> dict:
        """Desbloquea el controlador después de una alarma (``$X`` en grbl). No necesita habilitar:
        sirve justamente para salir de la alarma."""
        self.link.unlock()
        return {"ok": True}

    # --- MDI --------------------------------------------------------------------------------------
    def mdi(self, line: str) -> dict:
        """Ejecuta una línea MDI (Manual Data Input de Mach3, Alt+2).

        Parámetros:
            line: una línea de G-code (máx. 200 caracteres; vacía = no hace nada).

        Antes de enviarla la revisa como un programa de una línea (:meth:`_check_mdi`): un G0 fuera
        de los límites se rechaza acá, no en la máquina. Guarda la línea en el historial (sin repetir
        la última). Lanza :class:`SessionError` en RESET, si es muy larga o tiene errores."""
        self._require_enabled("MDI")
        line = line.strip()
        if not line:
            return {"ok": True}
        if len(line) > 200:
            raise SessionError("línea MDI demasiado larga")
        r = self._check_mdi(line)
        if r["errores"]:
            raise SessionError(r["errores"][0].replace("línea 1: ", ""))
        self.link.mdi(line)
        if not self.mdi_history or self.mdi_history[-1] != line:
            self.mdi_history.append(line)
        self.say(f"MDI: {line}")
        return {"ok": True}

    def _check_mdi(self, line: str) -> dict:
        """Revisa la línea con el estado modal de MDI (G91 sigue valiendo en la línea siguiente, como en
        el controlador) y con los orígenes reales; si hay error, el estado modal no cambia.

        Trabaja sobre una COPIA del intérprete de MDI con la posición y los orígenes que informa el
        controlador ahora; solo si la línea no tiene errores la copia pasa a ser el estado de MDI.
        Devuelve el informe de :func:`check_program`."""
        import copy
        try:
            real = self.link.offsets()
        except BackendError:
            real = {}
        it = copy.deepcopy(self.mdi_interp)
        for w, v in real.items():
            if w in it.offsets:
                it.offsets[w] = {a: float(v.get(a, 0.0)) for a in it.axes}
            elif w == "G92":
                it.g92 = {a: float(v.get(a, 0.0)) for a in it.axes}
        it.pos = dict(self.link.snapshot()["mpos"])
        it.ops, it.diags, it.ended, it.line = [], [], False, 0    # olvidar lo de la línea MDI anterior
        it.run(line)
        m = dataclasses.replace(it.m, offsets={w: dict(o) for w, o in it.offsets.items()})
        r = check_program(Program(it.ops, it.diags, 1, [], True), m)
        if not r["errores"]:
            self.mdi_interp = it
        return r

    # --- ceros, husillo, overrides ----------------------------------------------------------------
    def zero(self, axis: str) -> dict:
        """«Zero X/Y/Z» de Mach3: la posición actual pasa a ser 0 en G54 (``G10 L20 P1``) para un eje o
        para todos (``axis`` = «todo»/«all»). Pasa por :meth:`mdi`, así que pide máquina habilitada."""
        axes = list(self.m.axis_names) if axis.lower() in ("todo", "all") else [axis.upper()]
        if any(a not in self.m.axes for a in axes):
            raise SessionError(f"eje {axis} no configurado")
        return self.mdi("G10 L20 P1 " + " ".join(f"{a}0" for a in axes))

    def goto_zero(self) -> dict:
        """Como «Goto Z» de Mach3 pero prudente: sube Z al máximo y va a X0 Y0 (no baja Z).

        No baja Z a propósito: bajar en G0 hasta el cero de la pieza es la forma clásica de romper una
        fresa. Espera a que termine la subida antes de mover XY (hasta 120 s)."""
        self._require_enabled("ir a cero")
        zmax = self.m.axes["Z"].max if "Z" in self.m.axes else None
        if zmax is not None:
            self.mdi(f"G53 G0 Z{zmax:g}")
            self._wait_idle()
        return self.mdi("G90 G0 X0 Y0")

    def _wait_idle(self, timeout: float = 120.0):
        """Espera (bloqueando, consulta cada 50 ms) a que el controlador quede «Idle». ``timeout`` en s;
        al vencer vuelve sin error (quien llama sigue: el siguiente MDI lo encola el controlador)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.link.snapshot()["estado"] == "Idle":
                return
            time.sleep(0.05)

    def override(self, kind: str, action: str) -> dict:
        """Ajuste en marcha (los «FRO»/«SRO» de Mach3). ``kind``: «avance», «rapido» o «husillo»;
        ``action``: «100», «+10», «-10», «+1», «-1» (o «50»/«25» para rápidos). Ver ``links``."""
        self.link.override(kind, action)
        return {"ok": True}

    def spindle(self, on: bool, rpm: float | None = None) -> dict:
        """Botón del husillo: enciende en sentido horario (M3) a ``rpm`` (por defecto la mitad del máximo
        de la máquina) o apaga (M5). Va por :meth:`mdi`: pide máquina habilitada y se revisa."""
        if on:
            rpm = rpm or self.m.spindle_max / 2
            return self.mdi(f"M3 S{rpm:g}")
        return self.mdi("M5")

    # ==============================================================================================
    # ========================================== PLASMA ============================================
    # ==============================================================================================

    # --- plasma -----------------------------------------------------------------------------------
    def _plasma(self):
        """Parámetros de plasma de la máquina (``machine.Plasma``). Lanza :class:`SessionError` si la
        máquina no es de plasma."""
        if not self.m.is_plasma or self.m.plasma is None:
            raise SessionError("la máquina no es de plasma (tipo: plasma en la configuración)")
        return self.m.plasma

    def _sequence(self, lines: list[str], what: str):
        """Ejecuta varias líneas MDI una tras otra en un hilo de fondo, esperando «Idle» entre cada una
        (p. ej. G38.2 tiene que terminar de tocar la chapa antes del G10 que fija el cero). Vuelve
        enseguida; el resultado sale en el registro (ver :meth:`_safe`)."""
        def run():
            """Cuerpo del hilo: cada línea por :meth:`mdi` (revisada), esperando a que la anterior termine."""
            for ln in lines:
                self._wait_idle()
                self.mdi(ln)
            self._wait_idle()
        threading.Thread(target=self._safe, args=(run, what), daemon=True).start()

    def plasma_probe(self) -> dict:
        """Sensado de altura (IHS) en el lugar: baja hasta tocar la chapa y pone Z0 en la superficie.

        Secuencia: G91 → ``G38.2 Z-<probe_depth> F<probe_feed>`` (baja hasta que el sensor toca) →
        G90 → ``G10 L20 P1 Z-<float_offset>`` (descuenta la carrera del flotante: lo que sube la
        antorcha hasta que actúa el interruptor) → ``G0 Z<travel_height>`` (altura de traslado).
        Todas las medidas en mm, de la configuración de plasma de la máquina."""
        pl = self._plasma()
        self._require_enabled("sensar altura")
        self.say("Sensando la altura de la chapa…")
        self._sequence(["G91", f"G38.2 Z-{pl.probe_depth:g} F{pl.probe_feed:g}", "G90",
                        f"G10 L20 P1 Z-{pl.float_offset:g}", f"G0 Z{pl.travel_height:g}"], "sensado")
        return {"ok": True}

    def test_fire(self) -> dict:
        """Disparo de prueba: antorcha 0,3 s en el lugar (para ver el arco y la presión de aire)."""
        self._plasma()
        self._require_enabled("disparo de prueba")
        if self.link.snapshot()["trabajo"]["activo"]:
            raise SessionError("hay un programa en marcha")
        self.say("Disparo de prueba.")
        self._sequence(["M3", "G4 P0.3", "M5"], "disparo de prueba")
        return {"ok": True}

    def coolant(self) -> dict:
        """Botón del refrigerante (M8/M9 de Mach3). Con un programa en marcha usa el comando en tiempo
        real del controlador (no se puede meter un MDI en medio del programa); si no, M8 o M9 por MDI."""
        self._require_enabled("refrigerante")
        st = self.link.snapshot()
        if st["trabajo"]["activo"]:
            self.link.coolant_toggle()
            return {"ok": True}
        return self.mdi("M9" if st["refrigerante"] else "M8")

    # ==============================================================================================
    # ==================================== ESTADO PARA LA PANTALLA =================================
    # ==============================================================================================

    # --- estado para la pantalla ------------------------------------------------------------------
    def state(self) -> dict:
        """Todo lo que la pantalla muestra, en un solo dict (la pide varias veces por segundo).

        Claves principales: ``estado`` (snapshot del controlador: Idle/Run/Hold/Alarm, posiciones,
        trabajo en curso), ``trabajo_pos`` (posición en coordenadas de trabajo, mm), ``programa``
        (nombre, línea actual, tiempos en s, progreso 0–1, errores/avisos), ``plasma`` (antorcha y
        perforaciones hechas/total), ``pendiente`` (aprobación que espera), ``mensajes`` (últimos 40),
        ``opciones`` (LEDs), ``linea_siguiente``, ``modal``, ``herramienta`` y ``perfiles``.

        Progreso: el controlador informa ``confirmadas`` = líneas ENVIADAS ya confirmadas. Con
        ``run_map`` se pasan a índice en las emitidas, y con ``acumulado`` a segundos. No toma el
        lock: solo lee, y un valor de hace un instante alcanza para mostrar."""
        st = self.link.snapshot()
        wpos = {a: st["mpos"][a] - st["wco"].get(a, 0.0) for a in st["mpos"]}    # máquina − origen activo
        job = st["trabajo"]
        prog = self.program
        info = None
        if prog is not None:
            emitted, cum = prog["emitidas"], prog["acumulado"]
            ran = job["activo"] or job["terminado"] or job["error"]    # hay un trabajo (o lo hubo) que mostrar
            done = min(job["confirmadas"], len(emitted)) if ran else 0
            rm = self.run_map
            if ran and rm:
                # Enviadas → emitidas (salteando la preparación de «Run From Here»); terminado = todo.
                done = len(emitted) if job["terminado"] else rm[min(job["confirmadas"], len(rm) - 1)]
            src = 0                            # línea original en curso (0 = todavía nada)
            if ran and emitted:
                src = emitted[min(done, len(emitted) - 1)][1]
            total = cum[-1] if cum else 0.0
            elapsed = (time.monotonic() - self.started_at) if self.started_at and job["activo"] else None
            # Nota: al empezar desde la mitad, «tiempo_restante» y «progreso» cuentan desde la emitida
            # de retoma (cum[done]), así que reflejan lo que realmente falta.
            info = {"nombre": prog["nombre"], "lineas_total": len(prog["lineas"]), "linea_actual": src,
                    "tiempo_total": round(total, 1),
                    "tiempo_restante": round(max(0.0, total - cum[min(done, len(cum) - 1)]), 1) if cum else 0.0,
                    "transcurrido": None if elapsed is None else round(elapsed, 1),
                    "progreso": (cum[min(done, len(cum) - 1)] / total) if total else 0.0,
                    "runnable": prog["informe"]["runnable"], "errores": prog["informe"]["errores"][:50],
                    "avisos": prog["informe"]["avisos"][:50], "informe": report_text(prog["informe"]),
                    "sha256": prog["informe"]["sha256"]}
        plasma = None
        if self.m.is_plasma:
            idx = prog["perforaciones"] if prog else []
            ran = prog is not None and (job["activo"] or job["terminado"] or job["error"])
            k = job["confirmadas"]
            if ran and self.run_map:
                k = len(prog["emitidas"]) if job["terminado"] else self.run_map[min(k, len(self.run_map) - 1)]
            # Perforaciones hechas = M3 de las emitidas antes de la actual (al retomar cuentan las salteadas).
            done_n = sum(1 for i in idx if i < k) if ran else 0
            plasma = {"antorcha": st.get("husillo_dir", "off") != "off", "perforaciones_total": len(idx),
                      "perforaciones_hechas": min(done_n, len(idx))}
        return {"maquina": self.m.name, "marca": BRAND, "tipo": self.m.kind, "plasma": plasma,
                "ejes": list(self.m.axis_names), "habilitada": self.enabled,
                "fisica": self.link.physical, "estado": st, "trabajo_pos": {a: round(v, 4) for a, v in wpos.items()},
                "programa": info, "pendiente": self._pending_public(),
                "mensajes": [{"t": t, "m": m} for t, m in list(self.log)[-40:]],
                "mdi_historial": list(self.mdi_history), "opciones": dict(self.opts),
                "linea_siguiente": self.next_line, "desde_aqui": self.from_here,
                "modal": self.link.modal() or modal_text(self.mdi_interp), "herramienta": self._tool_info(st),
                "perfiles": None if self.selection is None else {"seleccion": self.selection,
                                                                 "resumen": self.profile_summary}}

    def _tool_info(self, st: dict) -> dict:
        """Herramienta actual para la pantalla: número (la del controlador; si no informa, la del estado
        modal de MDI), diámetro y largo (mm) de la tabla de herramientas, nombre, y ``cambio`` = True si
        el controlador está esperando un cambio manual de herramienta (M6)."""
        n = self.link.tool()
        if n is None:
            n = self.mdi_interp.tool
        t = self.m.tool(n) if n is not None else None
        return {"n": n or 0, "diametro": t.diameter if t else 0.0, "largo": t.length if t else 0.0,
                "nombre": t.name if t else "", "cambio": str(st.get("mensaje", "")).startswith("cambiar a la herramienta")}

    def program_lines(self) -> dict:
        """Texto del programa cargado, línea por línea (para la lista de G-code de la pantalla)."""
        p = self.program
        return {"nombre": p["nombre"], "lineas": p["lineas"]} if p else {"nombre": "", "lineas": []}

    # ==============================================================================================
    # =================================== PERFILES TIPO SLICER =====================================
    # ==============================================================================================
    # Como en un slicer 3D: la máquina efectiva sale de cuatro perfiles (máquina, material,
    # consumible, proceso) más el espesor de chapa. Cada perfil puede heredar de otro. Ver
    # :mod:`hope.cnc.profiles`. Cambiar la selección recalcula la máquina y recarga el programa.

    # --- perfiles tipo slicer ---------------------------------------------------------------------
    def _profile_line(self) -> str:
        """Resumen de una línea de los perfiles elegidos («Acero + 45 A + Corte fino: …»)."""
        sel, s = self.selection, self.profile_summary
        names = " + ".join(x for x in (sel.get("material"), sel.get("consumible"), sel.get("proceso")) if x)
        return f"{names}: {s['texto']}"

    def _selection_file(self):
        """Archivo donde se recuerda la selección entre sesiones (``<user_dir>/seleccion.json``), o None
        si el almacén de perfiles no tiene carpeta de usuario."""
        d = getattr(self.profiles, "user_dir", None)
        return None if d is None else Path(d) / "seleccion.json"

    def choose_profile(self, kind: str | None = None, name: str | None = None, thickness: float | None = None) -> dict:
        """Elegir un perfil (como los desplegables de un slicer) o el espesor; recalcula todo.

        Parámetros:
            kind: tipo de perfil («maquina», «material», «consumible», «proceso») o None.
            name: nombre del perfil de ese tipo.
            thickness: espesor de chapa (mm) o None para no cambiarlo.

        Sin argumentos solo recalcula (se usa cuando se editó el perfil en uso). Elegir otra máquina
        borra material/consumible/proceso (pueden no servir para ella) y ``fix_selection`` elige los
        compatibles. Aplica la máquina nueva (:meth:`_apply_machine`), guarda la selección y recarga
        el programa. Devuelve ``{"ok", "seleccion", "resumen"}``. Lanza :class:`SessionError` sin
        perfiles, con un programa en marcha o con un perfil inexistente/inválido."""
        from hope.cnc.profiles import KINDS, ProfileError, effective, fix_selection
        if self.profiles is None:
            raise SessionError("los perfiles no están activos")
        with self.lock:
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("no se cambian perfiles con un programa en marcha")
            sel = dict(self.selection)
            if kind is not None:
                if kind not in KINDS:
                    raise SessionError(f"tipo de perfil desconocido: {kind}")
                if name not in self.profiles.names(kind):
                    raise SessionError(f"no existe el perfil «{name}»")
                sel[kind] = name
                if kind == "maquina":
                    sel = {"maquina": name, "espesor": sel.get("espesor")}
            if thickness is not None:
                sel["espesor"] = float(thickness)
            try:
                sel = fix_selection(self.profiles, sel)
                m, summary = effective(self.profiles, sel)
            except (ProfileError, ValueError) as e:
                raise SessionError(str(e)) from None
            self._apply_machine(m)
            self.selection, self.profile_summary = sel, summary
            f = self._selection_file()
            if f is not None:
                try:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(sel, ensure_ascii=False), encoding="utf-8")
                except OSError as e:
                    self.say(f"no se pudo guardar la selección: {e}")
            if self.program is not None:
                self.load(self.program["nombre"], self.program["texto"])
            self.say(f"Perfiles: {sel['maquina']} · {self._profile_line()}")
            return {"ok": True, "seleccion": sel, "resumen": summary}

    def _apply_machine(self, m: Machine) -> None:
        """Pasa a usar la máquina efectiva ``m``.

        * Otro controlador (y hay ``link_factory``): para, cierra la conexión, abre una nueva y queda
          en RESET.
        * Otra máquina con el mismo controlador: queda en RESET. Por seguridad: lo aprobado para una
          máquina no vale para otra; el operador vuelve a habilitar a conciencia.
        * Misma máquina (cambió material, proceso…): sigue habilitada.

        Siempre reinicia el estado modal de MDI (puede cambiar ejes, unidades u orígenes)."""
        changed_machine = m.name != self.m.name
        if m.controller != self.m.controller and self.link_factory is not None:
            if self.enabled:
                self.link.stop()
            self.link.close()
            self.link = self.link_factory(m)
            self.enabled = False
            self.say(f"Conexión nueva: {self.link.name}. Pulsá RESET para habilitar.")
        elif changed_machine and self.enabled:
            self.enabled = False                  # otra máquina: volver a habilitar a conciencia
            self.say("Máquina cambiada: pulsá RESET para habilitarla.")
        self.m = m
        if hasattr(self.link, "set_machine"):    # el simulador necesita la máquina nueva (límites, velocidades)
            self.link.set_machine(m)
        self.mdi_interp = Interpreter(as_controller(m))

    def profile_list(self) -> dict:
        """Perfiles disponibles para los desplegables: los de cada tipo compatibles con el tipo de
        máquina actual (fresadora, plasma…), todas las máquinas, la selección, el resumen y, en plasma,
        los espesores que ofrecen el material y el consumible elegidos. ``{"activos": False}`` si los
        perfiles están desactivados."""
        from hope.cnc.profiles import KIND_NAMES, KINDS, thickness_options
        if self.profiles is None:
            return {"activos": False}
        kind = self.m.kind
        out = {k: self.profiles.list(k, kind) for k in KINDS}
        out["maquina"] = self.profiles.list("maquina")    # las máquinas no se filtran por tipo
        esp = []
        if self.m.is_plasma and self.selection.get("material"):
            esp = thickness_options(self.profiles.resolve("material", self.selection["material"]),
                                    self.profiles.resolve("consumible", self.selection["consumible"])
                                    if self.selection.get("consumible") else None)
        return {"activos": True, "tipos": out, "nombres": KIND_NAMES, "seleccion": self.selection,
                "resumen": self.profile_summary, "espesores": esp}

    def profile_detail(self, kind: str, name: str) -> dict:
        """Datos de un perfil para el editor: ``crudo`` (lo escrito en su archivo), ``datos`` (resuelto,
        con la herencia aplicada), ``padre`` (resuelto del que hereda), ``propios`` (claves que pisa),
        ``origen`` (de fábrica o del usuario) y ``esquema`` (campos con sus unidades).
        Lanza :class:`SessionError` si no existe o los perfiles están desactivados."""
        from hope.cnc.profiles import SCHEMA, ProfileError
        if self.profiles is None:
            raise SessionError("los perfiles no están activos")
        try:
            raw, origin = self.profiles.raw(kind, name)
            data = self.profiles.resolve(kind, name)
            parent = self.profiles.resolve(kind, raw["hereda"]) if raw.get("hereda") else None
            own = self.profiles.own_keys(kind, name)
        except ProfileError as e:
            raise SessionError(str(e)) from None
        return {"tipo": kind, "nombre": name, "origen": origin, "hereda": raw.get("hereda"), "datos": data,
                "padre": parent, "propios": own, "esquema": SCHEMA[kind], "crudo": raw}

    def profile_save(self, kind: str, name: str, data: dict, inherits: str | None) -> dict:
        """Guarda un perfil del usuario (``data`` = valores propios, ``inherits`` = perfil padre o None).
        Si es uno de los perfiles en uso, recalcula la máquina efectiva. Lanza :class:`SessionError` si
        los datos no son válidos."""
        from hope.cnc.profiles import ProfileError
        if self.profiles is None:
            raise SessionError("los perfiles no están activos")
        try:
            out = self.profiles.save(kind, name, data, inherits)
        except (ProfileError, ValueError) as e:
            raise SessionError(str(e)) from None
        self.say(f"Perfil guardado: {name}")
        if self.selection and self.selection.get(kind) == name or kind == "maquina" and self.selection["maquina"] == name:
            self.choose_profile()                 # el perfil en uso cambió: recalcular
        return {"ok": True, "perfil": out}

    def profile_delete(self, kind: str, name: str) -> dict:
        """Borra un perfil del usuario; si estaba en uso, ``choose_profile`` elige otro compatible.
        Lanza :class:`SessionError` si no se puede borrar (de fábrica, inexistente, perfiles inactivos:
        ``AttributeError`` cuando ``self.profiles`` es None)."""
        from hope.cnc.profiles import ProfileError
        try:
            self.profiles.delete(kind, name)
        except (ProfileError, AttributeError) as e:
            raise SessionError(str(e)) from None
        if self.selection.get(kind) == name:
            self.choose_profile()
        return {"ok": True}

    def profile_import(self, kind: str, text: str) -> dict:
        """Importa un perfil de tipo ``kind`` desde texto (archivo que compartió otro usuario).
        Lanza :class:`SessionError` si el texto no es un perfil válido."""
        from hope.cnc.profiles import ProfileError
        try:
            out = self.profiles.import_text(kind, text)
        except (ProfileError, ValueError, AttributeError) as e:
            raise SessionError(str(e)) from None
        return {"ok": True, "perfil": out}

    # ==============================================================================================
    # =============================== CONFIGURACIÓN Y ORÍGENES =====================================
    # ==============================================================================================

    def config(self) -> dict:
        """Configuración de la máquina efectiva para la pantalla «Config» (solo lectura): ejes (pasos/mm,
        velocidad máx. mm/min, aceleración mm/s², límites mm), desviación de unión y tolerancia de arco
        (mm), husillo (RPM), tabla de herramientas (mm), controlador, pasos de jog y, en plasma, alturas
        (mm), retardo de perforación (s), avance de bajada (mm/min) y opciones de sensado."""
        m = self.m
        return {"nombre": m.name, "tipo": m.kind, "archivo": self.machine_path,
                "ejes": {a: {"pasos_mm": ax.steps_per_mm, "vel_max": ax.max_rate, "acel": ax.accel, "min": ax.min,
                             "max": ax.max} for a, ax in m.axes.items()},
                "desviacion_union": m.junction_deviation, "tolerancia_arco": m.arc_tolerance,
                "husillo": {"rpm_min": m.spindle_min, "rpm_max": m.spindle_max},
                "herramientas": {n: {"largo": t.length, "diametro": t.diameter, "nombre": t.name}
                                 for n, t in sorted(m.tools.items())},
                "controlador": m.controller, "pasos_jog": JOG_STEPS, "marca": BRAND,
                "plasma": None if m.plasma is None or not m.is_plasma else {
                    "altura_perforacion": m.plasma.pierce_height, "altura_corte": m.plasma.cut_height,
                    "retardo_perforacion": m.plasma.pierce_delay, "altura_traslado": m.plasma.travel_height,
                    "avance_bajada": m.plasma.plunge_feed, "perforacion_auto": m.plasma.auto_pierce,
                    "sensado": m.plasma.probe, "holgura_flotante": m.plasma.float_offset}}

    def offsets_table(self) -> dict:
        """Tabla de orígenes (G54…G59 y G92 si hay) como la tiene AHORA el controlador; los que no
        informa salen de la configuración. ``{"ok": False, "mensaje"}`` si el controlador no responde."""
        try:
            real = self.link.offsets()
        except BackendError as e:
            return {"ok": False, "mensaje": str(e)}
        rows = {w: real.get(w, self.m.offset(w)) for w in WCS}
        if "G92" in real:
            rows["G92"] = real["G92"]
        return {"ok": True, "origenes": rows}

