"""Controlador de Hope CNC: revisar, simular y — con permiso humano — ejecutar.

Regla de seguridad (Carta de Hope, guardián): **mover una máquina real es ``hardware.control``,
nivel humano**. Cada ejecución en un controlador físico crea una solicitud ligada al hash del
programa, la máquina, el controlador y la envolvente calculada con los orígenes *reales* que se leen
del controlador; si algo cambia (otro programa, otro G54), hace falta otra aprobación. Hope puede
revisar y simular libremente, pero no puede mover la máquina por su cuenta.

La parada de emergencia es física (seta que corta la potencia de los drivers): ningún software la
reemplaza.

Cómo encaja
-----------
* **Usa:** ``gcode`` (interpretar), ``check`` (revisar: un programa con errores no se envía),
  ``dialect`` (traducir), un ``backends.Backend`` (simulador, grbl o LinuxCNC) y el guardián
  (``guardian.core``, paquete aparte; se importa recién cuando hace falta).
* **Lo usan:** la consola (``python -m hope.cnc enviar``) y la app (``app.server`` carga el guardián
  con ``load_guardian``).

Flujo de ``run``
----------------
1. Interpretar y revisar con los orígenes reales del controlador (no los del YAML).
2. Si el backend es físico: pedir permiso al guardián. Sin aprobación → «pendiente» (con el id de la
   solicitud para aprobar) o «rechazado». Sin guardián → «rechazado».
3. Traducir al dialecto y enviar; en grbl, atender las pausas M0 con el operador.

Ejemplo de uso
--------------
::

    ctl = Controller(machine, make_backend(machine.controller, machine), load_guardian("configs/guardian.yaml"))
    r = ctl.run(texto)                                  # 1.ª vez: {"estado": "pendiente", "solicitud": "…"}
    # una persona aprueba: python -m guardian.cli approve <id> --by <nombre>
    r = ctl.run(texto, approval_id=r["solicitud"])     # {"estado": "completado", …}
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from typing import TYPE_CHECKING

from hope.cnc.backends.base import Backend, StreamResult
from hope.cnc.check import check_program
from hope.cnc.dialect import emit
from hope.cnc.gcode import interpret
from hope.cnc.machine import Machine

if TYPE_CHECKING:                               # solo para las anotaciones (no existe en Python 3.4)
    from typing import Callable
    Operator = Callable[[str], bool]    # recibe el mensaje de la pausa; True = seguir, False = abortar


# ==================================================================================================
# =========================================== GUARDIÁN =============================================
# ==================================================================================================


def load_guardian(path: str):
    """Guardián desde su YAML sin pasar por ``guardian.cli`` (que trae PyTorch): liviano para PCs viejas.
    Devuelve None si el archivo no existe o falta pyyaml; sin guardián, la máquina real no se mueve.

    ``path``: archivo ``.yaml`` o ``.json`` con la configuración (``GuardianConfig``). Un JSON se
    reconoce por la extensión o porque empieza con «{» (no necesita pyyaml, que puede faltar en el
    paquete para PCs viejas). Puede lanzar los errores de ``GuardianConfig`` si el archivo tiene
    claves inválidas: mejor fallar fuerte que arrancar con un guardián mal configurado."""
    from pathlib import Path
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8")
    from guardian.core import Guardian, GuardianConfig
    if p.suffix == ".json" or text.lstrip().startswith("{"):
        import json
        return Guardian(GuardianConfig(**json.loads(text)))
    try:
        import yaml
    except ImportError:
        return None
    return Guardian(GuardianConfig(**(yaml.safe_load(text) or {})))


# ==================================================================================================
# ========================================= CONTROLADOR ============================================
# ==================================================================================================


class Controller:
    """Une máquina, backend y guardián: revisa, pide permiso y envía.

    Parámetros:
        machine:  la máquina (YAML/perfil) con límites y orígenes.
        backend:  el controlador (``backends.make_backend``); ``backend.physical`` dice si mueve algo real.
        guardian: ``guardian.core.Guardian`` o None (con None, un backend físico no se mueve nunca).
        dialect:  forzar un dialecto de ``dialect.DIALECTS``; por defecto LinuxCNC si el backend es
                  LinuxCNC y grbl en cualquier otro caso."""

    def __init__(self, machine: Machine, backend: Backend, guardian=None, dialect: str | None = None):
        self.machine = machine
        self.backend = backend
        self.guardian = guardian
        self.dialect = dialect or ("linuxcnc" if backend.name == "linuxcnc" else "grbl")

    def _machine_with_real_offsets(self) -> Machine:
        """Copia de la máquina con los orígenes (G54…) que tiene cargados el controlador real.

        Por seguridad: los límites de recorrido se revisan en coordenadas de máquina, y para eso hace
        falta el G54 con el que el controlador va a ejecutar, no el que dice el YAML (alguien pudo
        tocar el cero desde el panel). Con el simulador, o si el controlador no informa nada, queda la
        máquina tal cual."""
        real = self.backend.work_offsets() if self.backend.physical else {}
        if not real:
            return self.machine
        offsets = {**self.machine.offsets, **{k: dict(v) for k, v in real.items()}}
        return dataclasses.replace(self.machine, offsets=offsets)

    def review(self, text: str) -> dict:
        """Revisa ``text`` (G-code) con los orígenes reales; devuelve el informe de ``check_program``
        más ``sha256`` del texto. No mueve nada."""
        m = self._machine_with_real_offsets()
        r = check_program(interpret(text, m), m)
        r["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return r

    def action(self, report: dict):
        """La acción ``hardware.control`` que se le pide al guardián para este informe.

        Incluye el hash del programa, la máquina, el controlador y la envolvente en coordenadas de
        máquina: si cambia cualquiera de esas cosas, la acción es otra y la aprobación anterior no
        sirve (no se puede aprobar un programa y enviar otro, ni el mismo con otro G54)."""
        from guardian.core import Action
        return Action("hardware.control", {
            "dispositivo": "cnc", "maquina": self.machine.name, "controlador": self.backend.name,
            "programa_sha256": report["sha256"], "envolvente_maquina": report["envolvente_maquina"],
        }, requested_by="hope.cnc", justification=f"ejecutar un programa de {report['lineas']} líneas "
                                                   f"(~{report['tiempo_s']:.0f} s) en «{self.machine.name}»")

    def run(self, text: str, approval_id: str | None = None, operator: Operator | None = None,
            progress=None) -> dict:
        """Revisa, pide permiso (si la máquina es real) y envía el programa.

        Parámetros:
            text:        G-code tal como lo escribió el CAM o la persona.
            approval_id: id de la solicitud ya aprobada por una persona en el guardián (o None).
            operator:    ``operator(mensaje) -> bool`` para las pausas M0 (solo grbl); None = abortar
                         en la primera pausa.
            progress:    ``progress(enviadas, confirmadas, total)``, lo llama el backend.

        Devuelve un ``dict`` con ``estado`` («rechazado», «pendiente», «completado» o «abortado») y el
        ``informe`` de la revisión; según el caso también ``motivo``, ``solicitud`` (id para aprobar),
        ``enviadas``, ``confirmadas``, ``errores`` y ``alarma``. Los errores de conexión del backend
        (``BackendError``) no se atrapan acá: los ve quien llama."""
        m = self._machine_with_real_offsets()
        program = interpret(text, m)
        report = check_program(program, m)
        report["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not report["runnable"]:
            return {"estado": "rechazado", "motivo": "el programa tiene errores", "informe": report}
        if self.backend.physical:
            if self.guardian is None:
                return {"estado": "rechazado", "motivo": "sin guardián no se mueve una máquina real",
                        "informe": report}
            d = self.guardian.decide(self.action(report), approval_id=approval_id)
            if not d.allowed:
                return {"estado": "pendiente" if d.pending_request else "rechazado",
                        "solicitud": d.pending_request, "motivo": "; ".join(d.reasons), "informe": report}
        # Se envía la traducción del MISMO programa que se revisó (y cuyo hash se aprobó).
        lines = [t for t, _ in emit(program, m, self.dialect)]
        kw = {}
        if self.backend.name == "grbl":            # solo grbl acepta on_ack (LinuxCNC reanuda M0 desde su pantalla)
            kw["on_ack"] = self._pause_hook(lines, operator)
        res: StreamResult = self.backend.stream(lines, progress=progress, **kw)
        return {"estado": "completado" if res.ok else "abortado", "enviadas": res.sent,
                "confirmadas": res.acknowledged, "errores": res.errors, "alarma": res.alarm, "informe": report}

    def _pause_hook(self, lines: list[str], operator: Operator | None):
        """En grbl, M0 deja la máquina en Hold: se espera al operador y se reanuda con «~».

        Devuelve la función ``on_ack(índice, línea) -> bool`` para ``GrblBackend.stream``. ``lines``
        es el programa traducido (para leer el mensaje: el backend pasa la línea ya limpia, sin
        comentarios). Si la máquina entra en alarma o puerta abierta en vez de Hold, o el operador
        aborta, devuelve False y el backend reinicia grbl y corta el envío."""
        def hook(i: int, sent: str) -> bool:
            """``i``: índice de la línea confirmada; ``sent``: la línea limpia (mayúsculas, sin comentarios).
            True = seguir enviando."""
            if not sent.startswith("M0"):
                return True
            raw = lines[i]                         # emit nunca da líneas vacías: los índices coinciden
            msg = raw.split("MSG,", 1)[1].rstrip(")").strip() if "MSG," in raw else "pausa"
            # grbl confirma el M0 al leerlo, pero recién entra en Hold al terminar los movimientos
            # anteriores: se consulta el estado cada 0.2 s hasta que pare (o pase algo peor).
            while True:
                st = self.backend.status()
                if st.state in ("Hold", "Alarm", "Door"):
                    break
                time.sleep(0.2)
            if st.state != "Hold":
                return False
            if operator is None or not operator(msg):
                return False
            self.backend.resume()
            return True
        return hook
