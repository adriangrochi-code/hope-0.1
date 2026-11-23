"""Controlador de Hope CNC: revisar, simular y — con permiso humano — ejecutar.

Regla de seguridad (Carta de Hope, guardián): **mover una máquina real es ``hardware.control``,
nivel humano**. Cada ejecución en un controlador físico crea una solicitud ligada al hash del
programa, la máquina, el controlador y la envolvente calculada con los orígenes *reales* que se leen
del controlador; si algo cambia (otro programa, otro G54), hace falta otra aprobación. Hope puede
revisar y simular libremente, pero no puede mover la máquina por su cuenta.

La parada de emergencia es física (seta que corta la potencia de los drivers): ningún software la
reemplaza.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Callable

from hope.cnc.backends.base import Backend, StreamResult
from hope.cnc.check import check_program
from hope.cnc.dialect import emit
from hope.cnc.gcode import interpret
from hope.cnc.machine import Machine

Operator = Callable[[str], bool]          # recibe el mensaje de la pausa; True = seguir, False = abortar


class Controller:
    def __init__(self, machine: Machine, backend: Backend, guardian=None, dialect: str | None = None):
        self.machine = machine
        self.backend = backend
        self.guardian = guardian
        self.dialect = dialect or ("linuxcnc" if backend.name == "linuxcnc" else "grbl")

    def _machine_with_real_offsets(self) -> Machine:
        real = self.backend.work_offsets() if self.backend.physical else {}
        if not real:
            return self.machine
        offsets = {**self.machine.offsets, **{k: dict(v) for k, v in real.items()}}
        return dataclasses.replace(self.machine, offsets=offsets)

    def review(self, text: str) -> dict:
        m = self._machine_with_real_offsets()
        r = check_program(interpret(text, m), m)
        r["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return r

    def action(self, report: dict):
        from guardian.core import Action
        return Action("hardware.control", {
            "dispositivo": "cnc", "maquina": self.machine.name, "controlador": self.backend.name,
            "programa_sha256": report["sha256"], "envolvente_maquina": report["envolvente_maquina"],
        }, requested_by="hope.cnc", justification=f"ejecutar un programa de {report['lineas']} líneas "
                                                   f"(~{report['tiempo_s']:.0f} s) en «{self.machine.name}»")

    def run(self, text: str, approval_id: str | None = None, operator: Operator | None = None,
            progress=None) -> dict:
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
        lines = [t for t, _ in emit(program, m, self.dialect)]
        kw = {}
        if self.backend.name == "grbl":
            kw["on_ack"] = self._pause_hook(lines, operator)
        res: StreamResult = self.backend.stream(lines, progress=progress, **kw)
        return {"estado": "completado" if res.ok else "abortado", "enviadas": res.sent,
                "confirmadas": res.acknowledged, "errores": res.errors, "alarma": res.alarm, "informe": report}

    def _pause_hook(self, lines: list[str], operator: Operator | None):
        """En grbl, M0 deja la máquina en Hold: se espera al operador y se reanuda con «~»."""
        def hook(i: int, sent: str) -> bool:
            if not sent.startswith("M0"):
                return True
            raw = lines[i]
            msg = raw.split("MSG,", 1)[1].rstrip(")").strip() if "MSG," in raw else "pausa"
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
