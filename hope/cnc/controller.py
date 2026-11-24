"""Hope CNC controller: check, simulate and — with human permission — run.

Safety rule (Hope's Charter, guardian): **moving a real machine is ``hardware.control``, human
level**. Every run on a physical controller creates a request bound to the hash of the program, the
machine, the controller and the envelope computed with the *real* offsets read from the controller;
if anything changes (another program, another G54), a new approval is needed. Hope may check and
simulate freely, but it cannot move the machine on its own.

The emergency stop is physical (mushroom button that cuts power to the drivers): no software
replaces it.

How it fits
-----------
* **Uses:** ``gcode`` (interpret), ``check`` (check: a program with errors is not sent),
  ``dialect`` (translate), a ``backends.Backend`` (simulator, grbl or LinuxCNC) and the guardian
  (``guardian.core``, a separate package; imported only when needed).
* **Used by:** the console (``python -m hope.cnc enviar``) and the app (``app.server`` loads the
  guardian with ``load_guardian``).

``run`` flow
------------
1. Interpret and check with the controller's real offsets (not the ones in the YAML).
2. If the backend is physical: ask the guardian for permission. Without approval → «pendiente»
   (pending, with the request id to approve) or «rechazado» (rejected). No guardian → «rechazado».
3. Translate to the dialect and send; on grbl, handle M0 pauses with the operator.

Usage example
-------------
::

    ctl = Controller(machine, make_backend(machine.controller, machine), load_guardian("configs/guardian.yaml"))
    r = ctl.run(texto)                                  # 1st time: {"estado": "pendiente", "solicitud": "…"}
    # a person approves: python -m guardian.cli approve <id> --by <name>
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

if TYPE_CHECKING:                               # only for annotations (does not exist in Python 3.4)
    from typing import Callable
    Operator = Callable[[str], bool]    # receives the pause message; True = continue, False = abort


# ==================================================================================================
# ============================================ GUARDIAN ============================================
# ==================================================================================================


def load_guardian(path: str):
    """Guardian from its YAML without going through ``guardian.cli`` (which pulls in PyTorch): light
    for old PCs. Returns None if the file does not exist or pyyaml is missing; without a guardian, the
    real machine does not move.

    ``path``: ``.yaml`` or ``.json`` file with the configuration (``GuardianConfig``). A JSON file is
    recognized by its extension or because it starts with «{» (it does not need pyyaml, which may be
    missing from the package for old PCs). May raise ``GuardianConfig``'s errors if the file has
    invalid keys: better to fail loudly than to start with a misconfigured guardian."""
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
# =========================================== CONTROLLER ===========================================
# ==================================================================================================


class Controller:
    """Ties together machine, backend and guardian: checks, asks for permission and sends.

    Parameters:
        machine:  the machine (YAML/profile) with limits and offsets.
        backend:  the controller (``backends.make_backend``); ``backend.physical`` says whether it moves
                  something real.
        guardian: ``guardian.core.Guardian`` or None (with None, a physical backend never moves).
        dialect:  force a dialect from ``dialect.DIALECTS``; by default LinuxCNC if the backend is
                  LinuxCNC and grbl in any other case."""

    def __init__(self, machine: Machine, backend: Backend, guardian=None, dialect: str | None = None):
        self.machine = machine
        self.backend = backend
        self.guardian = guardian
        self.dialect = dialect or ("linuxcnc" if backend.name == "linuxcnc" else "grbl")

    def _machine_with_real_offsets(self) -> Machine:
        """Copy of the machine with the offsets (G54…) loaded in the real controller.

        For safety: travel limits are checked in machine coordinates, and that needs the G54 the
        controller will run with, not the one in the YAML (someone may have changed the zero from the
        panel). With the simulator, or if the controller reports nothing, the machine is left as is."""
        real = self.backend.work_offsets() if self.backend.physical else {}
        if not real:
            return self.machine
        offsets = {**self.machine.offsets, **{k: dict(v) for k, v in real.items()}}
        return dataclasses.replace(self.machine, offsets=offsets)

    def review(self, text: str) -> dict:
        """Check ``text`` (G-code) with the real offsets; returns the ``check_program`` report plus
        the text's ``sha256``. Moves nothing."""
        m = self._machine_with_real_offsets()
        r = check_program(interpret(text, m), m)
        r["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return r

    def action(self, report: dict):
        """The ``hardware.control`` action requested from the guardian for this report.

        It includes the hash of the program, the machine, the controller and the envelope in machine
        coordinates: if any of those changes, it is a different action and the previous approval does
        not apply (one cannot approve one program and send another, nor the same one with another G54)."""
        from guardian.core import Action
        return Action("hardware.control", {
            "dispositivo": "cnc", "maquina": self.machine.name, "controlador": self.backend.name,
            "programa_sha256": report["sha256"], "envolvente_maquina": report["envolvente_maquina"],
        }, requested_by="hope.cnc", justification=f"ejecutar un programa de {report['lineas']} líneas "
                                                   f"(~{report['tiempo_s']:.0f} s) en «{self.machine.name}»")

    def run(self, text: str, approval_id: str | None = None, operator: Operator | None = None,
            progress=None) -> dict:
        """Check, ask for permission (if the machine is real) and send the program.

        Parameters:
            text:        G-code as written by the CAM or the person.
            approval_id: id of the request already approved by a person in the guardian (or None).
            operator:    ``operator(message) -> bool`` for M0 pauses (grbl only); None = abort at the
                         first pause.
            progress:    ``progress(sent, acknowledged, total)``, called by the backend.

        Returns a ``dict`` with ``estado`` («rechazado», «pendiente», «completado» or «abortado») and
        the check's ``informe``; depending on the case also ``motivo``, ``solicitud`` (id to approve),
        ``enviadas``, ``confirmadas``, ``errores`` and ``alarma``. Backend connection errors
        (``BackendError``) are not caught here: the caller sees them."""
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
        # What is sent is the translation of the SAME program that was checked (and whose hash was approved).
        lines = [t for t, _ in emit(program, m, self.dialect)]
        kw = {}
        if self.backend.name == "grbl":            # only grbl accepts on_ack (LinuxCNC resumes M0 from its own screen)
            kw["on_ack"] = self._pause_hook(lines, operator)
        res: StreamResult = self.backend.stream(lines, progress=progress, **kw)
        return {"estado": "completado" if res.ok else "abortado", "enviadas": res.sent,
                "confirmadas": res.acknowledged, "errores": res.errors, "alarma": res.alarm, "informe": report}

    def _pause_hook(self, lines: list[str], operator: Operator | None):
        """On grbl, M0 leaves the machine in Hold: wait for the operator and resume with «~».

        Returns the ``on_ack(index, line) -> bool`` function for ``GrblBackend.stream``. ``lines`` is
        the translated program (to read the message: the backend passes the already cleaned line,
        without comments). If the machine goes into alarm or door-open instead of Hold, or the
        operator aborts, it returns False and the backend resets grbl and stops sending."""
        def hook(i: int, sent: str) -> bool:
            """``i``: index of the acknowledged line; ``sent``: the cleaned line (upper case, no
            comments). True = keep sending."""
            if not sent.startswith("M0"):
                return True
            raw = lines[i]                         # emit never yields empty lines: the indices match
            msg = raw.split("MSG,", 1)[1].rstrip(")").strip() if "MSG," in raw else "pausa"
            # grbl acknowledges M0 when it reads it, but only enters Hold once the previous moves are
            # done: poll the state every 0.2 s until it stops (or something worse happens).
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
