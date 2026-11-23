"""Simulador: «ejecuta» el programa sin hardware y lleva la posición y el tiempo virtual."""

from __future__ import annotations

from hope.cnc.backends.base import Backend, Status, StreamResult
from hope.cnc.gcode import interpret
from hope.cnc.machine import Machine
from hope.cnc.planner import plan


class SimBackend(Backend):
    name = "sim"
    physical = False

    def __init__(self, machine: Machine | None = None):
        self.machine = machine or Machine()
        self.pos = {a: 0.0 for a in self.machine.axis_names}
        self.clock = 0.0
        self.log: list[str] = []
        self.state = "Idle"

    def status(self) -> Status:
        g54 = self.machine.offset("G54")
        return Status(self.state, dict(self.pos), {a: v - g54[a] for a, v in self.pos.items()})

    def work_offsets(self):
        from hope.cnc.machine import WCS
        return {w: self.machine.offset(w) for w in WCS}

    def stream(self, lines, progress=None, stop_on_error=True) -> StreamResult:
        prog = interpret("\n".join(lines), self.machine, start=self.pos)
        errors = [str(d) for d in prog.errors]
        if errors and stop_on_error:
            return StreamResult(0, 0, errors, aborted=True)
        self.state = "Run"
        pl = plan(prog, self.machine)
        self.clock += pl.total_time
        for op in prog.ops:
            if op.end:
                self.pos = dict(op.end)
        self.log.extend(lines)
        self.state = "Idle"
        if progress:
            progress(len(lines), len(lines), len(lines))
        return StreamResult(len(lines), len(lines), errors)
