"""Simulator: «runs» the program without hardware and keeps track of position and virtual time.

WHAT IT IS
    A fake controller that fulfils the ``Backend`` interface. It moves nothing (``physical = False``,
    so the ``Controller`` does not ask the guardian for permission) and answers instantly.

WHAT IT IS FOR
    * Testing a program and the whole chain (profiles → check → translation → sending) without a machine.
    * The automated tests (``tests/test_cnc.py``).
    * It is the default controller: a profile without ``controlador.tipo`` starts in simulation.

HOW IT WORKS
    It re-interprets the translated program with ``hope.cnc.gcode.interpret`` (end position of each
    move) and runs it through ``hope.cnc.planner.plan`` to add up the time the real machine would take
    with its accelerations. There is no real time: ``stream`` finishes right away.

UNITS
    Positions in mm (machine coordinates), ``clock`` in seconds of accumulated virtual time.

EXAMPLE
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
# ============================================ SIMULATOR ===========================================
# ==================================================================================================

class SimBackend(Backend):
    """Simulated controller: keeps position, virtual clock and everything sent.

    Attributes:
        machine: the machine «as seen by the controller» (see ``as_controller``).
        pos:     current machine position per axis (mm). Starts at 0 on every axis.
        clock:   accumulated virtual time of all programs sent (s).
        log:     every line received, in order (so the tests can see what was sent).
        state:   «Idle» or «Run» (grbl style).
    """

    name = "sim"
    physical = False                                   # moves nothing: no guardian approval needed

    def __init__(self, machine: Machine | None = None):
        """``machine``: the profile's effective machine; None = generic ``Machine()``."""
        self.machine = as_controller(machine or Machine())   # receives the already translated program
        self.pos = {a: 0.0 for a in self.machine.axis_names}
        self.clock = 0.0
        self.log: list[str] = []
        self.state = "Idle"

    def status(self) -> Status:
        """Current state. The work position is always computed relative to G54 (the simulator does
        not track which coordinate system the program left active)."""
        g54 = self.machine.offset("G54")
        return Status(self.state, dict(self.pos), {a: v - g54[a] for a, v in self.pos.items()})

    def work_offsets(self):
        """The machine profile's G54–G59 offsets (in the simulator they are the «real» ones)."""
        from hope.cnc.machine import WCS
        return {w: self.machine.offset(w) for w in WCS}

    def stream(self, lines, progress=None, stop_on_error=True) -> StreamResult:
        """«Run» the program: update ``pos``, add its duration to ``clock`` and record it in ``log``.

        Parameters: as in ``Backend.stream``. If the interpreter finds errors and ``stop_on_error`` is
        True, nothing is run (0 sent, aborted); if False it runs anyway and the errors come back in
        the result. ``progress`` is called only once, at the end (there is no gradual progress).
        """
        prog = interpret("\n".join(lines), self.machine, start=self.pos)   # continues from where it stopped
        errors = [str(d) for d in prog.errors]
        if errors and stop_on_error:
            return StreamResult(0, 0, errors, aborted=True)
        self.state = "Run"
        pl = plan(prog, self.machine)                  # same planner that estimates the times on screen
        self.clock += pl.total_time
        for op in prog.ops:
            if op.end:                                 # only the operations that move have an end position
                self.pos = dict(op.end)
        self.log.extend(lines)
        self.state = "Idle"
        if progress:
            progress(len(lines), len(lines), len(lines))
        return StreamResult(len(lines), len(lines), errors)
