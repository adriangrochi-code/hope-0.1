"""Common interface of the real-time controllers (the «contract» every backend fulfils).

WHAT IT IS
    Hope CNC / TISMA CNC does not generate motor pulses: it checks, translates and **delivers** the
    program to a controller that does the real-time work (grbl, LinuxCNC) or to a simulator. This file
    defines the minimum the app needs from any of them, so the rest of the code does not know (or care)
    which one it is talking to.

HOW IT FITS
    * Implemented by ``sim.SimBackend``, ``grbl.GrblBackend`` and ``linuxcnc.LinuxCNCBackend``.
    * Created by ``hope.cnc.backends.make_backend`` from the profile's ``controlador`` section.
    * Used by ``hope.cnc.controller.Controller`` (sending with the guardian), ``hope.cnc.__main__``
      (console) and the app (``hope.cnc.session`` / ``hope.cnc.app.server``, which catch ``BackendError``).

CONVENTIONS
    * Positions in **mm**, feed in **mm/min**, spindle in **rpm** (the controller's units).
    * State names follow grbl (``Idle``, ``Run``, ``Hold``, ``Alarm``, ``Home``, ``Door``) even when
      the backend is a different one: that way the interface shows and decides the same for all of them.
    * ``physical = True`` means «moves a real machine»: with that, the ``Controller`` requires the
      guardian's approval before sending (see CLAUDE.md, «Seguridad» (Safety)).

EXAMPLE
    >>> from hope.cnc.backends import make_backend
    >>> b = make_backend({"tipo": "sim"})
    >>> b.connect(); r = b.stream(["G0 X10 Y5"]); r.ok, b.status().mpos["X"]
    (True, 10.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# ==================================================================================================
# ============================================= ERRORS =============================================
# ==================================================================================================

# Controller or communication failure (port, protocol, missing module, alarm when asked).
# The message is meant to be shown as-is to the operator (in Spanish and with the fix when it is
# known, e.g. «falta pyserial…: pip install pyserial»). The app catches it and shows it.
class BackendError(RuntimeError):
    pass


# ==================================================================================================
# ======================================== STATUS AND RESULT =======================================
# ==================================================================================================

@dataclass
class Status:
    """Snapshot of the controller's state at one instant (the answer to «how are you?»).

    Fields:
        state:   grbl-style state (see the inline comment). ``"desconocido"`` if it could not be read.
        mpos:    **machine** position per axis (mm), relative to the homing zero ($H).
        wpos:    **work** position per axis (mm) = mpos − active offset (G54…G59 + G92).
        feed:    actual feed rate at that moment (mm/min); 0 if the controller does not report it.
        spindle: spindle speed (rpm); 0 if it is off or not reported.
        raw:     the original response text (for debugging and for the log).
    """

    state: str = "desconocido"                         # Idle, Run, Hold, Alarm, Home, Door…
    mpos: dict[str, float] = field(default_factory=dict)
    wpos: dict[str, float] = field(default_factory=dict)
    feed: float = 0.0
    spindle: float = 0.0
    raw: str = ""


@dataclass
class StreamResult:
    """How sending a program ended (what ``Backend.stream`` returns).

    Fields:
        sent:         lines sent to the controller.
        acknowledged: lines the controller confirmed (``ok``/``error`` in grbl; line being executed
                      in LinuxCNC). If aborted, it tells how far the machine got (to resume).
        errors:       error messages already translated into Spanish, with the line that caused them.
        aborted:      True if sending was cut short before finishing (error, alarm, operator, timeout).
        alarm:        alarm text if the controller went into alarm (empty otherwise).
    """

    sent: int
    acknowledged: int
    errors: list[str]
    aborted: bool = False
    alarm: str = ""

    @property
    def ok(self) -> bool:
        """True if the whole program was sent without errors (what the app shows as «completado»)."""
        return not self.errors and not self.aborted


if TYPE_CHECKING:                                     # only for annotations (does not exist in Python 3.4)
    from typing import Callable
    Progress = Callable[[int, int, int], None]          # (sent, acknowledged, total)


# ==================================================================================================
# ============================================ INTERFACE ===========================================
# ==================================================================================================

class Backend:
    """What Hope CNC needs from a controller. ``physical`` tells whether it moves a real machine.

    It is a «hand-made» base class (no ``abc``, for compatibility with the Python 3.4 translator): the
    methods that do not apply to a controller are left as no-ops (``...``); ``status`` and ``stream``
    are mandatory and raise ``NotImplementedError`` if a subclass forgets to define them.

    Contract:
        connect()       opens/wakes the connection. No-op if not needed (simulator, LinuxCNC).
        close()         releases the port.
        status()        returns a ``Status``; raises ``BackendError`` if the controller does not answer.
        work_offsets()  the controller's actual G54–G59 offsets, to check the program against them.
        stream(lines, progress, stop_on_error)
                        sends the program already translated to the controller's dialect and waits
                        for it to finish; returns a ``StreamResult``. ``progress(sent, acknowledged,
                        total)`` is called as it advances (for the interface's progress bar).
        hold()/resume() feed hold and resume: the machine decelerates with its own acceleration,
                        without losing position.
        reset()         aborts the program (in grbl, Ctrl-X). It does NOT replace the physical
                        emergency stop: that one always cuts power in hardware.
    """

    name = "base"                                      # identifies the type («sim», «grbl», «linuxcnc»)
    physical = True                                    # to be safe, assume a real machine (asks the guardian)

    # Opens or wakes the connection to the controller. The base does nothing.
    def connect(self) -> None: ...

    # Closes the port or connection. The base does nothing.
    def close(self) -> None: ...

    def status(self) -> Status:
        """Current controller state (position, state, feed). Each backend implements it."""
        raise NotImplementedError

    def work_offsets(self) -> dict[str, dict[str, float]]:
        """G54–G59 offsets stored in the controller (to check against the real values).

        Returns ``{"G54": {"X": mm, ...}, ...}``. Empty (what the base does) = «unknown»: the program
        is then checked with the offsets stored in the machine profile.
        """
        return {}

    def stream(self, lines: list[str], progress: Progress | None = None, stop_on_error: bool = True) -> StreamResult:
        """Send ``lines`` (G-code already translated, one command per line) and wait for it to finish.

        ``stop_on_error``: if True, the first error stops the machine and the sending (the safe
        choice); if False it is recorded and sending continues. Each backend implements it.
        """
        raise NotImplementedError

    # Feed hold: the machine decelerates without losing position. The base does nothing.
    def hold(self) -> None: ...

    # Resumes after a hold. The base does nothing.
    def resume(self) -> None: ...

    # Aborts the running program (software reset). The base does nothing.
    def reset(self) -> None: ...
