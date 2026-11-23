"""Interfaz común de los controladores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


class BackendError(RuntimeError):
    pass


@dataclass
class Status:
    state: str = "desconocido"                         # Idle, Run, Hold, Alarm, Home, Door…
    mpos: dict[str, float] = field(default_factory=dict)
    wpos: dict[str, float] = field(default_factory=dict)
    feed: float = 0.0
    spindle: float = 0.0
    raw: str = ""


@dataclass
class StreamResult:
    sent: int
    acknowledged: int
    errors: list[str]
    aborted: bool = False
    alarm: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors and not self.aborted


Progress = Callable[[int, int, int], None]              # (enviadas, confirmadas, total)


class Backend:
    """Lo que Hope CNC necesita de un controlador. ``physical`` indica si mueve una máquina real."""

    name = "base"
    physical = True

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def status(self) -> Status:
        raise NotImplementedError

    def work_offsets(self) -> dict[str, dict[str, float]]:
        """Orígenes G54–G59 que tiene el controlador (para revisar con los valores reales)."""
        return {}

    def stream(self, lines: list[str], progress: Progress | None = None, stop_on_error: bool = True) -> StreamResult:
        raise NotImplementedError

    def hold(self) -> None: ...

    def resume(self) -> None: ...

    def reset(self) -> None: ...
