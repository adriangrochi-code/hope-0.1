"""HTE — catálogo de herramientas (Herramientas §3.1 y §16).

Cada herramienta tiene una ficha con el formato del documento de Herramientas
(§16) y un estado de ciclo de vida. Se puede buscar por objetivo, no solo por
nombre. Toda ejecución pasa por el guardián.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from guardian.core import Action, Guardian
from hope.epistemic import LifecycleStatus


@dataclass
class ToolSpec:
    id: str
    name: str
    version: str
    domain: str
    objective: str
    description: str
    inputs: dict
    outputs: dict
    status: LifecycleStatus = LifecycleStatus.ESTABLE
    units_and_formats: str = ""
    dependencies: list[str] = field(default_factory=list)
    execution: str = "en proceso"               # o "sandbox"
    cost: str = "bajo"
    expected_precision: str = ""
    limitations: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    usage_examples: list[str] = field(default_factory=list)
    do_not_use_when: list[str] = field(default_factory=list)
    compatible_with: list[str] = field(default_factory=list)
    changelog: list[str] = field(default_factory=list)
    provenance: str = ""


class ToolDenied(PermissionError):
    pass


class Catalog:
    def __init__(self, guardian: Guardian | None = None):
        self.guardian = guardian
        self._specs: dict[str, ToolSpec] = {}
        self._fns: dict[str, Callable] = {}

    def register(self, spec: ToolSpec, fn: Callable) -> None:
        if spec.id in self._specs:
            raise ValueError(f"herramienta duplicada: {spec.id}")
        self._specs[spec.id] = spec
        self._fns[spec.id] = fn

    def spec(self, tool_id: str) -> ToolSpec:
        return self._specs[tool_id]

    def __contains__(self, tool_id: str) -> bool:
        return tool_id in self._specs

    def ids(self) -> list[str]:
        return list(self._specs)

    def function(self, tool_id: str) -> Callable:
        """La función registrada (para inspeccionar su firma; ejecutar siempre con ``call``)."""
        return self._fns[tool_id]

    def search(self, objective: str) -> list[ToolSpec]:
        """Búsqueda por objetivo: palabras en común con objetivo, descripción y dominio."""
        words = set(objective.lower().split())
        scored = []
        for s in self._specs.values():
            if s.status == LifecycleStatus.BLOQUEADO:
                continue
            text = f"{s.objective} {s.description} {s.domain} {s.name}".lower()
            score = sum(w in text for w in words)
            if score:
                scored.append((score, s))
        return [s for _, s in sorted(scored, key=lambda t: -t[0])]

    def call(self, tool_id: str, **kwargs):
        spec = self._specs[tool_id]
        if spec.status in (LifecycleStatus.BLOQUEADO, LifecycleStatus.DEPRECADO):
            raise ToolDenied(f"{tool_id} está {spec.status.value}")
        if self.guardian is not None:
            d = self.guardian.decide(Action("tool.execute", {"tool": tool_id, "sandboxed": True},
                                            requested_by="hte"))
            if not d.allowed:
                raise ToolDenied("; ".join(d.reasons))
        return self._fns[tool_id](**kwargs)
