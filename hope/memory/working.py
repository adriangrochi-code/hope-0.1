"""Memoria de trabajo y continuidad de proyectos (AHMC; MDD §11).

``WorkingMemory`` — capacidad acotada para objetivos, variables y elementos
recuperados de la tarea actual. Cuando se llena, sale lo menos relevante. Nunca
contiene «toda la memoria»: solo lo que la tarea necesita ahora.

``ProjectState`` — lo necesario para retomar un trabajo largo después de un
reinicio: objetivos, estado, decisiones con su justificación, hipótesis
pendientes, errores, experimentos, archivos modificados (con hash) y próximos
pasos. Antes de retomar, ``validate`` comprueba que el estado guardado sigue
siendo válido (por ejemplo, que nadie cambió los archivos).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class WorkingItem:
    key: str
    content: str
    relevance: float
    source: dict | None = None


class WorkingMemory:
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.goal: str | None = None
        self.variables: dict[str, object] = {}
        self._items: dict[str, WorkingItem] = {}

    def add(self, item: WorkingItem) -> WorkingItem | None:
        """Agrega o actualiza un elemento. Devuelve el elemento desalojado, si lo hubo."""
        self._items[item.key] = item
        if len(self._items) <= self.capacity:
            return None
        worst = min(self._items.values(), key=lambda i: i.relevance)
        del self._items[worst.key]
        return worst

    def items(self) -> list[WorkingItem]:
        return sorted(self._items.values(), key=lambda i: i.relevance, reverse=True)

    def __contains__(self, key: str) -> bool:
        return key in self._items


def _sha(path: str | Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


@dataclass
class ProjectState:
    objective: str
    subgoals: list[str] = field(default_factory=list)
    state: str = ""
    progress: str = ""
    decisions: list[dict] = field(default_factory=list)          # {decision, justification, ts}
    pending_hypotheses: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    experiments: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    files: dict[str, str | None] = field(default_factory=dict)  # ruta → sha256 al guardar
    needed_resources: list[str] = field(default_factory=list)
    retrieve_before_resuming: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    saved_at: float = 0.0

    def decide(self, decision: str, justification: str) -> None:
        self.decisions.append({"decision": decision, "justification": justification, "ts": time.time()})

    def track_file(self, path: str | Path) -> None:
        self.files[str(path)] = _sha(path)

    def save(self, path: str | Path) -> None:
        self.saved_at = time.time()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProjectState":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> list[str]:
        """Problemas que impiden retomar sin revisar: archivos cambiados o ausentes."""
        issues = []
        for path, sha in self.files.items():
            now = _sha(path)
            if now is None:
                issues.append(f"falta el archivo {path}")
            elif now != sha:
                issues.append(f"el archivo {path} cambió desde que se guardó el estado")
        return issues
