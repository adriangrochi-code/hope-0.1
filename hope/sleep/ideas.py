"""Registro de ideas del sueño (SCA §6).

Cada idea conserva su procedencia y su estado epistémico. Nace como
``GENERADO_EN_SUENO`` o ``HIPOTESIS`` y solo cambia con evidencia de la vigilia.
Las ideas descartadas **no se borran**: quedan como estrategias que fallaron,
con las condiciones del fracaso (SCA §6).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hope.epistemic import EpistemicRecord, EpistemicStatus, Evidence


@dataclass
class Idea:
    dream_id: str
    mode: str                                   # ligero | asociativo | analitico | creativo | evolutivo
    description: str
    hypothesis: str
    generating_reasoning: str
    knowledge_used: list[str] = field(default_factory=list)
    related_experiences: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    possible_contradictions: list[str] = field(default_factory=list)
    proposed_tests: list[dict] = field(default_factory=list)     # {"tipo": ..., ...}
    initial_confidence: float = 0.2
    priority: float = 0.0
    status: EpistemicStatus = EpistemicStatus.GENERADO_EN_SUENO
    results: list[dict] = field(default_factory=list)
    conclusions: str = ""
    related_ideas: list[str] = field(default_factory=list)
    history: list[tuple[str, str, str]] = field(default_factory=list)
    idea_id: str = field(default_factory=lambda: f"idea_{uuid.uuid4().hex[:10]}")
    created: float = field(default_factory=time.time)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.mode}|{self.description}".encode()).hexdigest()[:16]

    def apply_result(self, outcome: str | None, evidence_ref: str, detail: str = "",
                     independent: bool = False) -> None:
        """``outcome``: 'respalda', 'refuta' o None (sin conclusión)."""
        rec = EpistemicRecord(self.status, list(self.history))
        if outcome == "respalda":
            rec.transition(EpistemicStatus.RESPALDADO, Evidence(evidence_ref, independent, detail))
        elif outcome == "refuta":
            rec.transition(EpistemicStatus.REFUTADO)
        elif self.status == EpistemicStatus.GENERADO_EN_SUENO:
            rec.transition(EpistemicStatus.HIPOTESIS)
        self.status, self.history = rec.status, rec.history
        self.results.append({"ts": time.time(), "outcome": outcome, "evidence": evidence_ref, "detail": detail})


class IdeaStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ideas: dict[str, Idea] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line:
                    raw = json.loads(line)
                    raw["status"] = EpistemicStatus(raw["status"])
                    raw["history"] = [tuple(h) for h in raw["history"]]
                    self._ideas[raw["idea_id"]] = Idea(**raw)

    def add(self, idea: Idea) -> Idea | None:
        """Agrega la idea si no es repetida (control contra la generación infinita, SCA §9)."""
        if any(i.fingerprint == idea.fingerprint for i in self._ideas.values()):
            return None
        self._ideas[idea.idea_id] = idea
        self.save()
        return idea

    def update(self, idea: Idea) -> None:
        self._ideas[idea.idea_id] = idea
        self.save()

    def all(self) -> list[Idea]:
        return list(self._ideas.values())

    def pending(self) -> list[Idea]:
        open_states = {EpistemicStatus.GENERADO_EN_SUENO, EpistemicStatus.HIPOTESIS, EpistemicStatus.PROBADO}
        return sorted((i for i in self._ideas.values() if i.status in open_states),
                      key=lambda i: -i.priority)

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for i in self._ideas.values():
                d = asdict(i)
                d["status"] = i.status.value
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
