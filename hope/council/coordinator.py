"""Coordinador–verificador del consejo interno (la «isla» donde se reúnen las propuestas).

1. **Elige qué agentes consultar**: todos, o los más útiles para el tema según su
   historial, con algo de exploración (UCB1, Auer et al. 2002), para no gastar
   cómputo en agentes que no aportan.
2. Los agentes resuelven **en paralelo**.
3. **Verifica** cada propuesta de forma independiente: crítica de CRE/HCRI
   (aritmética, unidades, dimensión, invariantes, datos faltantes o
   contradictorios) y, si se conoce, la respuesta esperada.
4. **Puntúa**: verificación > coincidencia con otros agentes (consenso) > velocidad.
5. Elige la mejor, registra quién ganó y aprende para el próximo tema.

Ningún agente se aprueba a sí mismo: el veredicto lo da el verificador.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hope.constraints.critic import critique
from hope.council.agents import Agent, Candidate

VERDICT_SCORE = {"ok": 1.0, "incierto": 0.5, "debe_declarar_limitacion": 0.1, "errores": 0.0}


def _norm(ans: str | None) -> str:
    return " ".join((ans or "").lower().replace(",", ".").split())


@dataclass
class Scored:
    candidate: Candidate
    verdict: str
    issues: list[str]
    correct: bool | None
    agreement: float
    score: float


@dataclass
class CouncilResult:
    chosen: Scored | None
    ranking: list[Scored]
    consulted: list[str]
    reasons: list[str] = field(default_factory=list)

    @property
    def answer(self) -> str | None:
        return self.chosen.candidate.answer if self.chosen else None


class AgentUsageMemory:
    """Cuánto rinde cada agente por tema (victorias y verificaciones), persistente."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.stats: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        if self.path and self.path.exists():
            self.stats.update(json.loads(self.path.read_text(encoding="utf-8")))

    def record(self, topic: str, agent: str, reward: float) -> None:
        s = self.stats[topic].setdefault(agent, {"n": 0, "reward": 0.0})
        s["n"] += 1
        s["reward"] += reward
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.stats, indent=2, ensure_ascii=False), encoding="utf-8")

    def select(self, topic: str, agents: list[str], k: int) -> list[str]:
        st = self.stats.get(topic, {})
        total = sum(v["n"] for v in st.values()) + 1

        def ucb(a: str) -> float:
            s = st.get(a)
            if not s or s["n"] == 0:
                return float("inf")                     # sin datos: probarlo
            return s["reward"] / s["n"] + math.sqrt(2 * math.log(total) / s["n"])
        return sorted(agents, key=ucb, reverse=True)[:k]


class Coordinator:
    def __init__(self, agents: list[Agent], *, memory: AgentUsageMemory | None = None, parallel: bool = True,
                 match: Callable[[str | None, str], bool] | None = None, speed_weight: float = 0.05):
        self.agents = {a.name: a for a in agents}
        self.memory = memory or AgentUsageMemory()
        self.parallel, self.speed_weight = parallel, speed_weight
        self.match = match

    def solve(self, topic: str, problem: str, *, expected: str | None = None, k: int | None = None) -> CouncilResult:
        names = list(self.agents)
        consulted = self.memory.select(topic, names, k) if k else names
        if self.parallel and len(consulted) > 1:
            with ThreadPoolExecutor(max_workers=len(consulted)) as pool:
                cands = list(pool.map(lambda n: self.agents[n].solve(topic, problem), consulted))
        else:
            cands = [self.agents[n].solve(topic, problem) for n in consulted]

        votes = Counter(_norm(c.answer) for c in cands if c.answer)
        fastest = min((c.seconds for c in cands), default=1.0) or 1e-9
        scored = []
        for c in cands:
            crit = critique(problem, c.text, c.answer)
            correct = None
            if expected is not None:
                correct = bool(self.match(c.answer, expected)) if self.match else _norm(c.answer) == _norm(expected)
            agreement = (votes[_norm(c.answer)] - 1) / max(len(cands) - 1, 1) if c.answer else 0.0
            score = (2.0 * (1.0 if correct else 0.0) if correct is not None else 0.0) \
                + VERDICT_SCORE.get(crit.verdict, 0.0) + 0.5 * agreement \
                + self.speed_weight * fastest / max(c.seconds, 1e-9)
            scored.append(Scored(c, crit.verdict, [i.detail for i in crit.issues], correct, agreement, score))
        scored.sort(key=lambda s: -s.score)
        chosen = scored[0] if scored else None
        reasons = []
        if chosen:
            reasons.append(f"elegido «{chosen.candidate.agent}»: veredicto {chosen.verdict}, "
                           f"consenso {chosen.agreement:.0%}, {chosen.candidate.seconds:.2f} s")
        for s in scored:                                   # aprendizaje: premio = verificado y/o correcto
            reward = (1.0 if s.correct else 0.0) if s.correct is not None else VERDICT_SCORE.get(s.verdict, 0.0)
            self.memory.record(topic, s.candidate.agent, reward)
        return CouncilResult(chosen, scored, consulted, reasons)
