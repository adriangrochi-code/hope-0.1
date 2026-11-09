"""Velocidad objetivo: responder a un ritmo fijo y usar el tiempo sobrante para mejorar la calidad.

Pedido del usuario: «quiero que respondas a un equivalente de 200 tokens por segundo; si
llegás antes, usá lo que sobra para revisar, buscar errores y mejorar el resultado».

**Cómo funciona** (algoritmo *anytime*: siempre hay una respuesta lista y mejora con el tiempo):

1. **Borrador rápido** con el agente veloz (1 iteración del núcleo, sin herramientas).
2. **Presupuesto de tiempo** = largo de la respuesta ÷ velocidad objetivo. Por ejemplo, 400
   bytes ≈ 100 tokens a 200 tokens/s = 0,5 s. Si se indica ``expected_bytes``, se usa ese
   largo; si no, el del borrador.
3. **Mientras sobre tiempo**, se escala en calidad: agente optimizado (herramientas), agente
   conservador (memoria, 3 revisiones con el crítico), agente innovador (otro enfoque) y,
   después, rondas cada vez más profundas: más iteraciones del núcleo y más revisiones.
   Antes de cada ronda se estima su duración (la última vez que corrió ese agente) y **no se
   empieza si no entra en el tiempo que queda**: la respuesta nunca llega tarde.
4. Cada candidato se verifica de forma independiente (CRE/HCRI) y se compara con los demás
   (consenso). Gana verificación > consenso > velocidad, como en el coordinador.
5. **Corte anticipado** opcional (``stop_when_confident``): si la mejor respuesta está
   verificada y otra independiente coincide, no hace falta seguir. Con ``consistent=True``
   se usa todo el presupuesto aunque haya confianza (más candidatos, más consenso).

Un «token equivalente» son 4 bytes (lo que ocupa en promedio un token de un BPE), así que
las velocidades se pueden comparar con las de modelos de tokens.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field, replace

from hope.constraints.critic import critique
from hope.council.agents import Agent, Candidate
from hope.council.coordinator import VERDICT_SCORE, _norm

BYTES_PER_TOKEN = 4.0
ESCALATION = ("veloz", "optimizado", "conservador", "innovador")


@dataclass
class SpeedTarget:
    tokens_per_s: float = 200.0
    expected_bytes: int | None = None        # largo esperado de la respuesta (None = el del borrador)
    min_seconds: float = 0.0
    max_seconds: float | None = None
    stop_when_confident: bool = True
    consistent: bool = False                 # usar todo el presupuesto aunque ya haya confianza
    safety: float = 0.9                      # fracción del presupuesto que se planifica (margen)

    @property
    def bytes_per_s(self) -> float:
        return self.tokens_per_s * BYTES_PER_TOKEN

    def seconds_for(self, nbytes: int) -> float:
        s = max(self.min_seconds, nbytes / self.bytes_per_s)
        return min(s, self.max_seconds) if self.max_seconds is not None else s


@dataclass
class Round:
    agent: str
    seconds: float
    verdict: str
    answer: str | None
    score: float


@dataclass
class BudgetReport:
    text: str
    answer: str | None
    agent: str
    budget_s: float
    elapsed_s: float
    output_bytes: int
    rounds: list[Round] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def effective_tokens_per_s(self) -> float:
        """Velocidad que percibe el usuario (largo entregado ÷ tiempo total)."""
        return self.output_bytes / BYTES_PER_TOKEN / max(self.elapsed_s, 1e-9)

    @property
    def improved(self) -> bool:
        return len(self.rounds) > 1 and self.agent != self.rounds[0].agent


def _deepen(agent: Agent, level: int) -> Agent:
    """Versión más profunda de un agente: más revisiones y más iteraciones del núcleo."""
    p = agent.profile
    core = (p.core_iterations or 4) + level
    return Agent(replace(p, name=f"{p.name}+{level}", review_iterations=p.review_iterations + level,
                         core_iterations=core, temperature=min(1.0, p.temperature + 0.1 * level)), agent.solver)


class BudgetedResponder:
    def __init__(self, agents: list[Agent], *, escalation: tuple[str, ...] = ESCALATION, max_deepening: int = 6,
                 clock=time.perf_counter):
        self.agents = {a.name: a for a in agents}
        self.order = [n for n in escalation if n in self.agents] or list(self.agents)
        self.max_deepening = max_deepening
        self.clock = clock

    def _plan(self):
        yield from (self.agents[n] for n in self.order)
        for level in range(1, self.max_deepening + 1):       # rondas más profundas, de mayor calidad
            for n in reversed(self.order[1:] or self.order):
                yield _deepen(self.agents[n], level)

    def _score(self, problem: str, cands: list[Candidate]) -> list[tuple[float, str, Candidate]]:
        votes = Counter(_norm(c.answer) for c in cands if c.answer)
        out = []
        for c in cands:
            verdict = critique(problem, c.text, c.answer).verdict
            agreement = (votes[_norm(c.answer)] - 1) / max(len(cands) - 1, 1) if c.answer else 0.0
            out.append((VERDICT_SCORE.get(verdict, 0.0) + 0.5 * agreement, verdict, c))
        return out

    def respond(self, topic: str, problem: str, target: SpeedTarget = SpeedTarget()) -> BudgetReport:
        t0 = self.clock()
        plan = self._plan()
        first = next(plan)
        cands = [first.solve(topic, problem)]
        budget = target.seconds_for(target.expected_bytes or cands[0].bytes_generated)
        cost = {first.profile.name.split("+")[0]: cands[0].seconds}
        rounds = [Round(first.name, cands[0].seconds, "", cands[0].answer, 0.0)]
        reason = "se agotaron las estrategias"
        for agent in plan:
            elapsed = self.clock() - t0
            base = agent.profile.name.split("+")[0]
            # Estimación: lo que tardó ese agente antes (más profundo = proporcionalmente más), o el
            # doble del borrador si todavía no corrió.
            level = int(agent.name.split("+")[1]) if "+" in agent.name else 0
            estimate = cost.get(base, 2.0 * cands[0].seconds) * (1 + 0.5 * level)
            if elapsed + estimate > budget * target.safety:
                reason = "presupuesto de tiempo cumplido"
                break
            scored = self._score(problem, cands)
            best = max(scored, key=lambda s: s[0])
            agree = sum(1 for s in scored if s[2].answer and _norm(s[2].answer) == _norm(best[2].answer))
            if (target.stop_when_confident and not target.consistent and best[1] == "ok" and agree >= 2):
                reason = "respuesta verificada con consenso"
                break
            c = agent.solve(topic, problem)
            cost[base] = c.seconds / (1 + 0.5 * level)
            cands.append(c)
            rounds.append(Round(agent.name, c.seconds, "", c.answer, 0.0))
        scored = self._score(problem, cands)
        for r, (s, v, _c) in zip(rounds, scored):
            r.score, r.verdict = s, v
        best_score, _v, best = max(scored, key=lambda s: (s[0], -cands.index(s[2])))
        return BudgetReport(best.text, best.answer, best.agent, budget, self.clock() - t0, len(best.text.encode()),
                            rounds, reason)
