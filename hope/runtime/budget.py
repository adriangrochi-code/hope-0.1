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
                 clock=time.perf_counter, knowledge=None):
        """``knowledge``: ``hope.memory.knowledge.KnowledgeBase`` opcional. Con ella, antes de responder
        se consultan soluciones y lecciones parecidas, y al terminar se guarda lo verificado y los
        errores encontrados (así el próximo reanálisis empieza sabiendo qué evitar)."""
        self.knowledge = knowledge
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
        self._issues = {}
        for c in cands:
            crit = critique(problem, c.text, c.answer)
            self._issues[id(c)] = [i.detail for i in crit.issues]
            agreement = (votes[_norm(c.answer)] - 1) / max(len(cands) - 1, 1) if c.answer else 0.0
            out.append((VERDICT_SCORE.get(crit.verdict, 0.0) + 0.5 * agreement, crit.verdict, c))
        return out

    def _with_review(self, base: str, problem: str, cands: list[Candidate]) -> str:
        """Reanálisis: el problema más los errores que la crítica encontró en los intentos anteriores."""
        self._score(problem, cands)
        notes = [f"- respuesta {c.answer}: {'; '.join(self._issues[id(c)][:2])}"
                 for c in cands if self._issues.get(id(c))]
        return base if not notes else f"{base}\nRevisión de intentos anteriores (no repetir estos errores):\n" + \
            "\n".join(notes[-4:])

    def respond(self, topic: str, problem: str, target: SpeedTarget = SpeedTarget(), *,
                context: str = "") -> BudgetReport:
        """``context``: texto extra para los agentes (p. ej. archivos de la carpeta en el harness F9).
        Se antepone al problema, pero la crítica se hace solo sobre ``problem``: así los números de un
        archivo no se confunden con los datos del enunciado."""
        t0 = self.clock()
        ctx = self.knowledge.context(topic, problem) if self.knowledge is not None else None
        base = "\n\n".join(x for x in (ctx.text if ctx is not None else "", context, problem) if x)
        plan = self._plan()
        first = next(plan)
        cands = [first.solve(topic, base)]
        budget = target.seconds_for(target.expected_bytes or cands[0].bytes_generated)
        cost = {first.profile.name.split("+")[0]: cands[0].seconds}
        rounds = [Round(first.name, cands[0].seconds, "", cands[0].answer, 0.0)]
        reason = "se agotaron las estrategias"
        prompt_base = base
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
            c = agent.solve(topic, self._with_review(prompt_base, problem, cands))
            cost[base] = c.seconds / (1 + 0.5 * level)
            cands.append(c)
            rounds.append(Round(agent.name, c.seconds, "", c.answer, 0.0))
        scored = self._score(problem, cands)
        for r, (s, v, _c) in zip(rounds, scored):
            r.score, r.verdict = s, v
        best_score, best_verdict, best = max(scored, key=lambda s: (s[0], -cands.index(s[2])))
        if self.knowledge is not None:
            agree = sum(1 for c in cands if c.answer and _norm(c.answer) == _norm(best.answer))
            ok = best_verdict == "ok"
            self.knowledge.learn_solution(topic, problem, best.text, best.answer, verified=ok and agree >= 1,
                                          source=f"consejo:{best.agent}")
            for s, v, c in scored:
                if v == "errores":
                    self.knowledge.learn_lesson(topic, problem, c.answer, self._issues.get(id(c), []))
            if ctx is not None and ctx.used:
                self.knowledge.feedback(ctx.used, helped=ok)
        return BudgetReport(best.text, best.answer, best.agent, budget, self.clock() - t0, len(best.text.encode()),
                            rounds, reason)
