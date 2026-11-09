"""Loop Controller v0 a nivel de tarea (glosario §8; ARS §7).

Intenta → CRE critica → si hay errores, Hope vuelve a intentar con la crítica
como contexto. Se detiene cuando:

* la crítica no encuentra errores (``verificado``);
* Hope repite la misma respuesta (``sin_progreso``: no tiene sentido insistir);
* se agotan las iteraciones (``presupuesto``).

La cantidad de iteraciones no es fija: depende de lo que encuentre la crítica.
Esto es adaptación temporal de la tarea: **nunca** modifica pesos ni estructura (ASG §11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from hope.constraints.critic import Critique


@dataclass
class LoopResult:
    text: str
    answer: str | None
    critique: Critique
    iterations: int
    stop_reason: str
    history: list[dict] = field(default_factory=list)


def solve_with_review(attempt: Callable[[str], tuple[str, str | None]],
                      review: Callable[[str, str | None], Critique],
                      max_iterations: int = 3) -> LoopResult:
    """``attempt(feedback)`` devuelve (razonamiento, respuesta); ``review`` la critica."""
    feedback, seen, history = "", set(), []
    best: tuple[str, str | None, Critique] | None = None
    for i in range(1, max_iterations + 1):
        text, answer = attempt(feedback)
        crit = review(text, answer)
        history.append({"iteration": i, "answer": answer, "verdict": crit.verdict,
                        "issues": [x.detail for x in crit.issues]})
        if best is None or _rank(crit) < _rank(best[2]):
            best = (text, answer, crit)
        if crit.verdict == "ok":
            return LoopResult(text, answer, crit, i, "verificado", history)
        if answer in seen:
            return LoopResult(*best, i, "sin_progreso", history)
        seen.add(answer)
        feedback = crit.feedback()
    return LoopResult(*best, max_iterations, "presupuesto", history)


def _rank(c: Critique) -> int:
    return {"ok": 0, "incierto": 1, "debe_declarar_limitacion": 2, "errores": 3}.get(c.verdict, 4)
