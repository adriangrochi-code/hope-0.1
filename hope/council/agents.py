"""Agentes internos de Hope con personalidades y objetivos distintos (decisión del usuario).

Cada agente es una **forma distinta de usar** el núcleo de Hope (o una parte de
él, p. ej. con sus propios adaptadores): cambia la exploración, la revisión, el
uso de memoria y herramientas, la profundidad del núcleo y el largo de la
respuesta. No son modelos completos separados (Language Core §7: evitar redes
duplicadas), aunque cualquiera puede correr como servidor aparte (``server.py``).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass
class AgentProfile:
    name: str
    goal: str
    instruction: str                 # se antepone al problema (su «personalidad»)
    temperature: float = 0.0
    review_iterations: int = 1       # intentos con crítica (CRE/HCRI)
    core_iterations: int | None = None   # iteraciones del núcleo recurrente (None = adaptativo)
    use_tools: bool = True
    use_memory: bool = True
    max_bytes: int = 400
    adapter: str | None = None       # adaptadores propios (hope/plasticity/adapters.py), si los hay


PROFILES: dict[str, AgentProfile] = {
    "innovador": AgentProfile(
        "innovador", "encontrar caminos nuevos",
        "Buscá un enfoque distinto al habitual; probá otra representación del problema.",
        temperature=0.9, review_iterations=1, use_memory=False),
    "conservador": AgentProfile(
        "conservador", "no equivocarse",
        "Usá procedimientos conocidos, revisá cada paso y declará si faltan datos.",
        temperature=0.0, review_iterations=3, use_memory=True),
    "optimizado": AgentProfile(
        "optimizado", "la solución con menos pasos",
        "Resolvé con la menor cantidad de pasos; usá herramientas para los cálculos.",
        temperature=0.3, review_iterations=2, max_bytes=250),
    "veloz": AgentProfile(
        "veloz", "responder lo antes posible",
        "Respondé directo.",
        temperature=0.0, review_iterations=1, core_iterations=1, use_tools=False, use_memory=False,
        max_bytes=120),
}


@dataclass
class Candidate:
    agent: str
    text: str
    answer: str | None
    seconds: float
    bytes_generated: int
    meta: dict = field(default_factory=dict)


# Un «solver» resuelve un problema con un perfil: (tema, problema, perfil) → (razonamiento, respuesta).
Solver = Callable[[str, str, AgentProfile], tuple[str, str | None]]


class Agent:
    def __init__(self, profile: AgentProfile, solver: Solver):
        self.profile, self.solver = profile, solver

    @property
    def name(self) -> str:
        return self.profile.name

    def solve(self, topic: str, problem: str) -> Candidate:
        t0 = time.perf_counter()
        text, answer = self.solver(topic, problem, self.profile)
        return Candidate(self.name, text, answer, time.perf_counter() - t0, len(text.encode()))

    def describe(self) -> dict:
        return asdict(self.profile)


def hope_solver(model, catalog=None, memory=None) -> Solver:
    """Solver que usa el núcleo HN-1 con las opciones de cada perfil."""
    from hope.constraints.critic import critique
    from hope.constraints.solver import analyze
    from hope.runtime.loop import solve_with_review
    from hope.runtime.toolcalls import ToolRunner, generate_with_tools
    from hope.teacher.cycle import extract_answer

    def solve(topic: str, problem: str, p: AgentProfile):
        base = f"{p.instruction}\nTema: {topic}\nProblema: {problem}\n"
        if p.use_memory and memory is not None:
            from hope.memory.retrieval import retrieve
            found = retrieve(memory, f"{topic} {problem}", k=1, min_score=0.3)
            if found.found:
                base = f"Ejemplo resuelto:\n{found.items[0].engram.content}\n\n" + base
        an = analyze(problem)
        if p.review_iterations > 1 and an.hint():
            base += an.hint() + "\n"
        runner = ToolRunner(catalog, memory) if (p.use_tools and catalog is not None) else None

        def gen(prompt: str, temp: float) -> str:
            if runner is not None:
                return generate_with_tools(model, prompt, runner, max_new=p.max_bytes, temperature=temp)
            return model.generate(prompt, max_new=p.max_bytes, temperature=temp,
                                  stop_after_line=b"Resultado:").decode("utf-8", errors="replace")

        def attempt(feedback: str):
            text = gen(base + (feedback + "\n" if feedback else ""), p.temperature)
            return text, extract_answer(text)
        res = solve_with_review(attempt, lambda t, a: critique(problem, t, a, an), max_iterations=p.review_iterations)
        return res.text, res.answer
    return solve
