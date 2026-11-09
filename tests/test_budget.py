"""Velocidad objetivo: usar el tiempo sobrante para mejorar la respuesta (hope/runtime/budget.py)."""

import time

from hope.council.agents import PROFILES, Agent
from hope.runtime.budget import BudgetedResponder, SpeedTarget

PROBLEM = "¿Cuánto es 6 * 7?"


def _agents(log):
    def solver(topic, problem, p):
        log.append(p.name)
        time.sleep(0.01 if p.name == "veloz" else 0.03)
        if p.name == "veloz":
            return "Paso 1: 6 * 7 = 41\nResultado: 41", "41"   # rápido pero equivocado
        return "Paso 1: 6 * 7 = 42\nResultado: 42", "42"
    return [Agent(PROFILES[n], solver) for n in ("veloz", "optimizado", "conservador", "innovador")]


def test_leftover_time_is_used_to_fix_the_draft():
    log = []
    r = BudgetedResponder(_agents(log)).respond("aritmética", PROBLEM,
                                                SpeedTarget(tokens_per_s=200, expected_bytes=400))   # 0,5 s
    assert r.budget_s == 0.5 and r.answer == "42" and r.agent != "veloz"
    assert r.elapsed_s <= r.budget_s
    assert r.stop_reason == "respuesta verificada con consenso" and log[:3] == ["veloz", "optimizado", "conservador"]
    assert r.rounds[0].verdict == "errores"


def test_tight_budget_returns_the_draft_on_time():
    log = []
    r = BudgetedResponder(_agents(log)).respond("aritmética", PROBLEM, SpeedTarget(tokens_per_s=1e6))
    assert log == ["veloz"] and r.answer == "41" and r.stop_reason == "presupuesto de tiempo cumplido"


def test_consistent_mode_uses_the_whole_budget_with_deeper_rounds():
    log = []
    r = BudgetedResponder(_agents(log)).respond("aritmética", PROBLEM,
                                                SpeedTarget(tokens_per_s=200, expected_bytes=320, consistent=True))
    assert any("+" in n for n in log)                      # rondas más profundas (más revisiones/iteraciones)
    assert r.answer == "42" and r.elapsed_s <= r.budget_s and len(r.rounds) >= 5
    assert r.effective_tokens_per_s > 0


def test_knowledge_base_learns_from_errors_and_dedups():
    from hope.epistemic import EpistemicStatus
    from hope.memory.knowledge import KnowledgeBase
    kb = KnowledgeBase()
    log = []

    def solver(topic, problem, p):
        log.append(problem)
        if "41" in problem and "no da 41" in problem:            # una lección guardada lo corrige
            return "Paso 1: 6 * 7 = 42\nResultado: 42", "42"
        return "Paso 1: 6 * 7 = 41\nResultado: 41", "41"
    agents = [Agent(PROFILES["veloz"], solver)]
    rb = BudgetedResponder(agents, knowledge=kb)
    r1 = rb.respond("aritmética", PROBLEM, SpeedTarget(tokens_per_s=1e6))    # solo el borrador: se equivoca
    assert r1.answer == "41"
    st = kb.stats()["por_tipo_y_estado"]
    assert st["leccion"]["RESPALDADO"] == 1 and st["solucion"]["INFERIDO"] == 1
    r2 = rb.respond("aritmética", PROBLEM, SpeedTarget(tokens_per_s=1e6))    # mismo presupuesto, con lo aprendido
    assert r2.answer == "42" and "Errores a evitar" in log[-1]
    # la solución verificada refuta a la anterior (misma pregunta, otra respuesta)
    st = kb.stats()["por_tipo_y_estado"]
    assert st["solucion"].get("RESPALDADO") == 1 and st["solucion"].get("REFUTADO") == 1
    ctx = kb.context("aritmética", PROBLEM)
    assert "Resultado: 42" in ctx.text and "Resultado: 41\n" not in ctx.text.split("Errores a evitar")[0]
    # sin duplicados: aprender lo mismo otra vez refuerza el recuerdo existente
    n = kb.store.stats()["engrams"]
    kb.learn_solution("aritmética", PROBLEM, "Paso 1: 6 * 7 = 42", "42", verified=True)
    assert kb.store.stats()["engrams"] == n
    rep = kb.optimize()
    assert "promoted_to_semantic" in rep and rep["engrams"] >= 2
    assert kb.store.get(next(i for i in ctx.used)).status != EpistemicStatus.REFUTADO
