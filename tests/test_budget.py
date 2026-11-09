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
