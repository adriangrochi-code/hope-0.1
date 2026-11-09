import pytest

from hope.constraints.critic import critique
from hope.constraints.dynamic import propose, test_cases as run_cases
from hope.constraints.model import extract_quantities, unknown_dimension
from hope.constraints.solver import analyze
from hope.epistemic import EpistemicStatus
from hope.runtime.loop import solve_with_review

P_OHM = "¿Qué tensión cae en un resistor de 220 ohm con 0.05 A?"


def test_quantity_recognition_avoids_false_units():
    qs = extract_quantities("Voy a comprar 3 manzanas a 2 pesos; el resistor tiene 220 ohm y circulan 5 A.")
    assert [(q.value, q.dimension) for q in qs] == [(220, "resistencia"), (5, "corriente")]
    assert unknown_dimension("¿Qué potencia disipa?") == "potencia"
    assert unknown_dimension("What current flows?") == "corriente"


def test_analysis_states():
    a = analyze(P_OHM)
    assert a.status == "determinado" and a.expected[0]["value"] == pytest.approx(11.0)
    assert "Ley de Ohm" in a.hint()
    assert analyze("¿Qué corriente circula por un resistor de 470 ohm?").status == "datos_faltantes"
    assert analyze("Un resistor de 100 ohm tiene 5 V y circulan 2 A. ¿Qué potencia disipa?").status == "datos_contradictorios"
    assert analyze("Una resistencia de -50 ohm con 2 A, ¿qué tensión cae?").out_of_range
    assert analyze("Si compro 3 manzanas, ¿cuánto pago?").status == "sin_restricciones_conocidas"


def test_critic_accepts_correct_reasoning():
    good = "Paso 1: 220 * 0,05 = 11 (ley de Ohm)\nResultado: 11 V"
    c = critique(P_OHM, good)
    assert c.verdict == "ok" and c.checked_steps == 1


@pytest.mark.parametrize("reasoning,kind", [
    ("Paso 1: 220 * 0,05 = 12 (ley de Ohm)\nResultado: 12 V", "aritmetica"),
    ("Paso 1: 220 * 0,05 = 11\nResultado: 11 W", "dimension"),
    ("Paso 1: 220 / 0,05 = 4400\nResultado: 4400 V", "restriccion"),
    ("Paso 1: 220 ohm * 0.05 A = 11 W\nResultado: 11 V", "unidades"),
    ("Pienso que da\nResultado: 11 V", "sin_pasos"),
    ("Paso 1: 220 * 0,05 = 11", "sin_resultado"),
    ("Paso 1: 220 * 0,05 = 11\nPaso 2: 220 * 0,05 = 11\nResultado: 11 V", "repeticion"),
])
def test_critic_detects_errors(reasoning, kind):
    assert kind in [i.kind for i in critique(P_OHM, reasoning).issues]


def test_critic_requires_declaring_limitations():
    p = "¿Qué corriente circula por un resistor de 470 ohm?"
    assert critique(p, "Paso 1: 470 / 2 = 235\nResultado: 235 A").verdict == "debe_declarar_limitacion"
    assert critique(p, "Resultado: falta la tensión aplicada").verdict == "ok"
    q = "Un resistor de 100 ohm tiene 5 V y circulan 2 A. ¿Qué potencia disipa?"
    assert critique(q, "Resultado: 10 W").verdict == "debe_declarar_limitacion"
    assert critique(q, "Resultado: los datos son contradictorios").verdict == "ok"


def test_loop_controller_retries_with_feedback():
    answers = iter([("Paso 1: 220 * 0,05 = 12\nResultado: 12 V", "12 V"),
                    ("Paso 1: 220 * 0,05 = 11\nResultado: 11 V", "11 V")])
    seen_feedback = []

    def attempt(feedback):
        seen_feedback.append(feedback)
        return next(answers)
    res = solve_with_review(attempt, lambda t, a: critique(P_OHM, t, a))
    assert res.stop_reason == "verificado" and res.iterations == 2 and res.answer == "11 V"
    assert seen_feedback[0] == "" and "220 * 0,05 no da 12" in seen_feedback[1]


def test_loop_controller_stops_without_progress():
    res = solve_with_review(lambda f: ("Paso 1: 220 * 0,05 = 12\nResultado: 12 V", "12 V"),
                            lambda t, a: critique(P_OHM, t, a), max_iterations=5)
    assert res.stop_reason == "sin_progreso" and res.iterations == 2


def test_new_constraint_starts_as_hypothesis_and_needs_evidence():
    ct = propose("Divisor de tensión", ["Vout - Vin*R2/(R1+R2)"],
                 {"Vout": ("tension", "volt"), "Vin": ("tension", "volt"), "R1": ("resistencia", "ohm"),
                  "R2": ("resistencia", "ohm")}, "carga despreciable", provenance="episodio de Qwen")
    assert ct.invariant.status == EpistemicStatus.HIPOTESIS
    good = [{"Vin": 10, "R1": r1, "R2": r2, "Vout": 10 * r2 / (r1 + r2)} for r1, r2 in
            [(1, 1), (2, 1), (10, 5), (3, 7), (100, 220)]]
    run_cases(ct, good[:3], source="simulación")
    assert ct.invariant.status == EpistemicStatus.HIPOTESIS          # pocos casos: sigue siendo hipótesis
    run_cases(ct, good[3:], source="simulación")
    assert ct.invariant.status == EpistemicStatus.RESPALDADO
    run_cases(ct, [{"Vin": 10, "R1": 1, "R2": 1, "Vout": 7}], source="medición")
    assert ct.invariant.status == EpistemicStatus.REFUTADO and ct.counterexamples
