import torch

from hope.bench import synthetic as S
from hope.core.toy import build_routed_model
from hope.growth.asg import Outcome, decide, plan_widening
from hope.growth.diagnosis import Cause, Signals, diagnose


def test_rule_order_prefers_cheaper_causes():
    # Etiquetas en conflicto ganan aunque también haya una herramienta que ajuste.
    s = Signals("t", val_acc=0.1, label_conflict_rate=0.3, tool_fit=1.0, best_tool="x")
    assert diagnose(s).cause == Cause.ETIQUETAS_ERRONEAS
    assert diagnose(Signals("t", val_acc=0.95)).cause == Cause.SIN_PROBLEMA
    assert diagnose(Signals("t", val_acc=0.1, tool_fit=1.0, best_tool="x")).cause == Cause.HERRAMIENTA_FALTANTE
    assert diagnose(Signals("t", val_acc=0.1, store_fit=1.0)).cause == Cause.MEMORIA_RECUPERACION


def test_capacity_requires_probe_against_control():
    d = diagnose(Signals("t", val_acc=0.1))
    assert d.cause == Cause.INCONCLUSO and d.needs_capacity_probe
    assert diagnose(Signals("t", val_acc=0.1, capacity_probe_gain=0.2)).cause == Cause.CAPACIDAD_INSUFICIENTE
    opt = diagnose(Signals("t", val_acc=0.1, capacity_probe_gain=0.0, optimization_gain=0.2))
    assert opt.cause == Cause.OPTIMIZACION
    both = diagnose(Signals("t", val_acc=0.1, capacity_probe_gain=0.2, optimization_gain=0.2))
    assert both.cause == Cause.CAPACIDAD_INSUFICIENTE
    assert [h.cause for h in both.hypotheses] == [Cause.CAPACIDAD_INSUFICIENTE, Cause.OPTIMIZACION]


def test_decision_rule():
    assert decide({"a": 0.2, "b": 0.0}, ["a"]).outcome == Outcome.CONSOLIDAR
    assert decide({"a": 0.2, "b": -0.1}, ["a"]).outcome == Outcome.REVERTIR
    assert decide({"a": 0.01, "b": 0.0}, ["a"]).outcome == Outcome.AJUSTAR
    assert decide({"a": 0.0, "b": 0.0}, ["a"]).outcome == Outcome.REVERTIR


def test_targeted_and_uniform_plans_add_equal_params():
    model, _ = build_routed_model(S.FAMILIES, vocab=S.VOCAB, seq_len=S.SEQ, n_classes=S.C)
    targeted = plan_widening(model, [2], 80).change(model).param_delta
    uniform = plan_widening(model, [1, 2, 3, 4, 5], 80).change(model).param_delta
    assert targeted == uniform


def test_bench_cheap_signals_identify_their_causes():
    """Las señales que no requieren entrenar ya separan datos, herramienta y memoria."""
    fams, res = S.make_families(0)
    by = {f.name: f for f in fams}
    assert S.label_conflict_rate(by["noisy"].train) > 0.05
    assert all(S.label_conflict_rate(by[n].train) == 0 for n in ("easy", "capacity", "tool", "memory"))
    assert S.tool_fit(by["tool"].train, res["tools"]) == ("mul_add_mod", 1.0)
    assert S.store_fit(by["memory"].train, res["store"]) == 1.0
    assert S.store_fit(by["capacity"].train, res["store"]) < 0.05
