"""TCG/TVE (creación y validación de herramientas) y HTO (composición), con el maestro simulado."""

from pathlib import Path

import pytest

from guardian.core import Guardian, GuardianConfig
from hope.epistemic import LifecycleStatus
from hope.runtime.toolcalls import ToolRunner
from hope.teacher.client import MockTeacher
from hope.tools.builtin import default_catalog
from hope.tools.compose import Composer, CompositionError, type_fits
from hope.tools.creation import (
    ToolCreationError,
    ToolForge,
    ToolProposal,
    security_review,
    validate,
)
from hope.tools.tlm import ToolUsageMemory

VOCALES = [{"entrada": {"texto": "murciélago"}, "salida": 5},
           {"entrada": {"texto": "AEIOU"}, "salida": 5},
           {"entrada": {"texto": "xyz"}, "salida": 0}]


@pytest.fixture
def guardian(tmp_path):
    budget = tmp_path / "configs" / "growth_budget.yaml"
    budget.parent.mkdir()
    budget.write_text(Path("configs/growth_budget.yaml").read_text())
    work = tmp_path / "runs"
    work.mkdir()
    return Guardian(GuardianConfig(workdir=str(work), protected_paths=[str(budget.parent)],
                                   budget_path=str(budget), audit_path=str(tmp_path / "g" / "audit.jsonl"),
                                   approvals_path=str(tmp_path / "g" / "approvals.json")))


@pytest.fixture
def forge(guardian, tmp_path):
    cat = default_catalog(guardian)
    return ToolForge(cat, MockTeacher(guardian=guardian), store_dir=tmp_path / "tcg", timeout_s=8)


def test_valid_tool_is_registered_as_experimental(forge):
    rec = forge.create("contar vocales de un texto", known_cases=VOCALES)
    spec = forge.catalog.spec(rec.tool_id)
    assert rec.tool_id == "tcg_contar_vocales_v1"
    assert spec.status == LifecycleStatus.EXPERIMENTAL
    assert spec.inputs == {"texto": "str"} and spec.outputs == {"cantidad": "int"}
    assert spec.execution == "sandbox" and "TCG" in spec.provenance and "sha256" in spec.provenance
    assert rec.report.checks["seguridad"] == "ok"
    assert rec.report.checks["pruebas_propias"] == "ok"
    assert rec.report.checks["pruebas_independientes"] == "ok"
    # Se ejecuta en el sandbox a través del catálogo (y del guardián), con tipos comprobados.
    assert forge.catalog.call(rec.tool_id, texto="canción") == 3
    with pytest.raises(TypeError):
        forge.catalog.call(rec.tool_id, texto=5)
    assert (forge.store / f"{rec.tool_id}.py").exists()


def test_malicious_tool_is_rejected_with_reason(forge):
    with pytest.raises(ToolCreationError) as e:
        forge.create("duplicar un número (maliciosa)")
    msg = str(e.value)
    assert "os" in msg and "socket" in msg and "open()" in msg
    assert not any(t.startswith("tcg_") for t in forge.catalog.ids())
    assert forge.failures                                      # el motivo queda guardado


def test_broken_tool_is_rejected_and_reason_reaches_next_attempt(forge):
    with pytest.raises(ToolCreationError, match="pruebas_propias"):
        forge.create("duplicar un número (rota)")
    prompt = forge.request_messages("duplicar un número (rota)")[-1]["content"]
    assert "FALLAS ANTERIORES" in prompt and "pruebas_propias" in prompt
    rec = forge.create("duplicar un número (rota)", known_cases=[{"entrada": {"x": 4}, "salida": 8}])
    assert rec.tool_id == "tcg_doble_v1"
    assert not forge.failures                                  # corregida: se limpian las fallas


def test_independent_cases_catch_what_own_tests_miss(forge):
    with pytest.raises(ToolCreationError, match="pruebas_independientes"):
        forge.create("contar vocales (sutil)", known_cases=VOCALES)


def test_time_limit(forge):
    forge.timeout_s = 3
    with pytest.raises(ToolCreationError, match="límite de tiempo"):
        forge.create("duplicar un número (lenta)")


def test_security_review_rules():
    assert security_review("import math\ndef f(x):\n    return math.sqrt(x)\n") == []
    for bad in ("import subprocess", "from urllib import request", "eval('1')", "().__class__.__bases__",
                "__import__('os')", "from . import x"):
        assert security_review(bad), bad


def test_spec_problems_are_reported():
    p = ToolProposal("Mal Nombre", "f", "", "", "", {"x": "tensor"}, {"a": "int", "b": "int"},
                     "def g(y):\n    return y\n", "def test_x():\n    pass\n")
    r = validate(p)
    text = " ".join(r.reasons)
    assert not r.ok and "nombre inválido" in text and "tipo desconocido" in text
    assert "exactamente una salida" in text and "no define la función" in text


def test_consolidation_needs_human_approval(forge, guardian):
    rec = forge.create("contar vocales de un texto", known_cases=VOCALES)
    d = forge.consolidate(rec.tool_id)
    assert not d.allowed and d.pending_request
    assert forge.catalog.spec(rec.tool_id).status == LifecycleStatus.EXPERIMENTAL
    assert not forge.consolidate(rec.tool_id, approval_id=d.pending_request).allowed   # aún sin aprobar
    pending = [k for k, v in guardian.pending().items() if v["action"]["kind"] == "tool.consolidate"]
    guardian.approve(pending[-1], approver="usuario")
    assert forge.consolidate(rec.tool_id, approval_id=pending[-1]).allowed
    assert forge.catalog.spec(rec.tool_id).status == LifecycleStatus.ESTABLE


def test_consolidation_requires_independent_tests(forge):
    rec = forge.create("invertir un texto")                   # sin casos conocidos
    assert rec.report.checks["pruebas_independientes"] == "omitido"
    with pytest.raises(ValueError, match="independientes"):
        forge.consolidate(rec.tool_id)


def test_forge_reloads_from_disk(forge, guardian):
    rec = forge.create("contar vocales de un texto", known_cases=VOCALES)
    cat2 = default_catalog(guardian)
    forge2 = ToolForge(cat2, MockTeacher(), store_dir=forge.store)
    assert rec.tool_id in cat2 and forge2.records[rec.tool_id].proposal.sha256 == rec.proposal.sha256


def test_repeated_task_detection(forge):
    task = "contar las vocales de una palabra"
    assert [forge.observe(task) for _ in range(3)] == [False, False, True]
    assert not forge.observe("calcular aritmética exacta suma") or forge.has_tool_for("aritmética exacta suma")
    forge.create("contar vocales de un texto", known_cases=VOCALES)
    assert forge.has_tool_for(task)


def test_types():
    assert type_fits("int", "number") and type_fits("int", "float") and type_fits("str", "str")
    assert not type_fits("int", "str") and not type_fits("str", "int") and not type_fits("dict", "list")


def test_composition_respects_types_and_records_times(forge, tmp_path):
    forge.create("invertir un texto", known_cases=[{"entrada": {"texto": "ab"}, "salida": "ba"}])
    forge.create("contar vocales de un texto", known_cases=VOCALES)
    forge.create("duplicar un número", known_cases=[{"entrada": {"x": 2}, "salida": 4}])
    tlm = ToolUsageMemory(tmp_path / "tlm.json")
    comp = Composer(forge.catalog, tlm=tlm)

    r = comp.run("invertir > contar_vocales > doble", "murciélago", topic="texto", expected=10)
    assert r.ok and r.value == 10 and r.exact
    assert [s.output for s in r.steps] == ["ogaléicrum", 5, 10]
    assert all(s.seconds > 0 for s in r.steps)

    with pytest.raises(CompositionError, match="produce int"):
        comp.run("contar_vocales > invertir", "hola")         # int → str: se rechaza antes de ejecutar
    with pytest.raises(CompositionError, match="desconocida"):
        comp.run("invertir > no_existe", "x")

    chain = "tcg_invertir_v1 > tcg_contar_vocales_v1 > tcg_doble_v1"
    s = tlm.composition_stats("texto", chain)
    assert s["uses"] == 1 and s["exact_rate"] == 1.0 and s["mean_seconds"] > 0
    assert len(s["step_seconds"]) == 3
    assert tlm.stats["texto"]["tcg_doble_v1"]["useful"] == 1
    assert ToolUsageMemory(tmp_path / "tlm.json").composition_stats("texto", chain)["uses"] == 1   # persiste

    # Una composición equivocada pesa menos que la exacta.
    comp.run("contar_vocales", "hola", topic="texto", expected=3)
    assert tlm.best_compositions("texto")[0] == chain


def test_composition_with_builtins_and_field_selection(guardian):
    comp = Composer(default_catalog(guardian))
    r = comp.run("python.stdout", "print(6 * 7)")
    assert r.ok and r.value.strip() == "42"
    with pytest.raises(CompositionError, match="campo"):
        comp.run("python.nada", "print(1)")


def test_text_calls(forge, tmp_path):
    tlm = ToolUsageMemory()
    runner = ToolRunner(forge.catalog, forge=forge, tlm=tlm, topic="texto")
    out = runner.run("crear_herramienta", "contar vocales de un texto")
    assert out.startswith("creada tcg_contar_vocales_v1 (EXPERIMENTAL")
    runner.run("crear_herramienta", "invertir un texto")
    assert runner.run("componer", "invertir > contar_vocales | murciélago") == "5"
    assert runner.run("componer", "contar_vocales > invertir | hola").startswith("error: tipos incompatibles")
    assert runner.run("crear_herramienta", "duplicar (maliciosa)").startswith("rechazada")
    assert tlm.composition_stats("texto", "tcg_invertir_v1 > tcg_contar_vocales_v1")["ok"] == 1
    text = "Paso 1: ⟦componer: invertir > contar_vocales | murciélago⟧"
    assert runner.run_text(text) == " → 5"
    assert ToolRunner(default_catalog()).run("crear_herramienta", "x") == "sin creador de herramientas disponible"
