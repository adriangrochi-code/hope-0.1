"""Uso de herramientas: banco sellado, currículo, selección aprendida (TLM) y refuerzo."""

import json

import pytest

from hope.data.tool_curriculum import (FAULT_MESSAGE, GLOSSARY, FaultInjectingRunner, ToolExample, Writer,
                                       answer_ok, b_cadena, b_trivial, build_examples, concept_split, export,
                                       make_runner)
from hope.eval.tool_bench import (BenchTampered, build_bench, load_bench, reference_generate, run_bench,
                                  save_bench, subset_per_category)
from hope.runtime.toolcalls import ToolRunner
from hope.tools.builtin import default_catalog
from hope.tools.tlm import ToolUsageMemory, parse_hint

FAST = {"calculo": 1, "unidades": 1, "trivial": 1, "error": 1, "definicion": 1, "cadena": 1, "fraccion": 1}


@pytest.fixture(scope="module")
def runner():
    return make_runner()


def test_answer_ok_numbers_units_and_text():
    assert answer_ok("0.05 A", "0.05 A") and answer_ok("1.32192E+6", "1321920")
    assert answer_ok("3.82609 V", "3.826086956 V") and not answer_ok("3.8 V", "3.82609 V")
    assert not answer_ok("11 V", "11 A") and not answer_ok(None, "1")
    assert answer_ok("x = 5", "x = 5", "texto") and answer_ok("dice: V = I · R ...", "V = I · R", "contiene")


def test_calc_prints_plain_numbers(runner):
    assert runner.run("calc", "6 * 5") == "30" and runner.run("calc", "4860 * 272") == "1321920"


def test_chain_executes_tools_with_real_results():
    import random
    w = Writer(make_runner())
    ex = b_cadena(random.Random(1), w)
    assert ex is not None and ex.cadena == ["circuito", "calc", "unidades"]
    text = ex.text
    assert text.count("⟧ → ") == 3 and text.rstrip().endswith(f"Resultado: {ex.respuesta}")
    # el resultado de cada paso es exactamente lo que devolvió la herramienta
    assert [c.name for c in w.runner.calls[-3:]] == ["circuito", "calc", "unidades"]
    assert all(c.output in text for c in w.runner.calls[-3:])


def test_trivial_has_no_tool_calls():
    import random
    ex = b_trivial(random.Random(0), Writer(make_runner()))
    assert not ex.necesita_herramienta and "⟦" not in ex.completion and ex.completion.startswith("Resultado:")


def test_fault_injection_and_retry_in_curriculum(runner):
    r = FaultInjectingRunner(runner.catalog, fail_next=1)
    assert r.run("calc", "2 + 2") == FAULT_MESSAGE and r.run("calc", "2 + 2") == "4"
    assert r.injected == [0] and [c.ok for c in r.calls] == [False, True]
    exs = build_examples(20, seed="t-error", mix={"error": 1.0}, runner=runner)
    assert exs and all(e.categoria == "error" for e in exs)
    assert all(e.completion.count("⟧ → ") >= 2 for e in exs)        # falla + reintento (o cambio de herramienta)


def test_curriculum_is_verified_and_excludes_given_problems(runner):
    exs = build_examples(60, seed="t", runner=runner, mix={**FAST, "codigo": 0}, hint_fraction=0.0)
    assert {e.categoria for e in exs} >= {"calculo", "trivial", "error", "definicion", "cadena"}
    for e in exs:
        assert e.text.endswith("\n") and "Resultado: " in e.completion
    defs = [e for e in exs if e.categoria == "definicion" and e.respuesta in GLOSSARY.values()]
    bank_concepts = {GLOSSARY[c] for c in GLOSSARY if concept_split(c) == "banco"}
    assert not any(e.respuesta in bank_concepts for e in defs)       # las definiciones del banco no se entrenan
    excl = {exs[0].problema}
    again = build_examples(60, seed="t", runner=runner, mix={**FAST, "codigo": 0}, exclude=excl)
    assert exs[0].problema not in {e.problema for e in again}


def test_export_writes_training_files(tmp_path):
    stats = export(tmp_path, 40, seed="exp", log=None)
    assert stats["train"] + stats["val"] >= 30
    raw = (tmp_path / "train.bin").read_bytes()
    assert raw.count(b"\x00") == stats["train"] and "⟦".encode() in raw
    from hope.data.tool_curriculum import bench_problems
    rows = [json.loads(x) for x in (tmp_path / "ejemplos.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not {r["problema"] for r in rows} & bench_problems()


def test_bench_reference_scores_perfectly_and_seal(tmp_path, runner):
    items = subset_per_category(build_bench(2, runner=runner), 1)
    assert {i.categoria for i in items} >= {"trivial", "error", "cadena", "codigo", "definicion"}
    path = tmp_path / "banco.jsonl"
    save_bench(items, path)
    loaded = load_bench(path)
    res = run_bench(reference_generate(loaded), loaded)["resumen"]
    assert res["exactitud"] == 1.0 and res["herramienta_correcta"] == 1.0
    assert res["llamadas_innecesarias"] == 0.0 and res["llamadas_faltantes"] == 0.0
    assert res["recuperacion"] == 1.0 and res["cadena_completa"] == 1.0 and res["argumentos_validos"] == 1.0
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BenchTampered):
        load_bench(path)


def test_bench_penalizes_bad_policies(runner):
    items = [i for i in build_bench(1, runner=runner) if i.categoria in ("trivial", "calculo", "error")]

    def always_calc(text, max_new=400, **kw):          # llama a calc siempre, aunque sea trivial
        produced = text.split("\n", 2)[-1]
        if "⟧" not in produced:
            return "Paso 1: ⟦calc: 1 + 1⟧".encode()
        return b"\nResultado: 2\n"

    def never_tools(text, max_new=400, **kw):
        return b"Resultado: 2\n"
    a = run_bench(always_calc, items)["resumen"]
    b = run_bench(never_tools, items)["resumen"]
    assert a["llamadas_innecesarias"] == 1.0 and a["llamadas_faltantes"] == 0.0
    assert b["llamadas_faltantes"] == 1.0 and b["llamadas_innecesarias"] == 0.0 and b["recuperacion"] == 0.0


def test_bench_uses_any_model_generate():
    from hope.core.hn1 import HN1Model, preset
    m = HN1Model(preset("hn1-tiny"))
    items = [i for i in load_bench() if i.categoria == "trivial"][:1]
    res = run_bench(m.generate, items, max_new=40)
    assert res["resumen"]["items"] == 1 and res["items"][0]["bytes_generados"] <= 3 * 40     # bytes inválidos → U+FFFD


def test_tlm_suggests_tools_from_similar_topics_with_time(tmp_path):
    from hope.runtime.toolcalls import ToolCall
    tlm = ToolUsageMemory(tmp_path / "uso.json")
    for _ in range(5):
        tlm.record([ToolCall("unidades", "x", "1 A", True, 0.02)], "ley de Ohm y potencia eléctrica", True)
        tlm.record([ToolCall("calc", "x", "error", False, 0.0)], "ley de Ohm y potencia eléctrica", False)
        tlm.record([ToolCall("circuito", "x", "{}", True, 3.0)], "ley de Ohm y potencia eléctrica", True)
        tlm.record_direct("aritmética de una cifra", True)
    # tema nuevo pero parecido (comparte «ohm»): sugiere unidades primero, circuito después (lento)
    hint = tlm.context_hint("corriente en un resistor", "¿Qué corriente circula por 100 ohm con 5 V?")
    tools = [t for t, _, _ in parse_hint(hint)]
    assert tools[0] == "unidades" and "calc" not in tools        # calc falló siempre en ese tema
    ranked = [t for t, _, _ in tlm.rank("corriente en un resistor", "¿Qué corriente circula por 100 ohm con 5 V?")]
    assert ranked.index("circuito") > ranked.index("unidades")      # mismo éxito, pero tarda 3 s
    assert parse_hint(hint)[0][1] == 1.0
    assert parse_hint(tlm.context_hint("aritmética de una cifra"))[0][0] == "ninguna"
    assert tlm.context_hint("tema sin relación alguna") == ""
    assert "unidades" in ToolUsageMemory(tmp_path / "uso.json").context_hint("ley de Ohm y potencia eléctrica")


def test_tool_quality_rewards_verified_answers():
    from hope.train.tool_rl import tool_quality
    it = ToolExample("calculo", "t", "¿Cuánto es 345 * 12?", "4140", ["calc"], True)
    assert tool_quality(it, "Paso 1: ⟦calc: 345 * 12⟧ → 4140\nResultado: 4140\n") == 1.0
    assert tool_quality(it, "Resultado: 4140\n") == 0.5                   # acertó sin verificar
    assert tool_quality(it, "Paso 1: ⟦calc: 345 * 12⟧ → error: x\nResultado: 4140\n") == 0.5
    assert tool_quality(it, "Resultado: 4141\n") == 0.0
    triv = ToolExample("trivial", "t", "¿Cuánto es 2 + 2?", "4", [], False)
    assert tool_quality(triv, "Resultado: 4\n") == 1.0


def test_tool_rl_smoke_on_cpu():
    from hope.train.tool_rl import main
    rep = main(["--prueba"])
    assert rep["pasos"] == 3 and rep["banco_items"] == 11
    assert set(rep["antes"]) >= {"exactitud", "herramienta_correcta", "recuperacion", "bytes_generados_por_item"}
    assert all("recompensa_media" in h for h in rep["historial"])


def test_runner_is_plain_toolrunner_subclass():
    assert issubclass(FaultInjectingRunner, ToolRunner)
    assert FaultInjectingRunner(default_catalog()).run("calc", "1 + 1") == "2"


def test_cost_only_charged_on_correct_answers():
    """Si el costo se cobra siempre, un grupo de respuestas incorrectas premia la más corta (sin herramientas)."""
    from hope.train.tool_rl import model_bytes, rewarded
    it = ToolExample("calculo", "t", "¿Cuánto es 345 * 12?", "4140", ["calc"], True)
    with_tool = "Paso 1: ⟦calc: 345 * 12⟧ → 4140\nResultado: 4140\n"
    assert model_bytes(with_tool) == len("Paso 1: ⟦calc: 345 * 12⟧\nResultado: 4140\n".encode())
    assert rewarded(it, "Resultado: 1\n") == rewarded(it, "x" * 300 + "\nResultado: 1\n") == 0.0
    assert 0.9 < rewarded(it, with_tool) < 1.0
    assert rewarded(it, with_tool) > rewarded(it, "Paso 1: ⟦calc: 345 * 12⟧ → 4140\n" + "relleno " * 30 + "\nResultado: 4140\n")
