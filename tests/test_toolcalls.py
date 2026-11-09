from fractions import Fraction

import pytest
import torch

from hope.core.hn1 import build_hn1
from hope.runtime.toolcalls import ToolRunner, generate_with_tools
from hope.tools import math_tools as mt
from hope.tools.builtin import default_catalog


def test_level0_level1_tools():
    assert mt.fraction_calc("1/3 + 1/6") == Fraction(1, 2)
    assert mt.stats("2 4 4 4 5 5 7 9")["desvio"] == 2.0
    assert mt.solve("2x + 3 = 7") == [{"x": "2"}]
    assert mt.solve("x + y = 10; x - y = 2", "x,y") == [{"x": "6", "y": "4"}]
    assert mt.derivative("x^3") == "3*x**2"
    assert mt.integral("x^2", "x", "0", "3") == "9"
    assert mt.limit("sin(x)/x", "x", "0") == "1"
    assert mt.linalg("det", "[[1,2],[3,4]]") == pytest.approx(-2.0)
    assert mt.linalg("solve", "[[2,0],[0,4]]", "[2,8]") == [1.0, 2.0]
    with pytest.raises(ValueError):
        mt.fraction_calc("__import__('os')")


def test_logic_with_z3():
    pytest.importorskip("z3")
    assert mt.logic("x + y == 10; x - y == 2", "x,y") == {"sat": True, "modelo": {"x": "6", "y": "4"}}
    assert mt.logic("x > 5; x < 3", "x") == {"sat": False}


def test_runner_formats_results_and_records_calls():
    r = ToolRunner(default_catalog())
    assert r.run("calc", "220 * 0,05") == "11"
    assert r.run("fraccion", "1/4 + 1/4") == "1/2"
    assert r.run("unidades", "11 V / (0.05 A) en ohm") == "220 ohm"
    assert r.run("resolver", "2x + 3 = 7") == "x = 2"
    assert r.run("matriz", "det [[1,2],[3,4]]").startswith("-2")
    assert r.run("python", "print(sum(range(10)))") == "45"
    assert r.run("noexiste", "x").startswith("error: herramienta desconocida")
    assert r.run("calc", "open('x')").startswith("error")
    assert [c.ok for c in r.calls] == [True] * 6 + [False, False]


def test_generation_inserts_tool_results():
    r = ToolRunner(default_catalog())
    script = iter([b"Paso 1: \xe2\x9f\xa6calc: 220 * 0,05\xe2\x9f\xa7", b"\nResultado: 11 V\n"])
    seen_prompts = []

    def fake_generate(text, **kw):
        seen_prompts.append(text)
        assert kw["stop_seq"] == "⟧".encode()
        return next(script)
    out = generate_with_tools(None, "Problema: tensión\n", r, generate=fake_generate)
    assert out == "Paso 1: ⟦calc: 220 * 0,05⟧ → 11\nResultado: 11 V\n"
    assert seen_prompts[1].endswith("⟧ → 11")                 # Hope sigue razonando con el resultado
    assert r.calls[0].name == "calc" and r.calls[0].ok


def test_real_model_stops_at_tool_call_marker():
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    out = model.generate("x", max_new=50, stop_seq=b"a")
    ref = model.generate_reference("x", max_new=50, stop_seq=b"a")
    assert out == ref
    assert len(out) == 50 or out.endswith(b"a")
    generate_with_tools(model, "Problema: 2+2\n", ToolRunner(default_catalog()), max_new=40)


def test_episodes_teach_tool_use_and_tool_results_match():
    from hope.data.generators import generate
    from hope.runtime.toolcalls import CALL_RE
    r = ToolRunner(default_catalog())
    checked = 0
    for fam in ("ohm", "aritmetica", "porcentajes", "unidades", "python"):
        for ep in generate(fam, 3, seed=2):
            text = ep.training_text(tool_calls=True)
            for m in CALL_RE.finditer(text):
                written = text[m.end():].split(" → ", 1)[1].split(" (")[0].strip()
                real = r.run(m.group(1), m.group(2))
                from hope.teacher.cycle import answers_match
                assert answers_match(real, written) or real == written, (m.group(0), real, written)
                checked += 1
    assert checked >= 10
