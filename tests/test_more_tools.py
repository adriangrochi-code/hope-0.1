import math

import pytest

from hope.runtime.toolcalls import ToolCall, ToolRunner
from hope.tools import code_tools as ct
from hope.tools import eng_tools as et
from hope.tools.builtin import default_catalog
from hope.tools.tlm import ToolUsageMemory

GOOD = "def suma(a, b):\n    return a + b\n"
TESTS = "from solucion import suma\n\ndef test_suma():\n    assert suma(2, 3) == 5\n\ndef test_neg():\n    assert suma(-1, 1) == 0\n"


def test_pytest_tool():
    r = ct.run_tests(GOOD, TESTS)
    assert r["ok"] and r["passed"] == 2 and r["failed"] == 0
    bad = ct.run_tests(GOOD.replace("a + b", "a - b"), TESTS)
    assert not bad["ok"] and bad["failed"] >= 1


def test_lint_tool():
    r = ct.lint("import os\nx = undefined_name\n")
    if "error" in r:
        pytest.skip(r["error"])
    codes = {i["code"] for i in r["issues"]}
    assert "F401" in codes and "F821" in codes
    assert ct.lint(GOOD)["ok"]


def test_compile_c():
    r = ct.compile_run('#include <stdio.h>\nint main(){int s=0;for(int i=0;i<5;i++)s+=i;printf("%d\\n",s);}')
    assert r["ok"] and r["stdout"].strip() == "10"
    err = ct.compile_run("int main(){ return x; }")
    assert err["stage"] == "compilación" and "x" in err["errors"]


def test_search_and_diff(tmp_path):
    (tmp_path / "a.py").write_text("def ley_de_ohm(i, r):\n    return i * r\n")
    hits = ct.search_code(r"def ley_de_ohm", str(tmp_path))
    assert hits and hits[0]["line"] == 1
    assert "+    return i * r * 1" in ct.diff("    return i * r\n", "    return i * r * 1\n")


def test_circuit_dc_divider_and_ac_rc():
    dc = et.circuit("V1 in 0 10; R1 in out 1k; R2 out 0 1k")
    assert dc["tensiones"]["out"] == pytest.approx(5.0)
    assert dc["corrientes_fuentes"]["V1"] == pytest.approx(0.005)
    # RC pasabajos en su frecuencia de corte: |Vout| = Vin/√2 y fase −45°
    r, c = 1000.0, 1e-6
    fc = 1 / (2 * math.pi * r * c)
    ac = et.circuit(f"V1 in 0 1; R1 in out {r}; C1 out 0 {c}; .ac {fc}")
    assert ac["tensiones"]["out"]["modulo"] == pytest.approx(1 / math.sqrt(2), rel=1e-6)
    assert ac["tensiones"]["out"]["fase_grados"] == pytest.approx(-45.0, abs=1e-4)
    with pytest.raises(ValueError):
        et.circuit("V1 a 0 5; R1 b c 100")                 # nodos «b» y «c» flotantes: sin solución única


def test_filter_and_data_tools():
    f = et.filter_response("rc_pasabajos 1k 1u")
    assert f["estable"] and f["corte_3db_hz_aprox"] == pytest.approx(159.155, rel=0.005)
    hp = et.filter_response("rc_pasaaltos 1k 1u")
    assert hp["corte_3db_hz_aprox"] == pytest.approx(159.155, rel=0.005)
    rows = et.sql_csv("nombre,valor\nR1,220\nR2,470\n", "SELECT nombre FROM t WHERE CAST(valor AS INT) > 300")
    assert rows == [{"nombre": "R2"}]
    with pytest.raises(ValueError):
        et.sql_csv("a\n1\n", "DROP TABLE t")
    assert '"valor": "220"' in et.convert("nombre,valor\nR1,220\n", "csv", "json")


def test_runner_new_tools_and_tlm(tmp_path):
    r = ToolRunner(default_catalog())
    assert r.run("pruebas", f"{GOOD} ||| {TESTS}") == "2 pasaron, 0 fallaron"
    assert r.run("compilar", 'c ||| #include <stdio.h>\nint main(){printf("hola");}') == "hola"
    assert '"out": 5.0' in r.run("circuito", "V1 in 0 10; R1 in out 1k; R2 out 0 1k")
    assert r.run("filtro", "rc_pasabajos 1k 1u").startswith("corte")
    tlm = ToolUsageMemory(tmp_path / "uso.json")
    for _ in range(4):
        tlm.record([ToolCall("unidades", "x", "y", True, 0.01)], "ley de Ohm", True)
        tlm.record([ToolCall("python", "x", "error", False, 0.01)], "ley de Ohm", False)
    assert tlm.suggest("ley de Ohm") == ["unidades"]
    assert "unidades" in ToolUsageMemory(tmp_path / "uso.json").hint("ley de Ohm")      # persiste
