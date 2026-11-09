from decimal import Decimal

import pytest

from hope.teacher.client import MockTeacher
from hope.teacher.episode import Episode, parse_episode
from hope.tools.builtin import arith, default_catalog, numbers_equal, symbolic_equal, units
from hope.tools.sandbox import run_python
from hope.tools.verify import verify_episode


def test_arith_is_exact_and_safe():
    assert arith("0,1 + 0,2") == Decimal("0.3")
    assert arith("220 * 0,05") == Decimal("11.00")
    assert numbers_equal("11", arith("220*0.05"))
    for bad in ("__import__('os').system('ls')", "open('x')", "2 ** 100000"):
        with pytest.raises(ValueError):
            arith(bad)


def test_symbolic_runs_in_sandbox():
    assert symbolic_equal("(x+1)^2", "x^2 + 2x + 1") is True
    assert symbolic_equal("(x+1)^2", "x^2 + 1") is False
    assert symbolic_equal("__import__('os').getcwd() +", "1") is None      # error → indeciso, no excepción


def test_units():
    r = units("220 ohm * 0.05 A", "11 V")
    assert r["equal"] and r["same_dimension"]
    assert not units("220 ohm * 0.05 A", "11 W")["same_dimension"]


def test_sandbox_limits():
    assert run_python("print(2+2)").stdout.strip() == "4"
    assert run_python("while True: pass", timeout_s=1).timed_out
    import os
    assert run_python("import os; print(os.environ.get('HOME'))").stdout.strip() == "None"
    assert os.getcwd() not in run_python("import os; print(os.getcwd())").stdout


def test_catalog_search_by_objective():
    cat = default_catalog()
    assert cat.search("comprobar dimensiones de tensión")[0].id == "hte_units_v1"
    assert cat.search("ejecutar un programa python")[0].id == "hte_python_v1"


@pytest.mark.parametrize("topic", ["suma", "ley de ohm"])
def test_verification_accepts_correct_and_rejects_wrong(topic):
    cat = default_catalog()
    good = MockTeacher(error_rate=0.0, seed=1).ask([{"role": "user", "content": f"TEMA: {topic}"}])
    bad = MockTeacher(error_rate=1.0, seed=1).ask([{"role": "user", "content": f"TEMA: {topic}"}])
    assert good.raw_thinking is None and "<think>" not in good.text
    assert verify_episode(cat, Episode(parse_episode(good.text))).verification["status"] == "RESPALDADO"
    assert verify_episode(cat, Episode(parse_episode(bad.text))).verification["status"] == "REFUTADO"


def test_unverifiable_episode_is_inferred():
    ep = Episode({"problema": "¿Por qué el cielo es azul?", "pasos": [{"n": 1, "operacion": "dispersión de Rayleigh"}],
                  "resultado": "porque la luz azul se dispersa más"})
    v = verify_episode(default_catalog(), ep).verification
    assert v["status"] == "INFERIDO" and v["weight"] == 0.3


def test_redondeo_correcto_se_acepta():
    from hope.tools.verify import check_step, rounded_ok
    cat = default_catalog()
    assert rounded_ok("0.333333333", "0.3333") and rounded_ok("1.08333", "1,0833")
    assert not rounded_ok("0.333333333", "0.3334") and not rounded_ok("3.5", "4") and not rounded_ok("3.5", "3.6")
    assert check_step(cat, {"tipo": "aritmetica", "expr": "1 / 3", "esperado": "0.3333"}) == "ok"
    assert check_step(cat, {"tipo": "aritmetica", "expr": "7 / 2", "esperado": "4"}) == "falla"
