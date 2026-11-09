"""Capacidad de programar: trazas, FIM, gramática, autodepuración, editor, buscar/reemplazar y banco."""

import json
import random
import shutil

import numpy as np
import pytest
import torch

from guardian.core import Guardian, GuardianConfig
from hope.core.gramatica import (JSONConstraint, PythonConstraint, generate_code, longest_valid_prefix,
                                 python_prefix_ok)
from hope.core.hn1 import HN1Model, preset
from hope.harness.tools import FolderTools
from hope.memory.pesam import ExperienceLog


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return HN1Model(preset("hn1-tiny")).eval()


@pytest.fixture
def env(tmp_path):
    folder = tmp_path / "proyecto"
    folder.mkdir()
    (folder / "calc.py").write_text("def doble(x):\n    return x + x + 1\n\n\ndef triple(x):\n    return 3 * x\n")
    (folder / "uso.py").write_text("from calc import doble\n\ny = doble(3)\n# doble en un comentario\n")
    (folder / "test_calc.py").write_text("from calc import doble\n\ndef test_doble():\n    assert doble(3) == 6\n")
    outside = tmp_path / "fuera"
    outside.mkdir()
    (outside / "otro.py").write_text("def secreto():\n    return 1\n")
    budget = tmp_path / "configs" / "growth_budget.yaml"
    budget.parent.mkdir()
    budget.write_text(open("configs/growth_budget.yaml").read())
    cfg = GuardianConfig(workdir=str(folder), protected_paths=[str(budget.parent)], budget_path=str(budget),
                         audit_path=str(tmp_path / "g" / "audit.jsonl"),
                         approvals_path=str(tmp_path / "g" / "approvals.json"))
    g = Guardian(cfg)
    pesam = ExperienceLog(tmp_path / "reg" / "pesam.jsonl")
    return {"folder": folder, "outside": outside, "guardian": g, "pesam": pesam, "tmp": tmp_path,
            "tools": FolderTools(folder, g, pesam)}


# --- gramática ---------------------------------------------------------------------
def test_json_automaton_accepts_exactly_valid_json():
    c = JSONConstraint()
    for text, viable, complete in [('{"a": [1, 2.5e-3, true, null, "x\\u00e9"]}', True, True), ('{"a":', True, False),
                                   ('{"a" 1', False, False), ("[1,]", False, False), ("01", False, False),
                                   ("-", True, False), ("12", True, True), ('"é"', True, True), ("{,", False, False),
                                   ("tru", True, False), ("trux", False, False)]:
        assert c.viable(text.encode()) == viable, text
        assert c.complete(text.encode()) == complete, text
    assert not JSONConstraint(root="objeto").viable(b"[1")
    for text in ['{"k": {"x": [1, {"y": -0.5}]}, "z": "a\\"b"}', "[]", "[[], {}]", '"\\n"']:
        assert c.complete(text.encode()) and json.loads(text) is not None


def test_python_prefix_check():
    ok = ["def f(x):\n    return x + 1\n", "x = (1,", "x = 'ab", "if x:\n", "try:\n    pass\n", "@d\n",
          "x = 1 !", "s = '''ab\ncd", "for i in range(3):\n    print(i)\nelse:\n", "x = [1, 2\n"]
    bad = ["x = 'ab\n", "x = )", "x = 1 $", "if x:\ny = 1\n", "try:\n    pass\nx=1\n", "a = 3\nb = a +\n",
           "x ! y", "def f():\n    x=1\n  y", "print(1))", "x =\n"]
    assert all(python_prefix_ok(s) for s in ok)
    assert not any(python_prefix_ok(s) for s in bad)
    assert longest_valid_prefix("x = 1\ny = 2\nz = (") == "x = 1\ny = 2\n"


def test_greedy_default_unchanged_and_constrained_json_is_valid(tiny):
    a = tiny.generate("hola", 20)
    assert a == tiny.generate("hola", 20, constraint=None)
    c = JSONConstraint(root="objeto")
    out = tiny.generate("x", 60, constraint=c)
    assert c.viable(out) and out.lstrip().startswith(b"{")          # todo prefijo generado es JSON viable
    out_r = tiny.generate_reference("x", 30, constraint=JSONConstraint())
    assert JSONConstraint().viable(out_r)


def test_generate_code_returns_compilable_python(tiny):
    head = "def f(x):\n"
    g = generate_code(tiny, "# código\n" + head, code_prefix=head, max_new=60, retries=1)
    assert g.text.startswith(head) or g.text == ""
    if g.text:
        compile(g.text, "<t>", "exec")
    c = PythonConstraint(prefix=head.encode())
    out = tiny.generate("# código\n" + head, 40, constraint=c, temperature=0.9)
    assert PythonConstraint().viable(head.encode() + out)


def test_terminator_only_after_complete_program():
    c = PythonConstraint(terminators=(b"\n```",))
    assert c.allowed(b"x = 1", ord("\n"))
    assert c.allowed(b"x = 1\n", ord("`"))                 # cierra el bloque: el programa está completo
    assert not c.allowed(b"x = (1\n", ord("`"))             # a la mitad no se puede cerrar
    assert not c.allowed(b"x = (", 0)                      # ni terminar (byte de parada)


# --- trazas -----------------------------------------------------------------------------
def test_execution_traces_are_real_and_verified():
    from hope.data.trazas import build, generated_programs, trace_programs
    rows = build(generated_programs(13, seed=1))
    assert len(rows) >= 12
    r = rows[0]
    assert r["traza"].startswith("⟦traza⟧") and "Ejecución:" in r["traza"] and r["salida"].strip()
    assert r["prediccion"].rstrip().endswith("⟦fin⟧") and "¿Qué imprime?" in r["prediccion"]
    t = trace_programs(["a = 2\nb = a * 3\nprint(b)\n"])[0]
    assert [s["linea"] for s in t.steps] == [1, 2, 3]
    assert t.steps[1]["vars"] == {"a": "2", "b": "6"} and t.steps[2]["salida"] == "6\n" and t.output == "6\n"
    f = trace_programs(["def g(n):\n    return n + 1\n\nprint(g(4))\n"])[0]
    assert any(s.get("retorna") == "5" and s["func"] == "g" for s in f.steps)


def test_traces_reject_unsafe_infinite_and_nondeterministic():
    from hope.data.trazas import check_program, trace_programs
    assert check_program("import os\nos.system('ls')") is not None
    assert check_program("open('x')") is not None
    assert trace_programs(["while True:\n    pass\n"], max_steps=50) == [None]          # truncado por pasos
    assert trace_programs(["print(1 / 0)\n"]) == [None]
    s = "s = {" + ", ".join(repr(chr(97 + i) * 3) for i in range(12)) + "}\nprint(s)\n"
    assert trace_programs([s]) == [None]                     # el orden de un set de cadenas cambia entre procesos


def test_trace_sources_corpus_and_exercises(tmp_path):
    from hope.bench.codigo import bank
    from hope.data.trazas import build, corpus_programs, exercise_programs
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "LICENSE").write_text("MIT License\n\nPermission is hereby granted, free of charge, ...")
    (lib / "m.py").write_text('def cuadrado(x):\n    """Cuadrado.\n\n    >>> cuadrado(3)\n    9\n    """\n    return x * x\n')
    other = tmp_path / "sin_licencia"
    other.mkdir()
    (other / "n.py").write_text('def f(x):\n    """>>> f(1)\n    """\n    return x\n')
    progs = corpus_programs(tmp_path)
    assert len(progs) == 1 and progs[0][1]["licencia"] == "MIT License"
    ex = exercise_programs(bank()[:3])
    rows = build(progs + ex)
    assert len(rows) == 4 and "print(repr(cuadrado(3)))" in rows[0]["codigo"] and rows[0]["salida"] == "9\n"


# --- FIM ---------------------------------------------------------------------------------
def test_fim_transform_roundtrip_and_byte_windows():
    from hope.data.fim import END, MID, PRE, SUF, apply_fim, fim_bytes, fim_prompt, fim_transform
    doc = "def f(x):\n    y = x + 1\n    return y\n\nprint(f(2))\n"
    rng = random.Random(0)
    for _ in range(20):
        t = fim_transform(doc, rng, rate=1.0)
        mid = t.split(MID, 1)[1][:-len(END)]
        if t.startswith(PRE):
            pre, suf = t[len(PRE):].split(MID)[0].split(SUF)
        else:
            suf, pre = t[len(SUF):].split(MID)[0].split(PRE)
        assert pre + mid + suf == doc
    assert fim_transform(doc, rng, rate=0.0) == doc
    row = ("ñandú " * 20).encode()
    out = fim_bytes(row, random.Random(1), mode="bytes")
    assert len(out) == len(row)
    x = torch.tensor([list(row[:64])] * 4)
    y, n = apply_fim(x, np.random.default_rng(0), rate=1.0, rows=[True, False, True, False])
    assert n == 2 and torch.equal(y[1], x[1]) and not torch.equal(y[0], x[0]) and y.shape == x.shape
    assert fim_prompt("a", "c") == f"{PRE}a{SUF}c{MID}"


def test_fim_option_in_training_and_generate(tiny):
    from hope.data.fim import fim_transform
    from hope.train.continual import Mixture, Source, Trainer, TrainConfig
    texts = [fim_transform("def f(x):\n    return x\n" * 4, random.Random(i), rate=0.0) for i in range(4)]
    tr = Trainer(HN1Model(preset("hn1-tiny")),
                 TrainConfig(batch=2, seq_bytes=48, precision="fp32", fim_rate=1.0, warmup=1), device="cpu")
    rep = tr.train(Mixture([Source.from_texts("codigo", texts, 1.0)]), 2)
    assert rep.steps == 2
    out = tiny.generate_fim("def f(x):\n", "\n    return y\n", max_new=10)
    assert isinstance(out, str)


# --- editor ------------------------------------------------------------------------------
def _check_for(tools):
    return lambda p, op: tools._check("file.read", {"path": p, "op": op})


def test_editor_structure_definition_references(env, monkeypatch):
    from hope.tools.editor import EditorTools
    e = EditorTools(env["folder"], _check_for(env["tools"]))
    st = e.estructura()
    assert {(s["ruta"], s["nombre"]) for s in st} >= {("calc.py", "doble"), ("calc.py", "triple"), ("uso.py", "y")}
    d = e.definicion("doble")
    assert d[0]["ruta"] == "calc.py" and d[0]["linea"] == 1 and "return x + x + 1" in d[0]["codigo"]
    refs = e.referencias("doble")
    assert {(r["ruta"], r["linea"]) for r in refs} >= {("calc.py", 1), ("uso.py", 3), ("test_calc.py", 4)}
    assert not any(r["ruta"] == "uso.py" and r["linea"] == 4 for r in refs)       # el comentario no cuenta
    monkeypatch.setattr(EditorTools, "_jedi_script", lambda self, p: None)        # sin jedi: índice propio
    assert e.definicion("uso.py:3:5")[0]["ruta"] == "calc.py"
    assert any(r["linea"] == 3 for r in e.referencias("uso.py:3:5"))


def test_editor_is_read_only_and_goes_through_the_guardian(env):
    from hope.harness.tools import NeedsApproval
    from hope.tools.editor import EditorTools
    e = EditorTools(env["folder"], _check_for(env["tools"]))
    before = {p.name: p.read_text() for p in env["folder"].iterdir()}
    e.estructura()
    e.recuperar("triple")
    with pytest.raises(NeedsApproval):
        e.estructura(str(env["outside"]))
    assert {p.name: p.read_text() for p in env["folder"].iterdir()} == before
    assert any(a["accion"] == "file.read" for a in env["tools"].actions)


def test_repo_retrieval_ranks_related_code(env):
    from hope.tools.editor import EditorTools, contexto
    e = EditorTools(env["folder"])
    found = e.recuperar("multiplicar por tres el valor triple", k=2)
    assert found[0]["nombre"] == "triple"
    calls = []
    it = e.recuperar_iterativo("una función", lambda p: calls.append(p) or "def doble(x): return x + x", k=2)
    assert calls and any(r["nombre"] == "doble" for r in it)
    assert "# De calc.py:" in contexto(found)


@pytest.mark.skipif(not (shutil.which("pyright") or shutil.which("mypy")), reason="sin pyright ni mypy")
def test_types_with_checker(env):
    from hope.tools.editor import EditorTools
    (env["folder"] / "tip.py").write_text("def f(x: int) -> int:\n    y = x + 1\n    return y\n\nz: str = f(2)\n")
    e = EditorTools(env["folder"])
    r = e.tipos("tip.py")
    assert not r["ok"] and r["errores"][0]["linea"] == 5
    r = e.tipos("tip.py", 2, "y")
    assert r["tipo"] == "int" and r["errores"][0]["linea"] == 5


def test_editor_tool_calls_in_harness(env):
    from hope.harness.session import HarnessToolRunner
    from hope.tools.builtin import default_catalog
    runner = HarnessToolRunner(default_catalog(env["guardian"]), folder=env["tools"])
    assert "calc.py:1" in runner.run("definicion", "doble")
    assert "uso.py:3" in runner.run("referencias", "doble")
    assert "funcion def triple" in runner.run("estructura", "calc.py")
    assert "def triple" in runner.run("recuperar", "triple tres")
    assert "pendiente" in runner.run("estructura", str(env["outside"]))


# --- buscar/reemplazar --------------------------------------------------------------------
def test_search_replace_parse_and_uniqueness():
    from hope.tools.diff_edits import EditError, SearchReplace, apply_blocks, parse_blocks
    t = ("calc.py\n<<<<<<< BUSCAR\ndef doble(x):\n    return x + x + 1\n=======\ndef doble(x):\n    return 2 * x\n"
         ">>>>>>> REEMPLAZAR\n\n```\nnuevo.py\n<<<<<<< SEARCH\n=======\nprint('hola')\n>>>>>>> REPLACE\n```\n")
    bl = parse_blocks(t)
    assert [b.path for b in bl] == ["calc.py", "nuevo.py"]
    files = {"calc.py": "import os\n\ndef doble(x):   \n    return x + x + 1\n"}
    out = apply_blocks(bl, files.get)
    assert out["calc.py"] == "import os\n\ndef doble(x):\n    return 2 * x\n" and out["nuevo.py"] == "print('hola')\n"
    with pytest.raises(EditError, match="aparece 2 veces"):
        apply_blocks([SearchReplace("a.py", "x", "y")], {"a.py": "x\nx\n"}.get)
    with pytest.raises(EditError, match="no aparece"):
        apply_blocks([SearchReplace("a.py", "zzz", "y")], {"a.py": "x\n"}.get)


def test_apply_checked_rolls_back_when_tests_fail(env):
    t = env["tools"]
    good = t.propose_search_replace("calc.py\n<<<<<<< BUSCAR\n    return x + x + 1\n=======\n    return 2 * x\n"
                                    ">>>>>>> REEMPLAZAR")
    bad = t.propose_search_replace("calc.py\n<<<<<<< BUSCAR\n    return x + x + 1\n=======\n    return x * x\n"
                                   ">>>>>>> REEMPLAZAR")
    orig = (env["folder"] / "calc.py").read_text()
    r = t.apply_edits_checked(bad, "python -m pytest -q -p no:cacheprovider")
    assert not r["aplicado"] and r["revertido"] and (env["folder"] / "calc.py").read_text() == orig
    r = t.apply_edits_checked(good, "python -m pytest -q -p no:cacheprovider")
    assert r["aplicado"] and "return 2 * x" in (env["folder"] / "calc.py").read_text()


def test_session_accepts_search_replace_answers(env):
    from hope.harness.session import HarnessSession
    s = HarnessSession(env["tools"], None, None, pesam=env["pesam"], episodes_path=env["tmp"] / "ep.jsonl",
                       test_command="python -m pytest -q -p no:cacheprovider")
    from hope.harness.session import Side
    side = Side("hope", "calc.py\n<<<<<<< BUSCAR\n    return x + x + 1\n=======\n    return 2 * x\n"
                        ">>>>>>> REEMPLAZAR\nResultado: corregido", "corregido")
    s._evaluate(side, "arreglá doble en calc.py", None, s.test_command)
    assert side.tests["ok"] and side.verified and "return 2 * x" in side.edits["calc.py"]
    s.edits.append(env["tools"].propose_edit("calc.py", side.edits["calc.py"]))
    r = s.apply_verified([0])
    assert r["aplicado"] and s.edits[0] is None


# --- autodepuración ----------------------------------------------------------------------------
def test_run_tests_detailed_and_mutation():
    from hope.runtime.autodepurar import mutants, mutation_score, run_tests_detailed
    rows = run_tests_detailed(["def d(x):\n    return x + x + 1\n", "def d(x):\n    y = 0\n    raise ValueError('no')\n",
                               "x = (\n"], ["assert d(2) == 4"])
    assert rows[0]["pruebas"][0]["error"] == "obtuvo 5, se esperaba 4"
    assert "línea 3: raise ValueError('no')" in rows[1]["pruebas"][0]["error"]
    assert rows[2]["carga"].startswith("SyntaxError")
    good = "def d(x):\n    return 2 * x\n"
    assert len(mutants(good)) >= 2
    mr = mutation_score(good, ["assert d(2) == 4", "assert d(0) == 0"])
    assert mr.score == 1.0 and mr.per_test["assert d(2) == 4"] >= mr.per_test["assert d(0) == 0"]


def test_self_debug_reads_error_fixes_and_validates_tests():
    from hope.runtime.autodepurar import CodeTask, ScriptedCoder, SelfDebugger
    bad, good = "def doble(x):\n    return x + x + 1\n", "def doble(x):\n    return 2 * x\n"
    c = ScriptedCoder([bad, good], tests="3) == 6\nassert doble(-1) == -2\nassert doble(1) == doble(1)\n")
    rep = SelfDebugger(c, rounds=4).solve(CodeTask("Devuelve el doble de x.", "doble", "def doble(x):",
                                                   ["assert doble(2) == 4"]))
    assert rep.ok and rep.verdict == "ok" and rep.code == good
    assert "obtuvo 5, se esperaba 4" in c.prompts[-1]                     # el error volvió al programador
    assert "assert doble(1) == doble(1)" in rep.dropped_tests            # no mata ningún mutante
    assert rep.mutation["puntaje"] >= 0.5


def test_anchors_win_over_a_wrong_generated_test():
    from hope.runtime.autodepurar import CodeTask, ScriptedCoder, SelfDebugger
    good = "def doble(x):\n    return 2 * x\n"
    c = ScriptedCoder([good, good, good], tests="5) == 11\n")
    rep = SelfDebugger(c, rounds=4, mutation=False).solve(
        CodeTask("Doble.", "doble", "def doble(x):", ["assert doble(2) == 4", "assert doble(3) == 6"]))
    assert rep.code == good and rep.ok and "assert doble(5) == 11" in rep.dropped_tests


def test_autodebug_agent_in_budgeted_responder():
    from hope.council.agents import PROFILES, Agent
    from hope.runtime.autodepurar import ScriptedCoder, SelfDebugger, autodebug_agent
    from hope.runtime.budget import BudgetedResponder, SpeedTarget
    fast = Agent(PROFILES["veloz"], lambda t, p, prof: ("def doble(x): return x\nResultado: listo", "listo"))
    dbg = autodebug_agent(SelfDebugger(ScriptedCoder(["def doble(x):\n    return 2 * x\n"]), reflect=False))
    r = BudgetedResponder([fast, dbg], escalation=("veloz", "autodepurador"), clock=iter(range(0, 1000)).__next__)
    rep = r.respond("código", "Escribí def doble(x): que devuelva el doble.\nassert doble(2) == 4",
                    SpeedTarget(tokens_per_s=1e-3))
    assert rep.agent == "autodepurador" and "return 2 * x" in rep.text


def test_folder_autodebug_proposes_passing_edit(env):
    from hope.harness.session import HarnessSession
    from hope.runtime.autodepurar import ScriptedCoder
    s = HarnessSession(env["tools"], None, None, pesam=env["pesam"], episodes_path=env["tmp"] / "ep.jsonl",
                       test_command="python -m pytest -q -p no:cacheprovider")
    coder = ScriptedCoder(replies=[
        "calc.py\n<<<<<<< BUSCAR\nno existe\n=======\nx\n>>>>>>> REEMPLAZAR",
        "calc.py\n<<<<<<< BUSCAR\n    return x + x + 1\n=======\n    return x * x\n>>>>>>> REEMPLAZAR",
        "calc.py\n<<<<<<< BUSCAR\n    return x + x + 1\n=======\n    return 2 * x\n>>>>>>> REEMPLAZAR"])
    orig = (env["folder"] / "calc.py").read_text()
    r = s.autodebug("arreglá doble", "calc.py", coder, rounds=3)
    assert r["ok"] and len(r["propuestas"]) == 1 and (env["folder"] / "calc.py").read_text() == orig
    assert "no aparece" in coder.prompts[1] and "assert" in coder.prompts[2]        # el error volvió
    assert "# De calc.py:" in coder.prompts[0]                                     # contexto recuperado
    s.apply(r["propuestas"][0])
    assert "return 2 * x" in (env["folder"] / "calc.py").read_text()


# --- banco ------------------------------------------------------------------------------------
def test_bank_is_validated_by_execution():
    from hope.bench.codigo import bank, generated_problems, pass_at_k, score_codes
    b = bank()
    assert len(b) >= 40 and all(len(p["ejemplos"]) == 2 and p["ocultas"] for p in b)
    assert all(ok for p in b[:10] for ok, _ in score_codes(p, [p["solucion"]]))
    g = generated_problems(5)
    assert len({p["id"] for p in g}) == 5 and all(score_codes(p, [p["solucion"]])[0][0] for p in g)
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2) and pass_at_k(5, 1, 5) == 1.0 and pass_at_k(5, 0, 1) == 0.0


def test_evaluate_metrics_with_a_fake_solver():
    from hope.bench.codigo import bank, evaluate
    probs = bank()[:3]
    solve = lambda p, i: (p["solucion"] if i == 0 else "def nada(:\n", 40)  # noqa: E731
    r = evaluate(probs, solve, n=2, ks=(1, 2))["resumen"]
    assert r["pass@1"] == pytest.approx(0.5) and r["pass@2"] == 1.0
    assert r["sintaxis_valida"] == pytest.approx(0.5) and r["tokens_por_problema"] == 20.0


def test_rl_with_execution_reward_runs_on_cpu():
    from hope.bench.codigo import code_reward, generated_problems, rl_step
    from hope.train.rlvr import GRPO, RLConfig
    p = generated_problems(1)[0]
    body = p["solucion"].split("\n", 1)[1]
    assert code_reward(p, body) == 1.0 and code_reward(p, "    return None\n") < 1.0
    torch.manual_seed(0)
    rl = GRPO(HN1Model(preset("hn1-tiny")), RLConfig(group=2, max_new=16, curriculum=(), temperature=1.0))
    st = rl_step(rl, [p])
    assert "recompensa_media" in st
