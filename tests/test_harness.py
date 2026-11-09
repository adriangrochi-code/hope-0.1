"""Harness de trabajo F9 (hope/harness/): carpeta fija, guardián, Hope vs. Qwen en paralelo, registro y episodios."""

import json
import sys
import threading
import time

import pytest

from guardian.core import Guardian, GuardianConfig
from hope.council.agents import PROFILES, Agent
from hope.harness.cli import Console, harness_guardian, main
from hope.harness.session import HarnessSession
from hope.harness.tools import FolderTools, HarnessDenied, NeedsApproval
from hope.memory.knowledge import KnowledgeBase, Notebook
from hope.memory.pesam import ExperienceLog
from hope.runtime.budget import BudgetedResponder, SpeedTarget
from hope.teacher.client import MockTeacher, Teacher, TeacherConfig
from hope.tools.builtin import default_catalog

PROBLEM = "¿Cuánto es 6 * 7?"
GOOD = "Paso 1: 6 * 7 = 42\nResultado: 42"
BAD = "Paso 1: 6 * 7 = 41\nResultado: 41"


@pytest.fixture
def env(tmp_path):
    folder = tmp_path / "proyecto"
    folder.mkdir()
    (folder / "notas.txt").write_text("hola desde la carpeta\n")
    (folder / "calc.py").write_text("def doble(x):\n    return x + x + 1\n")
    (folder / "test_calc.py").write_text("from calc import doble\n\ndef test_doble():\n    assert doble(3) == 6\n")
    (folder / ".git").mkdir()
    (folder / ".git" / "config").write_text("[core]\n")
    outside = tmp_path / "fuera"
    outside.mkdir()
    (outside / "secreto.txt").write_text("no deberías leer esto sin permiso")
    budget = tmp_path / "configs" / "growth_budget.yaml"
    budget.parent.mkdir()
    budget.write_text(open("configs/growth_budget.yaml").read())
    cfg = GuardianConfig(workdir=str(folder), protected_paths=[str(budget.parent), str(folder / ".git")],
                         budget_path=str(budget), audit_path=str(tmp_path / "g" / "audit.jsonl"),
                         approvals_path=str(tmp_path / "g" / "approvals.json"))
    guardian = Guardian(cfg)
    pesam = ExperienceLog(tmp_path / "reg" / "pesam.jsonl")
    return {"folder": folder, "outside": outside, "guardian": guardian, "pesam": pesam, "tmp": tmp_path,
            "tools": FolderTools(folder, guardian, pesam)}


class FixedTeacher(Teacher):
    """Qwen simulado que responde siempre lo mismo (y anota en qué hilo corrió)."""

    def __init__(self, text, guardian=None, delay=0.0):
        super().__init__(TeacherConfig(endpoint="http://127.0.0.1:8080/v1", model="qwen-falso"), guardian)
        self.text, self.delay, self.threads = text, delay, []

    def _complete(self, messages):
        self.threads.append(threading.current_thread().name)
        self.prompt = messages[-1]["content"]
        time.sleep(self.delay)
        return self.text, {}


def _responder(text, answer, delay=0.0, log=None):
    def solver(topic, problem, p):
        if log is not None:
            log.append(problem)
        time.sleep(delay)
        return text, answer
    return BudgetedResponder([Agent(PROFILES["veloz"], solver)])


def _session(env, responder, teacher, **kw):
    kb = kw.pop("knowledge", None)
    return HarnessSession(env["tools"], responder, teacher, pesam=env["pesam"],
                          catalog=default_catalog(env["guardian"]), knowledge=kb,
                          notebook=Notebook(kb) if kb else None,
                          target=SpeedTarget(tokens_per_s=1e6), episodes_path=env["tmp"] / "reg" / "episodios.jsonl",
                          **kw)


def _episodes(env):
    p = env["tmp"] / "reg" / "episodios.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# --- herramientas de carpeta y guardián --------------------------------------------------
def test_inside_folder_is_allowed(env):
    t = env["tools"]
    assert "hola" in t.read("notas.txt")
    assert "calc.py" in t.list(".") and ".git/" in t.list(".")
    hits = t.search("def doble")
    assert hits and hits[0]["path"] == "calc.py" and hits[0]["line"] == 1


def test_nothing_outside_the_folder_without_approval(env):
    t, secret = env["tools"], env["outside"] / "secreto.txt"
    for call in (lambda: t.read(str(secret)), lambda: t.read("../fuera/secreto.txt"),
                 lambda: t.list(str(env["outside"])), lambda: t.search("secreto", str(env["outside"]))):
        with pytest.raises(NeedsApproval):
            call()
    edit = t.propose_edit(str(env["outside"] / "nuevo.txt"), "x")       # archivo nuevo: no hay que leer nada
    with pytest.raises(NeedsApproval):
        t.apply_edit(edit)
    assert not (env["outside"] / "nuevo.txt").exists()
    link = env["folder"] / "atajo"                                         # un enlace no sirve para escapar
    link.symlink_to(env["outside"])
    with pytest.raises(NeedsApproval):
        t.read("atajo/secreto.txt")
    assert all(a["resultado"] == "pendiente" for a in t.actions)
    assert len(env["guardian"].pending()) == len(t.actions)


def test_protected_paths_are_denied_even_with_approval(env):
    t = env["tools"]
    edit = t.propose_edit(".git/config", "[core]\nhooksPath = /tmp\n")
    with pytest.raises(HarnessDenied):
        t.apply_edit(edit)
    assert (env["folder"] / ".git" / "config").read_text() == "[core]\n"
    budget = env["tmp"] / "configs" / "growth_budget.yaml"
    with pytest.raises(NeedsApproval):                         # leerlo ya está fuera de la carpeta
        t.propose_edit(str(budget), "max: 1e12\n")
    import hashlib
    from hope.harness.tools import Edit
    old = budget.read_text()
    edit = Edit(str(budget), "growth_budget.yaml", old, "max: 1e12\n", "", "hope",
                hashlib.sha256(old.encode()).hexdigest())
    with pytest.raises(HarnessDenied):                         # escribirlo: denegado, sin solicitud
        t.apply_edit(edit)
    assert budget.read_text() == old


def test_approval_is_for_that_action_and_used_once(env):
    t, secret = env["tools"], env["outside"] / "secreto.txt"
    with pytest.raises(NeedsApproval) as e:
        t.read(str(secret))
    s = _session(env, None, None)
    s.approve(e.value.request_id, "usuario")
    with pytest.raises(NeedsApproval):                       # otra acción: la aprobación no sirve
        t.read(str(env["outside"]))
    assert "no deberías" in t.read(str(secret))            # la aprobada: una sola vez
    with pytest.raises(NeedsApproval):
        t.read(str(secret))


def test_edit_shows_diff_and_writes_only_when_applied(env):
    t = env["tools"]
    edit = t.propose_edit("calc.py", "def doble(x):\n    return 2 * x\n")
    assert "-    return x + x + 1" in edit.diff and "+    return 2 * x" in edit.diff
    assert "x + x + 1" in (env["folder"] / "calc.py").read_text()
    t.apply_edit(edit)
    assert "2 * x" in (env["folder"] / "calc.py").read_text()
    stale = t.propose_edit("notas.txt", "otro")
    (env["folder"] / "notas.txt").write_text("cambió por fuera")
    with pytest.raises(RuntimeError):
        t.apply_edit(stale)


def test_commands_run_in_the_sandbox_on_a_copy(env):
    t = env["tools"]
    r = t.run([sys.executable, "-c", "import os; open('calc.py','w').write('roto'); print(sorted(os.listdir('.')))"])
    assert r.ok and "calc.py" in r.stdout and ".git" not in r.stdout
    assert "def doble" in (env["folder"] / "calc.py").read_text()          # el original no cambió
    fixed = {"calc.py": "def doble(x):\n    return 2 * x\n"}
    probe = "import calc; print(calc.doble(3))"
    assert t.run([sys.executable, "-c", probe]).stdout.strip() == "7"
    assert t.run([sys.executable, "-c", probe], overlay=fixed).stdout.strip() == "6"
    ex = [a for a in t.actions if a["accion"] == "tool.execute"]
    assert ex and all(a["sandboxed"] and a["resultado"] == "permitido" for a in ex)


# --- sesión: Hope vs. Qwen ---------------------------------------------------------------
def test_hope_and_qwen_run_in_parallel_and_verified_answer_wins(env):
    teacher = FixedTeacher(GOOD, env["guardian"], delay=0.2)
    s = _session(env, _responder(BAD, "41", delay=0.2), teacher)
    t0 = time.perf_counter()
    r = s.task(PROBLEM)
    assert time.perf_counter() - t0 < 0.38                     # en paralelo: ~0,2 s, no 0,4
    assert teacher.threads and teacher.threads[0] != threading.current_thread().name
    assert r.hope.answer == "41" and r.qwen.answer == "42"
    assert r.hope.verdict == "errores" and r.qwen.verified and not r.hope.verified
    assert r.differ and r.chosen == "qwen" and r.episode_saved
    [ep] = _episodes(env)
    assert ep["fuente"] == "qwen" and ep["resultado"] == "42" and ep["descartada"]["resultado"] == "41"
    assert ep["verificacion"]["veredicto"] == "ok" and ep["descartada"]["errores"]
    assert "Resultado: 42" in ep["texto"]


def test_episode_saved_only_if_verification_passes(env):
    s = _session(env, _responder("Resultado: 43", "43"), FixedTeacher("Resultado: 44"))
    r = s.task(PROBLEM)
    assert r.differ and not r.hope.verified and not r.qwen.verified
    assert not r.episode_saved and _episodes(env) == []
    s2 = _session(env, _responder(GOOD, "42"), FixedTeacher(GOOD))     # coinciden: nada que corregir
    r2 = s2.task(PROBLEM)
    assert r2.agree and r2.chosen == "hope" and not r2.episode_saved and _episodes(env) == []


def test_mock_teacher_episode_is_verified_with_tools(env):
    """MockTeacher responde con un episodio JSON: se verifica paso a paso con herramientas."""
    s = _session(env, _responder(GOOD, "42"), MockTeacher(guardian=env["guardian"], error_rate=1.0))
    r = s.task(PROBLEM)
    assert r.qwen.episode_status == "REFUTADO" and not r.qwen.verified
    assert r.hope.verified and r.chosen == "hope" and r.episode_saved
    assert _episodes(env)[0]["fuente"] == "hope"
    s2 = _session(env, _responder(BAD, "41"), MockTeacher(guardian=env["guardian"], error_rate=0.0))
    r2 = s2.task(PROBLEM)
    assert r2.qwen.episode_status == "RESPALDADO"                 # pasos coherentes…
    assert not r2.qwen.verified and "otro problema" in r2.qwen.issues[0]   # …pero sobre otro problema
    assert not r2.hope.verified and not r2.episode_saved


def test_tests_decide_between_edits_and_edit_is_only_proposed(env):
    good = "```editar calc.py\ndef doble(x):\n    return 2 * x\n```\nResultado: corregido"
    bad = "```editar calc.py\ndef doble(x):\n    return x * x\n```\nResultado: corregido de otra forma"
    s = _session(env, _responder(bad, "corregido de otra forma"), FixedTeacher(good),
                 test_command=f"{sys.executable} -m pytest -q -p no:cacheprovider")
    r = s.task("Arreglá doble en calc.py para que pase test_calc.py")
    assert r.context_files == ["calc.py", "test_calc.py"]
    assert r.hope.tests["ok"] is False and r.qwen.tests["ok"] is True
    assert r.chosen == "qwen" and r.episode_saved
    assert "x + x + 1" in (env["folder"] / "calc.py").read_text()      # nada escrito todavía
    [i] = r.proposed_edits
    assert "+    return 2 * x" in s.edits[i].diff
    s.apply(i)
    assert "2 * x" in (env["folder"] / "calc.py").read_text()


def test_full_log_in_pesam(env):
    teacher = FixedTeacher(GOOD, env["guardian"])
    s = _session(env, _responder(BAD, "41"), teacher)
    s.task(f"{PROBLEM} Mirá notas.txt y {env['outside'] / 'secreto.txt'}")
    rows = env["pesam"].query()
    [task] = [x for x in rows if x["kind"] == "harness.tarea"]
    d = task["data"]
    for key in ("pedido", "hope", "qwen", "elegida", "motivos", "archivos_contexto", "aprobaciones_pendientes"):
        assert key in d
    assert d["hope"]["answer"] == "41" and d["qwen"]["answer"] == "42" and d["elegida"] == "qwen"
    assert d["archivos_contexto"] == ["notas.txt"] and len(d["aprobaciones_pendientes"]) == 1
    acts = [x["data"] for x in rows if x["kind"] == "harness.accion"]
    assert any(a["resultado"] == "pendiente" and "secreto" in a["path"] for a in acts)
    assert [x for x in rows if x["kind"] == "harness.episodio"]
    assert "no deberías" not in teacher.prompt                  # lo de afuera no llegó a nadie
    assert "hola desde la carpeta" in teacher.prompt


def test_qwen_win_teaches_the_knowledge_base(env):
    kb = KnowledgeBase()
    s = _session(env, _responder(BAD, "41"), FixedTeacher(GOOD), knowledge=kb)
    s.task(PROBLEM)
    st = kb.stats()["por_tipo_y_estado"]
    assert st["solucion"]["RESPALDADO"] == 1 and st["leccion"]["RESPALDADO"] == 1


def test_hope_folder_tool_calls_go_through_the_guardian(env):
    s = _session(env, None, None)
    runner = s.runner_factory(default_catalog(env["guardian"]))
    assert "hola" in runner.run("leer", "notas.txt")
    assert "pendiente de aprobación" in runner.run("leer", str(env["outside"] / "secreto.txt"))
    assert len(s.tool_pending) == 1
    assert "calc.py:1" in runner.run("buscar", "def doble | .")
    assert "denegado" in runner.run("leer", ".git/../../fuera/secreto.txt") or s.tool_pending
    assert "código 0" in runner.run("ejecutar", f"{sys.executable} -c \"print(1)\"")


# --- consola ---------------------------------------------------------------------------
def test_console_commands(env):
    out = []
    s = _session(env, _responder(BAD, "41"), FixedTeacher(GOOD), knowledge=KnowledgeBase())
    c = Console(s, approver="usuario", print_fn=lambda *a: out.append(" ".join(map(str, a))))
    s.notebook.write("ley de Ohm", "V = I · R")
    assert c.handle("/ayuda") and "/aprobaciones" in out[-1]
    c.handle("/notas")
    assert "1 notas" in out[-1]
    c.handle("/notas ley de Ohm")
    assert "V = I · R" in out[-1]
    c.handle(f"leé {env['outside'] / 'secreto.txt'}")
    assert "Elegida: Qwen" in out[-1] and "/aprobar" in out[-1]
    c.handle("/aprobaciones")
    rid = out[-1].split()[0]
    c.handle(f"/aprobar {rid}")
    assert env["guardian"].pending() == {}
    c.handle("/cambios")
    assert "no hay cambios" in out[-1]
    c.handle("/pruebas pytest -q")
    assert s.test_command == "pytest -q"
    assert c.handle("/salir") is False


def test_harness_guardian_uses_the_folder_and_protects_git(env):
    gcfg = env["tmp"] / "configs" / "guardian.yaml"            # rutas relativas al repo de la config
    gcfg.write_text("workdir: x\nprotected_paths: [configs]\nbudget_path: configs/growth_budget.yaml\n"
                    "audit_path: g3/audit.jsonl\napprovals_path: g3/approvals.json\n")
    g = harness_guardian(gcfg, env["folder"])
    assert g.config.approvals_path == str(env["tmp"] / "g3" / "approvals.json")
    from guardian.core import Action
    assert g.decide(Action("file.write", {"path": str(env["folder"] / "a.txt")})).allowed
    assert not g.decide(Action("file.write", {"path": str(env["folder"] / ".git" / "HEAD")})).allowed
    assert g.decide(Action("file.write", {"path": "/tmp/x"})).needs_human


def test_cli_without_model_or_teacher(env, monkeypatch, capsys):
    lines = iter(["/aprobaciones", "/salir"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(lines))
    gcfg = env["tmp"] / "configs" / "guardian.yaml"
    gcfg.write_text("workdir: x\nprotected_paths: [configs]\nbudget_path: configs/growth_budget.yaml\n"
                    "audit_path: g2/audit.jsonl\napprovals_path: g2/approvals.json\n")
    main(["--carpeta", str(env["folder"]), "--maestro", "ninguno", "--guardian", str(gcfg),
          "--registro", str(env["tmp"] / "reg2")])
    assert "no hay solicitudes pendientes" in capsys.readouterr().out
    assert (env["tmp"] / "reg2" / "pesam.jsonl").read_text().count("harness.sesion") == 1


# --- con el núcleo real (hn1-tiny, sin entrenar) -------------------------------------------
def test_with_hn1_tiny_council(env):
    pytest.importorskip("torch")
    pytest.importorskip("hope.data.corpus", reason="hope/data/ no está en el repo (lo ignora «data/» en .gitignore)")
    from hope.core.hn1 import HN1Model, preset
    from hope.council.agents import build_local_council
    model = HN1Model(preset("hn1-tiny")).eval()
    kb = KnowledgeBase()
    s = _session(env, None, FixedTeacher(GOOD), knowledge=kb)
    agents = build_local_council(model, default_catalog(env["guardian"]), s.notebook,
                                 runner_factory=s.runner_factory)
    s.responder = BudgetedResponder(agents, knowledge=kb)
    r = s.task(PROBLEM)
    assert r.hope.error is None and r.hope.meta["agente"]
    assert r.qwen.verified and r.chosen in ("hope", "qwen")
    if r.differ:                                               # un modelo sin entrenar no sabe: gana Qwen
        assert r.chosen == "qwen" and r.episode_saved
    assert [x for x in env["pesam"].query() if x["kind"] == "harness.tarea"]
