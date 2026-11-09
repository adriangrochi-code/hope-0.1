import torch

from hope.core.hn1 import build_hn1
from hope.memory.pesam import ExperienceLog
from hope.memory.store import MemoryStore
from hope.teacher.client import MockTeacher, TeacherConfig, TeacherDenied, OpenAICompatibleTeacher
from hope.teacher.cycle import CycleConfig, SemiFunctional, TeachingCycle, Topic, answers_match
from hope.tools.builtin import default_catalog
from hope.train.continual import Source, TrainConfig, Trainer


def _cycle(tmp_path, error_rate=0.5, criterion=None, calls=40):
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    trainer = Trainer(model, TrainConfig(lr=3e-3, warmup=5, batch=4, seq_bytes=128), device="cpu")
    replay = Source.from_texts("corpus", ["texto general de repaso para no olvidar. " * 30], 1.0)
    cfg = CycleConfig(episodes_per_topic=3, consolidation_steps=30, max_teacher_calls=calls,
                      eval_items_per_topic=4, answer_max_bytes=60,
                      criterion=criterion or SemiFunctional(min_topic_accuracy=0.0, min_items_per_topic=1))
    return TeachingCycle(trainer, MockTeacher(error_rate=error_rate, seed=3), MemoryStore(), default_catalog(),
                         [Topic("suma", 1), Topic("ley de ohm", 2)], replay=[replay], cfg=cfg,
                         pesam=ExperienceLog(tmp_path / "pesam.jsonl"), out_dir=tmp_path / "out", log=lambda *_: None)


def test_answers_match():
    assert answers_match("42", "42") and answers_match("11,0 V", "11 V") and not answers_match("11 V", "11 W")
    assert not answers_match(None, "1") and answers_match("Verdadero.", "verdadero")


def test_full_cycle_with_mock_teacher(tmp_path):
    cyc = _cycle(tmp_path)
    hist = cyc.run(3)
    kinds = [r["kind"] for r in cyc.pesam.query()]
    # Verificación: los episodios con errores inyectados no se entrenan.
    eps = cyc.pesam.query("episodio")
    refuted = [e for e in eps if e["status"] == "REFUTADO"]
    assert refuted, "con error_rate=0.5 debería haber episodios refutados"
    assert len(cyc.training_texts) == len(eps) - len(refuted)
    # Memoria: todo lo enseñado quedó registrado, lo refutado como error del maestro.
    kinds_mem = {r[0] for r in cyc.memory.db.execute("SELECT kind FROM engrams")}
    assert {"procedimiento", "error_del_maestro"} <= kinds_mem
    # Consolidación: Hope aprende los episodios (baja su pérdida sobre ellos).
    cons = cyc.pesam.query("consolidacion")[0]["data"]
    assert cons["episode_bpb_after"] < cons["episode_bpb_before"]
    # Fases: con un criterio trivial pasa del currículo (nivel 1 → 2) a las preguntas.
    assert hist[0]["phase"] in ("arranque", "preguntas") and "cambio_de_fase" in kinds
    assert hist[-1]["phase"] == "preguntas" and cyc.level == 2
    assert (tmp_path / "out" / "rounds.json").exists() and (tmp_path / "out" / "model.pt").exists()
    # Las variaciones de evaluación nunca se usaron para entrenar.
    trained = "\n".join(t for t, _ in cyc.training_texts)
    for items in cyc.eval_pool.values():
        for it in items:
            assert f"Problema: {it['problema']}" not in trained


def test_strict_criterion_keeps_teaching(tmp_path):
    cyc = _cycle(tmp_path, error_rate=0.0, criterion=SemiFunctional(min_topic_accuracy=0.99))
    cyc.run(2)
    assert cyc.phase == "arranque" and cyc.level == 1       # un modelo sin entrenar no supera el nivel 1


def test_teacher_budget_is_respected(tmp_path):
    cyc = _cycle(tmp_path, calls=5)
    cyc.run(5)
    assert cyc.teacher.calls == 5


def test_remote_teacher_requires_human_approval(tmp_path):
    from guardian.core import Guardian, GuardianConfig
    budget = tmp_path / "b.yaml"
    budget.write_text("max_total_params: 1000\n")
    g = Guardian(GuardianConfig(str(tmp_path), [], str(budget), str(tmp_path / "a.jsonl"), str(tmp_path / "ap.json")))
    remote = OpenAICompatibleTeacher(TeacherConfig(endpoint="https://api.ejemplo.com/v1"), g)
    try:
        remote.ask([{"role": "user", "content": "hola"}])
        assert False
    except TeacherDenied as e:
        assert "pendiente" in str(e)
    local = MockTeacher(TeacherConfig(endpoint="http://127.0.0.1:8080/v1", model="mock"), g)
    assert local.ask([{"role": "user", "content": "TEMA: suma"}]).text
