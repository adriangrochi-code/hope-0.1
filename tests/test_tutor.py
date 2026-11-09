"""Tutoría on-policy (hope/teacher/tutor.py) con un tutor simulado y hn1-tiny."""

import json

import numpy as np
import pytest
import torch

from hope.core.hn1 import build_hn1
from hope.memory.pesam import ExperienceLog
from hope.memory.store import MemoryStore
from hope.teacher.client import OpenAICompatibleTeacher, TeacherConfig
from hope.teacher.cycle import Topic
from hope.teacher.tutor import (MockTutor, Problem, ReviewQueue, TutorConfig, TutorSession, combined_reward,
                                corrected_segments, leaks_answer, safe_hint)
from hope.tools.builtin import default_catalog
from hope.train.continual import Mixture, Source, TrainConfig, Trainer


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def _session(tmp_path, tutor=None, *, ask=False, steps=5, topics=None, **cfg):
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    trainer = Trainer(model, TrainConfig(lr=3e-3, warmup=2, batch=4, seq_bytes=128), device="cpu")
    base = dict(diagnosis_items=2, tutor_problems=1, problems_per_topic=2, topics_per_round=2,
                answer_max_bytes=40, consolidation_steps=steps, ask_threshold=1.0, mastery_min_items=3,
                mastery_window=3)
    base.update(cfg)
    replay = Source.from_texts("corpus", ["texto general de repaso para no olvidar. " * 30], 1.0)
    unc = (lambda prompt, text: (5.0, text.splitlines()[0] if text else "")) if ask else (lambda p, t: (0.0, ""))
    return TutorSession(trainer, tutor or MockTutor(seed=1), MemoryStore(), default_catalog(),
                        topics or [Topic("suma", 1), Topic("ley de Ohm y potencia eléctrica", 2)],
                        cfg=TutorConfig(**base), pesam=ExperienceLog(tmp_path / "pesam.jsonl"),
                        out_dir=tmp_path / "tutor", replay=[replay], clock=Clock(), uncertainty_fn=unc,
                        log=lambda *_: None)


# --- piezas -----------------------------------------------------------------------
def test_hint_never_contains_the_answer():
    assert leaks_answer("La respuesta es 42.", "42") and leaks_answer("da 11,5 V", "11.5 V")
    assert not leaks_answer("sumá 4 + 2 primero", "42") and not leaks_answer("revisá 142", "42")
    assert leaks_answer("es verdadero porque…", "Verdadero")
    h = safe_hint("Llevá 1 a las decenas; la respuesta es 131.", "131", step=1, kind="procedimiento")
    assert "131" not in h and "paso 1" in h
    assert safe_hint("Fijate el acarreo.", "131") == "Fijate el acarreo."


def test_combined_reward_verification_always_wins():
    assert combined_reward(0.0, {"pasos": 1, "herramienta": 1, "eficiencia": 1}) == 0.0   # el tutor no rescata
    assert combined_reward(1.0, {}) == 1.0
    lo = combined_reward(1.0, {"pasos": 0, "herramienta": 0, "eficiencia": 0}, 0.3)
    hi = combined_reward(1.0, {"pasos": 1, "herramienta": 1, "eficiencia": 1}, 0.3)
    assert lo == pytest.approx(0.7) and hi == 1.0
    assert combined_reward(0.2, {"pasos": 1}) <= 0.2 < lo              # acierto con pasos rotos < correcta


def test_corrected_attempt_mask_trains_only_correct_parts():
    first = "Plan: sumar.\nPaso 1: 7 + 5 = 11 (unidades)\nPaso 2: 37 + 25 = 51 (decenas)\nResultado: 51\n"
    segs = corrected_segments("Tema: suma\nProblema: ¿37 + 25?\n", first, 1, "Revisión: olvidó el acarreo",
                              "Paso 1: 7 + 5 = 12\nResultado: 62")
    trained = "".join(t for t, m in segs if m)
    context = "".join(t for t, m in segs if not m)
    assert "Plan: sumar." in trained and "Revisión: olvidó el acarreo" in trained and "Resultado: 62" in trained
    assert "= 11" in context and "Resultado: 51" in context and "Problema" in context
    src = Source.from_segments("t", [segs], 1.0)
    assert len(src.mask) == len(src.data)
    raw = bytes(src.data)
    i = raw.find("Resultado: 51".encode())
    assert src.mask[i] == 0 and src.mask[raw.find("Resultado: 62".encode())] == 1


def test_masked_source_trains_with_weights(tmp_path):
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    tr = Trainer(model, TrainConfig(lr=3e-3, warmup=1, batch=2, seq_bytes=64), device="cpu")
    segs = [[("contexto que no se entrena " * 3, False), ("parte correcta " * 3, True)]] * 4
    mix = Mixture([Source.from_segments("m", segs, 1.0)])
    assert mix.has_masks
    x, w, _ = mix.batch_weighted(2, 65, np.random.default_rng(0))
    assert x.shape == w.shape and 0 < float(w.mean()) < 1
    rep = tr.train(mix, 3)
    assert rep.steps == 3
    # run.py lee train.mask junto a train.bin
    from hope.teacher.tutor import TutorData
    from hope.train.run import _sources
    d = TutorData()
    for s in segs * 20:
        d.add("intento_corregido", s)
    d.write(tmp_path / "datos", val_percent=0)
    srcs = _sources({"tutor": f"{tmp_path / 'datos'}:1.0"})
    assert srcs[0].mask is not None and len(srcs[0].mask) == len(srcs[0].data)


def test_review_queue_is_spaced():
    clk = Clock()
    q = ReviewQueue(clk)
    assert q.add(Problem("suma", "¿1 + 1?", "2")) and not q.add(Problem("suma", "¿1 + 1?", "2"))
    item = q.due("suma")[0]
    q.record(item, True)
    assert not q.due("suma")
    clk.t += 2.5 * 86400
    assert q.due("suma")
    q.record(q.due("suma")[0], False)
    assert item.interval_days == 1.0 and item.streak == 0


# --- ciclo completo -------------------------------------------------------------------
def test_full_tutoring_cycle(tmp_path):
    s = _session(tmp_path, ask=True)
    hist = s.run(2)
    assert len(hist) == 2
    kinds = {r["kind"] for r in s.pesam.query()}
    # 1. diagnóstico y perfil
    assert "diagnostico" in kinds
    assert {"suma", "ley de Ohm y potencia eléctrica"} <= set(s.profile.topics)
    assert s.profile.get("suma").attempts >= 2 and s.profile.weaknesses()
    # 2. intento → revisión → pista → reintento; explicación y cuaderno al fallar dos veces
    tut = [r["data"] for r in s.pesam.query("tutoria")]
    failed = [t for t in tut if not t["primer_intento"]]
    assert failed and all(t["pista"] for t in failed)
    assert s.tutor.tasks["revision"] >= len(failed) and s.tutor.tasks["explicacion"] >= 1
    notes = {r[0] for r in s.memory.db.execute("SELECT concepto FROM notas_repaso")}
    assert "suma: idea clave" in notes and "suma: error a evitar" in notes
    assert s.memory.db.execute("SELECT COUNT(*) FROM engrams WHERE kind = 'leccion'").fetchone()[0] >= 1
    # 3. variaciones dirigidas al repaso espaciado
    assert s.tutor.tasks["problemas"] >= 2 and s.reviews.items
    assert all(i.problem["origen"] == "variacion" and i.problem["foco"] for i in s.reviews.items)
    # 4. recompensas: verificación + notas del tutor
    assert s.rewards and all(r["recompensa"] <= r["verificada"] for r in s.rewards)
    # 5. preguntas de Hope (incertidumbre alta), registradas
    assert s.questions and (tmp_path / "tutor" / "preguntas.jsonl").exists()
    assert s.questions[0]["estado"] == "verificada"
    # 7. datos: intentos corregidos con máscara, episodios del tutor; consolidación
    assert {"intento_corregido", "episodio_tutor"} <= set(s.data.kinds())
    d = tmp_path / "tutor" / "datos"
    assert (d / "train.bin").stat().st_size == (d / "train.mask").stat().st_size > 0
    cons = s.pesam.query("consolidacion_tutoria")[0]["data"]
    assert cons["pasos"] == 5 and cons["bpb_despues"] < cons["bpb_antes"]
    st = json.loads((tmp_path / "tutor" / "tutor_estado.json").read_text())
    assert st["perfil"] and st["repaso"]
    # reanudar: el estado se recupera
    s2 = _session(tmp_path)
    assert s2.load() and s2.rounds == 2 and s2.reviews.items and s2.data.records


def test_hints_from_a_leaky_tutor_are_replaced(tmp_path):
    s = _session(tmp_path, MockTutor(seed=2, leaky_hints=True), steps=0)
    s.run(1)
    tut = [r["data"] for r in s.pesam.query("tutoria") if r["data"]["pista"]]
    assert tut and all(t["pista_filtrada"] for t in tut)          # el tutor filtró la respuesta…
    for t in tut:                                                # …y Hope recibió una pista sin ella
        assert "La respuesta es" not in t["pista"] and "Revisá" in t["pista"]


def test_tutor_mistakes_are_verified_and_discarded(tmp_path):
    # Problemas con la respuesta equivocada y episodios con pasos equivocados: no entran.
    s = _session(tmp_path, MockTutor(seed=3, wrong_problems=1.0, error_rate=1.0, wrong_reviews=True), steps=0)
    s.run(1)
    kinds = {e["tipo"] for e in s.tutor_errors}
    assert {"problema_refutado", "explicacion_refutada", "revision_contradice_verificacion"} <= kinds
    assert not s.reviews.items                                   # ninguna variación equivocada al repaso
    assert "episodio_tutor" not in s.data.kinds()
    tut = [r["data"] for r in s.pesam.query("tutoria")]
    assert tut and all(t["origen"] == "generador" for t in tut)   # los problemas refutados nunca se usaron
    assert all(r.get("origen", "generador") == "generador" for r in s.data.records)
    # La verificación manda sobre el juicio del tutor: el perfil cuenta lo que verificaron las herramientas.
    tut = [r["data"] for r in s.pesam.query("tutoria")]
    assert sum(t["primer_intento"] for t in tut) <= sum(p.correct for p in s.profile.topics.values())


def test_adaptive_curriculum(tmp_path):
    s = _session(tmp_path, topics=[Topic("suma", 1), Topic("porcentajes", 1), Topic("ecuaciones", 2)])
    # «suma» dominada, «porcentajes» débil, «ecuaciones» sin datos (nivel 2)
    s.profile.get("suma").recent = [1, 1, 1, 1]
    s.profile.get("porcentajes").recent = [0, 1, 0]
    s.profile.get("porcentajes").sessions = [0.5, 0.3]           # empeora → más atención
    order = s.next_topics(3)
    assert "suma" not in order and order[0] == "porcentajes"
    assert s.mastered("suma") and not s.mastered("porcentajes")
    # Un repaso vencido de un tema dominado lo vuelve a traer.
    s.reviews.add(Problem("suma", "¿38 + 47?", "85", "variacion"))
    assert "suma" in s.next_topics(3) and not s.mastered("suma")
    # Si todo está dominado y sin repasos, no hay más temas.
    s.reviews.items.clear()
    for n in ("porcentajes", "ecuaciones"):
        s.profile.get(n).recent = [1, 1, 1]
    assert s.next_topics(3) == []


def test_remote_tutor_requires_human_approval(tmp_path):
    from guardian.core import Guardian, GuardianConfig
    budget = tmp_path / "b.yaml"
    budget.write_text("max_total_params: 1000\n")
    g = Guardian(GuardianConfig(str(tmp_path), [], str(budget), str(tmp_path / "a.jsonl"), str(tmp_path / "ap.json")))
    remote = OpenAICompatibleTeacher(TeacherConfig(endpoint="https://api.ejemplo.com/v1"), g)
    s = _session(tmp_path, remote, steps=0)
    s.run(1)                                                     # sigue con los generadores
    assert remote.calls == 0 and s.pesam.query("tutor_denegado")
    assert s.profile.get("suma").attempts > 0
    local = MockTutor(TeacherConfig(endpoint="http://127.0.0.1:8080/v1", model="mock"), g)
    assert local.ask([{"role": "user", "content": "TUTOR_TAREA: pregunta\nTEMA: suma"}]).text


def test_reinforcement_after_sft_uses_combined_reward(tmp_path):
    s = _session(tmp_path, steps=0, rl=True, rl_min_accuracy=0.0, rl_problems=1, rl_group=2, rl_max_new=20,
                 topics=[Topic("suma", 1)])
    s.profile.get("suma").recent = [1, 0, 0]
    out = s.reinforce()
    assert out["problemas"] == 1 and out["pasos"]
    s.cfg.rl_min_accuracy = 0.9                                   # SFT antes que RL: tema flojo → no hay RL
    assert "omitido" in s.reinforce()


def test_plan_stage_runs_tutoring(tmp_path):
    from hope.train.run import run
    (tmp_path / "tutor.yaml").write_text("endpoint: http://127.0.0.1/mock\nmodel: mock-tutor\n")
    (tmp_path / "cur.yaml").write_text("topics:\n  - {name: suma, level: 1}\n")
    (tmp_path / "plan.yaml").write_text(
        f"salida: {tmp_path / 'run'}\ncompilar: false\nprecision: fp32\nsecuencia: 64\n"
        f"guardian: {tmp_path / 'guardian.yaml'}\n"
        "etapas:\n  - nombre: t\n    preset: hn1-tiny\n    lote: 2\n    horas: 1\n    tutoria:\n"
        f"      maestro: {tmp_path / 'tutor.yaml'}\n      curriculo: {tmp_path / 'cur.yaml'}\n"
        f"      datos: {tmp_path / 'datos_tutor'}\n      rondas_max: 1\n"
        "      pasos_consolidacion: 2\n      config: {diagnosis_items: 1, problems_per_topic: 1, answer_max_bytes: 30}\n")
    budget = tmp_path / "b.yaml"
    budget.write_text("max_total_params: 100000000\n")
    (tmp_path / "guardian.yaml").write_text(
        f"workdir: {tmp_path / 'g'}\nprotected_paths: []\nbudget_path: {budget}\n"
        f"audit_path: {tmp_path / 'g' / 'audit.jsonl'}\napprovals_path: {tmp_path / 'g' / 'ap.json'}\n")
    run(str(tmp_path / "plan.yaml"), device="cpu")
    st = json.loads((tmp_path / "run" / "estado.json").read_text())
    assert st["etapas"]["t"]["hecha"]
    assert (tmp_path / "datos_tutor" / "train.mask").exists()
    assert (tmp_path / "run" / "t" / "tutoria" / "tutor_estado.json").exists()
