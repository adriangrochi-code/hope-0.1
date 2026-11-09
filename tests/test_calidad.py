"""Regresiones de errores encontrados en el control de calidad (docs/resultados/calidad.md)."""

import math

import numpy as np
import torch

from hope.core.hn1 import HN1Model, preset


# --- hope/train/rlvr.py ---------------------------------------------------------------
def test_grpo_group_of_one_is_skipped_instead_of_nan():
    from hope.train.rlvr import GRPO, RLConfig
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny"))
    before = [p.detach().clone() for p in m.parameters()]
    rl = GRPO(m, RLConfig(group=1, temperature=1.0, max_new=4, lr=1e-2))
    st = rl.step(["x: "], lambda i, t: float(len(t)))
    assert st["terminos"] == 0 and rl.report.skipped_groups == 1
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_grpo_keeps_the_training_mode_of_the_model():
    from hope.train.rlvr import GRPO, RLConfig
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny"))
    rl = GRPO(m, RLConfig(group=2, temperature=1.0, max_new=4, lr=1e-3))
    reward = lambda i, t: t.count("a") + 0.01 * len(t)  # noqa: E731
    m.train()
    rl.step(["x: "], reward)
    assert m.training
    m.eval()
    rl.step(["x: "], reward)
    assert not m.training


def test_completion_logprob_with_empty_prompt_counts_every_predicted_byte():
    from hope.train.rlvr import completion_logprob
    m = HN1Model(preset("hn1-tiny")).eval()
    _, n = completion_logprob(m, b"", b"hola")
    assert n == 3                       # el primer byte no tiene contexto; se predicen «ola»
    _, n = completion_logprob(m, b"P: ", b"hola")
    assert n == 4


# --- hope/core/hn1.py: long_loss ---------------------------------------------------------
def test_long_loss_weights_each_segment_by_its_bytes():
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny")).eval()
    x = torch.randint(0, 256, (2, 1 + 64 + 16))
    with torch.no_grad():
        total, st = m.long_loss(x, 64)
        l1, s1 = m.loss(x[:, :65])
        mem = m.summarize(m._last_global)
        l2, s2 = m.loss(x[:, 64:], mem)
    assert st["segmentos"] == 2
    assert math.isclose(float(total), (64 * float(l1) + 16 * float(l2)) / 80, rel_tol=1e-5)
    assert math.isclose(st["ce"], (64 * s1["ce"] + 16 * s2["ce"]) / 80, rel_tol=1e-5)


# --- hope/memory/knowledge.py: Notebook ---------------------------------------------------
def test_notebook_new_version_updates_the_search_vector():
    from hope.memory.knowledge import KnowledgeBase, Notebook
    kb = KnowledgeBase()
    nb = Notebook(kb)
    eid = nb.write("fotosíntesis", "proceso de las plantas")
    assert nb.write("fotosíntesis", "los cloroplastos convierten luz en glucosa con clorofila") == eid
    store = kb.store
    expected = store.embedder.embed(["fotosíntesis\nlos cloroplastos convierten luz en glucosa con clorofila"])[0]
    row = store._ids.index(eid)
    assert np.allclose(store._matrix[row], expected)
    store._load_vectors()                                   # también en la base (al reabrir)
    assert np.allclose(store._matrix[store._ids.index(eid)], expected)
    assert store.vector_search("cloroplastos glucosa clorofila", k=1)[0][0] == eid


# --- hope/core/fast.py --------------------------------------------------------------------
def test_find_batch_size_goes_below_the_start_when_it_does_not_fit():
    from hope.core.fast import find_batch_size
    tried = []

    def step(b):
        tried.append(b)
        if b > 2:
            raise RuntimeError("out of memory")
    assert find_batch_size(step, start=8) == 2
    assert find_batch_size(lambda b: None, start=4, maximum=16) == 16


# --- hope/train/run.py: enfriamiento por tramos ------------------------------------------
def test_cooldown_lr_follows_the_hours_not_each_chunk():
    from hope.train.continual import TrainConfig, Trainer
    from hope.train.run import cooldown_span
    tr = Trainer(HN1Model(preset("hn1-tiny")), TrainConfig(lr=1.0, warmup=0, schedule="wsd"), device="cpu")
    # Dos tramos de 30 min en una etapa de 1 h: el primero termina a mitad del enfriamiento,
    # el segundo empieza ahí (no vuelve a la tasa máxima) y termina en 0.
    a, b = cooldown_span(0.0, 30, 1.0), cooldown_span(0.5, 30, 1.0)
    assert a == (0.0, 0.5) and b == (0.5, 1.0)
    end_first = tr._lr(9, 10, True, a)
    start_second = tr._lr(0, 10, True, b)
    assert math.isclose(end_first, 1 - math.sqrt(0.5))
    assert start_second < end_first and math.isclose(tr._lr(9, 10, True, b), 0.0, abs_tol=1e-12)
    assert tr._lr(9, 10, True) == 0.0                        # sin tramo: igual que antes


# --- hope/constraints/solver.py: el mismo análisis no relanza el sandbox ----------------------
def test_constraint_analysis_reuses_the_sandbox_result(monkeypatch):
    from hope.constraints import solver
    calls = []
    real = solver.run_python

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(solver, "run_python", counting)
    monkeypatch.setattr(solver, "_SOLVE_CACHE", {})
    p = "Una resistencia de 220 ohm con 0,05 A de corriente. ¿Qué tensión tiene?"
    a = solver.analyze(p)
    a.contradictions.append({"x": 1})                    # modificar un resultado no contamina el siguiente
    b = solver.analyze(p)
    assert len(calls) == 1
    assert b.expected == a.expected and {"x": 1} not in b.contradictions


# --- hope/train/lowprec.py y muon.py: retomar el estado del optimizador ------------------------
def _train_save(tmp_path, precision, optimizer="adamw"):
    from hope.core.hn1 import build_hn1
    from hope.train.continual import Mixture, Source, TrainConfig, Trainer
    torch.manual_seed(0)
    m, _ = build_hn1("hn1-tiny")
    cfg = TrainConfig(lr=1e-3, warmup=1, batch=2, seq_bytes=32, precision=precision, optimizer=optimizer)
    tr = Trainer(m, cfg, device="cpu")
    src = Source.from_texts("a", ["hola mundo " * 40], 1.0)
    tr.train(Mixture([src]), 2, eval_sources=[])
    tr.save(tmp_path)
    return tr, Mixture([src])


def test_low_precision_resumes_a_fp32_adamw_checkpoint(tmp_path):
    from hope.train.continual import TrainConfig, Trainer
    tr, mix = _train_save(tmp_path, "fp32")
    exp_avg = next(iter(tr.opt.state.values()))["exp_avg"].clone()
    for prec in ("bf16-8bit", "fp32-8bit"):            # p. ej. de la CPU a la GTX 1060 o a la A770
        t2 = Trainer.resume(tmp_path, TrainConfig(lr=1e-3, warmup=1, batch=2, seq_bytes=32, precision=prec),
                            device="cpu")
        assert "exp_avg" in next(iter(t2.opt.state.values()))       # se cargó el estado de AdamW
        t2.train(mix, 1, eval_sources=[])                           # antes: KeyError 'm_q'
        st = next(iter(t2.opt.state.values()))
        assert "m_q" in st and st["step"] == 3
        assert st["m_q"].abs().sum() > 0 and exp_avg.abs().sum() > 0  # el momento siguió, no volvió a cero


def test_muon_resume_keeps_its_state_in_fp32(tmp_path):
    from hope.train.continual import TrainConfig, Trainer
    tr, _ = _train_save(tmp_path, "bf16-8bit", "muon")
    t2 = Trainer.resume(tmp_path, TrainConfig(lr=1e-3, warmup=1, batch=2, seq_bytes=32, precision="bf16-8bit",
                                              optimizer="muon"), device="cpu")
    dtypes = {v.dtype for st in t2.opt.state.values() for v in st.values() if torch.is_tensor(v)}
    assert dtypes == {torch.float32}                                 # antes: bfloat16
    a = next(iter(tr.opt.state.values()))
    b = next(iter(t2.opt.state.values()))
    assert all(torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))


# --- hope/bench/hardware.py: reconocer la placa al calibrar ------------------------------------
def test_calibration_recognizes_real_device_names():
    from hope.bench.hardware import PROFILES, match_profile
    assert match_profile("NVIDIA GeForce GTX 1060 6GB", 6.4) is PROFILES["gtx1060"]
    assert match_profile("NVIDIA GeForce GTX 1060 3GB", 3.2) is PROFILES["gtx1060-3gb"]
    assert match_profile("Intel(R) Arc(TM) A770 Graphics", 16.2) is PROFILES["a770"]
    assert match_profile("NVIDIA GeForce RTX 4090", 25.4) is PROFILES["rtx4090"]
    assert match_profile("CPU") is PROFILES["cpu"]           # antes, todas caían acá (40 GB/s)


# --- hope/train/qat.py y hope/core/fast.py: lo que se entrena es lo que se exporta ----------------
def test_qat_grid_matches_the_export_exactly_also_in_bf16_and_small_weights():
    from hope.core.fast import QuantLinear
    from hope.train.qat import fake_quant
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        for size in (0.02, 1e-6):
            lin = torch.nn.Linear(128, 64).to(dtype)
            lin.weight.data = (torch.randn(64, 128) * size).to(dtype)
            for bits in (4, 8):
                trained = fake_quant(lin.weight, bits, 64)
                exported = QuantLinear(lin, bits, 64).dequantized()
                # antes: hasta 12 % de diferencia en bf16 y 99 % con pesos chicos (escala fp16)
                assert torch.allclose(trained, exported, rtol=1e-6, atol=0), (dtype, size, bits)
    w = torch.randn(8, 64, requires_grad=True)
    fake_quant(w, 4).sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))                       # straight-through


def test_qat_turns_off_in_a_later_stage_without_qat():
    from hope.train.continual import Mixture, Source, TrainConfig, Trainer
    from hope.train.qat import QATLinear
    torch.manual_seed(0)
    tr = Trainer(HN1Model(preset("hn1-tiny")), TrainConfig(lr=1e-3, warmup=1, batch=2, seq_bytes=32,
                                                           qat_bits=(4, 8)), device="cpu")
    mix = Mixture([Source.from_texts("a", ["hola mundo " * 40], 1.0)])
    tr.train(mix, 2, eval_sources=[])
    qat = [mod for mod in tr.model.modules() if isinstance(mod, QATLinear)]
    assert qat and all(mod.qat_bits in (4, 8) for mod in qat)
    tr.cfg = TrainConfig(lr=1e-3, warmup=1, batch=2, seq_bytes=32)      # etapa siguiente (run.py)
    tr.train(mix, 1, eval_sources=[])
    assert all(mod.qat_bits is None for mod in qat)                      # antes: seguía en 4 u 8 bits
