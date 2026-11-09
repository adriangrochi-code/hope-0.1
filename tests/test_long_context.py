"""Contexto largo sin parámetros nuevos: escala de RoPE y memoria de segmentos."""

from dataclasses import replace

import torch

from hope.core.hn1 import HN1Model, preset, set_context_scale
from hope.train.continual import Mixture, Source, Trainer, TrainConfig

BASE = preset("hn1-tiny")


def test_rope_scale_changes_only_far_positions_and_adds_no_params():
    torch.manual_seed(0)
    m = HN1Model(BASE).eval()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randint(0, 256, (1, 300))
    ref = m(x)["logits"]
    set_context_scale(m, 1.0)
    assert torch.equal(m(x)["logits"], ref)
    set_context_scale(m, 4.0)
    out = m(x)["logits"]
    assert sum(p.numel() for p in m.parameters()) == n
    assert not torch.allclose(out, ref) and (out[:, :3] - ref[:, :3]).abs().max() < 0.05   # el inicio casi no cambia
    assert HN1Model(replace(BASE, rope_scale=4.0)).backbone[1].mixer.rope_base > 10000


def test_segment_memory_long_loss_and_generation():
    torch.manual_seed(0)
    m = HN1Model(BASE)
    x = torch.randint(0, 256, (2, 401))
    loss, st = m.long_loss(x, 100)
    assert st["segmentos"] == 4 and torch.isfinite(loss)
    loss.backward()
    assert m.memory.q.weight.grad is not None                 # la capa de memoria aprende a usar los resúmenes
    mem = m.eval().read_segments(x[:1, :300], 100)
    assert mem.shape[0] == 1 and mem.shape[2] == BASE.d and mem.shape[1] >= 3
    out = m.generate_long("hola " * 100, 10, segment=128)
    assert len(out) == 10
    # entrenamiento con secuencias más largas que el segmento
    text = Source.from_texts("t", ["el río fluye hacia el mar. " * 200], 1.0)
    tr = Trainer(HN1Model(BASE), TrainConfig(batch=2, seq_bytes=256, segment_bytes=64, precision="fp32"), device="cpu")
    rep = tr.train(Mixture([text]), 3, eval_sources=[])
    assert rep.steps == 3
