import numpy as np
import torch

from hope.core.hn1 import build_hn1
from hope.train.continual import Mixture, Source, TrainConfig, Trainer

TEXT_A = "la ley de ohm relaciona tension corriente y resistencia. " * 20
TEXT_B = "a voltage regulator keeps the output voltage constant. " * 20


def _trainer(seed=0):
    torch.manual_seed(seed)
    model, _ = build_hn1("hn1-tiny")
    return Trainer(model, TrainConfig(lr=3e-3, warmup=5, batch=4, seq_bytes=64, seed=seed), device="cpu")


def test_mixture_respects_weights():
    a, b = Source.from_texts("a", [TEXT_A], 3.0), Source.from_texts("b", [TEXT_B], 1.0)
    rng = np.random.default_rng(0)
    names = []
    for _ in range(200):
        names += Mixture([a, b]).batch(4, 32, rng)[1]
    frac_a = names.count("a") / len(names)
    assert 0.7 < frac_a < 0.8


def test_training_reduces_loss_and_reports_per_source():
    tr = _trainer()
    a = Source.from_texts("a", [TEXT_A], 1.0)
    rep = tr.train(Mixture([a]), 60)
    assert rep.steps == 60 and tr.step == 60
    assert rep.bpb_after["a"] < rep.bpb_before["a"]


def test_resume_continues_from_checkpoint(tmp_path):
    tr = _trainer()
    a = Source.from_texts("a", [TEXT_A], 1.0)
    tr.train(Mixture([a]), 30)
    loss = tr.bpb(a)
    tr.save(tmp_path / "ck")
    tr2 = Trainer.resume(tmp_path / "ck", TrainConfig(lr=3e-3, warmup=5, batch=4, seq_bytes=64), device="cpu")
    assert tr2.step == 30
    assert abs(tr2.bpb(a) - loss) < 1e-5
    assert tr2.opt.state_dict()["state"]                   # el optimizador conserva sus momentos


def test_replay_limits_forgetting():
    """Aprender B sin repasar A olvida A más que aprenderlo con repaso."""
    base = _trainer()
    a, b = Source.from_texts("a", [TEXT_A], 1.0), Source.from_texts("b", [TEXT_B], 1.0)
    base.train(Mixture([a]), 80)
    import copy
    no_replay, with_replay = copy.deepcopy(base), copy.deepcopy(base)
    r1 = no_replay.train(Mixture([b]), 80, eval_sources=[a, b])
    r2 = with_replay.train(Mixture([Source("b", b.data, 0.5), Source("a", a.data, 0.5)]), 80,
                           eval_sources=[a, b])
    assert r1.forgetting["a"] > r2.forgetting["a"]
