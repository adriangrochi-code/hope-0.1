import copy

import pytest
import torch

from hope.core.hn1 import build_hn1
from hope.growth.ops import AddExpert
from hope.plasticity.adapters import (adapter_parameters, attach_adapters, discard_adapters, has_adapters,
                                      merge_adapters)
from hope.plasticity.protection import ConsolidationPenalty, fisher_diagonal
from hope.plasticity.rates import PlasticityPolicy
from hope.train.continual import Mixture, Source, TrainConfig, Trainer

A = Source.from_texts("a", ["la ley de ohm relaciona tension corriente y resistencia. " * 30], 1.0)
B = Source.from_texts("b", ["a voltage regulator keeps the output voltage constant. " * 30], 1.0)


def _trainer(model=None):
    torch.manual_seed(0)
    if model is None:
        model, _ = build_hn1("hn1-tiny")
    return Trainer(model, TrainConfig(lr=3e-3, warmup=5, batch=4, seq_bytes=64, precision="fp32", seed=0),
                   device="cpu")


def test_consolidation_penalty_reduces_forgetting():
    base = _trainer()
    base.train(Mixture([A]), 80)
    x = A.sample(4, 65, __import__("numpy").random.default_rng(0))
    fisher = fisher_diagonal(base.model, [x[i:i + 1] for i in range(4)], lambda m, b: m.loss(b)[0])
    plain, protected = copy.deepcopy(base), copy.deepcopy(base)
    protected.penalties.append(ConsolidationPenalty(protected.model, fisher, strength=2000.0))
    r1 = plain.train(Mixture([B]), 60, eval_sources=[A, B])
    r2 = protected.train(Mixture([B]), 60, eval_sources=[A, B])
    assert r2.forgetting["a"] < r1.forgetting["a"]


def test_protected_regions_learn_slower():
    tr = _trainer()
    model = tr.model
    _, reg = build_hn1("hn1-tiny")
    reg.find_by_path("regions.0").protected = True
    scales = tr.set_plasticity(reg, PlasticityPolicy(protected=0.0))
    assert 0.0 in scales and 1.0 in scales
    before_zone = [p.detach().clone() for p in model.regions[0].parameters()]
    before_bb = [p.detach().clone() for p in model.backbone.parameters()]
    tr.train(Mixture([A]), 20)
    zone_change = sum((p - q).abs().sum() for p, q in zip(model.regions[0].parameters(), before_zone))
    bb_change = sum((p - q).abs().sum() for p, q in zip(model.backbone.parameters(), before_bb))
    assert zone_change == 0 and bb_change > 0


def test_fast_adapters_attach_train_merge_and_discard():
    tr = _trainer()
    model = tr.model
    model.eval()
    x = torch.tensor([list(b"adaptacion rapida sin reentrenar")])
    with torch.no_grad():
        original = model(x)["logits"].clone()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    n = attach_adapters(model, rank=4)
    assert n > 0 and has_adapters(model)
    with torch.no_grad():
        assert torch.allclose(model(x)["logits"], original)                   # al enganchar no cambia nada
    tr.refresh_optimizer()
    tr.train(Mixture([B]), 20)
    assert all(p.requires_grad for p in adapter_parameters(model))
    model.eval()
    with torch.no_grad():
        adapted = model(x)["logits"].clone()
    assert not torch.allclose(adapted, original)
    with pytest.raises(RuntimeError):
        AddExpert(0).apply(model, build_hn1("hn1-tiny")[1])                    # no crecer con adaptadores
    snapshot = copy.deepcopy(model)
    discard_adapters(snapshot)
    with torch.no_grad():
        assert torch.allclose(snapshot(x)["logits"], original)                 # descartar = volver exacto
    merge_adapters(model)
    assert not has_adapters(model)
    with torch.no_grad():
        assert torch.allclose(model(x)["logits"], adapted, atol=1e-5)          # consolidar = conservar lo aprendido
    assert any(not torch.equal(v, base_state[k]) for k, v in model.state_dict().items())
