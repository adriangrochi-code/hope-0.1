import random

import torch

from hope.core.genome import Genome
from hope.core.hn1 import HN1Model, build_hn1, build_registry, preset
from hope.growth.ops import AddExpert
from hope.plasticity.dynamic import (AddBackboneConnection, DynamicSparsity, HyperplasticityController,
                                     MorphBlock, MorphingBlock, finish_morphs, prune_connections)
from hope.train.continual import Mixture, Source, TrainConfig, Trainer

X = torch.tensor([list(b"hiperplasticidad: pesos, conexiones y estructura")])
SRC = Source.from_texts("a", ["la ley de ohm relaciona tension corriente y resistencia. " * 30], 1.0)


def _model():
    torch.manual_seed(0)
    return build_hn1("hn1-tiny")


def test_genome_specialists_preserve_function_and_mutate():
    model, reg = _model()
    model.eval()
    with torch.no_grad():
        before = model(X)["logits"]
        for ops in (["glu", "conv", "inhib"], ["scan", "max", "norm"], ["res_lin", "relu"]):
            AddExpert(0, genome=Genome(ops, hidden=16)).apply(model, reg)
        after = model(X)["logits"]
    assert (after - before).abs().max() < 1e-5
    assert model.regions[0].kinds[-3:] == ["compuesto"] * 3
    rng = random.Random(0)
    g = Genome(["gelu"], 32)
    seen = {g.key()}
    for _ in range(20):
        g = g.mutate(rng)
        seen.add(g.key())
    assert len(seen) > 5


def test_dynamic_sparsity_keeps_density_and_rewires():
    model, reg = _model()
    tr = Trainer(model, TrainConfig(lr=3e-3, warmup=2, batch=2, seq_bytes=48, precision="fp32"), device="cpu")
    ds = DynamicSparsity(model, density=0.5, update_fraction=0.3)
    ctl = HyperplasticityController(model, reg, sparsity=ds, rewire_every=5)
    tr.hooks.append(ctl)
    masks_before = {k: m.clone() for k, m in ds.masks.items()}
    tr.train(Mixture([SRC]), 12)
    assert ctl.log and ctl.log[-1]["connections_rewired"] > 0
    assert abs(ctl.log[-1]["density"] - 0.5) < 0.02
    assert any(not torch.equal(masks_before[k], ds.masks[k]) for k in ds.masks)
    for n, p in ds.params.items():
        assert torch.all(p[ds.masks[n] == 0] == 0)


def test_backbone_connections_are_born_silent_and_pruned():
    model, reg = _model()
    model.eval()
    with torch.no_grad():
        before = model(X)["logits"]
        rec = AddBackboneConnection(0, 1).apply(model, reg)
        assert (model(X)["logits"] - before).abs().max() < 1e-6
    assert reg.get(rec.created[0]).meta["directed"]
    assert prune_connections(model, reg) == ["0_1"] and len(model.connections) == 0


def test_block_type_morphs_gradually():
    model, reg = _model()
    model.eval()
    with torch.no_grad():
        before = model(X)["logits"]
        MorphBlock(0, "swa").apply(model, reg)                 # ssm → atención con ventana
        assert (model(X)["logits"] - before).abs().max() < 1e-5
    tr = Trainer(model, TrainConfig(lr=3e-3, warmup=2, batch=2, seq_bytes=48, precision="fp32"), device="cpu")
    ctl = HyperplasticityController(model, reg, morph_steps=5, prune_every=6)
    tr.hooks.append(ctl)
    tr.train(Mixture([SRC]), 6)
    assert not isinstance(model.backbone[0], MorphingBlock) and model.backbone[0].kind == "swa"
    assert reg.find_by_path("backbone.0").meta["kind"] == "swa"


def test_experimental_options_build_train_and_stay_causal():
    cfg = preset("hn1-tiny")
    cfg.kv_heads, cfg.local_kind, cfg.window, cfg.sinks = 2, "swa", 16, 2
    cfg.halting, cfg.balance, cfg.shared_expert = "convergencia", "sesgo", True
    cfg.pattern = ("ssm", "swa")
    torch.manual_seed(0)
    model = HN1Model(cfg)
    loss, _ = model.loss(X)
    loss.backward()
    model.eval()
    y = X.clone()
    y[:, 30:] = 7
    with torch.no_grad():
        assert (model(X)["logits"][:, :30] - model(y)["logits"][:, :30]).abs().max() < 1e-5
        assert model.generate("hiper", max_new=30) == model.generate_reference("hiper", max_new=30)
    assert build_registry(model).validate(model) == []
