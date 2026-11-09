import torch

from hope.core.toy import build_toy_model
from hope.registry.topology import NodeKind, TopologyRegistry


def test_ids_are_stable_and_unique():
    _, reg = build_toy_model(n_regions=2, n_experts=4)
    ids = [n.id for n in reg.nodes()]
    assert len(ids) == len(set(ids))
    assert reg.find_by_path("regions.1").id == "reg_region-1_0"


def test_param_report_matches_model():
    model, reg = build_toy_model(n_regions=2, n_experts=4, top_k=2)
    rep = reg.report(model)
    assert rep.total == sum(p.numel() for p in model.parameters())
    assert rep.trainable == rep.total
    assert rep.unregistered == 0
    # Con top-2 de 4, los especialistas cuentan a la mitad en parámetros activos.
    expert_params = sum(reg.node_params(model, n.id) for n in reg.nodes(NodeKind.EXPERT))
    assert abs((rep.total - rep.active_expected) - expert_params / 2) < 1e-6


def test_trainable_excludes_frozen():
    model, reg = build_toy_model()
    for p in model.embed.parameters():
        p.requires_grad_(False)
    rep = reg.report(model)
    assert rep.trainable == rep.total - model.embed.weight.numel()


def test_unregistered_params_are_detected():
    model, reg = build_toy_model()
    model.extra = torch.nn.Linear(4, 4)
    assert reg.report(model).unregistered == 20


def test_serialization_roundtrip():
    model, reg = build_toy_model()
    reg2 = TopologyRegistry.from_json(reg.to_json())
    assert reg2.to_json() == reg.to_json()
    assert reg2.report(model) == reg.report(model)
    assert reg2.validate(model) == []
