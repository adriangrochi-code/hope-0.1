import pytest
import torch

from hope.core.toy import build_toy_model
from hope.growth.ops import (AddConnection, AddExpert, RemoveMaskedExpert, WidenExpert,
                             mask_expert, rebuild_optimizer, unmask_expert)
from hope.registry.topology import NodeKind

EPS = 1e-5


def _setup(seed=0):
    torch.manual_seed(seed)
    model, reg = build_toy_model(vocab=50, d=16, n_regions=3, n_experts=4, hidden=32, top_k=2)
    x = torch.randint(0, 50, (4, 12))
    return model, reg, x


def _check_op(model, reg, x, op):
    model.eval()
    before = model(x)
    total_before = reg.report(model).total
    planned = op.plan(model).param_delta
    rec = op.apply(model, reg)
    after = model(x)
    assert rec.preserving
    assert (after - before).abs().max().item() < EPS
    assert reg.report(model).total - total_before == planned == rec.param_delta
    assert reg.validate(model) == []
    assert reg.report(model).unregistered == 0
    return rec


def test_add_expert_preserves_function():
    model, reg, x = _setup()
    rec = _check_op(model, reg, x, AddExpert(region_idx=1))
    new = reg.get(rec.created[0])
    assert new.kind == NodeKind.EXPERT and new.activation == 0.0


def test_widen_expert_preserves_function():
    model, reg, x = _setup()
    rec = _check_op(model, reg, x, WidenExpert(region_idx=0, expert_idx=2, extra=8))
    assert rec.detail == {"from": 32, "to": 40}


def test_add_connection_preserves_function():
    model, reg, x = _setup()
    rec = _check_op(model, reg, x, AddConnection(src=0, dst=2))
    node = reg.get(rec.created[0])
    assert node.kind == NodeKind.CONNECTION and node.meta["directed"]


def test_new_structures_can_learn():
    """Tras crecer, los parámetros nuevos reciben gradiente y el modelo entrena."""
    model, reg, x = _setup()
    AddConnection(0, 1).apply(model, reg)
    AddExpert(1).apply(model, reg)
    unmask_expert(model, reg, 1, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    y = torch.randint(0, 50, x.shape)
    model.train()
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert model.connections["0_1"].gate.item() != 0.0


def test_retire_is_two_steps():
    model, reg, x = _setup()
    model.eval()
    with pytest.raises(ValueError):
        RemoveMaskedExpert(0, 1).apply(model, reg)       # sin enmascarar primero
    kept_id = reg.find_by_path("regions.0.experts.3").id
    mask_expert(model, reg, 0, 1)
    rec = _check_op(model, reg, x, RemoveMaskedExpert(0, 1))
    assert rec.removed and rec.removed[0] not in reg
    # El especialista que estaba en el índice 3 conserva su ID estable en su nueva ruta.
    assert reg.get(kept_id).module_path == "regions.0.experts.2"


def test_protected_expert_cannot_be_masked():
    model, reg, _ = _setup()
    reg.find_by_path("regions.0.experts.0").protected = True
    with pytest.raises(PermissionError):
        mask_expert(model, reg, 0, 0)


def test_cannot_mask_below_top_k():
    model, reg, _ = _setup()
    mask_expert(model, reg, 0, 0)
    mask_expert(model, reg, 0, 1)
    with pytest.raises(ValueError):
        mask_expert(model, reg, 0, 2)


def test_optimizer_state_survives_growth():
    model, reg, x = _setup()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(x).sum().backward()
    opt.step()
    emb = model.embed.weight
    state_before = opt.state[emb]["exp_avg"].clone()
    AddExpert(0).apply(model, reg)
    opt2 = rebuild_optimizer(opt, model)
    assert torch.equal(opt2.state[emb]["exp_avg"], state_before)
    new_router = model.regions[0].router.weight
    assert new_router not in opt2.state
