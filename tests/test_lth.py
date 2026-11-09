import torch
import torch.nn.functional as F

from hope.bench import synthetic as S
from hope.core.hn1 import build_hn1
from hope.core.toy import build_routed_model
from hope.growth.lth import (ReinitUnits, apply_masks, expert_usage, find_ticket, hidden_layers,
                             idle_units, iteration_usage, ticket_overlap, unit_importance)


def _bench(seed=0):
    torch.manual_seed(seed)
    fams, _ = S.make_families(seed)
    model, reg = build_routed_model(S.FAMILIES, vocab=S.VOCAB, seq_len=S.SEQ, n_classes=S.C)
    return fams, model, reg


def _family_batches(f, n=3, size=128):
    g = torch.Generator().manual_seed(0)
    for _ in range(n):
        i = torch.randint(0, f.train.y.numel(), (size,), generator=g)
        yield f.train.x[i], f.train.y[i], torch.full((size,), f.index)


def _loss(model, batch):
    x, y, fam = batch
    return F.cross_entropy(model(x, fam), y)


def test_unit_importance_is_per_task_and_finds_dead_units():
    fams, model, _ = _bench()
    S.train(model, fams, 300, 0)
    # Apagamos a mano unidades de un especialista: deben salir con importancia 0.
    with torch.no_grad():
        model.regions[2].experts[0].fc2.weight[:, :3] = 0
    by_task = {f.name: unit_importance(model, _family_batches(f), _loss) for f in fams}
    layer = "regions.2.experts.0.fc2"
    assert by_task["capacity"][layer].taylor[:3].sum() == 0
    # Una región de familia solo trabaja para su familia.
    assert by_task["easy"]["regions.2.experts.0.fc2"].activity.sum() == 0
    idle = idle_units(by_task, quantile=0.2)
    assert idle[layer][:3].all()


def test_reinit_units_keeps_param_count_and_lets_them_learn():
    fams, model, reg = _bench()
    before = reg.report(model).total
    rec = ReinitUnits("regions.2.experts.0.fc2", [0, 1, 2]).apply(model, reg)
    assert rec.param_delta == 0 and reg.report(model).total == before
    assert (model.regions[2].experts[0].fc2.weight[:, :3] == 0).all()
    S.train(model, fams, 50, 0)
    assert model.regions[2].experts[0].fc2.weight[:, :3].abs().sum() > 0


def test_lottery_ticket_and_overlap():
    fams, model, _ = _bench()
    by = {f.name: f for f in fams}

    def trainer(f):
        def run(m, masks):
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            for batch in _family_batches(f, n=40, size=64):
                opt.zero_grad()
                _loss(m, batch).backward()
                opt.step()
                apply_masks(m, masks)
        return run

    prefixes = ("regions.0.experts",)
    tickets = {n: find_ticket(model, trainer(by[n]), prefixes, rounds=2, prune_per_round=0.4)
               for n in ("easy", "capacity")}
    for masks in tickets.values():
        kept = sum(m.sum() for m in masks.values()) / sum(m.numel() for m in masks.values())
        assert 0.3 < kept < 0.4                      # 0,6 × 0,6 = 0,36 de pesos vivos
    ov = ticket_overlap(tickets)
    assert 0 < ov["unused_fraction"] < 1 and 0 < ov["shared_fraction"] < 1


def test_expert_and_iteration_usage_on_hn1():
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    x = torch.randint(0, 256, (2, 60))
    out = model(x)
    usage = expert_usage(out["routes"], model.cfg.n_experts)
    assert abs(usage.sum().item() - 1) < 1e-6

    def loss_n(m, batch, n):
        o = m(batch, n_iters=n)
        return F.cross_entropy(o["logits"][:, :-1].reshape(-1, 256), batch[:, 1:].reshape(-1)), o
    rep = iteration_usage(model, [x], loss_n)
    assert len(rep["weight_per_iteration"]) == model.cfg.max_iters
    assert abs(sum(rep["weight_per_iteration"]) - 1) < 1e-4
    assert "regions.0.experts.0.fc2" in hidden_layers(model)
