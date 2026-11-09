import pytest
import torch

from hope.core.hn1 import HN1Config, build_hn1
from hope.core.zones import KINDS, Zone, make_specialist
from hope.growth.lth import expert_usage, hidden_layers
from hope.growth.ops import AddExpert, WidenExpert

D, H = 16, 24


@pytest.mark.parametrize("kind", KINDS)
def test_specialist_shapes_and_causality(kind):
    torch.manual_seed(0)
    sp = make_specialist(kind, D, H)
    x = torch.randn(2, 20, D)
    y = sp(x)
    assert y.shape == x.shape and sp.kind == kind
    if sp.sequence_aware:
        x2 = x.clone()
        x2[:, 12:] = torch.randn_like(x2[:, 12:])
        assert torch.allclose(sp(x2)[:, :12], y[:, :12], atol=1e-5)


@pytest.mark.parametrize("kind", [k for k in KINDS if k != "relacional"])
def test_widen_preserves_output(kind):
    torch.manual_seed(0)
    sp = make_specialist(kind, D, H)
    x = torch.randn(2, 20, D)
    before = sp(x)
    sp.widen(8, seed=1)
    assert sp.hidden == H + 8
    assert (sp(x) - before).abs().max().item() < 1e-5


def test_zone_mixes_heterogeneous_specialists():
    torch.manual_seed(0)
    zone = Zone(D, KINDS, H, top_k=2)
    x = torch.randn(3, 30, D)
    out, balance, idx = zone(x)
    assert out.shape == x.shape and idx.shape == (3, 30, 2) and balance.item() > 0
    used = expert_usage([idx], zone.n_experts)
    assert (used > 0).sum() >= 3                       # varios tipos trabajan a la vez


def test_add_specialist_of_a_given_kind_preserves_hn1():
    torch.manual_seed(0)
    model, reg = build_hn1("hn1-tiny")
    model.eval()
    x = torch.tensor([list("zonas de especialistas heterogeneos".encode())])
    with torch.no_grad():
        before = model(x)["logits"]
        rec = AddExpert(0, kind="recurrente").apply(model, reg)
        WidenExpert(0, 3, extra=8).apply(model, reg)           # el especialista recurrente original
        after = model(x)["logits"]
    assert (after - before).abs().max().item() < 1e-5
    assert reg.get(rec.created[0]).meta["kind"] == "recurrente"
    assert model.regions[0].kinds[-1] == "recurrente"
    assert reg.validate(model) == [] and reg.report(model).unregistered == 0


def test_registry_records_kinds_and_lth_sees_their_neurons():
    model, reg = build_hn1("hn1-tiny")
    kinds = [n.meta["kind"] for n in reg.nodes() if n.kind.value == "EXPERT"]
    assert kinds == list(KINDS)
    layers = hidden_layers(model)
    assert all(f"regions.0.experts.{i}.fc2" in layers for i in range(len(KINDS)))


def test_homogeneous_zone_is_still_possible_for_ablation():
    cfg = HN1Config(d_local=32, local_heads=2, local_mlp=64, d=48, heads=4, mlp_hidden=96,
                    pattern=("ssm",), max_iters=2, expert_hidden=32, zone_kinds=("mlp",) * 4)
    assert cfg.n_experts == 4
    from hope.core.hn1 import HN1Model
    m = HN1Model(cfg)
    assert m.regions[0].kinds == ["mlp"] * 4
    # Checkpoints anteriores (sin zone_kinds) se cargan como zona homogénea.
    old = cfg.to_dict()
    old.pop("zone_kinds")
    assert HN1Config.from_dict(old).zone_kinds == ("mlp",) * 4
