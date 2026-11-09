import pytest
import torch

from hope.core.hn1 import build_hn1, gather_patches, patch_boundaries, preset
from hope.core.layers import HNHSMLP, linear_scan
from hope.growth.ops import AddExpert, WidenExpert


def _bytes(text: str) -> torch.Tensor:
    return torch.tensor([list(text.encode())])


@pytest.mark.parametrize("train_mode", [False, True])
def test_causality(train_mode):
    """Cambiar bytes futuros no cambia ninguna predicción anterior."""
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    model.train(train_mode)
    text = "la resistencia del circuito se quema si la corriente es alta, revisar"
    x = _bytes(text).repeat(2, 1)
    k = 30
    y = x.clone()
    y[:, k:] = torch.randint(0, 256, y[:, k:].shape)
    with torch.no_grad():
        a, b = model(x)["logits"], model(y)["logits"]
    assert (a[:, :k] - b[:, :k]).abs().max().item() < 1e-5


def test_patches():
    x = _bytes("hola mundo, superextraordinario")
    end, pid = patch_boundaries(x, max_patch=8)
    text = x[0].tolist()
    patches, cur = [], []
    for byte, e in zip(text, end[0].tolist()):
        cur.append(byte)
        if e:
            patches.append(bytes(cur).decode())
            cur = []
    assert patches == ["hola ", "mundo,", " ", "superext", "raordina", "rio"]
    assert pid[0, -1].item() == len(patches) - 1
    states = torch.arange(x.shape[1]).float().view(1, -1, 1)
    gathered, valid = gather_patches(states, end, pid)
    assert gathered[0, :, 0].tolist() == [4, 10, 11, 19, 27, 30]
    assert valid.all()


def test_scan_matches_loop():
    torch.manual_seed(0)
    a, u = torch.rand(2, 100, 3), torch.randn(2, 100, 3)
    h, ref = torch.zeros(2, 3), []
    for t in range(100):
        h = a[:, t].clamp_min(torch.exp(torch.tensor(-5.0))) * h + u[:, t]
        ref.append(h)
    assert torch.allclose(linear_scan(torch.log(a), u), torch.stack(ref, 1), atol=1e-5)


def test_scan_gradients_finite_with_extreme_decay():
    log_a = torch.full((1, 40, 3), -1e4, requires_grad=True)   # decaimiento que en float32 sería exactamente 0
    u = torch.randn(1, 40, 3, requires_grad=True)
    linear_scan(log_a, u).sum().backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(log_a.grad).all()


def test_memory_gate_starts_closed():
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    model.eval()
    x = _bytes("memoria externa de hope")
    mem = torch.randn(1, 5, model.cfg.d)
    with torch.no_grad():
        assert torch.allclose(model(x)["logits"], model(x, memory=mem)["logits"])


def test_growth_ops_work_on_hn1_regions():
    torch.manual_seed(0)
    model, reg = build_hn1("hn1-tiny")
    model.eval()
    x = _bytes("crecimiento dirigido sin perder lo aprendido")
    with torch.no_grad():
        before = model(x)["logits"]
        AddExpert(0).apply(model, reg)
        WidenExpert(0, 1, extra=16).apply(model, reg)
        after = model(x)["logits"]
    assert (after - before).abs().max().item() < 1e-5
    assert reg.validate(model) == [] and reg.report(model).unregistered == 0


def test_presets_param_counts():
    for name, lo, hi in (("hn1-25m", 22e6, 26e6), ("hn1-200m", 160e6, 185e6)):
        model, reg = build_hn1(name, device="meta")
        rep = reg.report(model)
        assert lo < rep.total < hi and rep.unregistered == 0
        assert rep.active_expected < rep.total             # especialistas top-2 de 8


def test_halting_and_fixed_iterations():
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    x = _bytes("cuantas iteraciones necesita esta frase")
    out = model(x)
    assert 1.0 <= out["expected_iters"].item() <= model.cfg.max_iters
    assert model(x, n_iters=1)["iters_used"] == 1


def test_hnhs_signs_are_fixed():
    mlp = HNHSMLP(8, 16, inhibitory_frac=0.25)
    w = mlp.output_weight()
    assert (w[:, :12] >= 0).all() and (w[:, 12:] <= 0).all()
    cfg = preset("hn1-tiny")
    cfg.hnhs = True
    from hope.core.hn1 import HN1Model
    loss, _ = HN1Model(cfg).loss(_bytes("poblaciones excitadoras e inhibidoras"))
    loss.backward()


def test_tiny_model_learns():
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    x = _bytes("la ley de ohm: V = I * R. " * 4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        loss, stats = model.loss(x)
        first = first or stats["ce"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert stats["ce"] < first * 0.5


# --- bucle estable (informe «Razonamiento por parámetro» §1.1.1) -------------------------------
def test_stable_loop_defaults_keep_old_checkpoints_and_removes_step_embed():
    from dataclasses import replace

    from hope.core.hn1 import HN1Config, HN1Model, stable_loop
    cfg = preset("hn1-tiny")
    old = HN1Model(cfg)
    assert hasattr(old.core, "step_embed")                            # comportamiento anterior intacto
    HN1Model(HN1Config.from_dict(cfg.to_dict())).load_state_dict(old.state_dict())
    new = HN1Model(stable_loop(cfg))
    assert not hasattr(new.core, "step_embed")
    n_old = sum(p.numel() for p in old.parameters())
    n_new = sum(p.numel() for p in new.parameters())
    assert n_old - n_new == cfg.max_iters * cfg.d                     # no agrega: quita la tabla
    sen = HN1Model(replace(cfg, core_step="senoidal", core_inject="convexa"))
    assert sum(p.numel() for p in sen.parameters()) == n_new


@pytest.mark.parametrize("train_mode", [False, True])
def test_stable_loop_is_causal_and_iterates_beyond_training(train_mode):
    from hope.core.hn1 import HN1Model, stable_loop
    torch.manual_seed(0)
    model = HN1Model(stable_loop(preset("hn1-tiny"), train_max=4, infer_max=6)).train(train_mode)
    x = _bytes("la resistencia del circuito se quema si la corriente es alta, revisar").repeat(2, 1)
    y = x.clone()
    y[:, 30:] = torch.randint(0, 256, y[:, 30:].shape)
    with torch.no_grad():
        torch.manual_seed(1)
        a = model(x)["logits"]
        torch.manual_seed(1)                                           # misma profundidad muestreada
        b = model(y)["logits"]
    assert (a[:, :30] - b[:, :30]).abs().max().item() < 1e-5
    model.eval()
    with torch.no_grad():
        out = model(x, n_iters=24)                                     # 24 > tope de entrenamiento (4)
    assert torch.isfinite(out["logits"]).all() and out["iters_used"] <= 24


def test_convex_injection_keeps_the_state_bounded():
    """Con 0 < Ā < 1 el estado no explota aunque se itere mucho más que al entrenar."""
    from dataclasses import replace

    from hope.core.hn1 import ConvergentCore, stable_loop
    torch.manual_seed(0)
    cfg = stable_loop(preset("hn1-tiny"), tol=0.0)
    core = ConvergentCore(cfg).eval()
    plain = ConvergentCore(replace(cfg, core_inject="ninguna")).eval()
    plain.load_state_dict(core.state_dict())
    x = torch.randn(2, 10, cfg.d)
    with torch.no_grad():
        h_stable = core(x, n_iters=64)[0].norm(dim=-1).max()
        h_plain = plain(x, n_iters=64)[0].norm(dim=-1).max()
        h_8 = core(x, n_iters=8)[0].norm(dim=-1).max()
    assert torch.isfinite(h_stable) and h_stable < 3 * h_8        # se estabiliza
    assert h_plain > h_stable                                      # sin reinyección, el residuo crece


def test_heavy_tailed_depth_and_truncated_backprop():
    from hope.core.hn1 import ConvergentCore, stable_loop
    torch.manual_seed(0)
    core = ConvergentCore(stable_loop(preset("hn1-tiny"), mean_iters=4.0, train_max=12, bptt=2)).train()
    depths = [core.sample_depth() for _ in range(4000)]
    mean = sum(depths) / len(depths)
    assert 3.4 < mean < 4.6 and max(depths) <= 12 and min(depths) >= 1
    assert sum(d >= 8 for d in depths) > 20                        # cola pesada: a veces muy profundo
    x = torch.randn(2, 6, core.d, requires_grad=True)
    out, info = core(x)
    out.sum().backward()
    grads = [p.grad for p in core.block.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # backprop truncado: con más de bptt iteraciones, la entrada solo llega por la reinyección
    assert x.grad is not None and torch.isfinite(x.grad).all()
