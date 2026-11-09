import copy

import pytest
import torch

from hope.core.hn1 import build_hn1
from hope.train.continual import Mixture, Source, TrainConfig, Trainer
from hope.train.lowprec import (dequantize_blockwise, max_params_for, prepare,
                                quantize_blockwise, stochastic_round_bf16)

TEXT = "la ley de ohm relaciona tension corriente y resistencia. " * 30


def test_stochastic_rounding_is_unbiased():
    x = torch.full((100_000,), 1.0 + 2 ** -10)
    assert x.to(torch.bfloat16).float().mean().item() == 1.0          # al más cercano se pierde
    assert abs(stochastic_round_bf16(x).float().mean().item() - (1 + 2 ** -10)) < 5e-5


def test_blockwise_quantization_error_is_bounded():
    m = torch.randn(10_000) * 3
    q, s, n = quantize_blockwise(m)
    assert q.dtype == torch.int8
    err = (dequantize_blockwise(q, s, n, m.shape) - m).abs()
    assert err.max() <= s.max() / 127 + 1e-6


def test_low_precision_learns_like_fp32_with_much_less_state():
    torch.manual_seed(0)
    base, _ = build_hn1("hn1-tiny")
    losses = {}
    for mode in ("fp32", "bf16-8bit"):
        model = copy.deepcopy(base)
        tr = Trainer(model, TrainConfig(lr=3e-3, warmup=5, batch=4, seq_bytes=64, precision=mode, seed=0),
                     device="cpu")
        src = Source.from_texts("a", [TEXT], 1.0)
        rep = tr.train(Mixture([src]), 120)
        losses[mode] = rep.bpb_after["a"]
        assert rep.bpb_after["a"] < rep.bpb_before["a"] * 0.6
        if mode == "bf16-8bit":
            assert all(p.dtype == torch.bfloat16 for p in model.parameters())
            state = tr.opt.state_bytes()
            params = sum(p.numel() for p in model.parameters())
            assert state < params * 2.2                                  # vs. 8 bytes/param de AdamW fp32
    assert losses["bf16-8bit"] < losses["fp32"] * 1.25


def test_low_precision_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    cfg = TrainConfig(lr=3e-3, warmup=5, batch=4, seq_bytes=64, precision="bf16-8bit")
    tr = Trainer(model, cfg, device="cpu")
    src = Source.from_texts("a", [TEXT], 1.0)
    tr.train(Mixture([src]), 20)
    tr.save(tmp_path / "ck")
    tr2 = Trainer.resume(tmp_path / "ck", cfg, device="cpu")
    assert abs(tr2.bpb(src) - tr.bpb(src)) < 1e-4
    st1 = next(iter(tr.opt.state.values()))
    st2 = next(iter(tr2.opt.state.values()))
    assert st2["m_s"].dtype == torch.float32 and torch.equal(st1["m_q"], st2["m_q"])


def test_capacity_estimates():
    assert max_params_for(16, "fp32") < 1e9 < max_params_for(16, "bf16-8bit")
    with pytest.raises(ValueError):
        prepare(torch.nn.Linear(2, 2), "fp4", lr=1e-3)
