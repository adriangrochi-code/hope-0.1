"""Entrenamiento consciente de la cuantización y exportación a 4/8 bits."""

import copy

import torch

from hope.core.fast import QuantLinear, quantize
from hope.core.hn1 import HN1Model, preset
from hope.train.continual import Mixture, Source, Trainer, TrainConfig
from hope.train.qat import QATLinear, disable_qat, enable_qat, fake_quant, set_qat_bits


def test_fake_quant_matches_real_export_and_passes_gradient():
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 32)
    for bits in (4, 8):
        q = QuantLinear(lin, bits=bits, group=64)
        assert torch.allclose(q.dequantized(), fake_quant(lin.weight, bits, 64), atol=1e-3)
    assert QuantLinear(lin, 4).qweight.numel() == 32 * 64          # dos pesos de 4 bits por byte
    w = torch.randn(8, 64, requires_grad=True)
    fake_quant(w, 4).sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))                 # straight-through


def test_enable_qat_keeps_checkpoints_and_function():
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny")).eval()
    keys = list(m.state_dict())
    x = torch.randint(0, 256, (1, 40))
    ref = m(x)["logits"]
    n = enable_qat(m, bits=4)
    assert n > 10 and list(m.state_dict()) == keys
    assert any(isinstance(mod, QATLinear) for mod in m.modules())
    q4 = m(x)["logits"]
    assert not torch.allclose(q4, ref)
    set_qat_bits(m, None)
    assert torch.allclose(m(x)["logits"], ref)
    # el modelo con cuantización falsa de 4 bits = el exportado de verdad a 4 bits
    set_qat_bits(m, 4)
    real = copy.deepcopy(m)
    disable_qat(real)
    info = quantize(real, bits=4)
    assert info["bytes_despues"] < info["bytes_antes"] / 6
    assert torch.allclose(real(x)["logits"], q4, atol=2e-2)


def test_qat_training_reduces_int4_degradation():
    torch.manual_seed(0)
    text = Source.from_texts("t", ["el agua hierve a cien grados y se congela a cero. " * 80], 1.0)
    cfg = dict(batch=8, seq_bytes=96, precision="fp32", lr=3e-3, warmup=10)
    base = Trainer(HN1Model(preset("hn1-tiny")), TrainConfig(**cfg), device="cpu")
    base.train(Mixture([text]), 120, eval_sources=[])
    gaps = {}
    for name, qat in (("normal", None), ("qat", (4,))):
        tr = Trainer(copy.deepcopy(base.model), TrainConfig(**cfg, qat_bits=qat), device="cpu")
        tr.rng = __import__("numpy").random.default_rng(1)
        tr.train(Mixture([text]), 60, eval_sources=[])
        m = tr.model.eval()
        set_qat_bits(m, None)
        full = tr.bpb(text, windows=8, seq=96)
        disable_qat(m)
        quantize(m, bits=4)
        gaps[name] = tr.bpb(text, windows=8, seq=96) - full
    assert gaps["qat"] < gaps["normal"], gaps
