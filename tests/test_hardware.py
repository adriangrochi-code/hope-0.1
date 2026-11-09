"""Perfiles de hardware, precisión fp32-8bit (GTX 10xx) y continuación entre máquinas."""

import torch
import yaml

from hope.bench.hardware import PROFILES, recommend
from hope.core.hn1 import HN1Model, preset
from hope.device import supports_bf16
from hope.train.continual import Mixture, Source, Trainer, TrainConfig
from hope.train.lowprec import BYTES_PER_PARAM, LowPrecisionAdamW


def test_gtx1060_profile_recommends_200m_and_rejects_500m():
    rows = {r["preset"]: r for r in recommend(PROFILES["gtx1060"], months=2)}
    assert rows["hn1-200m"]["entra"] and not rows["hn1-500m"]["entra"]
    assert rows["hn1-500m"]["entra_con_offload"]                         # con el optimizador en la RAM, sí
    assert rows["hn1-200m"]["tokens_por_parametro"] >= 20               # por encima del óptimo de Chinchilla
    assert BYTES_PER_PARAM["fp32-8bit"] < BYTES_PER_PARAM["fp32"]
    assert not supports_bf16(torch.device("cpu"))


def test_fp32_8bit_trains_and_continues_in_bf16(tmp_path):
    torch.manual_seed(0)
    text = Source.from_texts("t", ["la lluvia cae sobre el campo verde. " * 60], 1.0)
    tr = Trainer(HN1Model(preset("hn1-tiny")), TrainConfig(precision="fp32-8bit", batch=4, seq_bytes=64,
                                                          lr=2e-3, warmup=5), device="cpu")
    assert next(tr.model.parameters()).dtype == torch.float32 and isinstance(tr.opt, LowPrecisionAdamW)
    before = tr.bpb(text, windows=8, seq=64)
    tr.train(Mixture([text]), 40, eval_sources=[])
    assert tr.bpb(text, windows=8, seq=64) < before
    tr.save(tmp_path / "ck")
    # la «A770» retoma el mismo modelo en bf16
    tr2 = Trainer.resume(tmp_path / "ck", TrainConfig(precision="bf16-8bit", batch=4, seq_bytes=64), device="cpu")
    assert next(tr2.model.parameters()).dtype == torch.bfloat16 and tr2.step == tr.step
    tr2.train(Mixture([text]), 3, eval_sources=[])


def test_home_plans_are_consistent():
    home = yaml.safe_load(open("configs/plan_gtx1060.yaml"))
    a770 = yaml.safe_load(open("configs/plan_a770_continuacion.yaml"))
    assert home["precision"] == "fp32-8bit" and not home["compilar"] and home["offload_optimizador"]
    assert home["etapas"][0]["preset"] == "hn1-500m" and home["etapas"][0]["vision"]
    alt = yaml.safe_load(open("configs/plan_gtx1060_200m.yaml"))
    assert alt["etapas"][0]["preset"] == "hn1-200m"
    assert a770["desde"].startswith(home["salida"]) and a770["etapas"][0]["crecer_a"] == "hn1-500m"


def test_optimizer_offload_keeps_state_on_cpu_and_matches():
    from hope.train.lowprec import prepare
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 40))
    outs = []
    for offload in (False, True):
        torch.manual_seed(0)
        m = HN1Model(preset("hn1-tiny"))
        opt = prepare(m, "fp32-8bit", lr=1e-3, offload=offload)
        for _ in range(3):
            opt.zero_grad()
            m.loss(x, full_depth=True)[0].backward()
            opt.step()
        if offload:
            assert all(v.device.type == "cpu" for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
        outs.append(torch.cat([p.detach().flatten() for p in m.parameters()]))
    assert torch.allclose(outs[0], outs[1], atol=1e-6)
    from hope.train.muon import make_muon
    m = HN1Model(preset("hn1-tiny"))
    opt = make_muon(m, lr=1e-3, offload=True)
    m.loss(x, full_depth=True)[0].backward()
    opt.step()
    assert all(v.device.type == "cpu" for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
