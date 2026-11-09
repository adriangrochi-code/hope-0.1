import torch

from hope.core.hn1 import build_hn1
from hope.growth.zone_composer import ZoneReport, evaluate, propose, zone_report
from hope.train.continual import Mixture, Source, TrainConfig, Trainer


def test_proposals_follow_usage():
    rep = ZoneReport(0, [0.40, 0.30, 0.20, 0.08, 0.02], ["mlp", "glu", "ei", "conv", "asociativa"])
    from collections import defaultdict
    agg = defaultdict(lambda: {"experts": 0, "usage": 0.0})
    for k, u in zip(rep.kinds, rep.usage):
        agg[k]["experts"] += 1
        agg[k]["usage"] += u
    for v in agg.values():
        v["usage_per_expert"] = v["usage"] / v["experts"]
    rep.by_kind = dict(agg)
    props = propose(rep)
    assert props[0].action == "agregar" and props[0].kind == "mlp"
    repl = [p for p in props if p.action == "reemplazar"]
    assert repl and repl[0].expert == 4 and repl[0].kind == "mlp"


def test_zone_report_and_controlled_experiment():
    torch.manual_seed(0)
    model, reg = build_hn1("hn1-tiny")
    x = torch.randint(0, 256, (2, 80))
    rep = zone_report(model, [model(x)["routes"][0]])
    assert abs(sum(rep.usage) - 1) < 1e-6 and set(rep.kinds) == set(rep.by_kind)
    tr = Trainer(model, TrainConfig(lr=3e-3, warmup=2, batch=2, seq_bytes=48, precision="fp32"), device="cpu")
    src = Source.from_texts("a", ["la ley de ohm relaciona tension y corriente. " * 20], 1.0)
    from hope.growth.zone_composer import ZoneProposal
    outcome, info = evaluate(tr, reg, ZoneProposal("agregar", 0, "recurrente"), Mixture([src]), 15, [src])
    assert info["params_added"] > 0 and outcome.outcome.value in ("CONSOLIDAR", "AJUSTAR", "REVERTIR")
    assert reg.report(tr.model).total == reg.report(model).total          # el original no cambió
