"""Composición automática de zonas (propuesta 5, aprobada; AANS §3.3, §5.4–5.5; ASG §7).

Lee cuánto se usa cada especialista (y de qué tipo de red es) y propone:

* **agregar** un especialista del tipo sobrecargado (su uso por especialista
  supera claramente lo esperado con un reparto parejo);
* **reemplazar** un especialista casi sin uso por uno del tipo más demandado:
  primero se enmascara (retiro en dos pasos) y se agrega el nuevo.

Ninguna propuesta se aplica sola: ``evaluate`` entrena dos copias iguales
(control y cambio) con el mismo entrenamiento y decide con la regla de ASG.
Uso bajo no prueba inutilidad (ASG §9): por eso se evalúa, no se borra.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from hope.growth.asg import DecisionRule, OutcomeReport, decide
from hope.growth.lth import expert_usage
from hope.growth.ops import AddExpert, mask_expert


@dataclass
class ZoneProposal:
    action: str                      # agregar | reemplazar
    zone: int
    kind: str                        # tipo a agregar
    expert: int | None = None        # especialista a enmascarar (reemplazar)
    reason: str = ""
    priority: float = 0.0


@dataclass
class ZoneReport:
    zone: int
    usage: list[float]
    kinds: list[str]
    by_kind: dict[str, dict] = field(default_factory=dict)


def zone_report(model, routes: list[torch.Tensor], zone: int = 0) -> ZoneReport:
    region = model.regions[zone]
    usage = expert_usage(routes, region.n_experts).tolist()
    kinds = list(region.kinds)
    agg: dict[str, dict] = defaultdict(lambda: {"experts": 0, "usage": 0.0})
    for k, u in zip(kinds, usage):
        agg[k]["experts"] += 1
        agg[k]["usage"] += u
    for v in agg.values():
        v["usage_per_expert"] = v["usage"] / v["experts"]
    return ZoneReport(zone, usage, kinds, dict(agg))


def propose(rep: ZoneReport, *, overload: float = 1.5, idle: float = 0.25) -> list[ZoneProposal]:
    n = len(rep.usage)
    expected = 1.0 / n
    out: list[ZoneProposal] = []
    hot = sorted(rep.by_kind.items(), key=lambda kv: -kv[1]["usage_per_expert"])
    hottest = hot[0][0]
    for kind, v in hot:
        if v["usage_per_expert"] >= overload * expected:
            out.append(ZoneProposal("agregar", rep.zone, kind,
                                    reason=f"«{kind}» recibe {v['usage_per_expert']:.1%} por especialista "
                                           f"(reparto parejo: {expected:.1%})",
                                    priority=v["usage_per_expert"] / expected))
    for e, (k, u) in enumerate(zip(rep.kinds, rep.usage)):
        if u < idle * expected and k != hottest:
            out.append(ZoneProposal("reemplazar", rep.zone, hottest, expert=e,
                                    reason=f"especialista {e} («{k}») casi sin uso ({u:.1%}); "
                                           f"probar un «{hottest}» en su lugar",
                                    priority=(expected - u) / expected))
    return sorted(out, key=lambda p: -p.priority)


def apply(model, reg, p: ZoneProposal) -> list:
    records = []
    if p.action == "reemplazar":
        records.append(mask_expert(model, reg, p.zone, p.expert))
    records.append(AddExpert(p.zone, kind=p.kind).apply(model, reg))
    return records


def evaluate(trainer, reg, proposal: ZoneProposal, mixture, steps: int, eval_sources,
             rule: DecisionRule = DecisionRule(min_target_gain=0.0, promising_gain=-1.0)) -> tuple[OutcomeReport, dict]:
    """Control vs. cambio con el mismo entrenamiento. Ganancia = BPB del control − BPB con el cambio."""
    from hope.train.lowprec import prepare
    control = copy.deepcopy(trainer)
    treated = copy.deepcopy(trainer)
    treg = copy.deepcopy(reg)
    apply(treated.model, treg, proposal)
    # Habilitar el especialista nuevo para que el router pueda elegirlo y entrene.
    zone = treated.model.regions[proposal.zone]
    zone.expert_mask[-1] = 0.0
    treated.opt = prepare(treated.model, treated.cfg.precision, lr=treated.cfg.lr,
                          weight_decay=treated.cfg.weight_decay, seed=treated.cfg.seed)
    control.rng = copy.deepcopy(trainer.rng)
    treated.rng = copy.deepcopy(trainer.rng)
    rc = control.train(mixture, steps, eval_sources=eval_sources)
    rt = treated.train(mixture, steps, eval_sources=eval_sources)
    gains = {k: rc.bpb_after[k] - rt.bpb_after[k] for k in rc.bpb_after}
    outcome = decide(gains, list(gains), rule)
    return outcome, {"control_bpb": rc.bpb_after, "treated_bpb": rt.bpb_after, "gains": gains,
                     "params_added": treg.report(treated.model).total - reg.report(trainer.model).total}
