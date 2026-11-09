"""ASG v0 — ciclo de crecimiento estructural dirigido (ASG §4 y §7; ARQUITECTURA_HN1 §6.3).

Piezas reutilizables del ciclo:

1. ``plan_widening`` — reparte un presupuesto de unidades entre las regiones
   elegidas. Con las regiones diagnosticadas es crecimiento **dirigido**; con
   todas, es la línea base de crecimiento **uniforme**. Mismo código, mismos
   parámetros: la comparación es justa.
2. ``request`` — pide permiso al guardián para experimentar o consolidar.
3. ``apply_plan`` — aplica las operaciones (todas preservan la función).
4. ``decide`` — decide consolidar, ajustar, revertir o mantener como
   experimento, comparando contra un **control entrenado lo mismo sin el cambio**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from guardian.core import Action, Decision, Guardian
from hope.growth.budget import ProposedChange
from hope.growth.ops import ChangeRecord, WidenExpert
from hope.registry.topology import TopologyRegistry


class Outcome(str, Enum):
    CONSOLIDAR = "CONSOLIDAR"
    AJUSTAR = "AJUSTAR"
    REVERTIR = "REVERTIR"
    MANTENER_EXPERIMENTO = "MANTENER_EXPERIMENTO"


@dataclass
class GrowthPlan:
    ops: list[WidenExpert]
    regions: list[int]
    units: int

    def change(self, model) -> ProposedChange:
        delta = sum(op.plan(model).param_delta for op in self.ops)
        return ProposedChange(f"ensanchar regiones {self.regions} (+{self.units} unidades)",
                              param_delta=delta, plasticity_level=4)


def plan_widening(model, regions: list[int], units: int, seed: int = 0) -> GrowthPlan:
    """Reparte ``units`` unidades ocultas entre los especialistas de ``regions``."""
    slots = [(r, e) for r in regions for e in range(model.regions[r].n_experts)]
    if not slots or units % len(slots):
        raise ValueError(f"{units} unidades no se reparten parejo entre {len(slots)} especialistas")
    per = units // len(slots)
    ops = [WidenExpert(r, e, per, seed=seed + i) for i, (r, e) in enumerate(slots)]
    return GrowthPlan(ops, list(regions), units)


def request(guardian: Guardian, model, reg: TopologyRegistry, plan: GrowthPlan, *,
            consolidate: bool, justification: str = "") -> Decision:
    rep = reg.report(model)
    change = plan.change(model)
    kind = "structural.consolidate" if consolidate else "structural.experiment"
    return guardian.decide(Action(kind, {
        "change": {"description": change.description, "param_delta": change.param_delta,
                   "plasticity_level": change.plasticity_level},
        "current_total": rep.total, "current_active": rep.active_expected,
    }, requested_by="asg", justification=justification))


def apply_plan(model, reg: TopologyRegistry, plan: GrowthPlan) -> list[ChangeRecord]:
    return [op.apply(model, reg) for op in plan.ops]


@dataclass
class DecisionRule:
    min_target_gain: float = 0.02      # mejora mínima en las tareas objetivo
    max_regression: float = 0.02       # caída máxima tolerada en otras tareas
    promising_gain: float = 0.005      # mejora chica pero positiva → ajustar


@dataclass
class OutcomeReport:
    outcome: Outcome
    target_gain: float
    worst_regression: float
    reasons: list[str] = field(default_factory=list)


def decide(gains: dict[str, float], targets: list[str], rule: DecisionRule = DecisionRule()) -> OutcomeReport:
    """``gains``: diferencia de exactitud de validación (experimento − control) por tarea."""
    if not targets:
        return OutcomeReport(Outcome.REVERTIR, 0.0, 0.0, ["sin tareas objetivo"])
    target_gain = sum(gains[t] for t in targets) / len(targets)
    others = [g for t, g in gains.items() if t not in targets]
    worst = min(others, default=0.0)
    reasons = [f"mejora objetivo {target_gain:+.3f}", f"peor cambio en otras tareas {worst:+.3f}"]
    if worst < -rule.max_regression:
        return OutcomeReport(Outcome.REVERTIR, target_gain, worst, reasons + ["regresión inaceptable"])
    if target_gain >= rule.min_target_gain:
        return OutcomeReport(Outcome.CONSOLIDAR, target_gain, worst, reasons)
    if target_gain >= rule.promising_gain:
        return OutcomeReport(Outcome.AJUSTAR, target_gain, worst, reasons + ["prometedor pero insuficiente"])
    return OutcomeReport(Outcome.REVERTIR, target_gain, worst, reasons + ["sin mejora demostrable"])
