"""Presupuesto y modos de crecimiento (ASG §3, §14 y §15; ARQUITECTURA_HN1 §6.1 y §8.3).

El límite autorizado es un **techo, no una meta**. Ninguna operación puede
superarlo, y el modo de crecimiento define qué puede hacer ASG por su cuenta.

Este módulo solo *evalúa* cambios. Quien hace cumplir las reglas es el
guardián (``guardian``), que corre de forma independiente del optimizador.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import yaml


class GrowthMode(str, Enum):
    EXPANSION_DISABLED = "EXPANSION_DISABLED"            # no puede iniciar crecimiento
    EXPERIMENT_ONLY = "EXPERIMENT_ONLY"                  # copias experimentales, sin consolidar
    USER_APPROVAL_REQUIRED = "USER_APPROVAL_REQUIRED"    # propone; consolidar requiere aprobación
    AUTO_CONSOLIDATION_WITHIN_LIMIT = "AUTO_CONSOLIDATION_WITHIN_LIMIT"
    PAUSED = "PAUSED"                                    # suspende todo y conserva el estado


@dataclass
class ConsolidationPolicy:
    """Qué cuenta como «modificación importante» (Carta art. 6.7).

    Un cambio importante **siempre** requiere aprobación humana, aunque el
    modo sea AUTO_CONSOLIDATION_WITHIN_LIMIT. La Carta prevalece sobre ASG.

    Valores por defecto conservadores, pendientes de que el usuario los fije.
    """

    max_auto_param_fraction: float = 0.02   # > 2% del total en un solo cambio → importante
    max_auto_plasticity_level: int = 4      # nivel 5 (regional/global) → importante
    protected_regions_require_human: bool = True


@dataclass
class GrowthBudget:
    max_total_params: int
    max_active_params: int | None = None
    max_regions: int | None = None
    max_connections: int | None = None
    training_gpu_hours: float | None = None
    max_vram_gb: float | None = None
    max_ram_gb: float | None = None
    max_disk_gb: float | None = None
    experiment_time_limit_hours: float | None = None
    mode: GrowthMode = GrowthMode.EXPERIMENT_ONLY
    consolidation: ConsolidationPolicy = field(default_factory=ConsolidationPolicy)

    @classmethod
    def from_yaml(cls, text: str) -> "GrowthBudget":
        data = yaml.safe_load(text)
        data["mode"] = GrowthMode(data.get("mode", GrowthMode.EXPERIMENT_ONLY))
        data["consolidation"] = ConsolidationPolicy(**data.get("consolidation", {}))
        return cls(**data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d


@dataclass(frozen=True)
class ProposedChange:
    """Resumen de un cambio estructural para evaluarlo contra el presupuesto."""

    description: str
    param_delta: int                     # positivo = agrega, negativo = retira
    active_delta: float = 0.0
    region_delta: int = 0
    connection_delta: int = 0
    plasticity_level: int = 4            # 1..5 según ARQUITECTURA_HN1 §6.4
    touches_protected: bool = False


@dataclass(frozen=True)
class BudgetCheck:
    allowed: bool
    reasons: tuple[str, ...]
    important: bool                      # requiere aprobación humana para consolidar


def check_change(budget: GrowthBudget, *, current_total: int, current_active: float,
                 current_regions: int, current_connections: int,
                 change: ProposedChange, experimental: bool = True) -> BudgetCheck:
    """Evalúa si un cambio puede aplicarse (en copia experimental o consolidarse)."""
    reasons: list[str] = []

    if budget.mode in (GrowthMode.PAUSED, GrowthMode.EXPANSION_DISABLED) and change.param_delta > 0:
        reasons.append(f"modo {budget.mode.value}: no se permite crecer")
    if budget.mode == GrowthMode.PAUSED:
        reasons.append("modo PAUSED: no se permiten cambios estructurales")

    new_total = current_total + change.param_delta
    if new_total > budget.max_total_params:
        reasons.append(f"parámetros totales {new_total:,} > techo {budget.max_total_params:,}")
    if budget.max_active_params is not None:
        new_active = current_active + change.active_delta
        if new_active > budget.max_active_params:
            reasons.append(f"parámetros activos {new_active:,.0f} > techo {budget.max_active_params:,}")
    if budget.max_regions is not None and current_regions + change.region_delta > budget.max_regions:
        reasons.append("se supera el máximo de regiones")
    if budget.max_connections is not None and current_connections + change.connection_delta > budget.max_connections:
        reasons.append("se supera el máximo de conexiones")

    if not experimental and budget.mode == GrowthMode.EXPERIMENT_ONLY:
        reasons.append("modo EXPERIMENT_ONLY: no se puede consolidar")

    pol = budget.consolidation
    important = (
        abs(change.param_delta) > pol.max_auto_param_fraction * max(current_total, 1)
        or change.plasticity_level > pol.max_auto_plasticity_level
        or (change.touches_protected and pol.protected_regions_require_human)
    )
    return BudgetCheck(allowed=not reasons, reasons=tuple(reasons), important=important)


def needs_human_approval(budget: GrowthBudget, check: BudgetCheck) -> bool:
    """¿Consolidar este cambio requiere aprobación humana?"""
    if budget.mode == GrowthMode.USER_APPROVAL_REQUIRED:
        return True
    if budget.mode == GrowthMode.AUTO_CONSOLIDATION_WITHIN_LIMIT:
        return check.important
    return True  # en cualquier otro modo, consolidar no es automático
