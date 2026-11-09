"""Presupuesto de parámetros como objetivo: «tu presupuesto ahora es X» (ARQUITECTURA_HN1 §6.5).

El usuario dice «ahora tenés 1B» (o «bajá a 300M») y Hope se adapta **sin perder lo
aprendido**. ``plan_to_budget(modelo, X, perfil)`` arma la secuencia de operaciones y
``apply_plan`` la ejecuta paso por paso con el guardián.

Crecer (operaciones que preservan la función, §6.2) — política, en este orden:

1. **especialistas** (``AddExpert``): son dispersos (top-k de n), así que agregan parámetros
   casi sin subir el cómputo por byte. Es lo que más capacidad da por cómputo. Hasta
   ``max_experts`` por zona: más allá cada especialista recibe muy pocos datos.
2. **bloques al final del backbone** (``AppendBackboneBlock``, salida en cero): suben el
   cómputo por byte, así que se agregan mientras el cómputo no pase ``block_flops_ratio``
   (1,4) veces el de partida.
3. **ensanchar especialistas** (``WidenExpert`` en modo ``zero_out``), en tandas de
   ``widen_chunks`` unidades, hasta ``max_hidden_factor`` veces el ancho de la configuración
   y mientras el cómputo no pase ``max_flops_ratio`` (1,6); las tandas chicas acercan al objetivo.
4. **zona visual** (``AddVisualZone``) si falta y entra (``vision_first=True`` la pone primera).

La eficiencia de cada familia se **mide** en el plan (parámetros agregados por FLOP de
entrenamiento por byte agregado) y se informa junto con la velocidad estimada antes y
después (``hope.bench.plan.speed``).

Reducir — en dos pasos (enmascarar → evaluar → eliminar), §6.2:

* bloques **agregados** al final del backbone cuya contribución relativa
  ``‖bloque(x) − x‖ / ‖x‖`` es casi nula (< ``identity_tol``);
* especialistas **menos usados** (fracción de elecciones del router medida con
  ``hope.growth.lth.expert_usage``), dejando al menos ``min_experts`` por zona.

Al planificar, los retiros elegidos se enmascaran sobre el modelo real, se mide la
pérdida sobre los lotes dados y se deshace: el plan informa la pérdida medida. Al aplicar
se vuelve a medir y se rechaza si se aleja de lo informado.

Reglas que no se negocian:

* nunca se supera ``min(objetivo, techo)``; el techo lo pone **solo el usuario**
  (``python -m guardian.cli limite X``) y se lee del guardián;
* el modelo tiene que poder entrenarse en la memoria del perfil de hardware
  (con o sin el optimizador en la RAM, según ``offload``);
* cada paso de más del 10 % de los parámetros pide aprobación humana (``structural.consolidate``).

Uso::

    python -m hope.growth.budget_target --checkpoint runs/hope/1_lenguaje --objetivo 1B --perfil a770
    python -m hope.growth.budget_target --checkpoint ... --objetivo 1B --aplicar --salida runs/hope/1b
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hope.growth.budget import ProposedChange
from hope.growth.hn1_ops import AddVisualZone, AppendBackboneBlock
from hope.growth.ops import AddExpert, ChangeRecord, RemoveMaskedExpert, WidenExpert, mask_expert, unmask_expert
from hope.core.toy import NEG_INF
from hope.registry.topology import TopologyRegistry

BLOCK_CYCLE = ("ssm", "ssm", "attn", "ssm")       # el mismo ritmo que los presets 500m/700m/1b
CATEGORY_NAMES = {"especialistas": "agregar especialistas", "bloques": "agregar bloques al final del backbone",
                  "ensanchar": "ensanchar especialistas", "vision": "agregar zona visual",
                  "retirar_bloques": "retirar bloques agregados casi identidad",
                  "retirar_especialistas": "retirar especialistas poco usados"}


def _n(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _meta_clone(model: nn.Module) -> nn.Module:
    """Copia de la estructura sin memoria (tensores «meta») para simular el plan."""
    memo: dict = {}
    for p in model.parameters():
        memo[id(p)] = nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=p.requires_grad)
    for b in model.buffers():
        memo[id(b)] = torch.empty_like(b, device="meta")
    for name in ("_last_global",):
        if hasattr(model, name):
            memo[id(getattr(model, name))] = None
    return copy.deepcopy(model, memo)


def _profile(hardware_profile):
    if hardware_profile is None:
        return None
    from hope.bench.hardware import PROFILES
    return PROFILES[hardware_profile] if isinstance(hardware_profile, str) else hardware_profile


def _speed(model, prof) -> dict:
    from hope.bench.plan import cost, speed
    c = cost(model)
    if prof is None:
        s = speed(c)
    else:
        from hope.bench.hardware import weight_bytes
        s = speed(c, prof.tflops_eff, prof.bandwidth_gbs, weight_bytes(prof))
    return {k: (round(v, 1) if k != "flops_por_byte" else v) for k, v in s.items()}


def _memory(total: int, prof) -> dict | None:
    if prof is None:
        return None
    from hope.bench.hardware import training_fit
    return training_fit(total, prof)


# ---------------------------------------------------------------------------
# Retiro de bloques: enmascarar (identidad) y después eliminar
# ---------------------------------------------------------------------------
class MaskedBlock(nn.Module):
    """Un bloque enmascarado se comporta como la identidad; guarda el original para deshacer."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.kind = getattr(inner, "kind", "?")

    def forward(self, x, cache=None):
        return x


def mask_last_blocks(model, count: int) -> None:
    for j in range(len(model.backbone) - count, len(model.backbone)):
        if not isinstance(model.backbone[j], MaskedBlock):
            model.backbone[j] = MaskedBlock(model.backbone[j])


def unmask_blocks(model) -> None:
    for j, blk in enumerate(model.backbone):
        if isinstance(blk, MaskedBlock):
            model.backbone[j] = blk.inner


class RemoveLastBlock:
    """Elimina el último bloque del backbone, ya enmascarado (exacto respecto del estado enmascarado)."""

    def plan(self, model) -> ProposedChange:
        blk = model.backbone[-1]
        return ProposedChange("retirar el último bloque del backbone", param_delta=-_n(blk), plasticity_level=4)

    @torch.no_grad()
    def apply(self, model, reg: TopologyRegistry) -> ChangeRecord:
        from dataclasses import replace
        j = len(model.backbone) - 1
        if not isinstance(model.backbone[j], MaskedBlock):
            raise ValueError("primero hay que enmascarar y evaluar el bloque")
        delta = self.plan(model).param_delta
        del model.backbone[j]
        model.cfg = replace(model.cfg, pattern=tuple(model.cfg.pattern)[:j])
        node = reg.find_by_path(f"backbone.{j}")
        if node is not None:
            reg.remove(node.id)
        return ChangeRecord("remove_last_block", [node.id] if node else [], delta, preserving=True,
                            removed=[node.id] if node else [], detail={"index": j})


def removable_blocks(model) -> int:
    """Cuántos bloques del final se pueden retirar: agregados, sin atención cruzada visual
    ni conexiones del conectoma que los usen."""
    base = model.cfg.base_blocks
    if base is None:
        return 0
    used = set(model.vision.at_blocks) if model.vision is not None else set()
    for key in model.connections.keys():
        used.update(int(v) for v in key.split("_"))
    count = 0
    for j in range(len(model.backbone) - 1, base - 1, -1):
        if j in used:
            break
        count += 1
    return count


# ---------------------------------------------------------------------------
# Mediciones
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure(model, batches, contribution_blocks: int = 0) -> dict:
    """Pérdida (bits por byte), uso de cada especialista y contribución relativa de los
    últimos ``contribution_blocks`` bloques del backbone sobre ``batches`` ([B, T] bytes)."""
    from hope.growth.lth import expert_usage
    was_training = model.training
    model.eval()
    routes: list[list] = [[] for _ in model.regions]
    contrib: dict[int, list[float]] = {}
    hooks = []
    first = len(model.backbone) - contribution_blocks
    for j in range(first, len(model.backbone)):
        def hook(mod, inp, out, j=j):
            x = inp[0].float()
            contrib.setdefault(j, []).append(((out.float() - x).norm() / x.norm().clamp_min(1e-12)).item())
        hooks.append(model.backbone[j].register_forward_hook(hook))
    ce, count = 0.0, 0
    try:
        for x in batches:
            x = x.to(model.head.weight.device)
            out = model(x[:, :-1])
            ce += F.cross_entropy(out["logits"].float().reshape(-1, out["logits"].shape[-1]),
                                  x[:, 1:].reshape(-1).long(), reduction="sum").item()
            count += x[:, 1:].numel()
            for r, idx in enumerate(out["routes"]):
                routes[r].append(idx)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return {"bpb": ce / max(count, 1) / math.log(2),
            "uso": [expert_usage(rs, model.regions[r].n_experts).tolist() for r, rs in enumerate(routes)],
            "contribucion": {j: sum(v) / len(v) for j, v in contrib.items()}}


def default_batches(n: int = 4, batch: int = 2, length: int = 129, seed: int = 0) -> list[torch.Tensor]:
    """Lotes de bytes aleatorios fijos (solo si no hay datos de validación: el uso que miden es pobre)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (batch, length), generator=g) for _ in range(n)]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
@dataclass
class Step:
    categoria: str
    ops: list
    param_delta: int
    total_antes: int
    total_despues: int
    requiere_aprobacion: bool
    flops_por_byte_delta: float = 0.0
    retiros: list = field(default_factory=list)   # reducir: [("bloque", j)] o [("especialista", r, e)]
    bpb_medido: float | None = None               # reducir: pérdida medida con todo lo anterior enmascarado

    def change(self) -> ProposedChange:
        return ProposedChange(f"presupuesto: {CATEGORY_NAMES[self.categoria]} ({len(self.ops)} operaciones)",
                              param_delta=self.param_delta, plasticity_level=4)

    def resumen(self) -> dict:
        d = {"paso": CATEGORY_NAMES[self.categoria], "operaciones": len(self.ops),
             "parametros": f"{self.total_antes:,} → {self.total_despues:,} ({self.param_delta:+,})".replace(",", "."),
             "requiere_aprobacion": self.requiere_aprobacion}
        if self.param_delta > 0:
            d["parametros_por_flop_byte"] = (round(self.param_delta / self.flops_por_byte_delta, 1)
                                             if self.flops_por_byte_delta > 0 else "sin costo de cómputo")
        if self.bpb_medido is not None:
            d["bpb_medido_tras_el_paso"] = round(self.bpb_medido, 4)
        return d


@dataclass
class BudgetPlan:
    objetivo: int
    techo: int | None
    limite: int
    direccion: str                  # "crecer" | "reducir" | "igual"
    total_antes: int
    total_despues: int
    pasos: list[Step]
    memoria_antes: dict | None
    memoria_despues: dict | None
    velocidad_antes: dict
    velocidad_despues: dict
    flops_base: float
    avisos: list[str] = field(default_factory=list)
    bpb_antes: float | None = None
    bpb_despues: float | None = None

    @property
    def ops(self) -> list:
        return [op for s in self.pasos for op in s.ops]

    def resumen(self) -> dict:
        h = _human
        out = {"direccion": self.direccion, "objetivo": h(self.objetivo),
               "techo_del_guardian": h(self.techo) if self.techo else None, "limite_efectivo": h(self.limite),
               "parametros": {"antes": self.total_antes, "despues": self.total_despues,
                              "despues_legible": h(self.total_despues),
                              "diferencia_con_el_limite": self.limite - self.total_despues},
               "pasos": [s.resumen() for s in self.pasos],
               "memoria_entrenamiento": {"antes": self.memoria_antes, "despues": self.memoria_despues},
               "velocidad": {"antes": self.velocidad_antes, "despues": self.velocidad_despues,
                             "relacion_entrenamiento": round(self.velocidad_despues["entrenamiento_bytes_s"]
                                                             / self.velocidad_antes["entrenamiento_bytes_s"], 3)},
               "avisos": self.avisos}
        if self.bpb_antes is not None:
            out["perdida_bpb"] = {"antes": round(self.bpb_antes, 4), "despues_medida": round(self.bpb_despues, 4),
                                  "diferencia": round(self.bpb_despues - self.bpb_antes, 4)}
        return out


def _human(n: float | None) -> str:
    from guardian.cli import human
    return human(n) if n else "0"


def _period(kinds: list[str]) -> int:
    for p in range(1, len(kinds) + 1):
        if all(kinds[i] == kinds[i % p] for i in range(len(kinds))):
            return p
    return len(kinds)


def plan_to_budget(model, target_params: int, hardware_profile=None, *, ceiling: int | None = None,
                   guardian=None, batches=None, offload: bool | str = "auto", max_experts: int = 64,
                   max_flops_ratio: float = 1.6, block_flops_ratio: float = 1.4, flops_base: float | None = None,
                   widen_chunks: tuple[int, ...] = (1024, 256, 64), max_hidden_factor: float = 2.0,
                   vision_cfg: dict | None = None, vision_first: bool = False, approval_fraction: float | None = None,
                   identity_tol: float = 0.02, min_experts: int | None = None) -> BudgetPlan:
    """Plan para llegar lo más cerca posible de ``target_params`` **sin pasarse** (ni del techo).

    ``hardware_profile``: clave de ``hope.bench.hardware.PROFILES`` o un ``Profile``; ``offload``:
    ``"auto"`` (se permite el optimizador en la RAM si hace falta), ``True`` o ``False`` (solo GPU).
    ``guardian``: si se da, el techo y el umbral de aprobación se leen de su presupuesto.
    ``batches``: lotes de validación para medir el uso y la pérdida al reducir.
    """
    from hope.bench.plan import cost
    prof = _profile(hardware_profile)
    if guardian is not None:
        b = guardian.budget()
        ceiling = b.max_total_params if ceiling is None else min(ceiling, b.max_total_params)
        if approval_fraction is None:
            approval_fraction = b.consolidation.max_auto_param_fraction
    approval_fraction = 0.10 if approval_fraction is None else approval_fraction
    limit = min(target_params, ceiling) if ceiling is not None else target_params
    total0 = _n(model)
    c0 = cost(model)
    flops_base = flops_base or c0.flops_per_byte
    avisos: list[str] = []
    if ceiling is not None and target_params > ceiling:
        avisos.append(f"el objetivo {_human(target_params)} supera el techo del guardián ({_human(ceiling)}): "
                      f"se planifica hasta el techo. Para subirlo: python -m guardian.cli limite "
                      f"{_human(target_params)} --by <tu-nombre>")

    def fits(total: int) -> bool:
        if prof is None:
            return True
        m = _memory(total, prof)
        if offload is True:
            return m["entra_con_offload"]
        if offload is False:
            return m["entra"]
        return m["entra"] or m["entra_con_offload"]

    if total0 <= limit:
        direction = "crecer" if total0 < limit else "igual"
        steps, sim = _plan_growth(model, total0, limit, fits, flops_base, prof, avisos, offload,
                                  max_experts=max_experts, max_flops_ratio=max_flops_ratio,
                                  block_flops_ratio=block_flops_ratio,
                                  widen_chunks=widen_chunks, max_hidden_factor=max_hidden_factor,
                                  vision_cfg=vision_cfg, vision_first=vision_first,
                                  approval_fraction=approval_fraction)
        if not steps:
            direction = "igual"
        total1 = total0 + sum(s.param_delta for s in steps)
        assert total1 <= limit
        if limit - total1 > 0.01 * limit and direction == "crecer":
            avisos.append(f"quedan {_human(limit - total1)} sin usar: ninguna operación más entra en el objetivo, "
                          "la memoria o el límite de cómputo")
        return BudgetPlan(target_params, ceiling, limit, direction, total0, total1, steps,
                          _memory(total0, prof), _memory(total1, prof), _speed(model, prof), _speed(sim, prof),
                          flops_base, avisos)
    return _plan_reduction(model, total0, limit, target_params, ceiling, prof, avisos, batches, flops_base,
                           identity_tol=identity_tol, min_experts=min_experts, approval_fraction=approval_fraction)


def _predict_flops(sim, op, delta: int, current: float, bytes_per_patch: float = 4.5) -> float:
    """Cómputo por byte después de ``op`` (la misma cuenta que ``hope.bench.plan.cost``) sin aplicarla."""
    if isinstance(op, AppendBackboneBlock):
        return current + 6.0 * delta / bytes_per_patch
    if isinstance(op, (AddExpert, WidenExpert)):
        z = sim.regions[op.region_idx]
        m = len(z.experts)
        total = sum(_n(e) for e in z.experts)
        old = min(z.top_k, m) * total / m
        if isinstance(op, AddExpert):
            n_e = delta - (z.d + 1)
            new = (z.d + 1) + min(z.top_k, m + 1) * (total + n_e) / (m + 1)
        else:
            new = min(z.top_k, m) * (total + delta) / m
        return current + 6.0 * (new - old) / bytes_per_patch
    return current            # zona visual: el codificador se paga por imagen, no por byte de texto


def _plan_growth(model, total0, limit, fits, flops_base, prof, avisos, offload, *, max_experts, max_flops_ratio,
                 block_flops_ratio, widen_chunks, max_hidden_factor, vision_cfg, vision_first, approval_fraction):
    from hope.bench.plan import cost
    from hope.core.hn1 import build_registry
    sim = _meta_clone(model)
    reg = build_registry(sim)                    # registro de la simulación (el real no se toca)
    state = {"total": total0, "flops": cost(sim).flops_per_byte}
    if prof is not None and not fits(total0):
        avisos.append("el modelo actual ya no entra en la memoria del perfil: no se puede crecer ahí")
        return [], sim

    def attempt(op, ratio: float = max_flops_ratio) -> bool:
        try:
            delta = op.plan(sim).param_delta
        except (NotImplementedError, ValueError):
            return False
        new = state["total"] + delta
        if delta <= 0 or new > limit or not fits(new):
            return False
        f = _predict_flops(sim, op, delta, state["flops"])
        if f > ratio * flops_base and f > state["flops"] * (1 + 1e-9):
            return False
        op.apply(sim, reg)
        state["total"], state["flops"] = new, f
        return True

    def experts():
        ops, active = [], True
        while active:
            active = False
            for r, region in enumerate(sim.regions):
                kinds = [getattr(e, "kind", "mlp") for e in region.experts]
                if len(kinds) >= max_experts:
                    continue
                period = _period(kinds[: len(sim.cfg.zone_kinds)] or kinds)
                op = AddExpert(r, hidden=sim.cfg.expert_hidden, kind=kinds[len(kinds) % period])
                if attempt(op):
                    ops.append(op)
                    active = True
        return ops

    def blocks():
        ops = []
        while attempt(op := AppendBackboneBlock(BLOCK_CYCLE[len(sim.backbone) % len(BLOCK_CYCLE)]),
                      min(block_flops_ratio, max_flops_ratio)):
            ops.append(op)
        return ops

    def widen():
        ops = []
        cap = int(max_hidden_factor * sim.cfg.expert_hidden)
        for chunk in widen_chunks:
            active = True
            while active:
                active = False
                for r, region in enumerate(sim.regions):
                    for e, ex in enumerate(region.experts):
                        h = getattr(ex, "hidden", None)
                        if h is None or h + chunk > cap:
                            continue
                        op = WidenExpert(r, e, chunk, seed=len(ops))
                        if attempt(op):
                            ops.append(op)
                            active = True
        return ops

    def vision():
        if sim.vision is not None:
            return []
        op = AddVisualZone(**(vision_cfg or _default_vision(sim)))
        return [op] if attempt(op) else []

    order = [("especialistas", experts), ("bloques", blocks), ("ensanchar", widen), ("vision", vision)]
    if vision_first:
        order = [order[3]] + order[:3]
    steps = []
    for name, fn in order:
        before, f_before = state["total"], state["flops"]
        ops = fn()
        if ops:
            delta = state["total"] - before
            steps.append(Step(name, ops, delta, before, state["total"], abs(delta) > approval_fraction * before,
                              flops_por_byte_delta=state["flops"] - f_before))
    return steps, sim


def _default_vision(model) -> dict:
    from hope.core.hn1 import preset
    if model.cfg.d == 1024:
        t = preset("hn1-500m")
        return {k: getattr(t, k) for k in ("vis_d", "vis_layers", "vis_heads", "vis_hidden", "vis_patch",
                                           "vis_image", "vis_pool")}
    return {"vis_d": model.cfg.d, "vis_layers": 2, "vis_heads": model.cfg.heads}


def _plan_reduction(model, total0, limit, target, ceiling, prof, avisos, batches, flops_base, *, identity_tol,
                    min_experts, approval_fraction) -> BudgetPlan:
    if batches is None:
        batches = default_batches()
        avisos.append("sin datos de validación: el uso y la pérdida se midieron sobre bytes aleatorios; "
                      "pasá --datos con texto real")
    nb = removable_blocks(model)
    base = measure(model, batches, contribution_blocks=nb)
    need = total0 - limit
    # 1. Bloques agregados casi identidad, desde el final (contiguos: se retiran de atrás para adelante).
    block_sizes: list[int] = []
    for j in range(len(model.backbone) - 1, len(model.backbone) - 1 - nb, -1):
        if base["contribucion"].get(j, 1.0) >= identity_tol:
            break
        block_sizes.append(_n(model.backbone[j]))
    # 2. Especialistas menos usados (uso medido), dejando al menos min_experts activos por zona.
    from hope.core.hn1 import build_registry
    reg = build_registry(model)
    cands = []
    for r, region in enumerate(model.regions):
        for e in range(region.n_experts):
            node = reg.find_by_path(f"regions.{r}.experts.{e}")
            if node is None or not node.protected:
                cands.append((base["uso"][r][e], r, e))
    cands.sort()

    def fill(freed: int) -> tuple[list[tuple[int, int]], int]:
        left = {r: int((z.expert_mask == 0).sum()) for r, z in enumerate(model.regions)}
        chosen = []
        for usage, r, e in cands:
            if freed >= need:
                break
            z = model.regions[r]
            floor = max(min_experts if min_experts is not None else z.top_k + 2, z.top_k + 1)
            if z.expert_mask[e] == 0:
                if left[r] - 1 < floor:
                    continue
                left[r] -= 1
            chosen.append((r, e))
            freed += _n(z.experts[e]) + z.d + 1
        return chosen, freed

    # Lo más cerca posible por debajo del objetivo: se prueba con k bloques (los más baratos en pérdida)
    # y el resto con especialistas; gana el que menos se pasa hacia abajo (a igualdad, más bloques).
    best = None
    for k in range(len(block_sizes), -1, -1):
        freed_k = sum(block_sizes[:k])
        if k and freed_k - block_sizes[k - 1] >= need:
            continue                                     # con un bloque menos ya alcanza
        chosen, freed_k = fill(freed_k)
        key = (freed_k < need, freed_k - need if freed_k >= need else need - freed_k)
        if best is None or key < best[0]:
            best = (key, k, chosen, freed_k)
    _, picked_blocks, experts, freed = best
    if freed < need:
        avisos.append(f"no se llega a {_human(limit)} retirando solo bloques agregados casi identidad y "
                      f"especialistas poco usados: faltan {_human(need - freed)}")
    steps: list[Step] = []
    total = total0
    saved_masks = [z.expert_mask.clone() for z in model.regions]
    try:
        if picked_blocks:
            mask_last_blocks(model, picked_blocks)
            delta = -sum(_n(b.inner) for b in list(model.backbone)[-picked_blocks:])
            bpb = measure(model, batches)["bpb"]
            steps.append(Step("retirar_bloques", [RemoveLastBlock() for _ in range(picked_blocks)], delta, total,
                              total + delta, abs(delta) > approval_fraction * total,
                              retiros=[("bloque", len(model.backbone) - 1 - i) for i in range(picked_blocks)],
                              bpb_medido=bpb))
            total += delta
        if experts:
            for r, e in experts:
                model.regions[r].expert_mask[e] = NEG_INF
            bpb = measure(model, batches)["bpb"]
            delta = -sum(_n(model.regions[r].experts[e]) + model.regions[r].d + 1 for r, e in experts)
            ops = [RemoveMaskedExpert(r, e) for r, e in sorted(experts, key=lambda t: (t[0], -t[1]))]
            steps.append(Step("retirar_especialistas", ops, delta, total, total + delta,
                              abs(delta) > approval_fraction * total,
                              retiros=[("especialista", r, e) for r, e in experts], bpb_medido=bpb))
            total += delta
    finally:                                            # deshacer: el plan no cambia el modelo
        unmask_blocks(model)
        for z, m in zip(model.regions, saved_masks):
            z.expert_mask.copy_(m)
    bpb_after = steps[-1].bpb_medido if steps else base["bpb"]
    sim = _meta_clone(model)
    _apply_removals_meta(sim, steps)
    after_speed = _speed(sim, prof)
    return BudgetPlan(target, ceiling, limit, "reducir", total0, total, steps, _memory(total0, prof),
                      _memory(total, prof), _speed(model, prof), after_speed, flops_base, avisos,
                      bpb_antes=base["bpb"], bpb_despues=bpb_after)


def _apply_removals_meta(sim, steps) -> None:
    """Estructura después de los retiros (solo para estimar la velocidad)."""
    for s in steps:
        for item in s.retiros:
            if item[0] == "bloque":
                del sim.backbone[-1]
    for s in steps:
        by_region: dict[int, list[int]] = {}
        for item in s.retiros:
            if item[0] == "especialista":
                by_region.setdefault(item[1], []).append(item[2])
        for r, es in by_region.items():
            z = sim.regions[r]
            z.experts = nn.ModuleList(x for i, x in enumerate(z.experts) if i not in es)


# ---------------------------------------------------------------------------
# Aplicar
# ---------------------------------------------------------------------------
@dataclass
class ApplyResult:
    hecho: bool
    pasos_aplicados: int
    pendiente: str | None = None            # id de solicitud del guardián
    bpb_medido: list = field(default_factory=list)
    diferencia_max_salida: float | None = None
    registros: list = field(default_factory=list)


@torch.no_grad()
def apply_plan(model, plan: BudgetPlan, reg: TopologyRegistry | None = None, *, guardian=None,
               approval: str | None = None, user_approved: bool = False, batches=None,
               loss_tol: float = 1e-3, unmask_new: bool = False) -> ApplyResult:
    """Aplica el plan paso por paso. Cada paso pasa por el guardián (``structural.consolidate``);
    sin guardián, un paso que requiere aprobación solo se aplica con ``user_approved=True``.
    Si un paso queda pendiente, se detiene ahí (lo anterior queda aplicado) y devuelve el id.

    Crecer: con ``batches`` se verifica que la salida no cambie. Reducir: se enmascara, se mide la
    pérdida y se compara con la informada (±``loss_tol`` bits por byte); recién después se elimina.
    ``unmask_new``: habilitar los especialistas nuevos al terminar (como ``hope.train.run``).
    """
    from guardian.core import Action
    from hope.core.hn1 import build_registry
    reg = reg or build_registry(model)
    res = ApplyResult(False, 0)
    ref = None
    if plan.direccion == "crecer" and batches is not None:
        was = model.training
        model.eval()
        ref = [model(x[:, :-1].to(model.head.weight.device))["logits"] for x in batches]
        model.train(was)
    for step in plan.pasos:
        total = _n(model)
        if total != step.total_antes:
            raise RuntimeError(f"el modelo cambió desde el plan ({total:,} ≠ {step.total_antes:,}): volvé a planificar")
        if guardian is not None:
            act = Action("structural.consolidate", {"change": step.change().__dict__, "current_total": total},
                         requested_by="hope.growth.budget_target",
                         justification=f"presupuesto de parámetros del usuario: {_human(plan.objetivo)}")
            d = guardian.decide(act, approval_id=approval)
            if not d.allowed:
                if d.pending_request:
                    res.pendiente = d.pending_request
                    return res
                raise PermissionError("el guardián rechazó el paso: " + "; ".join(d.reasons))
            approval = None                         # una aprobación vale para un solo paso
        elif step.requiere_aprobacion and not user_approved:
            raise PermissionError(f"el paso «{CATEGORY_NAMES[step.categoria]}» cambia más del 10 % de los "
                                  "parámetros: necesita aprobación humana")
        if step.categoria.startswith("retirar"):
            _apply_removal(model, reg, step, batches, loss_tol, res)
        else:
            for op in step.ops:
                rec = op.apply(model, reg)
                res.registros.append(rec)
        res.pasos_aplicados += 1
    if ref is not None:
        was = model.training
        model.eval()
        res.diferencia_max_salida = max(
            (model(x[:, :-1].to(model.head.weight.device))["logits"] - r).abs().max().item()
            for x, r in zip(batches, ref))
        model.train(was)
    if unmask_new and plan.direccion == "crecer":
        for z in model.regions:
            z.expert_mask.zero_()
    res.hecho = True
    return res


def _apply_removal(model, reg, step: Step, batches, loss_tol: float, res: ApplyResult) -> None:
    if batches is None:
        batches = default_batches()
    if step.categoria == "retirar_bloques":
        mask_last_blocks(model, len(step.ops))
    else:
        for item in step.retiros:
            mask_expert(model, reg, item[1], item[2])
    bpb = measure(model, batches)["bpb"]
    res.bpb_medido.append(bpb)
    if step.bpb_medido is not None and abs(bpb - step.bpb_medido) > loss_tol:
        if step.categoria == "retirar_bloques":
            unmask_blocks(model)
        else:
            for item in step.retiros:
                unmask_expert(model, reg, item[1], item[2])
        raise RuntimeError(f"la pérdida medida al enmascarar ({bpb:.4f}) no coincide con la del plan "
                           f"({step.bpb_medido:.4f}): se deshizo; volvé a planificar")
    for op in step.ops:
        res.registros.append(op.apply(model, reg))


# ---------------------------------------------------------------------------
# Checkpoints y línea de comandos
# ---------------------------------------------------------------------------
def load_model(path: str | Path):
    """Modelo de un checkpoint de ``hope.train`` (carpeta con model.pt) o de ``ligero.pt``."""
    from hope.core.hn1 import HN1Config, HN1Model
    p = Path(path)
    f = p / "model.pt" if p.is_dir() else p
    ck = torch.load(f, map_location="cpu", weights_only=False)
    if "summary" in ck:
        model = ck.get("structure") or HN1Model(HN1Config.from_dict(ck["summary"]["config"]))
    else:
        model = HN1Model(HN1Config.from_dict(ck["config"]))
    model.load_state_dict({k: (v.float() if v.is_floating_point() and v.dtype == torch.bfloat16 else v)
                           for k, v in ck["state_dict"].items()})
    return model, ck


def save_model(model, path: str | Path, source: dict | None = None, extra: dict | None = None) -> Path:
    """Guarda en el formato de ``Trainer.save`` (sin optimizador: tras un cambio estructural arranca de nuevo)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    src = source or {}
    summary = {**src.get("summary", {}), "config": model.cfg.to_dict(), "arch": "hn1", **(extra or {})}
    torch.save({"state_dict": model.state_dict(), "structure": model, "step": src.get("step", 0),
                "summary": summary}, p / "model.pt")
    return p


def _batches_from(path: str | None, n: int = 8, length: int = 513) -> list[torch.Tensor] | None:
    if not path:
        return None
    import numpy as np
    data = np.fromfile(path, dtype=np.uint8) if not Path(path).is_dir() else np.fromfile(Path(path) / "val.bin",
                                                                                        dtype=np.uint8)
    rows = [torch.from_numpy(data[i * length:(i + 1) * length].astype("int64")) for i in range(n)
            if (i + 1) * length <= len(data)]
    return [torch.stack(rows[i:i + 2]) for i in range(0, len(rows) - 1, 2)] or None


def _print_plan(plan: BudgetPlan) -> None:
    r = plan.resumen()
    print(f"\nPresupuesto: {r['direccion']} hacia {r['objetivo']} (límite efectivo {r['limite_efectivo']}"
          + (f", techo del guardián {r['techo_del_guardian']}" if r["techo_del_guardian"] else "") + ")")
    print(f"Parámetros: {_human(plan.total_antes)} → {_human(plan.total_despues)}")
    for i, s in enumerate(r["pasos"], 1):
        extra = f"  [{s.get('parametros_por_flop_byte')} parámetros por FLOP/byte]" if "parametros_por_flop_byte" in s else ""
        extra += f"  [bpb medido {s['bpb_medido_tras_el_paso']}]" if "bpb_medido_tras_el_paso" in s else ""
        print(f"  {i}. {s['paso']}: {s['operaciones']} operaciones, {s['parametros']}"
              f"{'  (pide aprobación)' if s['requiere_aprobacion'] else ''}{extra}")
    for k in ("antes", "despues"):
        m = r["memoria_entrenamiento"][k]
        if m:
            print(f"Memoria ({k}): {m['memoria_GB']} GB en GPU, {m['memoria_offload_GB']} GB con el optimizador "
                  f"en la RAM → {m['modo']}")
    v = r["velocidad"]
    print(f"Velocidad de entrenamiento: {v['antes']['entrenamiento_gb_dia']} → {v['despues']['entrenamiento_gb_dia']} "
          f"GB de texto por día (×{v['relacion_entrenamiento']}); generación ≈ "
          f"{v['antes']['generacion_bytes_s_aprox']} → {v['despues']['generacion_bytes_s_aprox']} bytes/s")
    if "perdida_bpb" in r:
        p = r["perdida_bpb"]
        print(f"Pérdida medida (bits por byte): {p['antes']} → {p['despues_medida']} ({p['diferencia']:+})")
    for a in plan.avisos:
        print(f"aviso: {a}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--objetivo", required=True, help="p. ej. 1B, 700M, 300M")
    ap.add_argument("--perfil", default=None, help="perfil de hardware (gtx1060, a770, ...)")
    ap.add_argument("--sin-offload", action="store_true", help="exigir que entre sin el optimizador en la RAM")
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    ap.add_argument("--sin-guardian", action="store_true", help="solo para ver un plan; nunca para aplicar")
    ap.add_argument("--datos", default=None, help="val.bin (o su carpeta) para medir uso y pérdida al reducir")
    ap.add_argument("--max-expertos", type=int, default=64)
    ap.add_argument("--max-computo", type=float, default=1.6, help="cómputo por byte máximo (× el actual)")
    ap.add_argument("--vision-primero", action="store_true")
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--aprobacion", default=None, help="id aprobado con python -m guardian.cli approve")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from guardian.cli import load as load_guardian, parse_count
    model, ck = load_model(args.checkpoint)
    guardian = None if args.sin_guardian else load_guardian(args.guardian)
    batches = _batches_from(args.datos)
    kw = dict(guardian=guardian, batches=batches, offload=not args.sin_offload and "auto",
              max_experts=args.max_expertos, max_flops_ratio=args.max_computo, vision_first=args.vision_primero)
    plan = plan_to_budget(model, parse_count(args.objetivo), args.perfil, **kw)
    if args.json:
        print(json.dumps(plan.resumen(), indent=2, ensure_ascii=False, default=str))
    else:
        _print_plan(plan)
    if not args.aplicar:
        print("\n(solo el plan; para aplicarlo: --aplicar --salida <carpeta>)")
        return 0
    if guardian is None:
        print("para aplicar hace falta el guardián (sin --sin-guardian)", file=sys.stderr)
        return 2
    if not args.salida:
        print("falta --salida", file=sys.stderr)
        return 2
    res = apply_plan(model, plan, guardian=guardian, approval=args.aprobacion, batches=batches)
    if res.pasos_aplicados:
        save_model(model, args.salida, ck, {"presupuesto": args.objetivo})
    if res.pendiente:
        print(f"\nEl paso {res.pasos_aplicados + 1} necesita tu aprobación: {res.pendiente}\n"
              f"  python -m guardian.cli approve {res.pendiente} --by <tu-nombre>\n"
              f"y después repetí el comando con --checkpoint {args.salida if res.pasos_aplicados else args.checkpoint} "
              f"--aprobacion {res.pendiente}")
        return 3
    print(f"\nListo: {_human(_n(model))} parámetros, guardado en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
