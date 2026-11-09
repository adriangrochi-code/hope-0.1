"""Diagnóstico de uso y Lottery Ticket Theory (ASG §9; ARQUITECTURA_HN1 §6.3).

Objetivo (decisión del usuario): usar LTH para **saber qué partes de Hope están
menos usadas y mejorar su uso**, tanto en las neuronas como en los bucles
internos. No para achicar el modelo: el presupuesto de parámetros se mantiene
y la capacidad ociosa se reasigna.

Herramientas:

* ``unit_importance`` — importancia de cada unidad oculta (neurona) de los
  especialistas y de los MLP: actividad media y |activación × gradiente|
  (aproximación de Taylor de cuánto subiría la pérdida si se apagara). Se mide
  **por familia de tareas**, para no confundir «poco usada en promedio» con
  «poco usada»: una unidad puede ser vital para una tarea rara (ASG §5 y §9).
* ``expert_usage`` — fracción de posiciones que el router manda a cada especialista.
* ``iteration_usage`` — peso medio de cada iteración del núcleo recurrente y
  cuánto mejora la pérdida cada iteración adicional.
* ``find_ticket`` — poda iterativa por magnitud con rebobinado (Frankle & Carbin
  2019; Frankle et al. 2020), **sobre una copia**. Devuelve la máscara de la
  subred que alcanza para una tarea.
* ``ticket_overlap`` — compara tickets de tareas distintas: pesos compartidos,
  específicos y los que no usa ninguna.
* ``ReinitUnits`` — operación estructural que reinicializa unidades ociosas no
  protegidas (entrada nueva aleatoria, salida en cero) para que vuelvan a
  aprender. Mantiene la cantidad de parámetros. Cambia la función, así que se
  evalúa contra un control como cualquier experimento (ASG §7).

Advertencia (ASG §9): poca magnitud o poca actividad **no prueban** que algo sea
inútil. Por eso la decisión final siempre pasa por un experimento con control.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn

from hope.growth.budget import ProposedChange
from hope.growth.ops import ChangeRecord
from hope.registry.topology import TopologyRegistry

# Capas cuya entrada son las unidades ocultas de una población.
HIDDEN_OUTPUT_NAMES = ("fc2", "w_out")


def hidden_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Capas de salida de cada población (su entrada = las neuronas a medir)."""
    return {name: mod for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and name.split(".")[-1] in HIDDEN_OUTPUT_NAMES}


@dataclass
class UnitStats:
    activity: torch.Tensor        # |a| medio por unidad
    taylor: torch.Tensor          # |a · dL/da| medio por unidad
    samples: int = 0


def unit_importance(model: nn.Module, batches: Iterable, loss_fn: Callable,
                    layers: dict[str, nn.Linear] | None = None) -> dict[str, UnitStats]:
    """Importancia por unidad oculta, promediada sobre ``batches``.

    ``loss_fn(model, batch) -> pérdida escalar``. El modelo no se modifica.
    """
    layers = layers or hidden_layers(model)
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for name, lin in layers.items():
        def pre(mod, inp, name=name):
            a = inp[0]
            if a.requires_grad:
                a.retain_grad()
            captured[name] = a
        hooks.append(lin.register_forward_pre_hook(pre))
    stats = {name: UnitStats(torch.zeros(lin.in_features), torch.zeros(lin.in_features))
             for name, lin in layers.items()}
    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            captured.clear()
            model.zero_grad(set_to_none=True)
            loss_fn(model, batch).backward()
            for name, a in captured.items():
                flat = a.detach().reshape(-1, a.shape[-1]).float()
                if flat.shape[0] == 0:
                    continue
                st = stats[name]
                st.activity += flat.abs().mean(0).cpu()
                if a.grad is not None:
                    st.taylor += (flat * a.grad.reshape(-1, a.shape[-1]).float()).abs().mean(0).cpu()
                st.samples += 1
    finally:
        for h in hooks:
            h.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    for st in stats.values():
        if st.samples:
            st.activity /= st.samples
            st.taylor /= st.samples
    return stats


def idle_units(stats_by_task: dict[str, dict[str, UnitStats]], quantile: float = 0.1,
               protect_quantile: float = 0.5) -> dict[str, torch.Tensor]:
    """Unidades ociosas: en el ``quantile`` inferior de importancia en **todas** las tareas.

    Una unidad que está por encima de ``protect_quantile`` en **alguna** tarea
    queda protegida aunque su promedio general sea bajo (tareas raras).
    """
    tasks = list(stats_by_task)
    layers = stats_by_task[tasks[0]].keys()
    out = {}
    for layer in layers:
        scores = torch.stack([stats_by_task[t][layer].taylor for t in tasks])     # [tareas, unidades]
        low = torch.ones(scores.shape[1], dtype=torch.bool)
        protected = torch.zeros_like(low)
        for row in scores:
            if row.numel() == 0 or row.max() == 0:
                continue
            low &= row <= torch.quantile(row, quantile)
            protected |= row >= torch.quantile(row, protect_quantile)
        out[layer] = low & ~protected
    return out


def expert_usage(routes: Iterable[torch.Tensor], n_experts: int) -> torch.Tensor:
    """Fracción de elecciones que recibe cada especialista (``routes``: índices top-k)."""
    counts = torch.zeros(n_experts)
    for r in routes:
        counts += torch.bincount(r.reshape(-1).cpu(), minlength=n_experts)[:n_experts].float()
    return counts / counts.sum().clamp_min(1)


@torch.no_grad()
def iteration_usage(model, batches: list, loss_with_iters: Callable) -> dict:
    """Peso medio de cada iteración del núcleo y pérdida al limitar las iteraciones.

    ``loss_with_iters(model, batch, n) -> (pérdida, salida del forward)``.
    Una iteración con poco peso **y** poca mejora de la pérdida es capacidad de
    cómputo ociosa: se puede reducir ``max_iters`` o reentrenar la parada.
    """
    model.eval()
    n_max = model.cfg.max_iters
    dist = torch.zeros(n_max)
    losses = torch.zeros(n_max)
    for batch in batches:
        for n in range(1, n_max + 1):
            loss, out = loss_with_iters(model, batch, n)
            losses[n - 1] += float(loss)
            if n == n_max:
                hd = out["halt_dist"].float().cpu()
                dist[:hd.numel()] += hd
    dist /= max(len(batches), 1)
    losses /= max(len(batches), 1)
    gain = torch.cat([torch.zeros(1), losses[:-1] - losses[1:]])
    return {"weight_per_iteration": dist.tolist(), "loss_at_iterations": losses.tolist(),
            "gain_of_each_extra_iteration": gain.tolist()}


# ---------------------------------------------------------------------------
# Lottery tickets
# ---------------------------------------------------------------------------
def prunable(model: nn.Module, prefixes: tuple[str, ...]) -> dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters()
            if p.dim() == 2 and any(n.startswith(pf) for pf in prefixes)}


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, m in masks.items():
            params[n].mul_(m)


def find_ticket(model: nn.Module, train_fn: Callable, prefixes: tuple[str, ...], *,
                rounds: int = 3, prune_per_round: float = 0.3,
                rewind_state: dict | None = None) -> dict[str, torch.Tensor]:
    """Poda iterativa por magnitud con rebobinado, sobre una **copia** del modelo.

    ``train_fn(model, masks)`` entrena la copia y debe reaplicar las máscaras
    después de cada paso (``apply_masks``). ``rewind_state`` es el estado al que
    se vuelve entre rondas (inicial o de un paso temprano); por defecto, el actual.
    """
    work = copy.deepcopy(model)
    start = copy.deepcopy(rewind_state or model.state_dict())
    params = prunable(work, prefixes)
    masks = {n: torch.ones_like(p, dtype=torch.float32) for n, p in params.items()}
    for _ in range(rounds):
        train_fn(work, masks)
        alive = torch.cat([params[n].detach().abs()[masks[n] > 0].flatten() for n in params])
        k = int(alive.numel() * prune_per_round)
        if k == 0:
            break
        thresh = alive.kthvalue(k).values
        for n, p in params.items():
            masks[n] = masks[n] * (p.detach().abs() > thresh).float()
        work.load_state_dict(start)
        apply_masks(work, masks)
    return masks


def ticket_overlap(tickets: dict[str, dict[str, torch.Tensor]]) -> dict:
    """Pesos compartidos por todas las tareas, específicos de una y sin uso en ninguna."""
    tasks = list(tickets)
    names = tickets[tasks[0]].keys()
    total = sum(tickets[tasks[0]][n].numel() for n in names)
    stack = {n: torch.stack([tickets[t][n] for t in tasks]) for n in names}
    used_by = {n: s.sum(0) for n, s in stack.items()}
    unused = sum(int((u == 0).sum()) for u in used_by.values())
    shared = sum(int((u == len(tasks)).sum()) for u in used_by.values())
    specific = {t: sum(int(((stack[n][i] > 0) & (used_by[n] == 1)).sum()) for n in names)
                for i, t in enumerate(tasks)}
    return {"total": total, "unused_fraction": unused / total, "shared_fraction": shared / total,
            "specific_fraction": {t: v / total for t, v in specific.items()}}


# ---------------------------------------------------------------------------
@dataclass
class ReinitUnits:
    """Reinicializa unidades ociosas de una población para que aprendan algo útil.

    ``layer``: nombre de la capa de salida (``...fc2`` o ``...w_out``); ``units``:
    índices de unidades. Entrada nueva aleatoria y salida en cero: la unidad
    deja de aportar lo poco que aportaba y empieza de nuevo. Parámetros: sin cambio.
    """

    layer: str
    units: list[int]
    seed: int = 0
    protected_ok: bool = field(default=False)

    def plan(self, model) -> ProposedChange:
        return ProposedChange(f"reinicializar {len(self.units)} unidades de {self.layer}",
                              param_delta=0, plasticity_level=3)

    @torch.no_grad()
    def apply(self, model: nn.Module, reg: TopologyRegistry) -> ChangeRecord:
        out_layer = model.get_submodule(self.layer)
        parent_path = self.layer.rsplit(".", 1)[0]
        node = reg.find_by_path(parent_path)
        if node is not None and node.protected and not self.protected_ok:
            raise PermissionError(f"{node.id} está protegido")
        parent = model.get_submodule(parent_path)
        in_layer = parent.fc1 if hasattr(parent, "fc1") else parent.w_in
        idx = torch.tensor(self.units, dtype=torch.long, device=out_layer.weight.device)
        g = torch.Generator().manual_seed(self.seed)
        bound = 1.0 / in_layer.in_features ** 0.5
        rows = idx if hasattr(parent, "fc1") else torch.cat([idx, idx + out_layer.in_features])  # SwiGLU: compuerta y valor
        new_w = (torch.rand(len(rows), in_layer.in_features, generator=g) * 2 - 1) * bound
        in_layer.weight[rows] = new_w.to(in_layer.weight)
        if in_layer.bias is not None:
            in_layer.bias[rows] = 0.0
        out_layer.weight[:, idx] = 0.0
        return ChangeRecord("reinit_units", [node.id] if node else [], 0, preserving=False,
                            detail={"layer": self.layer, "units": len(self.units)})
