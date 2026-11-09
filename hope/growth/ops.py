"""Operaciones estructurales que preservan la función (ARQUITECTURA_HN1 §6.2).

Cada operación tiene dos pasos:

* ``plan()`` — calcula el cambio (parámetros agregados o retirados, nivel de
  plasticidad) **sin tocar el modelo**, para que el presupuesto y el guardián
  puedan evaluarlo antes.
* ``apply()`` — modifica el modelo y el registro, y devuelve un ``ChangeRecord``.

Las operaciones de crecimiento están diseñadas para que, en el instante de
aplicarlas, el modelo calcule **exactamente lo mismo** (``preserving=True``).
Así no se destruye lo aprendido (ASG §2.3). La retirada se hace en dos pasos:
primero enmascarar (cambia la función y debe evaluarse) y después eliminar
(exacto respecto del estado enmascarado).

Por ahora operan sobre ``hope.core.toy.ToyModularModel``. El núcleo real de
HN-1 expondrá la misma interfaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from hope.core.toy import NEG_INF, Connection, Expert, Region, ToyModularModel
from hope.epistemic import LifecycleStatus
from hope.growth.budget import ProposedChange
from hope.registry.topology import NodeKind, TopologyRegistry


@dataclass
class ChangeRecord:
    op: str
    targets: list[str]                 # IDs estables afectados
    param_delta: int
    preserving: bool
    created: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def _no_adapters(model: nn.Module) -> None:
    from hope.plasticity.adapters import has_adapters
    if has_adapters(model):
        raise RuntimeError("hay adaptadores enganchados: consolidarlos (merge) o descartarlos antes de crecer")


def _n(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _region_node(reg: TopologyRegistry, region_idx: int):
    node = reg.find_by_path(f"regions.{region_idx}")
    if node is None:
        raise KeyError(f"región {region_idx} no registrada")
    return node


def _refresh_expert_activation(reg: TopologyRegistry, region: Region, region_idx: int) -> None:
    active = int((region.expert_mask == 0).sum())
    frac = min(region.top_k, active) / max(active, 1)
    for e in range(region.n_experts):
        node = reg.find_by_path(f"regions.{region_idx}.experts.{e}")
        if node is not None:
            node.activation = frac if region.expert_mask[e] == 0 else 0.0


# ---------------------------------------------------------------------------
class AddExpert:
    """Agrega un especialista a una región sin cambiar la función.

    El nuevo especialista tiene la proyección de salida en cero y su logit de
    router fijado en ``NEG_INF`` mediante la máscara, así que nunca se elige
    hasta que el entrenamiento lo habilita con ``unmask_expert``. Habilitarlo
    sí cambia la función y forma parte del experimento.
    """

    def __init__(self, region_idx: int, hidden: int | None = None, kind: str | None = None):
        self.region_idx = region_idx
        self.hidden = hidden
        self.kind = kind            # tipo de especialista (zonas heterogéneas); None = como el primero

    def _build(self, region):
        h = self.hidden or region.experts[0].fc2.in_features
        if hasattr(region, "kinds"):                     # zona heterogénea de HN-1
            from hope.core.zones import make_specialist
            return make_specialist(self.kind or region.experts[0].kind, region.d, h)
        return Expert(region.d, h)

    def plan(self, model: ToyModularModel) -> ProposedChange:
        region = model.regions[self.region_idx]
        with torch.device("meta"):
            n = _n(self._build(region))
        kind = self.kind or getattr(region.experts[0], "kind", "mlp")
        return ProposedChange(f"agregar especialista {kind} a región {self.region_idx}",
                              param_delta=n + region.d + 1, active_delta=0.0, plasticity_level=4)

    @torch.no_grad()
    def apply(self, model: ToyModularModel, reg: TopologyRegistry) -> ChangeRecord:
        _no_adapters(model)
        region = model.regions[self.region_idx]
        planned = self.plan(model).param_delta
        dev = region.router.weight.device
        expert = self._build(region).to(device=dev, dtype=region.router.weight.dtype)
        if hasattr(expert, "zero_output_"):
            expert.zero_output_()
        else:
            expert.fc2.weight.zero_()
            expert.fc2.bias.zero_()
        region.experts.append(expert)
        h = expert.fc2.in_features

        old = region.router
        router = nn.Linear(region.d, old.out_features + 1).to(dev)
        router.weight.zero_()
        router.bias.zero_()
        router.weight[:-1] = old.weight
        router.bias[:-1] = old.bias
        region.router = router
        region.expert_mask = torch.cat([region.expert_mask, torch.tensor([NEG_INF], device=dev)])

        rnode = _region_node(reg, self.region_idx)
        new_idx = region.n_experts - 1
        kind = getattr(expert, "kind", "mlp")
        node = reg.add(NodeKind.EXPERT, f"regions.{self.region_idx}.experts.{new_idx}",
                       parent=rnode.id, hint=f"{kind}-{self.region_idx}", activation=0.0, meta={"kind": kind})
        node.status = LifecycleStatus.EXPERIMENTAL
        _refresh_expert_activation(reg, region, self.region_idx)
        return ChangeRecord("add_expert", [rnode.id], planned, preserving=True,
                            created=[node.id], detail={"expert_index": new_idx, "hidden": h, "kind": kind})


# ---------------------------------------------------------------------------
class WidenExpert:
    """Ensancha la capa oculta de un especialista preservando la función.

    Dos modos:

    * ``"zero_out"`` (por defecto): las unidades nuevas tienen pesos de entrada
      aleatorios y pesos de salida en **cero**. La función se preserva exactamente
      y las unidades son distintas entre sí, así que aprenden cosas nuevas.
    * ``"net2net"`` (Net2WiderNet, Chen et al. 2015): duplica unidades existentes
      y divide su peso de salida entre las copias. Preserva la función, pero las
      copias reciben gradientes idénticos y **nunca se diferencian**: no agrega
      capacidad efectiva salvo que ``noise`` > 0 rompa la simetría (y entonces
      deja de preservar la función exactamente). Se mantiene para comparar.
    """

    def __init__(self, region_idx: int, expert_idx: int, extra: int, mode: str = "zero_out",
                 noise: float = 0.0, seed: int = 0):
        if mode not in ("zero_out", "net2net"):
            raise ValueError(f"modo desconocido: {mode}")
        self.region_idx, self.expert_idx, self.extra = region_idx, expert_idx, extra
        self.mode, self.noise, self.seed = mode, noise, seed

    def plan(self, model: ToyModularModel) -> ProposedChange:
        expert = model.regions[self.region_idx].experts[self.expert_idx]
        if hasattr(expert, "widen"):                     # especialista de zona: medir sobre una copia
            import copy
            probe = copy.deepcopy(expert)
            probe.widen(self.extra, self.seed)
            delta = _n(probe) - _n(expert)
        else:
            d = model.regions[self.region_idx].d
            delta = self.extra * (2 * d + 1)
        return ProposedChange(f"ensanchar especialista {self.region_idx}.{self.expert_idx}",
                              param_delta=delta, plasticity_level=4)

    @torch.no_grad()
    def apply(self, model: ToyModularModel, reg: TopologyRegistry) -> ChangeRecord:
        _no_adapters(model)
        expert = model.regions[self.region_idx].experts[self.expert_idx]
        if hasattr(expert, "widen"):
            planned = self.plan(model).param_delta
            h = expert.hidden
            expert.widen(self.extra, self.seed)
            node = reg.find_by_path(f"regions.{self.region_idx}.experts.{self.expert_idx}")
            return ChangeRecord("widen_expert", [node.id] if node else [], planned, preserving=True,
                                detail={"from": h, "to": expert.hidden, "kind": expert.kind})
        fc1, fc2 = expert.fc1, expert.fc2
        h = fc1.out_features
        dev = fc1.weight.device
        g = torch.Generator().manual_seed(self.seed)
        new1 = nn.Linear(fc1.in_features, h + self.extra).to(dev)
        new2 = nn.Linear(h + self.extra, fc2.out_features).to(dev)
        new2.bias.copy_(fc2.bias)

        if self.mode == "zero_out":
            bound = 1.0 / fc1.in_features ** 0.5          # misma escala que nn.Linear
            w_in = (torch.rand(self.extra, fc1.in_features, generator=g) * 2 - 1) * bound
            b_in = (torch.rand(self.extra, generator=g) * 2 - 1) * bound
            new1.weight.copy_(torch.cat([fc1.weight, w_in.to(dev)]))
            new1.bias.copy_(torch.cat([fc1.bias, b_in.to(dev)]))
            new2.weight.copy_(torch.cat([fc2.weight, torch.zeros(fc2.out_features, self.extra, device=dev)], dim=1))
            preserving = True
        else:
            src = torch.randint(0, h, (self.extra,), generator=g).to(dev)
            new1.weight.copy_(torch.cat([fc1.weight, fc1.weight[src]]))
            new1.bias.copy_(torch.cat([fc1.bias, fc1.bias[src]]))
            counts = torch.ones(h, device=dev)            # copias de cada unidad original
            counts.index_add_(0, src, torch.ones(self.extra, device=dev))
            scaled = fc2.weight / counts
            new2.weight.copy_(torch.cat([scaled, scaled[:, src]], dim=1))
            if self.noise > 0:
                new1.weight[h:] += self.noise * torch.randn(self.extra, fc1.in_features, generator=g).to(dev)
            preserving = self.noise == 0

        expert.fc1, expert.fc2 = new1, new2
        node = reg.find_by_path(f"regions.{self.region_idx}.experts.{self.expert_idx}")
        return ChangeRecord("widen_expert", [node.id] if node else [], self.plan(model).param_delta,
                            preserving=preserving,
                            detail={"from": h, "to": h + self.extra, "mode": self.mode})


# ---------------------------------------------------------------------------
class AddConnection:
    """Crea una conexión dirigida región ``src`` → región ``dst`` con compuerta en 0."""

    def __init__(self, src: int, dst: int):
        if src >= dst:
            raise ValueError("en el modelo de juguete las conexiones van hacia adelante (src < dst)")
        self.src, self.dst = src, dst

    @property
    def key(self) -> str:
        return f"{self.src}_{self.dst}"

    def plan(self, model: ToyModularModel) -> ProposedChange:
        d = model.d
        return ProposedChange(f"conexión {self.src}→{self.dst}", param_delta=d * d + 1,
                              active_delta=d * d + 1, connection_delta=1, plasticity_level=4)

    @torch.no_grad()
    def apply(self, model: ToyModularModel, reg: TopologyRegistry) -> ChangeRecord:
        if self.key in model.connections:
            raise ValueError(f"la conexión {self.key} ya existe")
        dev = model.head.weight.device
        model.connections[self.key] = Connection(model.d, gate_init=0.0).to(dev)
        root = reg.find_by_path("")
        src_node = _region_node(reg, self.src)
        dst_node = _region_node(reg, self.dst)
        node = reg.add(NodeKind.CONNECTION, f"connections.{self.key}", parent=root.id,
                       hint=f"conn-{self.src}-{self.dst}",
                       meta={"from": src_node.id, "to": dst_node.id, "directed": True})
        node.status = LifecycleStatus.EXPERIMENTAL
        return ChangeRecord("add_connection", [src_node.id, dst_node.id], self.plan(model).param_delta,
                            preserving=True, created=[node.id])


# ---------------------------------------------------------------------------
def mask_expert(model: ToyModularModel, reg: TopologyRegistry, region_idx: int,
                expert_idx: int) -> ChangeRecord:
    """Primer paso de retiro: deja de elegirse. **Cambia la función**; hay que evaluarlo."""
    region = model.regions[region_idx]
    active = int((region.expert_mask == 0).sum())
    if region.expert_mask[expert_idx] == 0 and active <= region.top_k:
        raise ValueError("no se puede dejar la región con menos especialistas activos que top_k")
    node = reg.find_by_path(f"regions.{region_idx}.experts.{expert_idx}")
    if node is not None and node.protected:
        raise PermissionError(f"{node.id} está protegido")
    region.expert_mask[expert_idx] = NEG_INF
    _refresh_expert_activation(reg, region, region_idx)
    return ChangeRecord("mask_expert", [node.id] if node else [], 0, preserving=False)


def unmask_expert(model: ToyModularModel, reg: TopologyRegistry, region_idx: int,
                  expert_idx: int) -> ChangeRecord:
    region = model.regions[region_idx]
    region.expert_mask[expert_idx] = 0.0
    _refresh_expert_activation(reg, region, region_idx)
    node = reg.find_by_path(f"regions.{region_idx}.experts.{expert_idx}")
    return ChangeRecord("unmask_expert", [node.id] if node else [], 0, preserving=False)


class RemoveMaskedExpert:
    """Segundo paso de retiro: elimina un especialista ya enmascarado (exacto)."""

    def __init__(self, region_idx: int, expert_idx: int):
        self.region_idx, self.expert_idx = region_idx, expert_idx

    def plan(self, model: ToyModularModel) -> ProposedChange:
        region = model.regions[self.region_idx]
        delta = -(_n(region.experts[self.expert_idx]) + region.d + 1)
        return ProposedChange(f"eliminar especialista {self.region_idx}.{self.expert_idx}",
                              param_delta=delta, plasticity_level=4)

    @torch.no_grad()
    def apply(self, model: ToyModularModel, reg: TopologyRegistry) -> ChangeRecord:
        r, e = self.region_idx, self.expert_idx
        region = model.regions[r]
        if region.expert_mask[e] == 0:
            raise ValueError("primero hay que enmascarar y evaluar (mask_expert)")
        delta = self.plan(model).param_delta
        keep = [i for i in range(region.n_experts) if i != e]

        old = region.router
        router = nn.Linear(region.d, len(keep)).to(old.weight.device)
        router.weight.copy_(old.weight[keep])
        router.bias.copy_(old.bias[keep])
        region.router = router
        region.experts = nn.ModuleList(region.experts[i] for i in keep)
        region.expert_mask = region.expert_mask[keep].clone()

        # Mantener IDs estables: el nodo eliminado sale, los demás cambian de ruta.
        removed = reg.find_by_path(f"regions.{r}.experts.{e}")
        if removed is not None:
            reg.remove(removed.id)
        for new_i, old_i in enumerate(keep):
            node = reg.find_by_path(f"regions.{r}.experts.{old_i}")
            if node is not None and old_i != new_i:
                node.module_path = f"__moving__{r}.{new_i}"
        for node in reg.nodes(NodeKind.EXPERT):
            if node.module_path and node.module_path.startswith(f"__moving__{r}."):
                node.module_path = f"regions.{r}.experts.{node.module_path.split('.')[-1]}"
        _refresh_expert_activation(reg, region, r)
        return ChangeRecord("remove_expert", [removed.id] if removed else [], delta,
                            preserving=True, removed=[removed.id] if removed else [])


# ---------------------------------------------------------------------------
def rebuild_optimizer(old: torch.optim.Optimizer, model: nn.Module) -> torch.optim.Optimizer:
    """Reconstruye el optimizador después de un cambio estructural (ASG §19).

    Conserva el estado (p. ej. momentos de Adam) de los parámetros que siguen
    siendo el mismo objeto. Los parámetros nuevos o reemplazados empiezan con
    estado vacío, que es lo correcto para pesos recién creados.
    """
    import inspect
    accepted = inspect.signature(type(old).__init__).parameters
    kwargs = {k: v for k, v in old.defaults.items() if k in accepted}
    new = type(old)(model.parameters(), **kwargs)
    for p in model.parameters():
        if p in old.state:
            new.state[p] = old.state[p]
    return new
