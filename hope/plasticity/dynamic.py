"""Hiperplasticidad: cambiar no solo pesos, sino conexiones, estructura y tipo de red
(decisión del usuario; HNHS §9 niveles 4–5; AANS §3.4 NSE, §6).

Tres mecanismos que se combinan durante el entrenamiento:

1. **Conectividad dinámica dentro de las capas** (``DynamicSparsity``; RigL, Evci et
   al. 2020): cada capa elegida mantiene una fracción de conexiones activas; cada
   cierto tiempo se podan las más débiles y **crecen** conexiones nuevas donde el
   gradiente indica que serían útiles. La cantidad de conexiones activas se
   mantiene: el presupuesto no crece, se reorganiza.
2. **Conectoma entre bloques** (``AddBackboneConnection`` / ``prune_connections``):
   conexiones dirigidas ``i → j`` entre bloques del backbone que nacen con
   compuerta en 0 (no cambian la función) y se eliminan si su compuerta queda
   en ~0 (no aportan).
3. **Cambio de tipo de estructura** (``MorphBlock``): un bloque pasa de un tipo de
   red a otro (p. ej. SSM → atención con ventana) **gradualmente**. El bloque nuevo
   corre en paralelo con peso 0, el peso se traslada de a poco y al final el
   viejo se retira. En el instante de empezar la función no cambia.

``HyperplasticityController`` los coordina como gancho del entrenamiento continuo
(``Trainer.hooks``). Toda modificación estructural sigue pasando por el
guardián y por el ciclo de evaluación con control de ASG antes de consolidarse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from hope.core.layers import Block
from hope.core.toy import Connection
from hope.epistemic import LifecycleStatus
from hope.growth.ops import ChangeRecord
from hope.registry.topology import NodeKind, TopologyRegistry


# --- 1. conectividad dinámica dentro de las capas ----------------------------------------
class DynamicSparsity:
    """Poda y recrecimiento de conexiones con densidad constante (RigL)."""

    def __init__(self, model: nn.Module, prefixes=("regions",), density: float = 0.5,
                 update_fraction: float = 0.2, seed: int = 0):
        self.density, self.update_fraction = density, update_fraction
        self.params: dict[str, nn.Parameter] = {
            n: p for n, p in model.named_parameters()
            if p.dim() == 2 and n.endswith("weight") and n.startswith(tuple(prefixes))}
        self.masks: dict[str, torch.Tensor] = {}
        self.history: list[dict] = []
        with torch.no_grad():
            for n, p in self.params.items():
                k = max(1, int(p.numel() * density))
                thresh = p.abs().flatten().kthvalue(p.numel() - k + 1).values
                self.masks[n] = (p.abs() >= thresh).to(p.dtype)
                p.mul_(self.masks[n])

    @torch.no_grad()
    def apply(self) -> None:
        """Mantener en cero las conexiones inactivas (después de cada paso del optimizador)."""
        for n, p in self.params.items():
            p.mul_(self.masks[n])

    @torch.no_grad()
    def update(self) -> dict:
        """Podar las más débiles y hacer crecer las que el gradiente señala; densidad constante."""
        changed = 0
        for n, p in self.params.items():
            m = self.masks[n]
            active = int(m.sum())
            k = int(active * self.update_fraction)
            if k == 0 or p.grad is None:
                continue
            score_drop = torch.where(m > 0, p.abs().float(), torch.full_like(p, float("inf"), dtype=torch.float))
            drop = score_drop.flatten().topk(k, largest=False).indices
            score_grow = torch.where(m > 0, torch.full_like(p, -1.0, dtype=torch.float), p.grad.abs().float())
            grow = score_grow.flatten().topk(k).indices
            flat = m.view(-1)
            flat[drop] = 0
            flat[grow] = 1
            p.view(-1)[grow] = 0.0                      # las conexiones nuevas nacen en cero
            changed += k
        self.apply()
        rec = {"connections_rewired": changed,
               "density": sum(float(m.sum()) for m in self.masks.values()) /
                          max(sum(m.numel() for m in self.masks.values()), 1)}
        self.history.append(rec)
        return rec


# --- 2. conectoma entre bloques -------------------------------------------------------
class AddBackboneConnection:
    def __init__(self, src: int, dst: int):
        if src >= dst:
            raise ValueError("las conexiones del backbone van hacia adelante (src < dst)")
        self.src, self.dst = src, dst

    def apply(self, model, reg: TopologyRegistry) -> ChangeRecord:
        key = f"{self.src}_{self.dst}"
        if key in model.connections:
            raise ValueError(f"la conexión {key} ya existe")
        ref = model.backbone[0].norm1.weight
        model.connections[key] = Connection(model.cfg.d, gate_init=0.0).to(device=ref.device, dtype=ref.dtype)
        root = reg.find_by_path("")
        node = reg.add(NodeKind.CONNECTION, f"connections.{key}", parent=root.id if root else None,
                       hint=f"conn-{self.src}-{self.dst}", meta={"from": f"backbone.{self.src}",
                                                                 "to": f"backbone.{self.dst}", "directed": True})
        node.status = LifecycleStatus.EXPERIMENTAL
        d = model.cfg.d
        return ChangeRecord("add_backbone_connection", [node.id], d * d + 1, preserving=True, created=[node.id])


def prune_connections(model, reg: TopologyRegistry, threshold: float = 1e-3) -> list[str]:
    """Elimina las conexiones entre bloques cuya compuerta quedó prácticamente en cero."""
    removed = []
    for key in list(model.connections):
        if abs(float(model.connections[key].gate.detach())) < threshold:
            del model.connections[key]
            node = reg.find_by_path(f"connections.{key}")
            if node is not None:
                reg.remove(node.id)
            removed.append(key)
    return removed


# --- 3. cambio gradual de tipo de bloque ---------------------------------------------------
class MorphingBlock(nn.Module):
    """``x + α·Δ_viejo + (1 − α)·Δ_nuevo``; con α = 1 es exactamente el bloque viejo."""

    def __init__(self, old: Block, new: Block):
        super().__init__()
        self.old, self.new = old, new
        self.kind = f"{old.kind}->{new.kind}"
        self.register_buffer("alpha", torch.tensor(1.0))

    @property
    def norm1(self):
        return self.old.norm1

    def forward(self, x, cache=None):
        a = self.alpha.to(x.dtype)
        return x + a * (self.old(x) - x) + (1 - a) * (self.new(x) - x)


class MorphBlock:
    def __init__(self, index: int, new_kind: str):
        self.index, self.new_kind = index, new_kind

    def apply(self, model, reg: TopologyRegistry) -> ChangeRecord:
        old = model.backbone[self.index]
        if isinstance(old, MorphingBlock):
            raise ValueError("ese bloque ya está en transición")
        cfg = model.cfg
        ref = old.norm1.weight
        new = Block(cfg.d, self.new_kind, cfg.heads, cfg.mlp_hidden, kv_heads=cfg.kv_heads,
                    window=cfg.window, sinks=cfg.sinks).to(device=ref.device, dtype=ref.dtype)
        model.backbone[self.index] = MorphingBlock(old, new)
        node = reg.find_by_path(f"backbone.{self.index}")
        if node is not None:
            node.meta["morph"] = {"from": old.kind, "to": self.new_kind}
            node.status = LifecycleStatus.EXPERIMENTAL
        added = sum(p.numel() for p in new.parameters())
        return ChangeRecord("morph_block", [node.id] if node else [], added, preserving=True,
                            detail={"from": old.kind, "to": self.new_kind})


def finish_morphs(model, reg: TopologyRegistry) -> list[int]:
    """Retira el bloque viejo de las transiciones que llegaron a α = 0."""
    done = []
    for i, blk in enumerate(model.backbone):
        if isinstance(blk, MorphingBlock) and float(blk.alpha) <= 0.0:
            model.backbone[i] = blk.new
            node = reg.find_by_path(f"backbone.{i}")
            if node is not None:
                node.meta["kind"] = blk.new.kind
                node.meta.pop("morph", None)
            done.append(i)
    return done


# --- controlador -------------------------------------------------------------------------
@dataclass
class HyperplasticityController:
    """Gancho de ``Trainer``: aplica las máscaras, reconecta, avanza las transiciones y poda."""

    model: nn.Module
    registry: TopologyRegistry
    sparsity: DynamicSparsity | None = None
    rewire_every: int = 100
    morph_steps: int = 500            # pasos para pasar de α = 1 a α = 0
    prune_every: int = 200
    log: list[dict] = field(default_factory=list)
    _step: int = 0

    def __call__(self, trainer) -> None:
        self._step += 1
        if self.sparsity is not None:
            self.sparsity.apply()
            if self._step % self.rewire_every == 0:
                self.log.append({"step": self._step, **self.sparsity.update()})
        for blk in self.model.backbone:
            if isinstance(blk, MorphingBlock):
                blk.alpha.fill_(max(0.0, float(blk.alpha) - 1.0 / self.morph_steps))
        if self._step % self.prune_every == 0:
            removed = prune_connections(self.model, self.registry)
            finished = finish_morphs(self.model, self.registry)
            if removed or finished:
                self.log.append({"step": self._step, "connections_removed": removed, "morphs_finished": finished})
