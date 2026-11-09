"""Operaciones de crecimiento del núcleo HN-1 completo (ARQUITECTURA_HN1 §6.2, §3.9).

Complementan a ``hope/growth/ops.py`` (especialistas y conexiones) con las que
hacen falta para llegar de ``hn1-200m`` a ``hn1-500m`` sin perder lo aprendido:

* ``AddVisualZone`` — agrega el codificador visual y la atención cruzada con
  compuertas en cero: la salida de texto no cambia.
* ``AppendBackboneBlock`` — agrega un bloque al final del backbone con sus
  proyecciones de salida en cero: el bloque nuevo es la identidad.

Las dos siguen el contrato de dos pasos (``plan`` sin tocar el modelo, ``apply``
que devuelve un ``ChangeRecord``) para que el presupuesto y el guardián las
evalúen antes. Un cambio de más del 10 % de los parámetros requiere aprobación
humana (decisión del usuario).

``plan_growth_to(model, "hn1-500m")`` arma la secuencia completa de operaciones.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from hope.core.layers import Block
from hope.core.vision import VisualZone
from hope.epistemic import LifecycleStatus
from hope.growth.budget import ProposedChange
from hope.growth.ops import AddExpert, ChangeRecord, WidenExpert, _n, _no_adapters
from hope.registry.topology import NodeKind, TopologyRegistry


def _model_dtype(model) -> tuple[torch.device, torch.dtype]:
    w = model.head.weight
    return w.device, w.dtype


class AddVisualZone:
    def __init__(self, **vision_cfg):
        self.vision_cfg = vision_cfg      # campos vis_* de HN1Config (vis_d, vis_layers, ...)

    def _cfg(self, model):
        return replace(model.cfg, vision=True, **self.vision_cfg)

    def _build(self, model):
        cfg = self._cfg(model)
        return VisualZone(cfg.d, cfg.heads, cfg.vision_blocks(), **cfg.vision_kwargs())

    def plan(self, model) -> ProposedChange:
        if model.vision is not None:
            raise ValueError("el modelo ya tiene zona visual")
        with torch.device("meta"):
            n = _n(self._build(model))
        return ProposedChange("agregar zona visual (codificador + atención cruzada)", param_delta=n,
                              active_delta=0.0, region_delta=1, plasticity_level=4)

    @torch.no_grad()
    def apply(self, model, reg: TopologyRegistry) -> ChangeRecord:
        _no_adapters(model)
        n = self.plan(model).param_delta
        dev, dtype = _model_dtype(model)
        zone = self._build(model).to(device=dev, dtype=dtype)
        for xa in zone.xattn.values():
            xa.out.weight.zero_()                 # salida en cero: la función de texto no cambia
        model.vision = zone
        model.cfg = self._cfg(model)
        from hope.core.hn1 import register_vision
        root = reg.find_by_path("")
        register_vision(reg, model, root.id)
        node = reg.find_by_path("vision")
        node.status = LifecycleStatus.EXPERIMENTAL
        return ChangeRecord("add_visual_zone", [root.id], n, preserving=True, created=[node.id],
                            detail={"tokens_per_image": zone.encoder.tokens_per_image(),
                                    "xattn_blocks": list(zone.at_blocks)})


class AppendBackboneBlock:
    def __init__(self, kind: str = "ssm"):
        self.kind = kind

    def _build(self, model):
        c = model.cfg
        return Block(c.d, self.kind, c.heads, c.mlp_hidden, hnhs=c.hnhs, kv_heads=c.kv_heads,
                     window=c.window, sinks=c.sinks)

    def plan(self, model) -> ProposedChange:
        with torch.device("meta"):
            n = _n(self._build(model))
        return ProposedChange(f"agregar bloque {self.kind} al final del backbone", param_delta=n,
                              active_delta=float(n), plasticity_level=4)

    @torch.no_grad()
    def apply(self, model, reg: TopologyRegistry) -> ChangeRecord:
        _no_adapters(model)
        n = self.plan(model).param_delta
        dev, dtype = _model_dtype(model)
        blk = self._build(model).to(device=dev, dtype=dtype)
        out = blk.mixer.out if hasattr(blk.mixer, "out") else blk.mixer.out_proj
        out.weight.zero_()
        mlp_out = getattr(blk.mlp, "w_out", None)
        if mlp_out is None:
            mlp_out = [m for m in blk.mlp.modules() if isinstance(m, nn.Linear)][-1]
        mlp_out.weight.zero_()
        if mlp_out.bias is not None:
            mlp_out.bias.zero_()
        model.backbone.append(blk)
        j = len(model.backbone) - 1
        model.cfg = replace(model.cfg, pattern=tuple(model.cfg.pattern) + (self.kind,))
        bb = reg.find_by_path("backbone")
        node = reg.add(NodeKind.POPULATION, f"backbone.{j}", parent=bb.id, hint=f"backbone-{self.kind}-{j}",
                       meta={"kind": self.kind})
        node.status = LifecycleStatus.EXPERIMENTAL
        return ChangeRecord("append_backbone_block", [bb.id], n, preserving=True, created=[node.id],
                            detail={"index": j, "kind": self.kind})


def plan_growth_to(model, target) -> list:
    """Secuencia de operaciones que preservan la función para llegar a ``target``
    (nombre de preset o ``HN1Config``) desde ``model``. Requisitos: mismo ``d``, misma
    parte local y que el patrón de ``target`` empiece con el patrón actual."""
    from hope.core.hn1 import preset
    t = preset(target) if isinstance(target, str) else target
    c = model.cfg
    if (t.d, t.d_local, t.enc_layers, t.dec_layers, t.n_regions) != (c.d, c.d_local, c.enc_layers,
                                                                     c.dec_layers, c.n_regions):
        raise ValueError("el destino cambia el ancho, la parte local o las regiones: no se llega creciendo")
    cur = tuple(c.pattern)
    if tuple(t.pattern[:len(cur)]) != cur:
        raise ValueError("el patrón destino no extiende al actual")
    ops: list = [AppendBackboneBlock(k) for k in t.pattern[len(cur):]]
    for r, region in enumerate(model.regions):
        have = [getattr(e, "kind", "mlp") for e in region.experts]
        missing = list(t.zone_kinds)
        for k in have:
            if k in missing:
                missing.remove(k)
        ops += [AddExpert(r, hidden=t.expert_hidden, kind=k) for k in missing]
        for e, ex in enumerate(region.experts):
            extra = t.expert_hidden - getattr(ex, "hidden", c.expert_hidden)
            if extra > 0:
                op = WidenExpert(r, e, extra)
                try:
                    op.plan(model)
                except NotImplementedError:          # p. ej. «relacional»: no se ensancha, ya tiene pares nuevos
                    continue
                ops.append(op)
    if t.vision and model.vision is None:
        ops.append(AddVisualZone(**{k: getattr(t, k) for k in ("vis_d", "vis_layers", "vis_heads", "vis_hidden",
                                                              "vis_patch", "vis_image", "vis_pool", "vis_blocks")}))
    return ops
