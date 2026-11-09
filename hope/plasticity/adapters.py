"""Adaptación rápida descartable — FAE (HLA §10; ARQUITECTURA_HN1 §6.4, nivel 2).

Para incorporar algo nuevo **sin reentrenar ni arriesgar toda la red**, se
enganchan adaptadores de bajo rango (LoRA, Hu et al. 2021) a capas lineales:

    y = W·x + (α/r) · B·A·x        con B = 0 al inicio → la función no cambia

* Durante la sesión solo se entrenan A y B (la base puede quedar congelada).
* Si lo aprendido demuestra ser útil y fiable → ``merge_adapters`` lo
  **consolida** en W (aprendizaje lento) y los adaptadores desaparecen.
* Si no → ``discard_adapters`` vuelve exactamente al modelo anterior.

Antes de una operación estructural (ASG) hay que consolidar o descartar los
adaptadores: las operaciones de crecimiento trabajan sobre capas lineales simples.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, seed: int = 0):
        super().__init__()
        self.base = base
        self.rank, self.scaling = rank, alpha / rank
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(rank, base.in_features, generator=g) / math.sqrt(base.in_features)
        self.lora_a = nn.Parameter(a.to(base.weight))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype,
                                               device=base.weight.device))

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling

    def merged(self) -> nn.Linear:
        with torch.no_grad():
            self.base.weight += (self.lora_b.float() @ self.lora_a.float() * self.scaling).to(self.base.weight)
        return self.base


def _targets(model: nn.Module, prefixes: Iterable[str]) -> list[tuple[nn.Module, str, nn.Linear]]:
    out = []
    prefixes = tuple(prefixes)
    for name, mod in model.named_modules():
        for child_name, child in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not isinstance(mod, LoRALinear) and full.startswith(prefixes):
                out.append((mod, child_name, child))
    return out


def attach_adapters(model: nn.Module, prefixes: Iterable[str] = ("regions", "backbone"), rank: int = 8,
                    alpha: float = 16.0, freeze_base: bool = True) -> int:
    """Engancha adaptadores a las capas lineales cuyo nombre empieza con ``prefixes``."""
    targets = _targets(model, prefixes)
    for i, (parent, name, lin) in enumerate(targets):
        setattr(parent, name, LoRALinear(lin, rank, alpha, seed=i))
    if freeze_base:
        for n, p in model.named_parameters():
            p.requires_grad_("lora_" in n)
    return len(targets)


def _wrapped(model: nn.Module) -> list[tuple[nn.Module, str, LoRALinear]]:
    return [(mod, n, c) for mod in model.modules() for n, c in mod.named_children() if isinstance(c, LoRALinear)]


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def merge_adapters(model: nn.Module) -> int:
    """Consolida lo aprendido en los pesos de base y quita los adaptadores."""
    items = _wrapped(model)
    for parent, name, w in items:
        setattr(parent, name, w.merged())
    for p in model.parameters():
        p.requires_grad_(True)
    return len(items)


def discard_adapters(model: nn.Module) -> int:
    """Descarta lo aprendido en la sesión: el modelo vuelve exactamente a como estaba."""
    items = _wrapped(model)
    for parent, name, w in items:
        setattr(parent, name, w.base)
    for p in model.parameters():
        p.requires_grad_(True)
    return len(items)


def has_adapters(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())
