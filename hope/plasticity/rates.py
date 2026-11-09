"""Velocidad de aprendizaje por región (plasticidad modulada; HNHS §8, §9, §17).

Cada región aprende a su ritmo según su estado en el registro de topología:

* **protegida** (circuitos de tareas raras, ASG §5) → aprende muy poco;
* **estable / consolidada** → aprende normal o más lento;
* **experimental** (lo recién creado por ASG/AANS) → aprende más rápido.

Es la versión de software de las neuronas moduladoras: modifica la tasa de
plasticidad de grupos enteros, con límites duros para que una sola señal no
cambie toda la arquitectura (HNHS §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hope.epistemic import LifecycleStatus
from hope.registry.topology import TopologyRegistry

MIN_SCALE, MAX_SCALE = 0.0, 5.0


@dataclass
class PlasticityPolicy:
    protected: float = 0.05
    stable: float = 1.0
    experimental: float = 3.0
    overrides: dict[str, float] = field(default_factory=dict)      # id de nodo → escala

    def scale_for(self, node) -> float:
        if node.id in self.overrides:
            s = self.overrides[node.id]
        elif node.protected:
            s = self.protected
        elif node.status == LifecycleStatus.EXPERIMENTAL:
            s = self.experimental
        else:
            s = self.stable
        return float(min(max(s, MIN_SCALE), MAX_SCALE))


def param_groups(model: torch.nn.Module, reg: TopologyRegistry, policy: PlasticityPolicy) -> list[dict]:
    """Agrupa los parámetros por escala de aprendizaje según el nodo más específico que los contiene.

    Un nodo protegido o experimental transmite su estado a sus hijos (salvo que el hijo lo defina).
    """
    bound = sorted((n for n in reg.nodes() if n.module_path is not None),
                   key=lambda n: len(n.module_path), reverse=True)

    def inherited(node):
        chain = [node]
        while chain[-1].parent is not None:
            chain.append(reg.get(chain[-1].parent))
        for n in chain:
            if n.id in policy.overrides or n.protected or n.status == LifecycleStatus.EXPERIMENTAL:
                return policy.scale_for(n)
        return policy.scale_for(node)

    groups: dict[float, list] = {}
    for name, p in model.named_parameters():
        node = next((n for n in bound if n.module_path == "" or name == n.module_path
                     or name.startswith(n.module_path + ".")), None)
        scale = inherited(node) if node is not None else policy.stable
        groups.setdefault(scale, []).append(p)
    return [{"params": ps, "lr_scale": s} for s, ps in sorted(groups.items())]
