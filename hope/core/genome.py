"""Especialistas compuestos por un «genoma» de operadores (HNHS §6, §12; AANS §3.2 NAG, §4).

En vez de elegir solo entre tipos de red conocidos, una zona puede tener
especialistas cuya arquitectura es una **secuencia de operadores** de la
biblioteca de HNHS. El genoma se puede **mutar** (agregar, quitar o cambiar un
operador, cambiar el ancho), así AANS explora arquitecturas nuevas, chicas y
especializadas, y las evalúa contra un control antes de integrarlas.

Operadores (sobre un ancho interno ``h``):

==============  ===========================================================
``gelu``        no linealidad
``relu``        no linealidad (tasas ≥ 0)
``glu``         compuerta multiplicativa: σ(W_g z) ⊙ W_v z           (GATE, MUL)
``max``         máximo entre dos proyecciones                         (MAX)
``conv``        convolución causal en el tiempo                       (DETECT, temporal)
``scan``        recurrencia con decaimiento aprendido                  (INTEGRATE, MEMORY)
``norm``        normalización RMS                                     (NORMALIZE)
``inhib``       normalización divisiva (inhibición lateral)            (INHIBIT)
``res_lin``     capa lineal con conexión residual                     (SUM)
==============  ===========================================================

La entrada se proyecta a ``h`` (``fc1``), pasa por los operadores y vuelve a
``d`` por ``fc2``; ``fc2`` en cero hace que agregar el especialista preserve la función.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from hope.core.layers import RMSNorm, causal_conv, scan

OPS = ("gelu", "relu", "glu", "max", "conv", "scan", "norm", "inhib", "res_lin")
TEMPORAL = {"conv", "scan"}


@dataclass
class Genome:
    ops: list[str] = field(default_factory=lambda: ["gelu"])
    hidden: int = 64

    def key(self) -> str:
        return f"{'-'.join(self.ops)}@{self.hidden}"

    def mutate(self, rng: random.Random, max_len: int = 6) -> "Genome":
        """Una mutación al azar: agregar, quitar o cambiar un operador, o cambiar el ancho."""
        ops = list(self.ops)
        kind = rng.choice(["add", "remove", "replace", "width"])
        if kind == "add" and len(ops) < max_len:
            ops.insert(rng.randint(0, len(ops)), rng.choice(OPS))
        elif kind == "remove" and len(ops) > 1:
            ops.pop(rng.randrange(len(ops)))
        elif kind == "replace":
            ops[rng.randrange(len(ops))] = rng.choice(OPS)
        else:
            return Genome(ops, max(8, int(self.hidden * rng.choice([0.5, 1.5, 2.0]))))
        return Genome(ops, self.hidden)


class _Op(nn.Module):
    def __init__(self, name: str, h: int):
        super().__init__()
        self.name = name
        if name in ("glu", "max"):
            self.a, self.b = nn.Linear(h, h), nn.Linear(h, h)
        elif name == "conv":
            self.conv = nn.Conv1d(h, h, 4, groups=h, padding=3)
        elif name == "scan":
            self.decay = nn.Parameter(torch.full((h,), -1.0))
            self.inp = nn.Linear(h, h)
        elif name == "norm":
            self.n = RMSNorm(h)
        elif name == "res_lin":
            self.lin = nn.Linear(h, h)

    def forward(self, z, cache: dict | None = None):
        n = self.name
        if n == "gelu":
            return F.gelu(z)
        if n == "relu":
            return F.relu(z)
        if n == "glu":
            return torch.sigmoid(self.a(z)) * self.b(z)
        if n == "max":
            return torch.maximum(self.a(z), self.b(z))
        if n == "conv":
            return causal_conv(self.conv, z, cache)
        if n == "scan":
            log_a = -F.softplus(-self.decay).expand_as(z) - 1e-3          # log a ∈ (−∞, 0)
            return scan(log_a, self.inp(z), cache)
        if n == "norm":
            return self.n(z)
        if n == "inhib":
            r = F.relu(z)
            return r / (1.0 + r.mean(-1, keepdim=True))
        return z + self.lin(z)                                             # res_lin


class ComposedSpecialist(nn.Module):
    kind = "compuesto"

    def __init__(self, d: int, genome: Genome):
        super().__init__()
        self.genome = genome
        self.sequence_aware = any(op in TEMPORAL for op in genome.ops)
        self.fc1 = nn.Linear(d, genome.hidden)
        self.ops = nn.ModuleList(_Op(o, genome.hidden) for o in genome.ops)
        self.fc2 = nn.Linear(genome.hidden, d)

    @property
    def hidden(self) -> int:
        return self.fc2.in_features

    def zero_output_(self) -> None:
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, x, cache: dict | None = None):
        squeeze = x.dim() == 2                    # especialistas por posición reciben [N, d]
        z = self.fc1(x.unsqueeze(1) if squeeze else x)
        for i, op in enumerate(self.ops):
            z = op(z, cache.setdefault(i, {}) if cache is not None else None)
        y = self.fc2(z)
        return y.squeeze(1) if squeeze else y
