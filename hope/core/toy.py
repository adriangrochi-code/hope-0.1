"""Modelo modular de juguete para probar la infraestructura de F0.

No es el núcleo de HN-1. Reproduce en miniatura las piezas que ASG necesita
modificar: regiones con especialistas y router top-k (ARQUITECTURA_HN1 §3.5) y
conexiones entre regiones con compuerta (§6.2). Sirve para verificar que las
operaciones estructurales preservan la función, que el registro cuenta bien y
que las versiones se pueden revertir. También es la base del banco sintético
de la §12.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hope.registry.topology import NodeKind, TopologyRegistry

NEG_INF = -1e9


class Expert(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Region(nn.Module):
    """Región con especialistas seleccionados por un router top-k."""

    def __init__(self, d: int, n_experts: int, hidden: int, top_k: int):
        super().__init__()
        self.d = d
        self.top_k = top_k
        self.norm = nn.LayerNorm(d)
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList(Expert(d, hidden) for _ in range(n_experts))
        # Máscara de especialistas retirados (0 = activo, NEG_INF = enmascarado).
        self.register_buffer("expert_mask", torch.zeros(n_experts))

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    def forward(self, x):
        h = self.norm(x)
        logits = self.router(h) + self.expert_mask
        k = min(self.top_k, self.n_experts)
        top_val, top_idx = logits.topk(k, dim=-1)
        weights = top_val.softmax(dim=-1)
        # Cómputo denso: suficiente para un modelo de juguete.
        all_out = torch.stack([e(h) for e in self.experts], dim=-2)          # [..., E, d]
        chosen = all_out.gather(-2, top_idx.unsqueeze(-1).expand(*top_idx.shape, self.d))
        return (weights.unsqueeze(-1) * chosen).sum(dim=-2)


class Connection(nn.Module):
    """Conexión dirigida entre regiones: ``gate · proj(salida_origen)``."""

    def __init__(self, d: int, gate_init: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x):
        return self.gate * self.proj(x)


class ToyModularModel(nn.Module):
    def __init__(self, vocab: int = 64, d: int = 32, n_regions: int = 2,
                 n_experts: int = 4, hidden: int = 64, top_k: int = 2):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.regions = nn.ModuleList(Region(d, n_experts, hidden, top_k) for _ in range(n_regions))
        self.connections = nn.ModuleDict()          # clave "i_j": de la región i a la j (i < j)
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens):
        h = self.embed(tokens)
        outputs: list[torch.Tensor] = []
        for j, region in enumerate(self.regions):
            inp = h
            for key, conn in self.connections.items():
                src, dst = (int(v) for v in key.split("_"))
                if dst == j:
                    inp = inp + conn(outputs[src])
            out = region(inp)
            outputs.append(out)
            h = h + out
        return self.head(h)


def build_toy_model(**kwargs) -> tuple[ToyModularModel, TopologyRegistry]:
    """Construye el modelo y su registro de topología con IDs estables."""
    model = ToyModularModel(**kwargs)
    reg = TopologyRegistry()
    root = reg.add(NodeKind.MODEL, "", node_id="mod_toy_0")
    reg.add(NodeKind.OPERATOR, "embed", parent=root.id, hint="embed")
    reg.add(NodeKind.HEAD, "head", parent=root.id, hint="head")
    for i, region in enumerate(model.regions):
        r = reg.add(NodeKind.REGION, f"regions.{i}", parent=root.id, hint=f"region-{i}")
        reg.add(NodeKind.HEAD, f"regions.{i}.router", parent=r.id, hint=f"router-{i}")
        reg.add(NodeKind.OPERATOR, f"regions.{i}.norm", parent=r.id, hint=f"norm-{i}")
        frac = min(region.top_k, region.n_experts) / region.n_experts
        for e in range(region.n_experts):
            reg.add(NodeKind.EXPERT, f"regions.{i}.experts.{e}", parent=r.id,
                    hint=f"expert-{i}", activation=frac)
    return model, reg


class RoutedModularModel(nn.Module):
    """Modelo del banco sintético (ARQUITECTURA_HN1 §12.2).

    Codifica la secuencia completa (concatenando posiciones), pasa por una
    región compartida y después por la región de su familia de tareas. Así cada
    familia tiene una región propia y se puede medir si ASG crece «en la región
    correcta». ``regions[0]`` es la compartida; ``regions[f + 1]`` es la de la familia f.
    """

    def __init__(self, vocab: int, seq_len: int, n_classes: int, n_families: int, d: int = 32,
                 n_experts: int = 2, hidden: int = 8, shared_hidden: int = 16, top_k: int = 1):
        super().__init__()
        self.d = d
        self.n_families = n_families
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.encode = nn.Linear(seq_len * d, d)
        self.regions = nn.ModuleList(
            [Region(d, n_experts, shared_hidden, top_k)]
            + [Region(d, n_experts, hidden, top_k) for _ in range(n_families)])
        self.head = nn.Linear(d, n_classes)

    def forward(self, tokens, family):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        h = self.encode((self.embed(tokens) + self.pos(positions)).flatten(1))
        h = h + self.regions[0](h)
        out = torch.zeros_like(h)
        for f in range(self.n_families):
            m = family == f
            if m.any():
                out[m] = self.regions[f + 1](h[m])
        return self.head(h + out)


def build_routed_model(family_names: list[str], **kwargs) -> tuple[RoutedModularModel, TopologyRegistry]:
    model = RoutedModularModel(n_families=len(family_names), **kwargs)
    reg = TopologyRegistry()
    root = reg.add(NodeKind.MODEL, "", node_id="mod_bench_0")
    for path in ("embed", "pos", "encode"):
        reg.add(NodeKind.OPERATOR, path, parent=root.id, hint=path)
    reg.add(NodeKind.HEAD, "head", parent=root.id, hint="head")
    names = ["shared"] + family_names
    for i, region in enumerate(model.regions):
        # La región de una familia solo se activa para su fracción de las muestras.
        r = reg.add(NodeKind.REGION, f"regions.{i}", parent=root.id, hint=f"region-{names[i]}",
                    activation=1.0 if i == 0 else 1.0 / len(family_names), meta={"task": names[i]})
        reg.add(NodeKind.HEAD, f"regions.{i}.router", parent=r.id, hint=f"router-{names[i]}")
        reg.add(NodeKind.OPERATOR, f"regions.{i}.norm", parent=r.id, hint=f"norm-{names[i]}")
        frac = min(region.top_k, region.n_experts) / region.n_experts
        for e in range(region.n_experts):
            reg.add(NodeKind.EXPERT, f"regions.{i}.experts.{e}", parent=r.id,
                    hint=f"expert-{names[i]}", activation=frac)
    return model, reg
