"""Protección de conocimientos consolidados (ASG §2.3; AMC §8; HNHS §17).

Penalización tipo EWC (Kirkpatrick et al. 2017): al aprender algo nuevo, mover
un parámetro cuesta en proporción a lo importante que fue para lo anterior,
medido con la diagonal de la información de Fisher (gradiente al cuadrado
promedio sobre datos de lo ya aprendido):

    penalización = λ/2 · Σ F_i · (θ_i − θ*_i)²

Complementa al repaso (replay): el repaso recuerda con datos; la protección,
con la geometría de los pesos. Para no duplicar la memoria del modelo en fp32,
F y θ* se guardan en bf16.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch


def fisher_diagonal(model: torch.nn.Module, batches: Iterable, loss_fn: Callable,
                    only: Callable[[str], bool] | None = None) -> dict[str, torch.Tensor]:
    """Promedio de grad² por parámetro sobre ``batches`` (``loss_fn(model, batch)``)."""
    fisher: dict[str, torch.Tensor] = {}
    n = 0
    was = model.training
    model.eval()
    for batch in batches:
        model.zero_grad(set_to_none=True)
        loss_fn(model, batch).backward()
        n += 1
        for name, p in model.named_parameters():
            if p.grad is None or (only and not only(name)):
                continue
            g2 = p.grad.detach().float().pow(2)
            fisher[name] = fisher.get(name, torch.zeros_like(g2)) + g2
    model.zero_grad(set_to_none=True)
    model.train(was)
    return {k: (v / max(n, 1)).to(torch.bfloat16) for k, v in fisher.items()}


class ConsolidationPenalty:
    def __init__(self, model: torch.nn.Module, fisher: dict[str, torch.Tensor], strength: float = 100.0,
                 normalize: bool = True):
        self.strength = strength
        params = dict(model.named_parameters())
        scale = 1.0
        if normalize and fisher:                         # que λ no dependa de la escala de los gradientes
            scale = float(torch.cat([f.float().flatten() for f in fisher.values()]).mean().clamp_min(1e-12))
        self.terms = [(params[k], (f.float() / scale).to(torch.bfloat16),
                       params[k].detach().clone().to(torch.bfloat16)) for k, f in fisher.items() if k in params]

    def __call__(self, model: torch.nn.Module | None = None) -> torch.Tensor:
        total = None
        for p, f, anchor in self.terms:
            if p.shape != anchor.shape:                  # la estructura creció: ese parámetro ya no se compara
                continue
            term = (f.float() * (p.float() - anchor.float()).pow(2)).sum()
            total = term if total is None else total + term
        if total is None:
            return torch.zeros(())
        return 0.5 * self.strength * total / max(sum(f.numel() for _, f, _ in self.terms), 1)
