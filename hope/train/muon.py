"""Optimizador Muon (Jordan et al. 2024; Liu et al. 2025, «Muon is Scalable», Moonlight).

Mejora sin parámetros n.º 3: aprender más por cada dato. Se adopta solo si gana
la ablación contra AdamW en la A770 (``hope/bench/ablation.py``, variante ``muon``).

* Las **matrices** de las capas ocultas se actualizan con el momento
  **ortogonalizado** por Newton–Schulz (5 iteraciones): todas las direcciones
  aprenden a la misma velocidad, en lugar de que unas pocas dominen.
* Con la escala de Moonlight (0,2·√max(filas, columnas)) el tamaño de la
  actualización es como el de AdamW, así que **sirve la misma tasa de aprendizaje**.
* Lo que no es matriz oculta (embeddings, cabeza de salida, normas, sesgos,
  compuertas, convoluciones) usa AdamW normal.
* Memoria: un solo momento por matriz (Adam guarda dos).
* Con pesos bf16 la escritura usa redondeo estocástico (como ``LowPrecisionAdamW``).
"""

from __future__ import annotations

import torch

from hope.train.lowprec import stochastic_round_bf16

NS_COEFS = (3.4445, -4.7750, 2.0315)
# ``embed``: embeddings de la línea base BPE; ``distill_head``: cabeza de destilación (vocabulario).
ADAM_PREFIXES = ("byte_embed", "head", "uncertainty", "embed", "distill_head")


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Aproxima U·Vᵀ de la descomposición en valores singulares de ``g`` (ortogonaliza)."""
    a, b, c = NS_COEFS
    x = g.float()
    x = x / (x.norm() + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    return x.T if tall else x


def muon_eligible(name: str, p: torch.nn.Parameter) -> bool:
    return p.ndim == 2 and min(p.shape) > 1 and not name.startswith(ADAM_PREFIXES)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, muon_ids: set[int] | None = None, lr: float = 3e-4, momentum: float = 0.95,
                 betas=(0.9, 0.95), weight_decay: float = 0.1, eps: float = 1e-8, ns_steps: int = 5, seed: int = 0,
                 offload: bool = False):
        super().__init__(params, dict(lr=lr, momentum=momentum, betas=betas, weight_decay=weight_decay,
                                      eps=eps, ns_steps=ns_steps))
        # Sin lista explícita (p. ej. al reconstruirlo tras un crecimiento) se decide por la forma
        # y por la marca ``_hope_adam`` que deja ``make_muon`` en embeddings y cabezas.
        self.muon_ids = set(muon_ids) if muon_ids is not None else None
        self.seed = seed
        self.offload = offload

    def __getstate__(self):
        d = super().__getstate__()
        d["muon_ids"], d["seed"], d["offload"] = self.muon_ids, self.seed, self.offload
        return d

    def __setstate__(self, state):
        super().__setstate__(state)

    def load_state_dict(self, state_dict):
        """Sin esto PyTorch convierte el momento fp32 al tipo del parámetro (bf16) al retomar."""
        from hope.train.lowprec import load_state_exact
        load_state_exact(self, state_dict, getattr(self, "offload", False))

    def _write(self, p: torch.Tensor, new: torch.Tensor) -> None:
        p.copy_(stochastic_round_bf16(new) if p.dtype == torch.bfloat16 else new.to(p.dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if getattr(self, "offload", False):
                    from hope.train.lowprec import fetch_state
                    fetch_state(st, p.device)
                w = p.float()
                is_muon = (id(p) in self.muon_ids if self.muon_ids is not None else
                           p.ndim == 2 and min(p.shape) > 1 and not getattr(p, "_hope_adam", False))
                if is_muon:
                    buf = st.setdefault("momentum", torch.zeros_like(g))
                    buf.mul_(group["momentum"]).add_(g)
                    upd = newton_schulz(g.add(buf, alpha=group["momentum"]), group["ns_steps"])
                    upd = upd * (0.2 * max(p.shape) ** 0.5)
                    self._write(p, w * (1 - lr * wd) - lr * upd)
                else:
                    b1, b2 = group["betas"]
                    st["step"] = st.get("step", 0) + 1
                    m = st.setdefault("exp_avg", torch.zeros_like(g))
                    v = st.setdefault("exp_avg_sq", torch.zeros_like(g))
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    mh = m / (1 - b1 ** st["step"])
                    vh = v / (1 - b2 ** st["step"])
                    decay = wd if p.ndim >= 2 else 0.0
                    self._write(p, w * (1 - lr * decay) - lr * mh / (vh.sqrt() + group["eps"]))
                if getattr(self, "offload", False):
                    from hope.train.lowprec import park_state
                    park_state(st)
        return loss


def make_muon(model: torch.nn.Module, *, lr: float, weight_decay: float = 0.1, seed: int = 0,
              param_groups: list[dict] | None = None, offload: bool = False) -> Muon:
    ids = set()
    for n, p in model.named_parameters():
        if muon_eligible(n, p):
            ids.add(id(p))
        else:
            p._hope_adam = True
    params = param_groups if param_groups is not None else [{"params": list(model.parameters()), "lr_scale": 1.0}]
    return Muon(params, ids, lr=lr, weight_decay=weight_decay, seed=seed, offload=offload)
