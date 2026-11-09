"""Parcheo por entropía (propuesta 4; inspirado en BLT, Pagnoni et al. 2024).

Idea: gastar cómputo donde el texto es difícil. Un modelo de n-gramas de bytes
estima la entropía H(próximo byte | últimos bytes). Cuando es alta, el parche
se cierra: el próximo byte empieza un parche nuevo y recibe atención propia
del modelo global. Los tramos predecibles (el final de una palabra conocida,
espacios repetidos) quedan dentro de parches largos.

* **Causal:** la frontera en *t* depende solo de los bytes ≤ *t*.
* **Liviano:** una tabla de 65.536 entropías en float16 (128 KB) guardada como
  buffer del modelo; no es una red neuronal extra.
* **Calibrado:** el umbral se elige para lograr un largo medio de parche
  objetivo (por defecto 4,5 bytes, como el parcheo por espacios).

BLT usa un pequeño modelo de lenguaje de bytes para la entropía; acá se usa un
modelo de n-gramas por simplicidad y velocidad. Si mejora el BPB por cómputo
frente al parcheo por espacios se decide midiendo en la A770.
"""

from __future__ import annotations

import numpy as np
import torch

MULT = 257


def context_hash(x: torch.Tensor, order: int, buckets: int) -> torch.Tensor:
    """Hash de los últimos ``order`` bytes que terminan en cada posición ([B, T] → [B, T])."""
    x = x.long()
    acc = torch.zeros_like(x)
    for k in range(order - 1, -1, -1):
        shifted = torch.nn.functional.pad(x, (k, 0), value=0)[:, : x.shape[1]]
        acc = (acc * MULT + shifted + 1) % buckets
    return acc


def fit_entropy_table(data: np.ndarray, order: int = 3, buckets: int = 65536, alpha: float = 0.01,
                      max_bytes: int = 50_000_000) -> np.ndarray:
    """Entropía (bits) del próximo byte para cada contexto, estimada con conteos suavizados."""
    data = np.asarray(data[:max_bytes], dtype=np.int64)
    x = torch.from_numpy(data).unsqueeze(0)
    ctx = context_hash(x, order, buckets)[0, :-1].numpy()
    nxt = data[1:]
    counts = np.zeros((buckets, 256), dtype=np.float32)
    np.add.at(counts, (ctx, nxt), 1.0)
    probs = (counts + alpha) / (counts.sum(1, keepdims=True) + 256 * alpha)
    return (-(probs * np.log2(probs)).sum(1)).astype(np.float16)


def entropy_boundaries(x: torch.Tensor, table: torch.Tensor, threshold: float, order: int) -> torch.Tensor:
    """``True`` donde el próximo byte es difícil (el parche se cierra en *t*)."""
    return table[context_hash(x, order, table.numel())].float() > threshold


def calibrate_threshold(data: np.ndarray, table: np.ndarray, order: int, max_patch: int,
                        target_bytes_per_patch: float = 4.5, sample: int = 2_000_000) -> float:
    """Busca el umbral que da el largo medio de parche deseado (búsqueda binaria)."""
    from hope.core.hn1 import patch_boundaries
    x = torch.from_numpy(np.asarray(data[:sample], dtype=np.int64)).unsqueeze(0)
    t = torch.from_numpy(table.astype(np.float32))
    lo, hi = 0.0, 8.0
    for _ in range(25):
        mid = (lo + hi) / 2
        end, _ = patch_boundaries(x, max_patch, entropy=(t, mid, order))
        length = x.shape[1] / max(int(end.sum()), 1)
        if length < target_bytes_per_patch:
            lo = mid                                   # parches cortos → subir el umbral
        else:
            hi = mid
    return (lo + hi) / 2
