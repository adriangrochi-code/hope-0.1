"""Codificadores de texto a vector para la recuperación por similitud.

``HashingEmbedder`` no necesita entrenamiento: n-gramas de caracteres
hasheados y proyectados a un vector normalizado. Es determinista y robusto
a errores ortográficos, y sirve como base mientras el núcleo neuronal no está
entrenado. Cuando lo esté, ``ModelEmbedder`` (pendiente) usará sus estados
internos. Ambos cumplen la misma interfaz: ``embed(list[str]) -> np.ndarray[N, dim]``.
"""

from __future__ import annotations

import hashlib
import unicodedata

import numpy as np


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


class HashingEmbedder:
    def __init__(self, dim: int = 256, ngram: tuple[int, ...] = (3, 4, 5), seed: int = 0):
        self.dim, self.ngram, self.seed = dim, ngram, seed

    def _index(self, gram: str) -> tuple[int, float]:
        h = hashlib.blake2b(f"{self.seed}:{gram}".encode(), digest_size=8).digest()
        v = int.from_bytes(h, "little")
        return v % self.dim, 1.0 if (v >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = f" {_strip_accents(text.lower())} "
            for n in self.ngram:
                for j in range(max(len(t) - n + 1, 0)):
                    k, sign = self._index(t[j:j + n])
                    out[i, k] += sign
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out
