"""FIM — completar el medio (*fill-in-the-middle*; Bavarian et al. 2022, StarCoder, DeepSeek-Coder).

Al programar casi nunca se escribe al final del archivo: se completa **entre** código que ya
existe (el cuerpo de una función entre su firma y lo que sigue). Un modelo causal aprende eso si
parte de sus documentos se reordenan así:

    PSM:  ⟦prefijo⟧ <antes> ⟦sufijo⟧ <después> ⟦medio⟧ <lo del medio> ⟦fin⟧
    SPM:  ⟦sufijo⟧ <después> ⟦prefijo⟧ <antes> ⟦medio⟧ <lo del medio> ⟦fin⟧

La pérdida es la de siempre (todos los bytes), así el modelo sigue aprendiendo a escribir de
izquierda a derecha con el resto del lote (``TrainConfig.fim_rate``; 0,5 es lo habitual y casi no
cuesta capacidad de izquierda a derecha según Bavarian et al.).

* Cortes por **líneas** (por defecto para código: el medio son líneas completas) o por
  **bytes** (en cualquier posición que no parta un carácter UTF-8).
* ``fim_bytes`` transforma una ventana de bytes del lote **sin cambiar su largo** (las marcas
  ocupan lugar: se recorta el final del medio), así se aplica en ``Trainer.train`` sin tocar
  el resto del lote.
* Generación: ``fim_prompt(prefijo, sufijo)`` y ``complete_middle(model, prefijo, sufijo)``
  (también ``HN1Model.generate_fim``), que corta en ``⟦fin⟧``.
"""

from __future__ import annotations

import random

import numpy as np
import torch

PRE, SUF, MID, END = "⟦prefijo⟧", "⟦sufijo⟧", "⟦medio⟧", "⟦fin⟧"
PRE_B, SUF_B, MID_B, END_B = (m.encode("utf-8") for m in (PRE, SUF, MID, END))


def fim_format(prefix: str, middle: str, suffix: str, *, spm: bool = False, end: bool = True) -> str:
    tail = middle + (END if end else "")
    if spm:
        return f"{SUF}{suffix}{PRE}{prefix}{MID}{tail}"
    return f"{PRE}{prefix}{SUF}{suffix}{MID}{tail}"


def fim_prompt(prefix: str, suffix: str, *, spm: bool = False) -> str:
    """Lo que se le da al modelo para que escriba el medio."""
    return fim_format(prefix, "", suffix, spm=spm, end=False)


def split_points(text: str, rng: random.Random, mode: str = "lineas") -> tuple[int, int]:
    """Dos posiciones ``a ≤ b`` (medio = ``text[a:b]``)."""
    if mode == "lineas":
        cuts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"] + [len(text)]
        cuts = sorted(set(cuts))
        if len(cuts) >= 3:
            a, b = sorted(rng.sample(cuts, 2))
            return a, b
    a, b = sorted(rng.randint(0, len(text)) for _ in range(2))
    return a, b


def fim_transform(text: str, rng: random.Random, *, rate: float = 0.5, spm_rate: float = 0.5,
                  mode: str = "lineas") -> str:
    """Con probabilidad ``rate`` reordena el documento en formato FIM; si no, lo deja igual."""
    if rng.random() >= rate or len(text) < 4:
        return text
    a, b = split_points(text, rng, mode)
    return fim_format(text[:a], text[a:b], text[b:], spm=rng.random() < spm_rate)


def _cut_ok(row: bytes, i: int) -> bool:
    return i <= 0 or i >= len(row) or (row[i] & 0xC0) != 0x80          # no partir un carácter UTF-8


def fim_bytes(row: bytes, rng: random.Random, *, spm_rate: float = 0.5, mode: str = "lineas") -> bytes:
    """Reordena una ventana de bytes en formato FIM **conservando el largo** (se recorta el medio)."""
    n = len(row)
    if n < 32:
        return row
    if mode == "lineas":
        cuts = [0] + [i + 1 for i, c in enumerate(row) if c == 10] + [n]
        cuts = sorted(set(cuts))
        if len(cuts) >= 3:
            a, b = sorted(rng.sample(cuts, 2))
        else:
            mode = "bytes"
    if mode != "lineas":
        cands = [i for i in range(n + 1) if _cut_ok(row, i)]
        a, b = sorted(rng.sample(cands, 2))
    pre, mid, suf = row[:a], row[a:b], row[b:]
    out = (SUF_B + suf + PRE_B + pre if rng.random() < spm_rate else PRE_B + pre + SUF_B + suf) + MID_B + mid + END_B
    return out[:n] if len(out) >= n else out + b"\x00" * (n - len(out))


def apply_fim(x: torch.Tensor, rng: np.random.Generator, *, rate: float, spm_rate: float = 0.5,
              mode: str = "lineas", rows: list[bool] | None = None) -> tuple[torch.Tensor, int]:
    """Aplica FIM a una fracción ``rate`` de las filas de un lote de bytes ``[B, T]``.

    ``rows``: qué filas son elegibles (p. ej. solo las de fuentes de código). Devuelve el lote y
    cuántas filas se transformaron."""
    if rate <= 0:
        return x, 0
    out = x.clone()
    n = 0
    for i in range(x.shape[0]):
        if rows is not None and not rows[i]:
            continue
        if rng.random() >= rate:
            continue
        r = random.Random(int(rng.integers(1 << 62)))
        row = bytes(x[i].tolist())
        new = fim_bytes(row, r, spm_rate=spm_rate, mode=mode)
        out[i] = torch.tensor(list(new), dtype=x.dtype)
        n += 1
    return out, n


def complete_middle(model, prefix: str, suffix: str, *, max_new: int = 256, spm: bool = False, **kw) -> str:
    """Pide al modelo el medio entre ``prefix`` y ``suffix`` (corta en ``⟦fin⟧``)."""
    out = model.generate(fim_prompt(prefix, suffix, spm=spm), max_new, stop_seq=END_B, **kw)
    text = out.decode("utf-8", "replace")
    return text[:-len(END)] if text.endswith(END) else text
