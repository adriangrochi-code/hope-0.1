"""Corpus de bytes en disco (``train.bin`` / ``val.bin``).

Cada fuente es una carpeta con dos archivos de bytes crudos (UTF-8): ``train.bin`` y ``val.bin``.
Los documentos se separan con ``\\x00``. ``ByteData`` los abre con ``np.memmap`` (sin cargarlos
en memoria), así una fuente de varios GB no ocupa RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SEP = b"\x00"


class ByteData:
    """Arreglo ``uint8`` de solo lectura sobre un ``.bin`` de bytes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        size = self.path.stat().st_size
        if size == 0:
            self.data = np.zeros(0, dtype=np.uint8)
        else:
            self.data = np.memmap(self.path, dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"ByteData({str(self.path)!r}, {len(self)} bytes)"


class BinWriter:
    """Escribe documentos a ``train.bin``/``val.bin`` con un presupuesto de bytes.

    La partición es por documento y determinista (``val_every``: uno de cada N documentos va a
    validación), así ningún documento queda partido entre entrenamiento y validación.
    """

    def __init__(self, folder: str | Path, budget: int, val_every: int = 100):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget)
        self.val_every = max(int(val_every), 2)
        self._train = (self.folder / "train.bin").open("wb")
        self._val = (self.folder / "val.bin").open("wb")
        self.written = 0
        self.docs = 0

    @property
    def full(self) -> bool:
        return self.written >= self.budget

    def add(self, text: str) -> bool:
        """Agrega un documento. Devuelve ``False`` si ya se alcanzó el presupuesto."""
        if self.full:
            return False
        raw = text.replace("\x00", "").encode("utf-8") + SEP
        self.docs += 1
        (self._val if self.docs % self.val_every == 0 else self._train).write(raw)
        self.written += len(raw)
        return True

    def close(self) -> dict:
        self._train.close()
        self._val.close()
        return {"docs": self.docs, "bytes": self.written,
                "train_bytes": (self.folder / "train.bin").stat().st_size,
                "val_bytes": (self.folder / "val.bin").stat().st_size}
