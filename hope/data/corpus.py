"""Preparación del corpus de bytes (ARQUITECTURA_HN1 §10.1.1).

Pasos:
  1. Leer documentos (.txt, .md, código fuente y .pdf si ``pdftotext`` está instalado).
  2. Normalizar: Unicode NFC, finales de línea, sin caracteres de control
     salvo salto de línea y tabulación.
  3. Deduplicar documentos exactos por hash.
  4. Partición **determinista por hash del documento**: el mismo documento cae
     siempre en la misma partición aunque se agreguen otros. La validación
     queda separada del entrenamiento.
  5. Escribir ``train.bin`` / ``val.bin`` (bytes UTF-8; el byte 0 separa
     documentos), ``manifest.json`` y ``provenance.jsonl`` (origen, hash,
     tamaño y partición de cada documento, para revisar licencias).

Para varios TB esto se reemplaza por un proceso por lotes y en paralelo, con
deduplicación aproximada (MinHash) y filtros de calidad. El formato de salida
es el mismo.

Uso:
    python -m hope.data.corpus --input carpeta1 carpeta2 --out data/smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

EOD = 0                      # separador de documentos
TEXT_EXT = {".txt", ".md", ".py", ".c", ".h", ".cpp", ".hpp", ".rst", ".tex", ".json", ".csv"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def read_document(path: Path) -> str | None:
    if path.suffix.lower() == ".pdf":
        if shutil.which("pdftotext") is None:
            return None
        res = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, timeout=300)
        return res.stdout.decode("utf-8", errors="replace") if res.returncode == 0 else None
    if path.suffix.lower() in TEXT_EXT:
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def split_of(digest: str, val_percent: float) -> str:
    return "val" if int(digest[:8], 16) % 10_000 < val_percent * 100 else "train"


def prepare(inputs: list[str | Path], out: str | Path, *, val_percent: float = 5.0,
            min_chars: int = 200, source_label: str | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    files = {"train": open(out / "train.bin", "wb"), "val": open(out / "val.bin", "wb")}
    stats = {"train": {"docs": 0, "bytes": 0}, "val": {"docs": 0, "bytes": 0},
             "skipped_duplicate": 0, "skipped_short": 0, "skipped_unreadable": 0}
    with open(out / "provenance.jsonl", "w", encoding="utf-8") as prov:
        for root in inputs:
            root = Path(root)
            paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
            for path in paths:
                text = read_document(path)
                if text is None:
                    stats["skipped_unreadable"] += 1
                    continue
                text = normalize(text).strip()
                if len(text) < min_chars:
                    stats["skipped_short"] += 1
                    continue
                data = text.encode("utf-8")
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    stats["skipped_duplicate"] += 1
                    continue
                seen.add(digest)
                split = split_of(digest, val_percent)
                files[split].write(data + bytes([EOD]))
                stats[split]["docs"] += 1
                stats[split]["bytes"] += len(data) + 1
                prov.write(json.dumps({"path": str(path), "sha256": digest, "bytes": len(data),
                                       "split": split, "source": source_label or str(root),
                                       "license": None}, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    manifest = {"format": "utf-8 bytes, 0x00 separa documentos", "val_percent": val_percent, **stats}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class ByteData:
    """Ventanas aleatorias de bytes sobre un archivo ``.bin`` (memoria mapeada)."""

    def __init__(self, path: str | Path):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        if len(self.data) < 2:
            raise ValueError(f"{path} está vacío")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch: int, length: int, rng: np.random.Generator):
        import torch
        length = min(length, len(self.data) - 1)
        starts = rng.integers(0, len(self.data) - length, size=batch)
        arr = np.stack([self.data[s:s + length] for s in starts]).astype(np.int64)
        return torch.from_numpy(arr)

    def windows(self, length: int, max_windows: int | None = None):
        """Ventanas consecutivas sin solapamiento, para evaluar siempre lo mismo."""
        import torch
        n = (len(self.data) - 1) // length
        if max_windows:
            n = min(n, max_windows)
        for i in range(n):
            yield torch.from_numpy(self.data[i * length:(i + 1) * length].astype(np.int64)).unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-percent", type=float, default=5.0)
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()
    print(json.dumps(prepare(args.input, args.out, val_percent=args.val_percent,
                             min_chars=args.min_chars), indent=2))


if __name__ == "__main__":
    main()
