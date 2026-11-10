"""Datos largos de verdad para el currículo de contexto (ARQUITECTURA_HN1 §3.9.2; pedido del usuario).

ProLong (Gao et al. 2024) encontró que las mejores fuentes largas son **libros** y **repositorios de
código completos**, siempre **mezcladas con al menos la mitad de datos cortos** de calidad. Este
módulo arma esas fuentes largas, con documentos enteros y un índice de dónde empieza cada uno:

* **Wikipedia**: artículos completos de al menos ``--min`` bytes (por defecto 16 KB);
* **libros**: los de Project Gutenberg que ya bajó ``hope.data.web_sources`` (dominio público);
* **repositorios**: todos los archivos de un repositorio concatenados, cada uno precedido por su
  ruta (``⟦repo: nombre⟧ ⟦archivo: ruta⟧``), como los arma ``hope.data.repos``;
* cualquier carpeta con ``train.bin``/``val.bin`` (documentos separados por 0x00).

Salida por fuente: ``train.bin``, ``val.bin`` y ``train.docs.npy`` / ``val.docs.npy`` (comienzo y
largo de cada documento). ``hope.train.run`` reconoce el índice y muestrea con ``LongDocSource``:
ventanas **dentro de un mismo documento** (si el documento es más corto que la ventana, sigue con
el siguiente de la misma fuente).

Uso::

    python -m hope.data.largos --out data/largos \\
        --desde-bin libros=data/web/consolidado/gutenberg \\
        --desde-bin wiki_es=data/pre/wiki_es --desde-bin wiki_en=data/pre/wiki_en \\
        --repos-bin repos=data/pre/repos/repos_codigo --min 16K
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from hope.data.corpus import EOD
from hope.train.continual import Source

REPO_HEADER = re.compile(rb"^\xe2\x9f\xa6repo: ([^\xe2]+)\xe2\x9f\xa7")      # ⟦repo: nombre⟧


def iter_docs(path: str | Path, block: int = 1 << 26):
    """Documentos (bytes) de un ``.bin`` separados por 0x00, leyendo por bloques."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    start = 0
    for s in range(0, len(data), block):
        zeros = np.flatnonzero(data[s:s + block] == EOD) + s
        for z in zeros:
            if z > start:
                yield bytes(data[start:z])
            start = z + 1
    if start < len(data):
        yield bytes(data[start:])


def group_repos(docs):
    """Junta los archivos consecutivos de un mismo repositorio en un solo documento."""
    cur, parts = None, []
    for d in docs:
        m = REPO_HEADER.match(d)
        name = m.group(1) if m else None
        if parts and name != cur:
            yield b"\n\n".join(parts)
            parts = []
        cur = name
        parts.append(d)
    if parts:
        yield b"\n\n".join(parts)


class _Writer:
    def __init__(self, folder: Path, split: str):
        folder.mkdir(parents=True, exist_ok=True)
        self.f = open(folder / f"{split}.bin", "wb")
        self.folder, self.split = folder, split
        self.starts: list[int] = []
        self.lens: list[int] = []
        self.pos = 0

    def add(self, doc: bytes) -> None:
        self.starts.append(self.pos)
        self.lens.append(len(doc))
        self.f.write(doc + bytes([EOD]))
        self.pos += len(doc) + 1

    def close(self) -> dict:
        self.f.close()
        np.save(self.folder / f"{self.split}.docs.npy", np.array([self.starts, self.lens], dtype=np.int64))
        lens = np.array(self.lens or [0])
        return {"docs": len(self.lens), "bytes": int(self.pos),
                "mediana": int(np.median(lens)), "p90": int(np.percentile(lens, 90)), "max": int(lens.max())}


def build_long(out: str | Path, name: str, splits: dict[str, object], min_bytes: int = 16_000,
               max_bytes: int | None = None, max_doc: int = 4_000_000) -> dict:
    """Escribe ``out/name`` con los documentos de al menos ``min_bytes`` de cada partición
    (``splits = {"train": docs, "val": docs}``; ``docs`` itera bytes o textos)."""
    folder = Path(out) / name
    summary = {}
    for split, docs in splits.items():
        w = _Writer(folder, split)
        skipped = 0
        for d in docs:
            d = d.encode("utf-8") if isinstance(d, str) else d
            d = d.replace(bytes([EOD]), b"")[:max_doc]
            if len(d) < min_bytes:
                skipped += 1
                continue
            w.add(d)
            if max_bytes and split == "train" and w.pos >= max_bytes:
                break
        summary[split] = {**w.close(), "cortos_descartados": skipped}
    (folder / "manifest.json").write_text(json.dumps({"fuente": name, "min_bytes": min_bytes, **summary},
                                                     indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


class LongDocSource(Source):
    """Fuente que muestrea ventanas dentro de documentos largos (con su índice ``*.docs.npy``).
    Cada byte de la fuente tiene la misma probabilidad de caer en una ventana: los documentos se
    eligen en proporción a su largo."""

    def __init__(self, name: str, folder: str | Path, weight: float, split: str = "train", mask=None):
        from hope.data.corpus import ByteData
        folder = Path(folder)
        super().__init__(name, ByteData(folder / f"{split}.bin"), weight, mask)
        idx = np.load(folder / f"{split}.docs.npy")
        self.starts, self.lens = idx[0], idx[1]
        self.p = self.lens / self.lens.sum()

    def sample_weighted(self, batch: int, length: int,
                        rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        arr = self.data.data
        length = min(length, len(arr) - 1)
        rows, ws = [], []
        for i in rng.choice(len(self.lens), size=batch, p=self.p):
            start, n = int(self.starts[i]), int(self.lens[i])
            s = start + int(rng.integers(0, n - length + 1)) if n >= length else start
            s = min(s, len(arr) - length)
            rows.append(np.asarray(arr[s:s + length], dtype=np.int64))
            ws.append(np.ones(length, np.float32) if self.mask is None
                      else np.asarray(self.mask[s:s + length], dtype=np.float32))
        return torch.from_numpy(np.stack(rows)), torch.from_numpy(np.stack(ws))


def source_for(name: str, folder: str | Path, weight: float, split: str = "train", mask=None) -> Source:
    """``LongDocSource`` si la carpeta tiene índice de documentos; si no, la fuente de siempre.
    ``mask``: máscara de pérdida opcional (``train.mask``, un byte por byte)."""
    from hope.data.corpus import ByteData
    folder = Path(folder)
    if (folder / f"{split}.docs.npy").exists():
        return LongDocSource(name, folder, weight, split, mask)
    return Source(name, ByteData(folder / f"{split}.bin"), weight, mask)


def main(argv: list[str] | None = None) -> dict:
    from hope.data.sources import parse_size, wikipedia
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--desde-bin", action="append", default=[], help="nombre=carpeta con train.bin/val.bin")
    ap.add_argument("--repos-bin", action="append", default=[],
                    help="nombre=carpeta de hope.data.repos (repos_codigo): un documento por repositorio")
    ap.add_argument("--wiki", action="append", default=[], help="idioma:bytes (descarga los Parquet)")
    ap.add_argument("--min", default="16K", help="largo mínimo de un documento")
    ap.add_argument("--max", default=None, help="bytes máximos por fuente")
    args = ap.parse_args(argv)
    mn, mx = parse_size(args.min), parse_size(args.max) if args.max else None
    out = {}
    for spec in args.desde_bin:
        name, folder = spec.split("=", 1)
        out[name] = build_long(args.out, name, {sp: iter_docs(Path(folder) / f"{sp}.bin")
                                                for sp in ("train", "val") if (Path(folder) / f"{sp}.bin").exists()},
                               mn, mx)
    for spec in args.repos_bin:
        name, folder = spec.split("=", 1)
        out[name] = build_long(args.out, name, {sp: group_repos(iter_docs(Path(folder) / f"{sp}.bin"))
                                                for sp in ("train", "val") if (Path(folder) / f"{sp}.bin").exists()},
                               mn, mx)
    for spec in args.wiki:
        lang, size = spec.split(":")
        import hashlib
        from hope.data.corpus import split_of

        def docs(split, lang=lang):
            for text, _meta in wikipedia(lang):
                if split_of(hashlib.sha256(text.encode()).hexdigest(), 1.0) == split:
                    yield text
        out[f"wiki_{lang}_largos"] = build_long(args.out, f"wiki_{lang}_largos",
                                                {"train": docs("train"), "val": docs("val")}, mn, parse_size(size))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
