"""Fuentes reales para el preentrenamiento, con procedencia y presupuesto de bytes.

- **Wikipedia** (es, en, …) desde Hugging Face (``wikimedia/wikipedia``, volcado ``20231101``),
  leída en Parquet. Licencia CC BY-SA 4.0: se guarda la procedencia por documento.
- **Código** de carpetas locales (``.py`` por defecto), con la ruta relativa como encabezado.

Cada fuente queda en ``<out>/<nombre>/{train,val}.bin`` (ver ``hope/data/corpus.py``) más
``procedencia.jsonl`` y ``resumen.json``.

Uso::

    python -m hope.data.sources --out data/pre --wiki es:120M --wiki en:80M \\
        --code /usr/lib/python3.13:60M
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

from hope.data.corpus import BinWriter

HF = "https://huggingface.co"
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_DUMP = "20231101"
WIKI_LICENSE = "CC BY-SA 4.0"


def parse_size(s: str) -> int:
    """``"120M"`` → 120_000_000 (también K y G; sin sufijo = bytes)."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMG]?)B?\s*", s.upper())
    if not m:
        raise ValueError(f"tamaño inválido: {s!r}")
    return int(float(m.group(1)) * {"": 1, "K": 10**3, "M": 10**6, "G": 10**9}[m.group(2)])


def split_spec(spec: str) -> tuple[str, int]:
    """``"ruta:60M"`` → (``"ruta"``, 60_000_000)."""
    head, size = spec.rsplit(":", 1)
    return head, parse_size(size)


# --- Wikipedia -------------------------------------------------------------------------------

def _get(url: str, timeout: float = 60.0):
    import requests      # respeta HTTPS_PROXY y REQUESTS_CA_BUNDLE
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - red inestable: reintentar
            if attempt == 4:
                raise
            print(f"  reintento {attempt + 1} ({e})", file=sys.stderr)
            time.sleep(2 ** (attempt + 1))


def wiki_shards(lang: str, dump: str = WIKI_DUMP) -> list[str]:
    """URLs de los Parquet del volcado de Wikipedia en ``lang``."""
    r = _get(f"{HF}/api/datasets/{WIKI_DATASET}/parquet/{dump}.{lang}/train")
    return list(r.json())


def _download(url: str, dest: Path) -> Path:
    r = _get(url, timeout=300)
    with dest.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return dest


def clean_wiki(text: str) -> str:
    """Limpieza mínima: sin líneas vacías repetidas ni espacios al final."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def wiki_docs(lang: str, min_chars: int = 500, max_shards: int | None = None) -> Iterator[tuple[str, dict]]:
    """Documentos ``(texto, procedencia)`` de Wikipedia, shard por shard."""
    import pyarrow.parquet as pq
    shards = wiki_shards(lang)[:max_shards]
    with tempfile.TemporaryDirectory(prefix="wiki_") as tmp:
        for i, url in enumerate(shards):
            path = Path(tmp) / f"{lang}_{i}.parquet"
            print(f"  wiki {lang}: bajando shard {i + 1}/{len(shards)}", file=sys.stderr)
            _download(url, path)
            pf = pq.ParquetFile(path)
            for g in range(pf.num_row_groups):
                tbl = pf.read_row_group(g, columns=["id", "url", "title", "text"]).to_pylist()
                for row in tbl:
                    text = clean_wiki(row["text"] or "")
                    if len(text) < min_chars:
                        continue
                    yield f"{row['title']}\n\n{text}", {"id": row["id"], "titulo": row["title"],
                                                        "url": row["url"], "licencia": WIKI_LICENSE,
                                                        "fuente": f"{WIKI_DATASET}/{WIKI_DUMP}.{lang}"}
            path.unlink()


# --- código local ----------------------------------------------------------------------------

SKIP_DIRS = {"__pycache__", ".git", "site-packages", "dist-packages", "node_modules"}


def code_docs(root: str | Path, exts: tuple[str, ...] = (".py",), max_file: int = 200_000,
              ) -> Iterator[tuple[str, dict]]:
    """Archivos de código ``(texto, procedencia)`` en orden determinista."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix not in exts or p.is_symlink():
                continue
            try:
                if p.stat().st_size > max_file:
                    continue
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.strip():
                continue
            rel = p.relative_to(root).as_posix()
            yield f"# archivo: {rel}\n{text}", {"ruta": rel, "raiz": str(root)}


# --- construcción ----------------------------------------------------------------------------

def build(name: str, docs: Iterator[tuple[str, dict]], out: Path, budget: int, val_every: int = 100) -> dict:
    folder = out / name
    w = BinWriter(folder, budget, val_every)
    t0 = time.time()
    with (folder / "procedencia.jsonl").open("w", encoding="utf-8") as prov:
        for text, meta in docs:
            if not w.add(text):
                break
            prov.write(json.dumps(meta, ensure_ascii=False) + "\n")
    summary = w.close() | {"nombre": name, "presupuesto": budget, "segundos": round(time.time() - t0, 1)}
    (folder / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: {summary['docs']} documentos, {summary['bytes'] / 1e6:.1f} MB", file=sys.stderr)
    return summary


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiki", action="append", default=[], help="idioma:presupuesto, p. ej. es:120M")
    ap.add_argument("--code", action="append", default=[], help="carpeta:presupuesto, p. ej. /usr/lib/python3.13:60M")
    ap.add_argument("--val-cada", type=int, default=100, help="uno de cada N documentos va a validación")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for spec in args.wiki:
        lang, budget = split_spec(spec)
        res.append(build(f"wiki_{lang}", wiki_docs(lang), out, budget, args.val_cada))
    for spec in args.code:
        folder, budget = split_spec(spec)
        if not Path(folder).is_dir():
            raise FileNotFoundError(folder)
        res.append(build(f"codigo_{Path(folder).name}", code_docs(folder), out, budget, args.val_cada))
    return res


if __name__ == "__main__":
    main()
