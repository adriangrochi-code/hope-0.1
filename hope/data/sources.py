"""Fuentes del corpus y construcción de ``train.bin`` / ``val.bin`` por presupuesto de bytes.

* ``wikipedia(lang)`` — artículos de Wikipedia (español o inglés) desde los archivos
  Parquet de ``wikimedia/wikipedia`` en Hugging Face (licencia CC BY-SA; se registra la
  procedencia de cada documento).
* ``code_files(paths)`` — código fuente de carpetas locales.
* ``build(out, [(nombre, documentos, bytes_max), ...])`` — aplica las reglas de calidad
  (``hope.data.quality``) y la deduplicación exacta, parte en entrenamiento/validación
  de forma determinista y escribe una carpeta por fuente (``out/<nombre>/train.bin``),
  lista para ``--source nombre=out/nombre:peso`` en ``hope.train.continual``.

Uso::

    python -m hope.data.sources --out data/pre --wiki es:300M --wiki en:200M --code /usr/lib/python3.13:80M
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from hope.data.corpus import EOD, normalize, split_of
from hope.data.quality import rules

HF_WIKI = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.{lang}/train-{i:05d}-of-{n:05d}.parquet"
WIKI_SHARDS = {"es": 13, "en": 41}
CODE_EXT = {".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".rs", ".go", ".js", ".ts", ".java", ".cs", ".ino",
            ".sh", ".lua", ".kt", ".swift", ".m", ".scad", ".gcode_tpl"}


def parse_size(text: str) -> int:
    t = text.strip().upper()
    mult = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}.get(t[-1], 1)
    return int(float(t[:-1] if t[-1] in "KMGT" else t) * mult)


def wikipedia(lang: str, shards: list[int] | None = None, cache: str | Path = "data/raw/wiki"):
    """Genera (texto, procedencia). Descarga los Parquet una vez y los guarda en ``cache``."""
    import pyarrow.parquet as pq
    n = WIKI_SHARDS[lang]
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for i in shards if shards is not None else range(n):
        path = cache / f"{lang}-{i:05d}.parquet"
        if not path.exists():
            url = HF_WIKI.format(lang=lang, i=i, n=n)
            with urllib.request.urlopen(url, timeout=600) as r:
                path.write_bytes(r.read())
        table = pq.read_table(path, columns=["title", "text", "url"])
        for title, text, url in zip(*(table.column(c).to_pylist() for c in ("title", "text", "url"))):
            yield f"{title}\n\n{text}", {"fuente": f"wikipedia-{lang}", "url": url, "licencia": "CC BY-SA 4.0"}


HF_ROWS = ("https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia&config=20231101.{lang}"
           "&split=train&offset={offset}&length={length}")
WIKI_ROWS = {"es": 1_841_155, "en": 6_407_814}


def wikipedia_rows(lang: str, pages: int = 40, per_page: int = 100, seed: int = 0):
    """Muestra liviana de Wikipedia sin descargar los Parquet (API de filas de Hugging Face).

    Toma ``pages`` páginas de ``per_page`` artículos repartidas a lo largo de todo el volcado
    (no solo los primeros títulos en orden alfabético). Sirve para corpus chicos de experimento
    (``hope.bench.tokens_vs_bytes``); para preentrenar de verdad, ``wikipedia``."""
    import random
    import time
    total = WIKI_ROWS[lang]
    rnd = random.Random(seed)
    step = total // max(pages, 1)
    for i in range(pages):
        offset = i * step + rnd.randrange(max(step - per_page, 1))
        url = HF_ROWS.format(lang=lang, offset=offset, length=per_page)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    rows = json.loads(r.read())["rows"]
                break
            except Exception:                                   # noqa: BLE001 — red inestable: reintentar
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        for row in rows:
            d = row["row"]
            yield f"{d['title']}\n\n{d['text']}", {"fuente": f"wikipedia-{lang}", "url": d["url"],
                                                    "licencia": "CC BY-SA 4.0"}


def code_files(paths: list[str | Path], max_file: int = 200_000):
    for root in paths:
        for f in sorted(Path(root).rglob("*")):
            if f.is_file() and f.suffix.lower() in CODE_EXT and f.stat().st_size <= max_file:
                text = f.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    yield f"# archivo: {f.name}\n{text}", {"fuente": "codigo", "ruta": str(f)}


def build(out: str | Path, sources: list[tuple[str, object, int]], *, val_percent: float = 1.0,
          apply_rules: bool = True) -> dict:
    out = Path(out)
    summary = {}
    seen: set[str] = set()
    for name, docs, budget in sources:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        files = {"train": open(d / "train.bin", "wb"), "val": open(d / "val.bin", "wb")}
        prov = open(d / "provenance.jsonl", "w", encoding="utf-8")
        written = {"train": 0, "val": 0}
        stats = {"docs": 0, "descartados_reglas": 0, "duplicados": 0}
        for text, meta in docs:
            text = normalize(text)
            if apply_rules and not rules(text).keep:
                stats["descartados_reglas"] += 1
                continue
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in seen:
                stats["duplicados"] += 1
                continue
            seen.add(h)
            split = split_of(h, val_percent)
            raw = text.encode("utf-8") + bytes([EOD])
            files[split].write(raw)
            written[split] += len(raw)
            stats["docs"] += 1
            prov.write(json.dumps({**meta, "sha256": h, "bytes": len(raw), "particion": split},
                                  ensure_ascii=False) + "\n")
            if written["train"] + written["val"] >= budget:
                break
        for f in files.values():
            f.close()
        prov.close()
        summary[name] = {**stats, "bytes_train": written["train"], "bytes_val": written["val"]}
        (d / "manifest.json").write_text(json.dumps(summary[name], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Construye el corpus por fuentes con presupuesto de bytes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiki", action="append", default=[], help="idioma:bytes, p. ej. es:300M")
    ap.add_argument("--code", action="append", default=[], help="carpeta:bytes (repetible)")
    ap.add_argument("--wiki-rows", action="append", default=[],
                    help="idioma:bytes con la API de filas (sin pyarrow; muestra repartida en todo el volcado)")
    ap.add_argument("--val-percent", type=float, default=1.0)
    args = ap.parse_args(argv)
    sources = []
    for spec in args.wiki:
        lang, size = spec.split(":")
        sources.append((f"wiki_{lang}", wikipedia(lang), parse_size(size)))
    for spec in args.wiki_rows:
        lang, size = spec.split(":")
        n = parse_size(size)
        sources.append((f"wiki_{lang}", wikipedia_rows(lang, pages=max(4, n // 250_000)), n))
    for spec in args.code:
        folder, size = spec.rsplit(":", 1)
        sources.append((f"codigo_{Path(folder).name}", code_files([folder]), parse_size(size)))
    print(json.dumps(build(args.out, sources, val_percent=args.val_percent), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
