"""Internet como fuente de entrenamiento, con filtrado (ARQUITECTURA_HN1 §10.1.1 y §10.4).

A partir de **temas** (se buscan) o **semillas** (URL directas):

    buscar → leer → neutralizar → reglas de calidad → clasificador → deduplicar → guardar

* Todo acceso a la red pasa por ``hope.tools.web`` y por el guardián (``configs/web.yaml``):
  en modo ``preguntar`` cada página queda como solicitud pendiente; para juntar corpus conviene
  ``lista_permitida`` con pocos dominios de calidad. Se respeta ``robots.txt`` y un ritmo bajo
  (``intervalo_min_por_dominio_s``; por defecto 2 s entre pedidos al mismo dominio).
* **Calidad**: ``hope.data.quality.rules`` (reglas estilo Gopher/C4) y, si se da un modelo,
  el clasificador estilo FineWeb-Edu con su umbral.
* **Deduplicación**: exacta (SHA-256 del texto normalizado) contra lo ya guardado aquí y
  contra los ``provenance.jsonl`` de otras fuentes del corpus (``--existente``), y aproximada
  (similitud de Jaccard de 5-gramas de palabras) dentro de lo recolectado.
* **Procedencia y licencia**: cada documento registra URL, título, fecha, licencia (conocida
  por dominio o declarada en la página; si no se sabe, ``null``) y nota de calidad.
* **Salida**: ``out/train.bin`` y ``out/val.bin`` (bytes UTF-8, byte 0 entre documentos,
  partición determinista por hash), ``provenance.jsonl`` y ``manifest.json``. Es una fuente más
  para ``hope/train/run.py``: ``web: data/pre/web:0.05`` en el plan. Se puede correr de nuevo:
  agrega sin repetir.

Uso::

    python -m hope.data.web_corpus --temas "ley de Ohm" "transistor bipolar" --out data/pre/web \\
        --bytes 20M --modelo-calidad data/calidad.npz --umbral 2.5 --existente data/pre/wiki_es
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from hope.data.corpus import EOD, normalize, split_of
from hope.data.quality import QualityClassifier, rules
from hope.tools.web import REMOVED, WebDenied, WebError


def _shingles(text: str, n: int = 5) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    return {hash(" ".join(words[i:i + n])) for i in range(max(1, len(words) - n + 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def known_hashes(dirs) -> set[str]:
    out: set[str] = set()
    for d in dirs:
        p = Path(d) / "provenance.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("sha256"):
                        out.add(r["sha256"])
    return out


def collect(tools, out: str | Path, *, topics=(), seeds=(), per_topic: int = 5, budget_bytes: int = 20_000_000,
            classifier: QualityClassifier | None = None, threshold: float = 2.5, existing=(),
            val_percent: float = 1.0, known_license_only: bool = False, near_dup: float = 0.8,
            min_chars: int = 400) -> dict:
    """Recolecta, filtra y escribe. ``tools`` es un ``WebTools`` (o un objeto con ``search`` y ``read``)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prov_path = out / "provenance.jsonl"
    seen = known_hashes([out, *existing])
    seen_urls = {json.loads(line)["url"] for line in prov_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()} if prov_path.exists() else set()
    rep = {"urls": 0, "guardados": 0, "bytes": 0, "pendientes": [], "denegados": 0, "errores": 0,
           "reglas": 0, "nota_baja": 0, "duplicados": 0, "casi_duplicados": 0, "sin_licencia": 0,
           "instrucciones_eliminadas": 0}

    urls = list(dict.fromkeys(seeds))
    for t in topics:
        try:
            urls += [r.url for r in tools.search(t, k=per_topic)]
        except WebDenied as e:
            rep["denegados"] += 1
            if e.pending:
                rep["pendientes"].append(e.pending)
        except (WebError, ValueError):
            rep["errores"] += 1
    urls = [u for u in dict.fromkeys(urls) if u not in seen_urls]

    kept_shingles: list[set] = []
    with open(out / "train.bin", "ab") as ftr, open(out / "val.bin", "ab") as fval, \
            open(prov_path, "a", encoding="utf-8") as prov:
        for url in urls:
            if rep["bytes"] >= budget_bytes:
                break
            rep["urls"] += 1
            try:
                page = tools.read(url)
            except WebDenied as e:
                rep["denegados"] += 1
                if e.pending:
                    rep["pendientes"].append(e.pending)
                continue
            except (WebError, ValueError):
                rep["errores"] += 1
                continue
            rep["instrucciones_eliminadas"] += page.removed
            body = "\n".join(line for line in page.text.replace(REMOVED, "").splitlines())
            text = normalize(f"{page.title}\n\n{body}".strip())
            if len(text) < min_chars or not rules(text).keep:
                rep["reglas"] += 1
                continue
            score = float(classifier.score([text])[0]) if classifier is not None else None
            if score is not None and score < threshold:
                rep["nota_baja"] += 1
                continue
            if known_license_only and not page.license:
                rep["sin_licencia"] += 1
                continue
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in seen:
                rep["duplicados"] += 1
                continue
            sh = _shingles(text)
            if any(_jaccard(sh, k) >= near_dup for k in kept_shingles):
                rep["casi_duplicados"] += 1
                continue
            seen.add(h)
            kept_shingles.append(sh)
            raw = text.encode("utf-8") + bytes([EOD])
            split = split_of(h, val_percent)
            (fval if split == "val" else ftr).write(raw)
            prov.write(json.dumps({"fuente": "web", "url": page.url, "titulo": page.title,
                                   "recuperado": page.fetched, "licencia": page.license, "nota_calidad": score,
                                   "sha256": h, "bytes": len(raw), "particion": split}, ensure_ascii=False) + "\n")
            rep["guardados"] += 1
            rep["bytes"] += len(raw)

    manifest = out / "manifest.json"
    prev = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    totals = {"bytes_train": (out / "train.bin").stat().st_size, "bytes_val": (out / "val.bin").stat().st_size,
              "docs": sum(1 for line in prov_path.read_text(encoding="utf-8").splitlines() if line.strip()),
              "corridas": prev.get("corridas", []) + [{"fecha": time.time(), **{k: v for k, v in rep.items()
                                                                                  if k != "pendientes"}}]}
    manifest.write_text(json.dumps(totals, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


def main(argv: list[str] | None = None) -> dict:
    from guardian.cli import load as load_guardian
    from hope.data.sources import parse_size
    from hope.tools.web import load
    ap = argparse.ArgumentParser(description="Corpus de entrenamiento desde internet, filtrado y con procedencia")
    ap.add_argument("--temas", nargs="*", default=[])
    ap.add_argument("--semillas", nargs="*", default=[], help="URL para leer directamente")
    ap.add_argument("--por-tema", type=int, default=5)
    ap.add_argument("--out", default="data/pre/web")
    ap.add_argument("--bytes", default="20M")
    ap.add_argument("--modelo-calidad", default=None, help="clasificador de hope.data.quality (.npz)")
    ap.add_argument("--umbral", type=float, default=2.5)
    ap.add_argument("--existente", nargs="*", default=[], help="carpetas del corpus para deduplicar contra ellas")
    ap.add_argument("--solo-licencia-conocida", action="store_true")
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    args = ap.parse_args(argv)
    tools = load(load_guardian(args.guardian), requested_by="web_corpus")
    clf = QualityClassifier.load(args.modelo_calidad) if args.modelo_calidad else None
    rep = collect(tools, args.out, topics=args.temas, seeds=args.semillas, per_topic=args.por_tema,
                  budget_bytes=parse_size(args.bytes), classifier=clf, threshold=args.umbral,
                  existing=args.existente, known_license_only=args.solo_licencia_conocida)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if rep["pendientes"]:
        print(f"{len(rep['pendientes'])} accesos esperan aprobación: python -m guardian.cli pending")
    return rep


if __name__ == "__main__":
    main()
