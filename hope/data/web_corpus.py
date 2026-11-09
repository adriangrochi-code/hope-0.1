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


class CorpusWriter:
    """Calidad → deduplicación → ``train.bin``/``val.bin`` + ``provenance.jsonl`` (+ ``indice.jsonl``).

    ``add(texto, meta)`` devuelve ``"guardado"`` o el motivo del descarte. Lo que no tiene una licencia
    apta para entrenar (``trainable=False``) no entra a los ``.bin``: queda solo su ficha en
    ``indice.jsonl`` (metadatos consultables)."""

    def __init__(self, out: str | Path, *, existing=(), classifier: QualityClassifier | None = None,
                 threshold: float = 2.5, val_percent: float = 1.0, near_dup: float | None = 0.8,
                 near_dup_window: int = 2000, min_chars: int = 400, apply_rules: bool = True):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.prov_path = self.out / "provenance.jsonl"
        self.seen = known_hashes([self.out, *existing])
        self.seen_urls = {json.loads(line).get("url") for line in self.prov_path.read_text(
            encoding="utf-8").splitlines() if line.strip()} if self.prov_path.exists() else set()
        self.classifier, self.threshold, self.val_percent = classifier, threshold, val_percent
        self.near_dup, self.window, self.min_chars, self.apply_rules = near_dup, near_dup_window, min_chars, apply_rules
        self.kept_shingles: list[set] = []
        self.rep = {"guardados": 0, "bytes": 0, "reglas": 0, "nota_baja": 0, "duplicados": 0,
                    "casi_duplicados": 0, "solo_indice": 0}
        self._train = open(self.out / "train.bin", "ab")
        self._val = open(self.out / "val.bin", "ab")
        self._prov = open(self.prov_path, "a", encoding="utf-8")
        self._index = None

    def index_only(self, meta: dict) -> None:
        if self._index is None:
            self._index = open(self.out / "indice.jsonl", "a", encoding="utf-8")
        self._index.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self.rep["solo_indice"] += 1

    def add(self, text: str, meta: dict, *, trainable: bool = True) -> str:
        text = normalize(text.replace(REMOVED, "")).strip()
        if not trainable:
            self.index_only({**meta, "resumen": text[:500]})
            return "solo_indice"
        if len(text) < self.min_chars or (self.apply_rules and not rules(text).keep):
            self.rep["reglas"] += 1
            return "reglas"
        score = float(self.classifier.score([text])[0]) if self.classifier is not None else None
        if score is not None and score < self.threshold:
            self.rep["nota_baja"] += 1
            return "nota_baja"
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in self.seen:
            self.rep["duplicados"] += 1
            return "duplicado"
        if self.near_dup:
            sh = _shingles(text)
            if any(_jaccard(sh, k) >= self.near_dup for k in self.kept_shingles):
                self.rep["casi_duplicados"] += 1
                return "casi_duplicado"
            self.kept_shingles = (self.kept_shingles + [sh])[-self.window:]
        self.seen.add(h)
        raw = text.encode("utf-8") + bytes([EOD])
        split = split_of(h, self.val_percent)
        (self._val if split == "val" else self._train).write(raw)
        self._prov.write(json.dumps({**meta, "nota_calidad": score, "sha256": h, "bytes": len(raw),
                                     "particion": split}, ensure_ascii=False) + "\n")
        if meta.get("url"):
            self.seen_urls.add(meta["url"])
        self.rep["guardados"] += 1
        self.rep["bytes"] += len(raw)
        return "guardado"

    def close(self, extra: dict | None = None) -> dict:
        for f in (self._train, self._val, self._prov, self._index):
            if f is not None:
                f.close()
        manifest = self.out / "manifest.json"
        prev = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        totals = {"bytes_train": (self.out / "train.bin").stat().st_size,
                  "bytes_val": (self.out / "val.bin").stat().st_size,
                  "docs": sum(1 for line in self.prov_path.read_text(encoding="utf-8").splitlines() if line.strip()),
                  "corridas": prev.get("corridas", []) + [{"fecha": time.time(), **self.rep, **(extra or {})}]}
        manifest.write_text(json.dumps(totals, indent=2, ensure_ascii=False), encoding="utf-8")
        return totals


def collect(tools, out: str | Path, *, topics=(), seeds=(), per_topic: int = 5, budget_bytes: int = 20_000_000,
            classifier: QualityClassifier | None = None, threshold: float = 2.5, existing=(),
            val_percent: float = 1.0, known_license_only: bool = False, near_dup: float = 0.8,
            min_chars: int = 400, max_pages: int | None = None) -> dict:
    """Recolecta, filtra y escribe. ``tools`` es un ``WebTools`` (o un objeto con ``search`` y ``read``)."""
    w = CorpusWriter(out, existing=existing, classifier=classifier, threshold=threshold, val_percent=val_percent,
                     near_dup=near_dup, min_chars=min_chars)
    rep = {"urls": 0, "pendientes": [], "denegados": 0, "errores": 0, "sin_licencia": 0,
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
    urls = [u for u in dict.fromkeys(urls) if u not in w.seen_urls]

    for url in urls:
        if w.rep["bytes"] >= budget_bytes or (max_pages is not None and rep["urls"] >= max_pages):
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
        if known_license_only and not page.license:
            rep["sin_licencia"] += 1
            continue
        w.add(f"{page.title}\n\n{page.text}", {"fuente": "web", "url": page.url, "titulo": page.title,
                                              "recuperado": page.fetched, "licencia": page.license})
    w.close({k: v for k, v in rep.items() if k != "pendientes"})
    return {**rep, **w.rep}


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
