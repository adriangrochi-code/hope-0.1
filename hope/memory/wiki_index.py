"""Wikipedia completa (es + en) consultable sin red: ``⟦buscar_wiki: consulta⟧``.

``MemoryStore`` carga todos sus vectores en RAM: sirve para lo que Hope aprende, no para
millones de fragmentos. Este índice vive **en disco** (SQLite) y no carga nada entero:

* **Fragmentos con procedencia**: cada artículo se parte en trozos de ~1000 caracteres por
  párrafos; cada trozo guarda título, sección, URL, idioma y licencia (CC BY-SA 4.0: la
  atribución es el título y la URL del artículo, que se devuelven siempre con el texto).
* **Índice de texto completo FTS5** (BM25, sin acentos, con prefijos) para encontrar candidatos.
* **Vectores livianos**: n-gramas de caracteres con hashing (``HashingEmbedder``) de 64
  dimensiones en float16 (128 bytes por fragmento) para reordenar los candidatos por similitud.
* Ingesta por lotes y reanudable (un artículo ya indexado no se repite).

Tamaño orientativo: Wikipedia es+en en texto ≈ 25 GB → ≈ 25 M fragmentos; el índice ocupa
del orden del texto más un 30–50 %. Se puede indexar solo una parte (``--max-articulos``).

Uso::

    python -m hope.memory.wiki_index ingerir --parquet data/raw/wiki/es-*.parquet --lang es --out data/wiki.sqlite
    python -m hope.memory.wiki_index buscar --db data/wiki.sqlite "ley de Ohm"
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hope.memory.embed import HashingEmbedder

LICENSE = "CC BY-SA 4.0"
SCHEMA = """
CREATE TABLE IF NOT EXISTS articulos (id INTEGER PRIMARY KEY, titulo TEXT, lang TEXT, url TEXT, licencia TEXT,
                                      UNIQUE(lang, titulo));
CREATE TABLE IF NOT EXISTS fragmentos (id INTEGER PRIMARY KEY, articulo INTEGER, seccion TEXT, texto TEXT,
                                       vector BLOB);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(texto, titulo, content='', tokenize='unicode61 remove_diacritics 2',
                                                  prefix='3');
"""


@dataclass
class Hit:
    title: str
    section: str
    text: str
    url: str
    lang: str
    license: str
    score: float


def chunk(text: str, size: int = 1000) -> list[tuple[str, str]]:
    """(sección, texto) por párrafos, sin cortar a mitad de párrafo salvo que sea enorme."""
    out, section, buf = [], "", ""
    for para in re.split(r"\n\s*\n", text):
        p = para.strip()
        if not p:
            continue
        m = re.fullmatch(r"=+\s*(.+?)\s*=+", p) or (re.fullmatch(r"[^\n.]{2,80}", p) if len(p) < 80 else None)
        if m and "\n" not in p:
            if buf:
                out.append((section, buf))
                buf = ""
            section = m.group(1) if m.lastindex else p
            continue
        while len(p) > size * 2:
            cut = p.rfind(". ", 0, size) + 1 or size
            out.append((section, (buf + " " + p[:cut]).strip()))
            buf, p = "", p[cut:].strip()
        if len(buf) + len(p) > size and buf:
            out.append((section, buf))
            buf = ""
        buf = (buf + "\n\n" + p).strip()
    if buf:
        out.append((section, buf))
    return out


def _fts_query(q: str) -> str:
    """Palabras como prefijos (plural, género y derivaciones: «transistores» → «transis*»)."""
    words = re.findall(r"\w+", unicodedata.normalize("NFC", q.lower()))
    terms = [f'"{w[:max(4, len(w) - 3)]}" *' if len(w) > 5 else f'"{w}"' for w in words if len(w) > 1]
    return " OR ".join(terms) or '""'


class WikiIndex:
    def __init__(self, path: str | Path, dim: int = 64):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.executescript(SCHEMA)
        self.embedder = HashingEmbedder(dim=dim)

    def add_article(self, title: str, text: str, *, lang: str, url: str, license: str = LICENSE,
                    commit: bool = True) -> int:
        """Agrega un artículo partido en fragmentos. Devuelve cuántos fragmentos (0 si ya estaba)."""
        try:
            cur = self.db.execute("INSERT INTO articulos (titulo, lang, url, licencia) VALUES (?,?,?,?)",
                                  (title, lang, url, license))
        except sqlite3.IntegrityError:
            return 0
        aid = cur.lastrowid
        parts = chunk(text)
        if not parts:
            return 0
        vecs = self.embedder.embed([f"{title} {s} {t[:600]}" for s, t in parts]).astype(np.float16)
        for (section, frag), v in zip(parts, vecs):
            fid = self.db.execute("INSERT INTO fragmentos (articulo, seccion, texto, vector) VALUES (?,?,?,?)",
                                  (aid, section, frag, v.tobytes())).lastrowid
            self.db.execute("INSERT INTO fts (rowid, texto, titulo) VALUES (?,?,?)", (fid, frag, title))
        if commit:
            self.db.commit()
        return len(parts)

    def ingest(self, docs, *, lang: str, batch: int = 500, max_articles: int | None = None) -> dict:
        """``docs``: iterable de (texto, meta) como ``hope.data.sources.wikipedia`` (título en la 1.ª línea)."""
        n = frags = 0
        for text, meta in docs:
            title, _, body = text.partition("\n\n")
            frags += self.add_article(title.strip(), body, lang=lang, url=meta.get("url", ""),
                                      license=meta.get("licencia", LICENSE), commit=False)
            n += 1
            if n % batch == 0:
                self.db.commit()
            if max_articles and n >= max_articles:
                break
        self.db.commit()
        return {"articulos": n, "fragmentos": frags}

    def search(self, query: str, k: int = 3, candidates: int = 50, lang: str | None = None) -> list[Hit]:
        rows = self.db.execute(
            "SELECT f.id, f.seccion, f.texto, f.vector, a.titulo, a.url, a.lang, a.licencia, bm25(fts) "
            "FROM fts JOIN fragmentos f ON f.id = fts.rowid JOIN articulos a ON a.id = f.articulo "
            "WHERE fts MATCH ? " + ("AND a.lang = ? " if lang else "") + "ORDER BY bm25(fts) LIMIT ?",
            (_fts_query(query), *([lang] if lang else []), candidates)).fetchall()
        if not rows:
            return []
        q = self.embedder.embed([query])[0]
        bm = np.array([-r[8] for r in rows], dtype=np.float32)
        bm = (bm - bm.min()) / (bm.max() - bm.min() + 1e-9)
        sims = np.array([float(np.frombuffer(r[3], dtype=np.float16).astype(np.float32) @ q) for r in rows])
        title_bonus = np.array([0.3 if r[4].lower() in query.lower() or query.lower() in r[4].lower() else 0.0
                                for r in rows])
        score = 0.6 * bm + 0.4 * sims + title_bonus
        order = np.argsort(-score)[:k]
        return [Hit(rows[i][4], rows[i][1], rows[i][2], rows[i][5], rows[i][6], rows[i][7], float(score[i]))
                for i in order]

    def search_text(self, query: str, k: int = 2) -> str:
        """Para ``⟦buscar_wiki⟧``: fragmentos con título, URL y licencia (la atribución de CC BY-SA)."""
        from hope.tools.web import clip
        hits = self.search(query, k=k)
        if not hits:
            return "no encontrado en la Wikipedia local"
        return " | ".join(f"«{h.title}»{' § ' + h.section if h.section else ''} <{h.url}> ({h.license}): "
                          f"{clip(h.text, 220)}" for h in hits)

    def stats(self) -> dict:
        one = lambda q: self.db.execute(q).fetchone()[0]  # noqa: E731
        return {"articulos": one("SELECT COUNT(*) FROM articulos"), "fragmentos": one("SELECT COUNT(*) FROM fragmentos"),
                "bytes": self.path.stat().st_size if self.path.exists() else 0}


def main(argv: list[str] | None = None) -> None:
    import json
    ap = argparse.ArgumentParser(description="Wikipedia local consultable (FTS5 + vectores livianos)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingerir")
    a.add_argument("--parquet", nargs="+", required=True)
    a.add_argument("--lang", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--max-articulos", type=int, default=None)
    b = sub.add_parser("buscar")
    b.add_argument("--db", required=True)
    b.add_argument("consulta")
    args = ap.parse_args(argv)
    if args.cmd == "ingerir":
        import pyarrow.parquet as pq
        idx = WikiIndex(args.out)

        def docs():
            for f in args.parquet:
                t = pq.read_table(f, columns=["title", "text", "url"])
                for title, text, url in zip(*(t.column(c).to_pylist() for c in ("title", "text", "url"))):
                    yield f"{title}\n\n{text}", {"url": url}
        print(json.dumps(idx.ingest(docs(), lang=args.lang, max_articles=args.max_articulos), ensure_ascii=False))
    else:
        for h in WikiIndex(args.db).search(args.consulta, k=5):
            print(f"{h.score:.3f} {h.title} § {h.section} <{h.url}>\n  {h.text[:200]}")


if __name__ == "__main__":
    main()
