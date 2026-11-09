"""Adquisición de datos web por vías legales, identificada y controlada, con cuota diaria.

Decisión del usuario: datos solo de fuentes con licencia conocida, por el método más amable con
los servidores (**dump oficial > API oficial > lectura de páginas**), con un bot que se identifica
y nunca se hace pasar por una persona, y un flujo diario constante que se puede dejar corriendo.

* ``configs/fuentes_web.yaml`` (del usuario): registro de fuentes con tipo, método, licencia,
  términos, ritmo, cuota diaria y si su contenido puede usarse para entrenar.
* Todo pasa por ``hope.tools.web.WebClient`` y por el guardián (``configs/web.yaml``): solo
  GET/HEAD, dominios autorizados, límites por hora y por día, robots.txt, User-Agent con contacto.
* **Cuota diaria** (``--diario``): por fuente y en total (bytes y documentos por día), con pausas.
  Las descargas grandes (dumps, Parquet) son **reanudables**: cada día baja un tramo y sigue.
* **Licencia por documento**: CORE y arXiv la informan por artículo; solo entra a entrenamiento
  lo que tiene una licencia de ``licencias_entrenamiento`` (CC BY, CC BY-SA, CC0, dominio
  público, PSF…). Lo demás queda como ficha consultable en ``indice.jsonl``.
* **Deduplicación**: por hash del texto (también contra lo ya consolidado) y entre agregadores
  de papers por DOI y título normalizado (CORE contra arXiv) en ``data/web/vistos.sqlite``.
* **Salida**: ``data/web/<fecha>/<fuente>/`` (``train.bin``, ``val.bin``, ``provenance.jsonl``,
  ``indice.jsonl``, ``manifest.json``) y la suma por fuente en ``consolidado/<fuente>/`` para el plan
  de entrenamiento. Las fuentes con ``indexar: true`` (Wikipedia) además entran al índice local
  ``hope.memory.wiki_index`` para ``⟦buscar_wiki⟧`` sin red.

Uso::

    python -m hope.data.web_sources --fuentes            # tabla de fuentes, licencias y estado
    python -m hope.data.web_sources --diario              # una corrida (la cuota de hoy)
    python -m hope.data.web_sources --diario --repetir    # queda corriendo: una corrida por día
"""

from __future__ import annotations

import argparse
import bz2
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hope.data.web_corpus import CorpusWriter
from hope.tools.web import WebDenied, WebError, extract_text, neutralize

PRIORITY = {"dump": 0, "api": 1, "paginas": 2}


# ---------------------------------------------------------------------------
# Registro y licencias
# ---------------------------------------------------------------------------
@dataclass
class Source:
    name: str
    kind: str
    method: str
    license: str
    terms: str
    trainable: object                 # True | False | "por_documento"
    rate_s: float
    bytes_per_day: int
    docs_per_day: int
    active: bool = True
    index: bool = False
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        q = d.get("cuota") or {}
        return cls(d["nombre"], d["tipo"], d.get("metodo", "api"), str(d.get("licencia", "")), d.get("terminos", ""),
                   d.get("entrenamiento", False), float(d.get("ritmo_s", 3)), int(q.get("bytes_por_dia", 0)),
                   int(q.get("paginas_por_dia", 0)), bool(d.get("activo", True)), bool(d.get("indexar", False)), d)


@dataclass
class Registry:
    sources: list[Source]
    daily: dict

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls([Source.from_dict(x) for x in d.get("fuentes", [])], d.get("diario", {}))

    def active(self) -> list[Source]:
        return sorted((s for s in self.sources if s.active), key=lambda s: PRIORITY.get(s.method, 9))

    def allowed_licenses(self) -> list[str]:
        return self.daily.get("licencias_entrenamiento", ["CC BY", "CC BY-SA", "CC0", "dominio público"])


CC_PARTS = {"by", "sa", "nc", "nd"}


def license_kind(text: str | None) -> str | None:
    """Normaliza una licencia: «cc0», «pd», «cc by-sa», «cc by-nc»… o el texto en minúsculas."""
    if not text:
        return None
    t = text.lower()
    if re.search(r"publicdomain/zero|cc[\s-]*0\b|cc[\s-]*zero|\bcc0\b", t):
        return "cc0"
    if re.search(r"public\s*domain|dominio\s*p[uú]blico|publicdomain/mark|\bpdm\b", t):
        return "pd"
    t = re.sub(r"creative\s*commons", "cc", t)
    t = re.sub(r"attribution", "by", t)
    t = re.sub(r"share\s*-?\s*alike", "sa", t)
    t = re.sub(r"non\s*-?\s*commercial", "nc", t)
    t = re.sub(r"no\s*-?\s*deriv\w*", "nd", t)
    toks = re.findall(r"[a-z]+", t)
    if "cc" in toks:
        parts = [x for x in toks[toks.index("cc") + 1:] if x in CC_PARTS]
        if parts:
            return "cc " + "-".join(sorted(dict.fromkeys(parts), key=["by", "nc", "nd", "sa"].index))
    return " ".join(re.findall(r"[a-z0-9]+", re.sub(r"\d+\.\d+", " ", t))).strip() or None


def license_allows(license: str | None, allowed: list[str]) -> bool:
    """¿La licencia permite entrenar? Tiene que coincidir con una de ``allowed`` (NC y ND nunca
    pasan salvo que el usuario las ponga explícitamente). Sin licencia conocida: no."""
    kind = license_kind(license)
    if kind is None:
        return False
    for a in allowed:
        ak = license_kind(a)
        if ak and (kind == ak or (not ak.startswith(("cc", "pd")) and kind.startswith(ak))):
            return True
    return False


# ---------------------------------------------------------------------------
# Dedup entre agregadores (DOI y título)
# ---------------------------------------------------------------------------
def norm_title(t: str) -> str:
    return " ".join(re.findall(r"\w+", (t or "").lower()))


class Seen:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS vistos (clave TEXT PRIMARY KEY, fuente TEXT)")

    def keys(self, doi: str | None, title: str | None) -> list[str]:
        out = []
        if doi:
            out.append("doi:" + doi.lower().removeprefix("https://doi.org/").strip())
        if title and len(norm_title(title)) > 10:
            out.append("titulo:" + norm_title(title))
        return out

    def seen(self, doi=None, title=None) -> str | None:
        for k in self.keys(doi, title):
            r = self.db.execute("SELECT fuente FROM vistos WHERE clave = ?", (k,)).fetchone()
            if r:
                return r[0]
        return None

    def add(self, source: str, doi=None, title=None) -> None:
        for k in self.keys(doi, title):
            self.db.execute("INSERT OR IGNORE INTO vistos VALUES (?, ?)", (k, source))
        self.db.commit()


# ---------------------------------------------------------------------------
# Texto de cada formato
# ---------------------------------------------------------------------------
WIKI_DROP = re.compile(r"\{\{[^{}]*\}\}|<ref[^>]*/>|<ref[^>]*>.*?</ref>|<!--.*?-->|\[\[(?:Archivo|File|Imagen|Image|"
                       r"Categoría|Category):[^\]]*\]\]", re.S | re.I)


def wikitext_to_text(w: str) -> str:
    """Wikitexto → texto (aproximado, sin dependencias): plantillas, referencias y archivos fuera."""
    for _ in range(3):                                         # plantillas anidadas
        w = WIKI_DROP.sub("", w)
    w = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", w)       # [[enlace|texto]] → texto
    w = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", w)
    w = re.sub(r"'{2,}", "", w)
    w = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\n\1\n", w, flags=re.M)
    w = re.sub(r"^\s*[{|!].*$", "", w, flags=re.M)             # tablas
    w = re.sub(r"<[^>]+>", "", w)
    return re.sub(r"\n{3,}", "\n\n", w).strip()


def iter_mediawiki_dump(stream, skip: int = 0):
    """Páginas (título, texto) del espacio principal de un dump XML de MediaWiki (en flujo)."""
    n = 0
    title = ns = text = None
    for _, el in ET.iterparse(stream, events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "title":
            title = el.text
        elif tag == "ns":
            ns = el.text
        elif tag == "text":
            text = el.text or ""
        elif tag == "page":
            if ns == "0" and text and not text.lower().startswith(("#redirect", "#redirección")):
                n += 1
                if n > skip:
                    yield title, text
            el.clear()
            title = ns = text = None


def gutenberg_body(text: str) -> str | None:
    """Saca el encabezado y el pie de licencia de Project Gutenberg. None si el libro tiene copyright."""
    head = text[:5000].lower()
    if "copyrighted project gutenberg" in head:
        return None
    m1 = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", text)
    m2 = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", text)
    return text[m1.end() if m1 else 0: m2.start() if m2 else len(text)].strip()


def iter_stackexchange_posts(stream, site: str):
    """(texto, meta) de preguntas con su respuesta aceptada, desde ``Posts.xml`` de un dump de Stack Exchange."""
    questions, answers = {}, {}
    for _, el in ET.iterparse(stream, events=("end",)):
        if el.tag != "row":
            continue
        a = el.attrib
        if a.get("PostTypeId") == "1":
            questions[a["Id"]] = (a.get("Title", ""), a.get("Body", ""), a.get("AcceptedAnswerId"))
        elif a.get("PostTypeId") == "2":
            answers[a["Id"]] = a.get("Body", "")
        el.clear()
    for qid, (title, body, acc) in questions.items():
        if acc and acc in answers:
            q, ans = extract_text(body)[1], extract_text(answers[acc])[1]
            yield (f"{title}\n\nPregunta:\n{q}\n\nRespuesta aceptada:\n{ans}",
                   {"url": f"{site}/questions/{qid}", "titulo": title})


# ---------------------------------------------------------------------------
# Recolectores: cada uno genera (texto, meta, apto_para_entrenar) y avanza su estado
# ---------------------------------------------------------------------------
class Budget:
    def __init__(self, bytes_left: int, docs_left: int):
        self.bytes, self.docs = bytes_left, docs_left

    def ok(self) -> bool:
        return self.bytes > 0 and self.docs > 0


class Collector:
    def __init__(self, client, registry: Registry, raw_dir: Path, seen: Seen, *, sleep=time.sleep):
        self.client, self.registry, self.raw, self.seen, self.sleep = client, registry, raw_dir, seen, sleep

    def _get(self, src: Source, url: str, **kw):
        self.client.settings.min_interval = max(self.client.settings.min_interval, src.rate_s)
        return self.client.get(url, purpose=f"fuente:{src.name}", **kw)

    def _download(self, src: Source, url: str, dest: Path, budget: Budget) -> bool:
        self.client.settings.min_interval = max(self.client.settings.min_interval, src.rate_s)
        r = self.client.download(url, dest, purpose=f"fuente:{src.name}", max_bytes=budget.bytes)
        budget.bytes -= r["bytes"]
        return r["completo"]

    def run(self, src: Source, state: dict, budget: Budget):
        fn = getattr(self, "_" + src.kind, None)
        if fn is None:
            raise ValueError(f"tipo de fuente desconocido: {src.kind}")
        yield from fn(src, state, budget)

    # --- dumps y archivos -------------------------------------------------------
    def _hf_parquet(self, src, state, budget):
        import pyarrow.parquet as pq
        files = src.raw.get("archivos") or []
        if isinstance(files, str):
            files = [files.format(i=i) for i in range(int(src.raw.get("n_archivos", 1)))]
        i = state.get("archivo", 0)
        while i < len(files) and budget.ok():
            dest = self.raw / src.name / Path(urllib.parse.urlsplit(files[i]).path).name
            if not self._download(src, files[i], dest, budget):
                return                                            # sigue mañana
            cols = [src.raw.get("columna_texto", "text")] + [c for c in (src.raw.get("columna_titulo"),
                                                                       src.raw.get("columna_url")) if c]
            table = pq.read_table(dest, columns=cols)
            data = [table.column(c).to_pylist() for c in cols]
            row = state.get("fila", 0)
            for j in range(row, table.num_rows):
                if not budget.ok():
                    state["fila"] = j
                    return
                text = data[0][j]
                title = data[1][j] if len(data) > 1 else ""
                url = data[2][j] if len(data) > 2 else files[i]
                budget.docs -= 1 if src.docs_per_day else 0
                yield (f"{title}\n\n{text}" if title else text), {"url": url, "titulo": title}, True
            i += 1
            state.update(archivo=i, fila=0)

    def _wikimedia_dump(self, src, state, budget):
        dest = self.raw / src.name / Path(urllib.parse.urlsplit(src.raw["url"]).path).name
        if not self._download(src, src.raw["url"], dest, budget):
            return
        skip = state.get("paginas", 0)
        with bz2.open(dest, "rb") as f:
            for title, wt in iter_mediawiki_dump(f, skip=skip):
                if not budget.ok():
                    return
                state["paginas"] = state.get("paginas", 0) + 1
                budget.docs -= 1
                url = src.raw.get("pagina", "").format(title=urllib.parse.quote(title.replace(" ", "_")))
                yield f"{title}\n\n{wikitext_to_text(wt)}", {"url": url, "titulo": title}, True

    def _stackexchange_dump(self, src, state, budget):
        dest = self.raw / src.name / Path(urllib.parse.urlsplit(src.raw["url"]).path).name
        if not self._download(src, src.raw["url"], dest, budget):
            return
        if state.get("procesado"):
            return
        try:
            import py7zr
        except ImportError:
            state["aviso"] = "descargado; falta py7zr (pip install py7zr) para procesarlo"
            return
        out = dest.with_suffix("")
        with py7zr.SevenZipFile(dest) as z:
            z.extract(path=out, targets=["Posts.xml"])
        with open(out / "Posts.xml", "rb") as f:
            for text, meta in iter_stackexchange_posts(f, src.raw.get("sitio", "")):
                yield text, meta, True
        state["procesado"] = True

    def _gutenberg(self, src, state, budget):
        mirror = src.raw["mirror"].rstrip("/")
        book = state.get("libro", int(src.raw.get("desde_id", 1)))
        misses = 0
        while budget.ok() and misses < 20:
            digits = str(book)
            folder = "/".join(digits[:-1]) if len(digits) > 1 else "0"
            text = None
            for name in (f"{book}-0.txt", f"{book}.txt"):
                try:
                    text = self._get(src, f"{mirror}/{folder}/{book}/{name}", respect_robots=True).text
                    break
                except WebError:
                    continue
            book += 1
            state["libro"] = book
            if text is None:
                misses += 1
                continue
            misses = 0
            budget.docs -= 1
            budget.bytes -= len(text.encode())
            body = gutenberg_body(text)
            title = re.search(r"Title:\s*(.+)", text[:3000])
            meta = {"url": f"https://www.gutenberg.org/ebooks/{book - 1}", "titulo": title.group(1).strip() if title else ""}
            if body is None:
                yield "", {**meta, "licencia": "con copyright"}, False
            else:
                yield body, meta, True

    # --- APIs ------------------------------------------------------------------
    def _mediawiki_api(self, src, state, budget):
        cont = state.get("continuar", "")
        while budget.ok():
            q = {"action": "query", "generator": "allpages", "gapnamespace": 0, "gaplimit": 1,
                 "gapfilterredir": "nonredirects", "prop": "extracts", "explaintext": 1, "format": "json"}
            if cont:
                q["gapcontinue"] = cont
            r = self._get(src, src.raw["api"] + "?" + urllib.parse.urlencode(q), respect_robots=False)
            data = json.loads(r.text)
            budget.bytes -= len(r.text.encode())
            for page in data.get("query", {}).get("pages", {}).values():
                budget.docs -= 1
                title = page.get("title", "")
                url = src.raw.get("pagina", "").format(title=urllib.parse.quote(title.replace(" ", "_")))
                yield f"{title}\n\n{neutralize(page.get('extract', ''))[0]}", {"url": url, "titulo": title}, True
            cont = data.get("continue", {}).get("gapcontinue")
            state["continuar"] = cont or ""
            if not cont:
                state["terminado"] = True
                return

    def _arxiv_api(self, src, state, budget):
        cats = src.raw.get("categorias") or ["cs.LG"]
        ci, start = state.get("categoria", 0), state.get("inicio", 0)
        ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        while budget.ok() and ci < len(cats):
            q = {"search_query": f"cat:{cats[ci]}", "start": start, "max_results": 50,
                 "sortBy": "submittedDate", "sortOrder": "descending"}
            r = self._get(src, src.raw["api"] + "?" + urllib.parse.urlencode(q), respect_robots=False)
            budget.bytes -= len(r.text.encode())
            entries = ET.fromstring(r.text).findall("a:entry", ns)
            if not entries:
                ci, start = ci + 1, 0
                state.update(categoria=ci, inicio=0)
                continue
            for e in entries:
                budget.docs -= 1
                title = " ".join(e.findtext("a:title", "", ns).split())
                doi = e.findtext("arxiv:doi", None, ns)
                url = e.findtext("a:id", "", ns).strip()
                self.seen.add("arxiv", doi, title)
                abstract = " ".join(e.findtext("a:summary", "", ns).split())
                yield (f"{title}\n\n{abstract}",
                       {"url": url, "titulo": title, "doi": doi, "licencia": "CC0 (metadatos de arXiv)"}, True)
            start += len(entries)
            state.update(categoria=ci, inicio=start)

    def _core_api(self, src, state, budget):
        key = os.environ.get(src.raw.get("variable_clave", "HOPE_CORE_API_KEY"))
        if not key:
            state["aviso"] = f"falta la clave en la variable {src.raw.get('variable_clave', 'HOPE_CORE_API_KEY')}"
            return
        queries = src.raw.get("consultas") or ["science"]
        qi, offset = state.get("consulta", 0), state.get("desde", 0)
        allowed = self.registry.allowed_licenses()
        while budget.ok() and qi < len(queries):
            q = {"q": queries[qi], "limit": 20, "offset": offset}
            r = self._get(src, src.raw["api"] + "?" + urllib.parse.urlencode(q), respect_robots=False,
                          headers={"Authorization": f"Bearer {key}"})
            budget.bytes -= len(r.text.encode())
            results = json.loads(r.text).get("results", [])
            if not results:
                qi, offset = qi + 1, 0
                state.update(consulta=qi, desde=0)
                continue
            for w in results:
                budget.docs -= 1
                title, doi = w.get("title") or "", w.get("doi")
                lic = w.get("license") or w.get("licence") or (w.get("rights") if isinstance(w.get("rights"), str)
                                                               else None)
                dup = self.seen.seen(doi, title)
                meta = {"url": (w.get("links") or [{}])[0].get("url") if w.get("links") else
                        f"https://core.ac.uk/works/{w.get('id')}", "titulo": title, "doi": doi, "licencia": lic,
                        "core_id": w.get("id")}
                if dup:
                    yield "", {**meta, "duplicado_de": dup}, None
                    continue
                self.seen.add("core", doi, title)
                text = w.get("fullText") or w.get("abstract") or ""
                yield f"{title}\n\n{text}", meta, license_allows(lic, allowed)
            offset += len(results)
            state.update(consulta=qi, desde=offset)

    # --- páginas ----------------------------------------------------------------
    def _paginas(self, src, state, budget):
        from hope.tools.web import WebTools
        tools = WebTools(self.client)
        queue = state.get("cola") or list(src.raw.get("semillas") or [])
        done = set(state.get("hechas") or [])
        prefixes = tuple(src.raw.get("semillas") or [])
        while queue and budget.ok():
            url = queue.pop(0)
            if url in done:
                continue
            done.add(url)
            try:
                page = tools.read(url)
            except (WebError, WebDenied):
                continue
            budget.docs -= 1
            budget.bytes -= len(page.text.encode())
            if src.raw.get("seguir_enlaces"):
                base = prefixes[0].rsplit("/", 1)[0] if prefixes else ""
                queue += [u for u, _ in page.links if u.startswith(base) and u not in done][:50]
            yield f"{page.title}\n\n{page.text}", {"url": page.url, "titulo": page.title}, True
            state.update(cola=queue[:2000], hechas=sorted(done)[-5000:])


# ---------------------------------------------------------------------------
# Corrida diaria
# ---------------------------------------------------------------------------
def daily_bytes(d: dict) -> int:
    return int(float(d["cuota_diaria_mb"]) * 1e6) if "cuota_diaria_mb" in d else int(d.get("bytes_por_dia", 2e9))


def in_window(d: dict, when: dt.datetime) -> bool:
    h = d.get("horario") or {}
    return int(h.get("desde", 0)) <= when.hour < int(h.get("hasta", 24))


def daily(registry: Registry, client, *, date: str | None = None, root: str | Path | None = None,
          classifier=None, sleep=time.sleep, wiki_index=None, now=dt.datetime.now) -> dict:
    d = registry.daily
    root = Path(root or d.get("carpeta", d.get("salida", "data/web")))
    date = date or dt.date.today().isoformat()
    state_path = root / "estado.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("_ultima_corrida") == date:
        return {"fecha": date, "aviso": "la cuota de hoy ya se usó"}
    if not in_window(d, now()):
        return {"fecha": date, "aviso": "fuera del horario permitido (diario.horario)", "fuentes": {}}
    seen = Seen(root / "vistos.sqlite")
    col = Collector(client, registry, root / "raw", seen, sleep=sleep)
    total = Budget(daily_bytes(d), int(d.get("paginas_por_dia", 3000)))
    consolidated = Path(d.get("consolidado", root / "consolidado"))
    allowed = registry.allowed_licenses()
    report = {"fecha": date, "fuentes": {}}
    if wiki_index is None and d.get("indice_wikipedia") and any(s.index for s in registry.active()):
        from hope.memory.wiki_index import WikiIndex
        wiki_index = WikiIndex(d["indice_wikipedia"])
    for i, src in enumerate(registry.active()):
        if not total.ok():
            break
        if not in_window(d, now()):                       # el horario terminó: lo que falta, otro día
            report["aviso"] = "se cortó al terminar el horario permitido (diario.horario)"
            break
        if i:
            sleep(float(d.get("pausa_entre_fuentes_s", 30)))
        st = state.setdefault(src.name, {})
        budget = Budget(min(src.bytes_per_day or total.bytes, total.bytes),
                        min(src.docs_per_day or total.docs, total.docs) if src.docs_per_day else 10**12)
        start_bytes, start_docs = budget.bytes, budget.docs
        w = CorpusWriter(root / date / src.name, existing=[consolidated / src.name], classifier=classifier,
                         threshold=float(d.get("umbral_calidad", 2.5)), near_dup=None if src.method == "dump" else 0.8,
                         min_chars=200)
        rep = {"errores": [], "denegados": 0}
        try:
            for text, meta, trainable in col.run(src, st, budget):
                meta = {"fuente": src.name, "licencia": meta.get("licencia") or src.license,
                        "terminos": src.terms, "recuperado": time.time(), **meta}
                if trainable is None:                               # duplicado de otro agregador
                    rep["duplicados_agregador"] = rep.get("duplicados_agregador", 0) + 1
                    continue
                ok = trainable and (src.trainable is True or (src.trainable == "por_documento"
                                                              and license_allows(meta["licencia"], allowed)))
                if not ok:                                          # solo ficha consultable
                    w.add(text, meta, trainable=False)
                    continue
                if w.add(text, meta) == "guardado" and src.index and wiki_index is not None:
                    title, _, body = text.partition("\n\n")
                    wiki_index.add_article(title, body, lang=src.raw.get("idioma", "es"), url=meta.get("url", ""),
                                           license=meta["licencia"], commit=False)
        except WebDenied as e:
            rep["denegados"] += 1
            rep["pendiente"] = e.pending
            rep["errores"].append(str(e)[:200])
        except (WebError, ValueError, ET.ParseError) as e:
            rep["errores"].append(str(e)[:200])
        totals = w.close()
        if wiki_index is not None:
            wiki_index.db.commit()
        used_bytes, used_docs = start_bytes - budget.bytes, start_docs - budget.docs
        total.bytes -= max(0, used_bytes)
        total.docs -= max(0, used_docs) if src.docs_per_day else 0
        _consolidate(root / date / src.name, consolidated / src.name)
        report["fuentes"][src.name] = {**w.rep, **rep, "bytes_red": used_bytes, "docs_vistos": used_docs
                                       if src.docs_per_day else None, "bytes_train_dia": totals["bytes_train"],
                                       **({"aviso": st["aviso"]} if st.get("aviso") else {})}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    state["_ultima_corrida"] = date
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / date / "informe.json").parent.mkdir(parents=True, exist_ok=True)
    (root / date / "informe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _consolidate(day_dir: Path, dest: Path) -> None:
    """Suma lo del día a la carpeta acumulada de la fuente (lo que lee el plan de entrenamiento)."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("train.bin", "val.bin", "provenance.jsonl", "indice.jsonl"):
        src = day_dir / name
        if src.exists() and src.stat().st_size:
            with open(dest / name, "ab") as out, open(src, "rb") as f:
                shutil.copyfileobj(f, out)


def table(registry: Registry) -> list[dict]:
    return [{"fuente": s.name, "tipo": s.kind, "metodo": s.method, "licencia": s.license,
             "entrenamiento": s.trainable, "ritmo_s": s.rate_s, "gb_dia": round(s.bytes_per_day / 1e9, 3),
             "docs_dia": s.docs_per_day, "activa": s.active, "terminos": s.terms} for s in registry.sources]


def main(argv: list[str] | None = None) -> dict | list:
    ap = argparse.ArgumentParser(description="Datos web legales con cuota diaria")
    ap.add_argument("--config", default="configs/fuentes_web.yaml")
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    ap.add_argument("--diario", action="store_true", help="traer la cuota de hoy")
    ap.add_argument("--repetir", action="store_true", help="con --diario: una corrida por día, sin parar")
    ap.add_argument("--fuentes", action="store_true", help="mostrar el registro de fuentes")
    args = ap.parse_args(argv)
    reg = Registry.load(args.config)
    if args.fuentes or not args.diario:
        rows = table(reg)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return rows
    from guardian.cli import load as load_guardian
    from hope.tools.web import WebClient
    client = WebClient(load_guardian(args.guardian), requested_by="web_sources")
    clf = None
    if reg.daily.get("modelo_calidad"):
        from hope.data.quality import QualityClassifier
        clf = QualityClassifier.load(reg.daily["modelo_calidad"])
    h = reg.daily.get("horario") or {}
    while True:
        if args.repetir and not in_window(reg.daily, dt.datetime.now()):
            time.sleep(600)                                  # esperar el horario permitido
            continue
        rep = daily(reg, client, classifier=clf)
        print(json.dumps(rep, indent=2, ensure_ascii=False), flush=True)
        if not args.repetir:
            return rep
        start = int(h.get("desde", 0))
        tomorrow = dt.datetime.combine(dt.date.today() + dt.timedelta(days=1), dt.time(start % 24, 17))
        time.sleep(max(60.0, (tomorrow - dt.datetime.now()).total_seconds()))


if __name__ == "__main__":
    main()
