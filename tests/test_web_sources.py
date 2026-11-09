"""Fuentes legales con cuota diaria, CORE, licencias por documento e índice de Wikipedia — sin red real."""

import bz2
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from hope.data.web_sources import (Registry, daily, gutenberg_body, iter_mediawiki_dump, iter_stackexchange_posts,
                                   license_allows, wikitext_to_text)
from hope.memory.wiki_index import WikiIndex
from hope.runtime.toolcalls import ToolRunner
from hope.tools.catalog import Catalog
from hope.tools.web import WebClient, WebSettings
from tests.test_web import _guardian

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

PROSE = ("{t} es un tema de física y electrónica que se explica con cuidado en este artículo. La corriente "
         "eléctrica circula por un conductor cuando hay una diferencia de potencial entre sus extremos. "
         "La resistencia limita esa corriente y se mide en ohmios. Los ingenieros usan estas relaciones "
         "para diseñar circuitos seguros, calcular la potencia disipada y elegir componentes adecuados.")


OTHER = ("Un circuito es un camino cerrado por el que pueden moverse cargas. Tiene una fuente, conductores y "
         "componentes como resistores, capacitores o diodos. Se dibuja con símbolos normalizados y se analiza con "
         "las leyes de Kirchhoff: la suma de corrientes en un nodo es cero y la suma de tensiones en una malla "
         "también. Así se calculan corrientes y tensiones en cada rama del circuito.")


def _parquet() -> bytes:
    rows = {"title": [f"Artículo {i}" for i in range(6)], "text": [PROSE.format(t=f"Artículo {i}") for i in range(6)],
            "url": [f"https://es.wikipedia.org/wiki/Art%C3%ADculo_{i}" for i in range(6)]}
    buf = io.BytesIO()
    pq.write_table(pa.table(rows), buf)
    return buf.getvalue()


DUMP = bz2.compress(("""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
<page><title>Ley de Ohm</title><ns>0</ns><revision><text>'''La ley de Ohm''' relaciona [[tensión eléctrica|tensión]],
corriente y resistencia.{{Cita|x}}&lt;ref&gt;Libro&lt;/ref&gt; """ + PROSE.format(t="La ley") + """
== Historia ==
[[Archivo:Ohm.jpg|miniatura]]Georg Ohm la publicó en 1827.</text></revision></page>
<page><title>Usuario:X</title><ns>2</ns><revision><text>nada</text></revision></page>
<page><title>Ohm</title><ns>0</ns><revision><text>#REDIRECCIÓN [[Ley de Ohm]]</text></revision></page>
</mediawiki>""").encode())

ATOM = ("<feed xmlns='http://www.w3.org/2005/Atom' xmlns:arxiv='http://arxiv.org/schemas/atom'><entry>"
        "<id>http://arxiv.org/abs/2401.00001v1</id><title>Learning Circuits From Data</title>"
        "<arxiv:doi>10.1000/circ.1</arxiv:doi><summary>" + PROSE.format(t="El aprendizaje") + "</summary></entry></feed>")

CORE = {"results": [
    {"id": 1, "title": "Open Physics Notes", "doi": "10.2000/a", "license": "https://creativecommons.org/licenses/by/4.0/",
     "abstract": PROSE.format(t="La física abierta")},
    {"id": 2, "title": "Closed Paper", "doi": "10.2000/b", "license": "all rights reserved",
     "abstract": PROSE.format(t="El cerrado")},
    {"id": 3, "title": "Learning circuits from data", "doi": "10.1000/CIRC.1", "license": "CC BY",
     "abstract": PROSE.format(t="Duplicado")},
    {"id": 4, "title": "No License Info", "doi": None, "abstract": PROSE.format(t="Sin licencia")}]}

GUT = ("Title: Un libro viejo\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n" + PROSE.format(t="El libro") * 3
       + "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicencia...")


class _H(BaseHTTPRequestHandler):
    files: dict = {}
    seen: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlsplit(self.path)
        q = parse_qs(u.query)
        _H.seen.append((u.path, dict(self.headers)))
        if u.path == "/robots.txt":
            return self._send(b"User-agent: *\nAllow: /\n", "text/plain")
        if u.path in _H.files:
            data = _H.files[u.path]
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                if start >= len(data):
                    self.send_response(416)
                    self.end_headers()
                    return
                part = data[start:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                self.wfile.write(part)
                return
            return self._send(data, "application/octet-stream")
        if u.path == "/w/api.php":
            cont = q.get("gapcontinue", [""])[0]
            title = {"": "Electrónica básica", "B": "Circuitos"}.get(cont)
            text = PROSE.format(t=title) if cont == "" else OTHER
            body = {"query": {"pages": {"1": {"title": title, "extract": text}}}}
            if cont == "":
                body["continue"] = {"gapcontinue": "B"}
            return self._send(json.dumps(body).encode(), "application/json")
        if u.path == "/arxiv":
            return self._send((ATOM if q["start"] == ["0"] else "<feed xmlns='http://www.w3.org/2005/Atom'/>")
                              .encode(), "application/atom+xml")
        if u.path == "/core":
            if self.headers.get("Authorization") != "Bearer clave-core":
                return self._send(b"{}", "application/json", 401)
            body = CORE if q["offset"] == ["0"] else {"results": []}
            return self._send(json.dumps(body).encode(), "application/json")
        if u.path == "/1/1.txt" or u.path == "/0/1/1-0.txt":
            return self._send(GUT.encode(), "text/plain; charset=utf-8")
        return self._send(b"no", "text/plain", 404)

    def _send(self, data, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def srv():
    _H.files = {"/es-0.parquet": _parquet(), "/eswiki.xml.bz2": DUMP}
    s = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def _registry(srv, tmp_path, **daily_kw):
    def src(name, tipo, metodo, **kw):
        return {"nombre": name, "tipo": tipo, "metodo": metodo, "activo": True, "ritmo_s": 0, "licencia": "CC BY-SA 4.0",
                "terminos": "https://ejemplo.invalid/terminos", "entrenamiento": True, **kw}
    d = {"diario": {"cuota_diaria_mb": 10, "paginas_por_dia": 100, "carpeta": str(tmp_path / "web"),
                    "consolidado": str(tmp_path / "pre"), "indice_wikipedia": str(tmp_path / "wiki.sqlite"),
                    "pausa_entre_fuentes_s": 0, "licencias_entrenamiento": ["CC BY", "CC BY-SA", "CC0", "dominio público"],
                    **daily_kw},
         "fuentes": [
             src("wiki_hf", "hf_parquet", "dump", indexar=True, idioma="es", archivos=[f"{srv}/es-0.parquet"],
                 columna_titulo="title", columna_url="url", cuota={"bytes_por_dia": 10_000_000}),
             src("wiki_dump", "wikimedia_dump", "dump", url=f"{srv}/eswiki.xml.bz2",
                 pagina="https://es.wikipedia.org/wiki/{title}", cuota={"bytes_por_dia": 10_000_000,
                                                                       "paginas_por_dia": 10}),
             src("libros", "mediawiki_api", "api", api=f"{srv}/w/api.php", pagina="https://es.wikibooks.org/wiki/{title}",
                 cuota={"paginas_por_dia": 5}),
             src("arxiv", "arxiv_api", "api", api=f"{srv}/arxiv", categorias=["cs.LG"], licencia="CC0",
                 cuota={"paginas_por_dia": 50}),
             src("core", "core_api", "api", api=f"{srv}/core", entrenamiento="por_documento", licencia="por artículo",
                 consultas=["circuitos"], cuota={"paginas_por_dia": 50}),
             src("gutenberg", "gutenberg", "dump", mirror=srv, desde_id=1, licencia="dominio público",
                 cuota={"paginas_por_dia": 1}),
         ]}
    p = tmp_path / "fuentes.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True))
    return Registry.load(p)


def _client(tmp_path):
    return WebClient(_guardian(tmp_path), WebSettings(timeout=5, min_interval=0, retries=0, cache_dir=None),
                     proxies={}, sleep=lambda s: None)


def test_licenses():
    allowed = ["CC BY", "CC BY-SA", "CC0", "dominio público"]
    assert license_allows("https://creativecommons.org/licenses/by-sa/4.0/", allowed)
    assert license_allows("CC0 1.0", allowed) and license_allows("Public Domain", allowed)
    for bad in ("CC BY-NC 4.0", "CC BY-ND", "all rights reserved", None, ""):
        assert not license_allows(bad, allowed)


def test_text_helpers():
    t = wikitext_to_text("'''Ohm''' fue [[físico|un físico]] {{cita}}<ref>x</ref> [[Archivo:a.jpg|m]].\n== Vida ==\nNació.")
    assert t.startswith("Ohm fue un físico") and "cita" not in t and "Archivo" not in t and "Vida" in t
    pages = list(iter_mediawiki_dump(io.BytesIO(bz2.decompress(DUMP))))
    assert [p[0] for p in pages] == ["Ley de Ohm"]                   # sin usuario ni redirecciones
    assert "START" not in gutenberg_body(GUT) and gutenberg_body("Copyrighted Project Gutenberg eBook ...") is None
    posts = io.BytesIO(b"<posts><row Id='1' PostTypeId='1' Title='Q' Body='&lt;p&gt;Pregunta&lt;/p&gt;' "
                       b"AcceptedAnswerId='2'/><row Id='2' PostTypeId='2' Body='&lt;p&gt;Respuesta&lt;/p&gt;'/>"
                       b"<row Id='3' PostTypeId='1' Title='Sin aceptada' Body='x'/></posts>")
    out = list(iter_stackexchange_posts(posts, "https://electronics.stackexchange.com"))
    assert len(out) == 1 and "Respuesta aceptada:\nRespuesta" in out[0][0]
    assert out[0][1]["url"] == "https://electronics.stackexchange.com/questions/1"


def test_daily_run_end_to_end(srv, tmp_path, monkeypatch):
    monkeypatch.setenv("HOPE_CORE_API_KEY", "clave-core")
    reg = _registry(srv, tmp_path)
    assert [s.method for s in reg.active()][:3] == ["dump", "dump", "dump"]        # dumps primero
    rep = daily(reg, _client(tmp_path), date="2026-11-09", sleep=lambda s: None)
    f = rep["fuentes"]
    assert f["wiki_hf"]["guardados"] == 6 and f["wiki_dump"]["guardados"] == 1
    assert f["libros"]["guardados"] == 2 and f["arxiv"]["guardados"] == 1 and f["gutenberg"]["guardados"] == 1
    core = f["core"]
    assert core["guardados"] == 1 and core["solo_indice"] == 2 and core["duplicados_agregador"] == 1
    day = tmp_path / "web" / "2026-11-09"
    prov = [json.loads(x) for x in (day / "core" / "provenance.jsonl").read_text().splitlines()]
    assert prov[0]["titulo"] == "Open Physics Notes" and "creativecommons" in prov[0]["licencia"]
    idx = [json.loads(x) for x in (day / "core" / "indice.jsonl").read_text().splitlines()]
    assert {i["titulo"] for i in idx} == {"Closed Paper", "No License Info"}
    assert all(not (day / "core" / "train.bin").read_bytes().count(b"cerrado") for _ in [0])
    w = [json.loads(x) for x in (day / "wiki_dump" / "provenance.jsonl").read_text().splitlines()]
    assert w[0]["url"] == "https://es.wikipedia.org/wiki/Ley_de_Ohm" and w[0]["licencia"] == "CC BY-SA 4.0"
    assert (tmp_path / "pre" / "wiki_hf" / "train.bin").stat().st_size > 0      # consolidado para el plan
    # Bot identificado; claves solo en cabeceras, nunca en las URL.
    assert all("HopeHN1" in h.get("User-Agent", "") for _, h in _H.seen)
    assert not any("clave-core" in p for p, _ in _H.seen)
    # Mismo día: no se repite la cuota.
    assert daily(reg, _client(tmp_path), date="2026-11-09", sleep=lambda s: None)["aviso"]
    # Índice local de Wikipedia, consultable sin red.
    wi = WikiIndex(tmp_path / "wiki.sqlite")
    hits = wi.search("Artículo 3 corriente")
    assert hits and hits[0].title == "Artículo 3" and hits[0].license == "CC BY-SA 4.0"
    out = ToolRunner(Catalog(None), wiki=wi).run("buscar_wiki", "Artículo 3")
    assert "«Artículo 3»" in out and "CC BY-SA" in out and "wikipedia.org" in out


def test_resumable_download_and_daily_quota(srv, tmp_path, monkeypatch):
    monkeypatch.delenv("HOPE_CORE_API_KEY", raising=False)
    size = len(_H.files["/es-0.parquet"])
    reg = _registry(srv, tmp_path, cuota_diaria_mb=size / 2 / 1e6)
    reg.sources = [s for s in reg.sources if s.name in ("wiki_hf", "core")]
    r1 = daily(reg, _client(tmp_path), date="2026-11-09", sleep=lambda s: None)
    assert r1["fuentes"]["wiki_hf"]["guardados"] == 0                         # a medio bajar
    part = tmp_path / "web" / "raw" / "wiki_hf" / "es-0.parquet"
    assert 0 < part.stat().st_size <= size // 2 + 1
    r2 = daily(reg, _client(tmp_path), date="2026-11-10", sleep=lambda s: None)
    assert r2["fuentes"]["wiki_hf"]["guardados"] == 0 and part.stat().st_size == size
    ranges = [h.get("Range") for p, h in _H.seen if p == "/es-0.parquet"]
    assert any(r and r.startswith("bytes=") for r in ranges)                   # reanudó, no empezó de nuevo
    r3 = daily(reg, _client(tmp_path), date="2026-11-11", sleep=lambda s: None)
    assert r3["fuentes"]["wiki_hf"]["guardados"] == 6
    assert "HOPE_CORE_API_KEY" in r3["fuentes"].get("core", {}).get("aviso", "HOPE_CORE_API_KEY")


def test_docs_quota_per_source(srv, tmp_path):
    reg = _registry(srv, tmp_path)
    reg.sources = [s for s in reg.sources if s.name == "libros"]
    reg.sources[0].docs_per_day = 1
    rep = daily(reg, _client(tmp_path), date="2026-11-09", sleep=lambda s: None)
    assert rep["fuentes"]["libros"]["guardados"] == 1
    rep = daily(reg, _client(tmp_path), date="2026-11-10", sleep=lambda s: None)
    assert rep["fuentes"]["libros"]["guardados"] == 1                          # sigue desde donde quedó
    titles = [json.loads(x)["titulo"] for x in (tmp_path / "pre" / "libros" / "provenance.jsonl").read_text().splitlines()]
    assert titles == ["Electrónica básica", "Circuitos"]


def test_registry_file_is_users_and_sane():
    reg = Registry.load("configs/fuentes_web.yaml")
    names = {s.name for s in reg.sources}
    assert {"wikipedia_es_hf", "wikipedia_en_hf", "wikipedia_es_dump", "wikilibros_es", "gutenberg", "arxiv",
            "core", "stackexchange", "docs_python", "mdn"} <= names
    core = next(s for s in reg.sources if s.name == "core")
    assert core.trainable == "por_documento" and core.raw["variable_clave"] == "HOPE_CORE_API_KEY"
    text = open("configs/fuentes_web.yaml").read()
    assert "api_key:" not in text.lower() and "bearer" not in text.lower()
    assert all(s.license and s.terms for s in reg.sources)


def test_hope_cannot_modify_sources_registry(tmp_path):
    from guardian.core import Action, Guardian, GuardianConfig
    real = yaml.safe_load(open("configs/guardian.yaml"))
    g = Guardian(GuardianConfig(**{**real, "audit_path": str(tmp_path / "a.jsonl"),
                                   "approvals_path": str(tmp_path / "ap.json")}))
    d = g.decide(Action("file.write", {"path": "configs/fuentes_web.yaml"}))
    assert not d.allowed and not d.needs_human


def test_schedule_window(srv, tmp_path):
    import datetime as dt
    reg = _registry(srv, tmp_path, horario={"desde": 1, "hasta": 7})
    rep = daily(reg, _client(tmp_path), date="2026-11-09", sleep=lambda s: None,
                now=lambda: dt.datetime(2026, 11, 9, 15, 0))
    assert "horario" in rep["aviso"] and not rep["fuentes"]
    rep = daily(reg, _client(tmp_path), date="2026-11-09", sleep=lambda s: None,
                now=lambda: dt.datetime(2026, 11, 9, 3, 0))            # dentro del horario: corre ese mismo día
    assert rep["fuentes"]
