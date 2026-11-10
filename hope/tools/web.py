"""Búsqueda y lectura en internet: ``⟦buscar_web: consulta⟧`` y ``⟦leer_web: url⟧``.

Nivel 6 del plan de herramientas (docs/HERRAMIENTAS_HN1.md §3 y §7). Reglas:

* **Permiso del usuario.** Cada acceso a la red es la acción ``network.access`` (nivel 3) y lo
  decide el guardián con ``configs/web.yaml`` (modo ``preguntar`` / ``lista_permitida`` /
  ``apagado``, dominios, consultas por hora, bytes por día). Sin guardián no hay red. Si el
  acceso queda pendiente, la herramienta devuelve el id de la solicitud; cuando una persona la
  aprueba, el mismo pedido se puede repetir y usa esa aprobación (una vez).
* **Redirecciones** no se siguen solas: cada destino nuevo vuelve a pasar por el guardián
  (una redirección no sirve para salir de la lista permitida).
* **Contenido no confiable.** Lo que viene de la web es un **dato**, nunca una instrucción:
  se extrae solo el texto (sin scripts ni estilos; nada se ejecuta), se eliminan las frases que
  intentan darle órdenes al modelo («ignorá lo anterior…»), los delimitadores de llamadas a
  herramientas y los marcadores de plantilla de chat, se recorta y se devuelve marcado
  ``⟪dato web no confiable · URL⟫ … ⟪fin⟫``.
* **Límites**: tamaño máximo por página (y lo que quede del día), tiempo máximo, reintentos
  con espera creciente solo ante errores transitorios (red, 429, 5xx), ritmo mínimo por
  dominio y ``robots.txt`` para las páginas (las API públicas se usan según sus normas, con
  un User-Agent identificable).
* **Caché local en disco**: lo ya leído no vuelve a salir a la red mientras no venza.
* **Procedencia**: lo leído se puede guardar en la memoria como OBSERVADO con la URL, enlazado
  en el grafo; un dato que otra fuente da distinto queda como contradicción registrada.

Buscadores sin clave: API de Wikipedia, API de arXiv, DuckDuckGo HTML y SearxNG (instancia
configurable). Se elige en ``configs/web.yaml`` o con prefijo: ``⟦buscar_web: arxiv: mamba ssm⟧``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from guardian.core import Action
from guardian.web import READ_METHODS
from hope.tools.catalog import ToolDenied, ToolSpec

OPEN_MARK, CLOSE_MARK = "⟪dato web no confiable · {url}⟫", "⟪fin⟫"
SNIPPET_CHARS = 420                    # cabe en el resultado de una llamada (toolcalls.MAX_OUT = 600)
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/atom+xml", "application/xhtml+xml",
              "application/rss+xml")

DEFAULT_URLS = {
    "wikipedia": "https://{lang}.wikipedia.org/w/api.php",
    "wikipedia_pagina": "https://{lang}.wikipedia.org/wiki/{title}",
    "arxiv": "https://export.arxiv.org/api/query",
    "duckduckgo": "https://html.duckduckgo.com/html/",
    "brave": "https://api.search.brave.com/res/v1/web/search",
    "google_cse": "https://www.googleapis.com/customsearch/v1",
}

# Licencias conocidas por dominio (para el corpus; None = desconocida, se registra así).
KNOWN_LICENSES = {
    "wikipedia.org": "CC BY-SA 4.0",
    "stackoverflow.com": "CC BY-SA 4.0",
    "docs.python.org": "PSF-2.0 (documentación de Python)",
    "developer.mozilla.org": "CC BY-SA 2.5",
}


class WebError(RuntimeError):
    pass


class WebDenied(ToolDenied):
    def __init__(self, reasons, pending: str | None = None):
        self.reasons, self.pending = tuple(reasons), pending
        msg = "; ".join(self.reasons)
        if pending:
            msg += (f" — solicitud {pending} pendiente de aprobación humana "
                    f"(python -m guardian.cli approve {pending} --by <nombre>)")
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Configuración (la parte de comportamiento de configs/web.yaml)
# ---------------------------------------------------------------------------
@dataclass
class WebSettings:
    backend: str = "wikipedia"
    lang: str = "es"
    searxng_url: str | None = None
    user_agent: str = "HopeHN1/0.1 (asistente de investigación local; respeta robots.txt)"
    timeout: float = 15.0
    max_page_bytes: int = 2_000_000
    retries: int = 2
    retry_wait: float = 1.0
    min_interval: float = 2.0
    cache_dir: str | None = "runs/web/cache"
    cache_days: float = 7.0
    respect_robots: bool = True
    urls: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, d: dict) -> "WebSettings":
        lim = d.get("limites") or {}
        return cls(backend=str(d.get("buscador", "wikipedia")), lang=str(d.get("idioma_wikipedia", "es")),
                   searxng_url=d.get("searxng_url"),
                   user_agent=str(d.get("user_agent", cls.user_agent)),
                   timeout=float(lim.get("segundos_por_pedido", d.get("tiempo_max_s", 15))),
                   max_page_bytes=int(float(lim["mb_por_pagina"]) * 1e6) if "mb_por_pagina" in lim
                   else int(d.get("max_bytes_por_pagina", 2e6)),
                   respect_robots=bool(d.get("respetar_robots", True)),
                   retries=int(d.get("reintentos", 2)), retry_wait=float(d.get("espera_reintento_s", 1.0)),
                   min_interval=float(d.get("intervalo_min_por_dominio_s", 2.0)),
                   cache_dir=d.get("cache_dir", "runs/web/cache"), cache_days=float(d.get("cache_dias", 7)),
                   urls=dict(d.get("urls_buscadores") or {}))

    def url(self, name: str) -> str:
        return self.urls.get(name) or DEFAULT_URLS[name]


# ---------------------------------------------------------------------------
# Texto: extracción de HTML y neutralización de instrucciones
# ---------------------------------------------------------------------------
SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "canvas", "template", "object", "embed",
             "nav", "footer", "header", "form", "button", "select", "aside"}
BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section",
              "article", "pre", "blockquote", "dd", "dt", "hr", "main", "figcaption"}


class _Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.skip = 0
        self.in_title = False
        self.license: str | None = None
        self.links: list[tuple[str, str]] = []          # (href, texto)
        self._a: list | None = None

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "a" and a.get("href") and not self.skip:
            self._a = [a["href"], []]
        if tag in ("link", "a") and "license" in a.get("rel", "").lower().split() and a.get("href"):
            self.license = self.license or a["href"]
        if tag == "meta" and a.get("name", "").lower() in ("license", "dc.rights", "dcterms.license"):
            self.license = self.license or a.get("content") or None
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag in SKIP_TAGS:
            self.skip -= 1

    def handle_endtag(self, tag):
        if tag == "a" and self._a is not None:
            self.links.append((self._a[0], " ".join("".join(self._a[1]).split())))
            self._a = None
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self.in_title = False
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        elif not self.skip:
            self.parts.append(data)
            if self._a is not None:
                self._a[1].append(data)


def extract_links(html: str, base: str) -> list[tuple[str, str]]:
    """Enlaces http(s) de la página como (url absoluta, texto neutralizado), sin repetir."""
    p = _Extractor()
    try:
        p.feed(html)
        p.close()
    except Exception:                                   # noqa: BLE001
        pass
    out, seen = [], set()
    for href, text in p.links:
        u = urllib.parse.urljoin(base, href).split("#")[0]
        if urllib.parse.urlsplit(u).scheme in ("http", "https") and u not in seen:
            seen.add(u)
            out.append((u, neutralize(text)[0]))
    return out


def extract_text(html: str) -> tuple[str, str, str | None]:
    """HTML → (título, texto limpio, licencia declarada). Nunca ejecuta nada: solo lee el texto."""
    p = _Extractor()
    try:
        p.feed(html)
        p.close()
    except Exception:                                   # noqa: BLE001 — HTML roto: lo que se haya podido leer
        pass
    lines = [" ".join(line.split()) for line in "".join(p.parts).splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return " ".join("".join(p.title).split()), text, p.license


# Frases que le hablan al modelo en lugar de informar. Se elimina la oración entera.
INJECTION_PATTERNS = [
    r"\b(ignor[aáe]\w*|olvid[aáe]\w*|descart[aáe]\w*|omit[ií]\w*)\b.{0,40}\b(instruccion\w*|indicacion\w*|"
    r"órdenes|ordenes|reglas|lo anterior|todo lo previo|lo que te dijeron)",
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|prompts?|rules|above|previous|prior|"
    r"everything)",
    r"\b(ahora (sos|eres|serás|vas a ser)|a partir de ahora (sos|eres|respond\w*|actu\w*|ten[eé]s que))",
    r"\b(you are now|from now on,? you|act as (an?|the)\b|pretend to be|you must now)",
    r"\b(nuevas instrucciones|new instructions|instrucciones del sistema|system prompt|mensaje del sistema|"
    r"prompt del sistema|developer mode|modo desarrollador|jailbreak)",
    r"\b(asistente|modelo|ia|ai|assistant|language model|llm|hope)\b.{0,30}\b(deb[eé]s?|ten[eé]s que|tienes que|"
    r"must|should|shall)\b.{0,60}\b(responder|decir|escribir|ejecutar|enviar|revelar|reply|say|write|"
    r"execute|run|send|reveal)",
    r"\b(ejecut[aáe]\w*|corr[eé]\w*|run|execute)\b.{0,20}\b(este|el siguiente|this|the following)\b.{0,15}"
    r"\b(código|comando|script|code|command)",
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)
TEMPLATE_TOKENS = re.compile(r"<\|[^|>]{1,40}\|>|\[/?INST\]|<</?SYS>>|^\s*#{2,}\s*(instruction|system|"
                             r"assistant|user)\b.*$|^\s*(system|assistant|user)\s*:", re.IGNORECASE | re.MULTILINE)
REMOVED = "[instrucción eliminada]"


def neutralize(text: str) -> tuple[str, int]:
    """Quita del texto web lo que intenta dirigir al modelo. Devuelve (texto, oraciones eliminadas)."""
    # Delimitadores propios: el contenido no puede abrir llamadas a herramientas, simular un
    # resultado («→») ni cerrar antes de tiempo el recuadro de dato no confiable.
    text = text.replace("⟦", "[").replace("⟧", "]").replace("⟪", "«").replace("⟫", "»").replace("→", "->")
    text, n_tokens = TEMPLATE_TOKENS.subn(" ", text)
    removed = 0
    out_lines = []
    for line in text.split("\n"):
        sentences = re.split(r"(?<=[.!?¡¿;])\s+", line)
        kept = []
        for s in sentences:
            if INJECTION_RE.search(s):
                removed += 1
                if not kept or kept[-1] != REMOVED:
                    kept.append(REMOVED)
            else:
                kept.append(s)
        out_lines.append(" ".join(kept))
    return "\n".join(out_lines), removed + n_tokens


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut + " …"


def format_untrusted(text: str, url: str, max_chars: int = SNIPPET_CHARS) -> str:
    """Recuadro de dato no confiable, ya neutralizado y recortado, con su procedencia."""
    clean, _ = neutralize(text)
    return f"{OPEN_MARK.format(url=url)} {clip(clean, max_chars)} {CLOSE_MARK}"


def license_for(url: str, declared: str | None = None) -> str | None:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    for dom, lic in KNOWN_LICENSES.items():
        if host == dom or host.endswith("." + dom):
            return lic
    return declared


# ---------------------------------------------------------------------------
# Cliente HTTP con guardián, caché, límites, reintentos, ritmo y robots.txt
# ---------------------------------------------------------------------------
@dataclass
class Response:
    url: str
    status: int
    content_type: str
    text: str
    fetched: float
    from_cache: bool = False
    truncated: bool = False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):     # noqa: D401 — se maneja afuera
        return None


class _Redirect(Exception):
    def __init__(self, location: str):
        self.location = location


class WebClient:
    """Toda salida a la red pasa por aquí y, antes, por el guardián (``network.access``)."""

    def __init__(self, guardian, settings: WebSettings | None = None, *, proxies: dict | None = None,
                 sleep=time.sleep, clock=time.time, requested_by: str = "hope"):
        if guardian is None:
            raise WebDenied(("sin guardián no hay acceso a la red (Carta art. 5)",))
        self.guardian = guardian
        perms = guardian.web_config()
        self.settings = settings or WebSettings.from_raw(perms.raw)
        if perms.contact and "contacto" not in self.settings.user_agent:     # bot identificado, nunca disfrazado
            self.settings.user_agent += f" contacto: {perms.contact}"
        handlers = [_NoRedirect()]
        if proxies is not None:
            handlers.append(urllib.request.ProxyHandler(proxies))
        self.opener = urllib.request.build_opener(*handlers)
        self.sleep, self.clock, self.requested_by = sleep, clock, requested_by
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.cache = Path(self.settings.cache_dir) if self.settings.cache_dir else None
        self.log: list[dict] = []                       # pedidos de esta sesión (red o caché)

    # --- caché ----------------------------------------------------------------
    def _cache_path(self, key: str) -> Path | None:
        return self.cache / f"{hashlib.sha256(key.encode()).hexdigest()}.json" if self.cache else None

    def _cache_get(self, key: str) -> Response | None:
        p = self._cache_path(key)
        if p is None or not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        if self.clock() - d["fetched"] > self.settings.cache_days * 86400:
            return None
        return Response(**{**d, "from_cache": True})

    def _cache_put(self, key: str, r: Response) -> None:
        p = self._cache_path(key)
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({**asdict(r), "from_cache": False}, ensure_ascii=False), encoding="utf-8")

    # --- permiso ----------------------------------------------------------------
    def _authorize(self, url: str, host: str, purpose: str, method: str = "GET") -> None:
        action = Action("network.access", {"host": host, "url": url, "proposito": purpose, "metodo": method},
                        requested_by=self.requested_by, justification=purpose)
        d = self.guardian.decide(action, approval_id=self.guardian.find_approval(action))
        if not d.allowed:
            raise WebDenied(d.reasons, d.pending_request)

    # --- HTTP crudo ----------------------------------------------------------------
    def _http(self, url: str, host: str, *, max_bytes: int, method: str = "GET",
              headers: dict | None = None) -> tuple[int, str, bytes, bool]:
        if method not in READ_METHODS:                         # solo lectura, por diseño
            raise WebDenied((f"método {method} prohibido: solo GET/HEAD",))
        extra = {k: v for k, v in (headers or {}).items() if k.lower() not in ("cookie", "content-type")}
        req = urllib.request.Request(url, data=None, method=method,
                                     headers={"User-Agent": self.settings.user_agent,
                                              "Accept": "text/html,text/plain,application/json,"
                                                        "application/xml;q=0.9,*/*;q=0.1", **extra})
        wait = self.settings.retry_wait
        last_err: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            gap = self._last.get(host, -1e18) + self.settings.min_interval - self.clock()
            if gap > 0:
                self.sleep(gap)                                    # ritmo bajo por dominio
            self._last[host] = self.clock()
            try:
                with self.opener.open(req, timeout=self.settings.timeout) as r:
                    ctype = r.headers.get("Content-Type", "")
                    deadline = time.monotonic() + self.settings.timeout
                    chunks, size = [], 0
                    while size <= max_bytes:
                        if time.monotonic() > deadline:
                            raise WebError("tiempo máximo de lectura superado")
                        b = r.read(min(65536, max_bytes + 1 - size))
                        if not b:
                            break
                        chunks.append(b)
                        size += len(b)
                    body = b"".join(chunks)
                    return r.status, ctype, body[:max_bytes], len(body) > max_bytes
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                    raise _Redirect(urllib.parse.urljoin(url, e.headers["Location"])) from None
                if e.code == 429 or e.code >= 500:
                    last_err = WebError(f"HTTP {e.code}")
                else:
                    raise WebError(f"HTTP {e.code} en {url}") from None
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
                last_err = WebError(f"error de red: {getattr(e, 'reason', e)}")
            if attempt < self.settings.retries:
                self.sleep(wait)                                   # espera creciente entre reintentos
                wait *= 2
        raise last_err or WebError("sin respuesta")

    def _robots_ok(self, scheme: str, host: str, netloc: str, url: str) -> bool:
        if host not in self._robots:
            key = f"robots:{scheme}://{netloc}"
            cached = self._cache_get(key)
            rp = urllib.robotparser.RobotFileParser()
            if cached is not None:
                rp.parse(cached.text.splitlines())
            else:
                try:
                    status, ctype, body, _ = self._http(f"{scheme}://{netloc}/robots.txt", host, max_bytes=500_000)
                    self.guardian.record_network_bytes(host, len(body))
                    text = body.decode("utf-8", "replace")
                except WebError as e:
                    # 401/403: el sitio no quiere robots; 404: sin restricciones; otros: prudencia.
                    text = "" if "HTTP 404" in str(e) or "HTTP 410" in str(e) else "User-agent: *\nDisallow: /"
                except _Redirect:
                    text = ""
                rp.parse(text.splitlines())
                self._cache_put(key, Response(key, 200, "text/plain", text, self.clock()))
            self._robots[host] = rp
        return self._robots[host].can_fetch(self.settings.user_agent, url)

    # --- interfaz ------------------------------------------------------------------
    def authorize_page(self, url: str, purpose: str, *, respect_robots: bool = True) -> tuple[str, str]:
        """Para el navegador de solo lectura: valida la URL, pide permiso y mira robots.txt.
        Devuelve (url normalizada, host). La red del navegador se limita después a ese sitio."""
        parts = urllib.parse.urlsplit(url.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise WebError(f"URL no válida (solo http/https): {url[:120]}")
        if parts.username or parts.password:
            raise WebDenied(("credenciales en la URL: prohibido iniciar sesión",))
        url, host = urllib.parse.urlunsplit(parts), parts.hostname.lower()
        self._authorize(url, host, purpose)
        if respect_robots and self.settings.respect_robots and not self._robots_ok(parts.scheme, host,
                                                                                    parts.netloc, url):
            raise WebDenied((f"robots.txt de {host} no permite leer {parts.path or '/'}",))
        return url, host

    def download(self, url: str, dest: str | Path, *, purpose: str, max_bytes: int, headers: dict | None = None,
                 respect_robots: bool = True, _hops: int = 0) -> dict:
        """Descarga reanudable (GET con ``Range``) de un archivo de una fuente registrada (dumps oficiales,
        Parquet de conjuntos abiertos). Baja como máximo ``max_bytes`` por llamada y lo que quede del día;
        si se corta, la próxima llamada sigue donde quedó. Devuelve {completo, bytes, total}."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        done_mark = dest.with_suffix(dest.suffix + ".completo")
        if done_mark.exists():
            return {"completo": True, "bytes": 0, "total": dest.stat().st_size}
        url, host = self.authorize_page(url, purpose, respect_robots=respect_robots)
        have = dest.stat().st_size if dest.exists() else 0
        cap = min(max_bytes, self.guardian.network_usage()["bytes_restantes"])
        hdr = {"User-Agent": self.settings.user_agent, **{k: v for k, v in (headers or {}).items()
                                                          if k.lower() != "cookie"}}
        if have:
            hdr["Range"] = f"bytes={have}-"
        gap = self._last.get(host, -1e18) + self.settings.min_interval - self.clock()
        if gap > 0:
            self.sleep(gap)
        self._last[host] = self.clock()
        req = urllib.request.Request(url, data=None, method="GET", headers=hdr)
        got, total = 0, None
        try:
            with self.opener.open(req, timeout=self.settings.timeout) as r:
                if have and r.status == 200:                    # el servidor no acepta Range: empezar de nuevo
                    have = 0
                    dest.write_bytes(b"")
                cr = r.headers.get("Content-Range", "")
                m = re.search(r"/(\d+)$", cr)
                total = int(m.group(1)) if m else (have + int(r.headers["Content-Length"])
                                                    if r.headers.get("Content-Length") else None)
                with open(dest, "ab") as f:
                    while got < cap:
                        b = r.read(min(1 << 20, cap - got))
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
        except urllib.error.HTTPError as e:
            if e.code == 416:                                   # ya estaba todo
                done_mark.touch()
                return {"completo": True, "bytes": 0, "total": have}
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                # Redirección (p. ej. Hugging Face → su CDN): se sigue con permiso del guardián para el destino.
                # A otro host no viajan las credenciales (Authorization) del usuario.
                if _hops >= 3:
                    raise WebError("demasiadas redirecciones") from None
                new = urllib.parse.urljoin(url, e.headers["Location"])
                same = (urllib.parse.urlsplit(new).hostname or "").lower() == host
                hdr2 = {k: v for k, v in (headers or {}).items() if same or k.lower() != "authorization"}
                return self.download(new, dest, purpose=purpose, max_bytes=max_bytes, headers=hdr2,
                                     respect_robots=respect_robots, _hops=_hops + 1)
            raise WebError(f"HTTP {e.code} en {url}") from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            raise WebError(f"error de red: {getattr(e, 'reason', e)}") from None
        finally:
            if got:
                self.guardian.record_network_bytes(host, got)
        size = have + got
        complete = total is not None and size >= total
        if complete:
            done_mark.touch()
        return {"completo": complete, "bytes": got, "total": total}

    def request(self, method: str, url: str, **kw) -> Response:
        """Único punto para elegir el método: cualquier cosa que no sea GET/HEAD se rechaza aquí."""
        method = method.upper()
        if method not in READ_METHODS:
            raise WebDenied((f"método {method} prohibido: Hope solo lee en la red (GET/HEAD)",))
        return self.get(url, method=method, **kw)

    def get(self, url: str, *, purpose: str = "leer_web", respect_robots: bool = True, method: str = "GET",
            headers: dict | None = None, secret_query: dict | None = None, _hops: int = 0) -> Response:
        """GET (o HEAD). Sin cuerpo, sin cookies, sin credenciales en la URL. ``headers`` solo para
        claves de API que el usuario puso en variables de entorno (nunca cookies). ``secret_query``: claves
        que la API exige en la URL; se agregan solo al pedir, nunca a la auditoría ni a la caché."""
        if method not in READ_METHODS:
            raise WebDenied((f"método {method} prohibido: solo GET/HEAD",))
        parts = urllib.parse.urlsplit(url.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise WebError(f"URL no válida (solo http/https): {url[:120]}")
        if parts.username or parts.password:
            raise WebDenied(("credenciales en la URL: prohibido iniciar sesión",))
        url, host = urllib.parse.urlunsplit(parts), parts.hostname.lower()
        cached = self._cache_get(url) if method == "GET" else None
        if cached is not None:
            self.log.append({"url": url, "cache": True})
            return cached
        self._authorize(url, host, purpose, method)
        if respect_robots and self.settings.respect_robots and not self._robots_ok(parts.scheme, host,
                                                                                    parts.netloc, url):
            raise WebDenied((f"robots.txt de {host} no permite leer {parts.path or '/'}",))
        cap = min(self.settings.max_page_bytes, self.guardian.network_usage()["bytes_restantes"])
        try:
            fetch = url + (("&" if "?" in url else "?") + urllib.parse.urlencode(secret_query) if secret_query else "")
            status, ctype, body, truncated = self._http(fetch, host, max_bytes=cap, method=method, headers=headers)
        except _Redirect as r:
            if _hops >= 3:
                raise WebError("demasiadas redirecciones") from None
            return self.get(r.location, purpose=purpose, respect_robots=respect_robots, method=method,
                            headers=headers, secret_query=secret_query, _hops=_hops + 1)
        self.guardian.record_network_bytes(host, len(body))
        base = ctype.split(";")[0].strip().lower()
        if base and not base.startswith(TEXT_TYPES):
            raise WebError(f"tipo de contenido no soportado: {base}")
        m = re.search(r"charset=([\w-]+)", ctype, re.I)
        try:
            text = body.decode(m.group(1) if m else "utf-8", "replace")
        except LookupError:
            text = body.decode("utf-8", "replace")
        resp = Response(url, status, base, text, self.clock(), truncated=truncated)
        self._cache_put(url, resp)
        self.log.append({"url": url, "cache": False, "bytes": len(body)})
        return resp


# ---------------------------------------------------------------------------
# Buscar y leer
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str


@dataclass
class Page:
    url: str
    title: str
    text: str                       # ya neutralizado
    fetched: float
    license: str | None = None
    from_cache: bool = False
    truncated: bool = False
    removed: int = 0                # oraciones con instrucciones eliminadas
    links: list = field(default_factory=list)     # [(url, texto)] para seguir enlaces


BACKEND_PREFIX = re.compile(r"^\s*(wikipedia|wiki|arxiv|duckduckgo|ddg|searxng|brave|google_cse|google)"
                            r"(?:\s+(\w\w))?\s*:\s*", re.I)


def format_results(res: list[SearchResult]) -> str:
    """Resultado de ``⟦buscar_web⟧`` tal como lo ve Hope (mismo formato en vivo y en los episodios grabados)."""
    if not res:
        return "sin resultados"
    body = " | ".join(f"{i}. {clip(r.title, 70)} <{r.url}>: {clip(r.snippet, 90)}" for i, r in enumerate(res, 1))
    return format_untrusted(body, f"búsqueda {res[0].source}", max_chars=SNIPPET_CHARS + 60)


def format_page(p: Page) -> str:
    """Resultado de ``⟦leer_web⟧``: texto ya neutralizado, recortado y con la URL."""
    head = f"{p.title}: " if p.title else ""
    return format_untrusted(head + p.text, p.url)


class WebTools:
    def __init__(self, client: WebClient, *, trajectories: str | Path | None = None):
        self.client = client
        self.settings = client.settings
        self.trajectory: dict | None = None
        self.trajectories = Path(trajectories) if trajectories else None

    # --- trayectorias de búsqueda (consulta → resultados → páginas → conclusión) ---------
    def start(self, question: str, topic: str = "búsqueda web") -> None:
        self.trajectory = {"pregunta": question, "tema": topic, "pasos": [], "inicio": time.time()}

    def _trace(self, kind: str, **data) -> None:
        if self.trajectory is not None:
            self.trajectory["pasos"].append({"tipo": kind, **data})

    def conclude(self, conclusion: str, sources: list[str] | None = None) -> dict:
        """Cierra la trayectoria; se guarda (JSONL) como episodio de entrenamiento de búsqueda."""
        if self.trajectory is None:
            raise ValueError("no hay una búsqueda en curso (llamar start)")
        t = {**self.trajectory, "conclusion": conclusion, "fuentes": sources or [
            p["url"] for p in self.trajectory["pasos"] if p["tipo"] == "leer"], "fin": time.time()}
        self.trajectory = None
        if self.trajectories:
            self.trajectories.parent.mkdir(parents=True, exist_ok=True)
            with self.trajectories.open("a", encoding="utf-8") as f:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        return t

    def links(self, url: str, *, allowed_only: bool = True, k: int = 20) -> list[tuple[str, str]]:
        """Enlaces de una página para seguir navegando (por defecto, solo a sitios permitidos o al mismo sitio)."""
        page = self.read(url)
        host = urllib.parse.urlsplit(page.url).hostname or ""
        perms = self.client.guardian.web_config()
        out = [(u, t) for u, t in page.links
               if not allowed_only or (urllib.parse.urlsplit(u).hostname or "") == host
               or perms.is_allowed(urllib.parse.urlsplit(u).hostname or "")]
        return out[:k]

    def links_text(self, url: str) -> str:
        links = self.links(url, k=8)
        if not links:
            return "sin enlaces a sitios permitidos"
        return format_untrusted(" | ".join(f"{clip(t, 40) or '(sin texto)'} <{u}>" for u, t in links), url,
                                max_chars=SNIPPET_CHARS + 60)

    # --- buscar ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        backend, lang = self.settings.backend, self.settings.lang
        m = BACKEND_PREFIX.match(query)
        if m:
            backend = {"wiki": "wikipedia", "ddg": "duckduckgo", "google": "google_cse"}.get(
                m.group(1).lower(), m.group(1).lower())
            lang = (m.group(2) or lang).lower()
            query = query[m.end():]
        query = query.strip()
        if not query:
            raise ValueError("consulta vacía")
        fn = {"wikipedia": self._wikipedia, "arxiv": self._arxiv, "duckduckgo": self._duckduckgo,
              "searxng": self._searxng, "brave": self._brave, "google_cse": self._google_cse}.get(backend)
        if fn is None:
            raise ValueError(f"buscador desconocido «{backend}»")
        res = fn(query, k, lang)[:k]
        self._trace("buscar", consulta=query, buscador=backend,
                    resultados=[{"titulo": r.title, "url": r.url, "fragmento": clip(r.snippet, 160)} for r in res])
        return res

    def _get_api(self, url: str, purpose: str) -> str:
        return self.client.get(url, purpose=purpose, respect_robots=False).text

    def _wikipedia(self, q: str, k: int, lang: str) -> list[SearchResult]:
        api = self.settings.url("wikipedia").format(lang=lang)
        url = api + "?" + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": q,
                                                  "srlimit": k, "format": "json", "utf8": 1})
        data = json.loads(self._get_api(url, "buscar_web:wikipedia"))
        page = self.settings.url("wikipedia_pagina")
        out = []
        for r in data.get("query", {}).get("search", []):
            title = r.get("title", "")
            out.append(SearchResult(title, page.format(lang=lang, title=urllib.parse.quote(title.replace(" ", "_"))),
                                    extract_text(r.get("snippet", ""))[1], f"wikipedia-{lang}"))
        return out

    def _arxiv(self, q: str, k: int, lang: str) -> list[SearchResult]:
        import xml.etree.ElementTree as ET
        url = self.settings.url("arxiv") + "?" + urllib.parse.urlencode(
            {"search_query": f"all:{q}", "start": 0, "max_results": k})
        root = ET.fromstring(self._get_api(url, "buscar_web:arxiv"))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in root.findall("a:entry", ns):
            out.append(SearchResult(" ".join((e.findtext("a:title", "", ns)).split()),
                                    (e.findtext("a:id", "", ns)).strip(),
                                    " ".join((e.findtext("a:summary", "", ns)).split()), "arxiv"))
        return out

    def _duckduckgo(self, q: str, k: int, lang: str) -> list[SearchResult]:
        html = self._get_api(self.settings.url("duckduckgo") + "?" + urllib.parse.urlencode({"q": q}),
                             "buscar_web:duckduckgo")
        out = []
        for m in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
                             r'(?:.*?class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>)?', html, re.S):
            href = urllib.parse.unquote(m.group(1).replace("&amp;", "&"))
            u = re.search(r"uddg=([^&]+)", href)
            href = urllib.parse.unquote(u.group(1)) if u else href
            if href.startswith("//"):
                href = "https:" + href
            out.append(SearchResult(extract_text(m.group(2))[1], href, extract_text(m.group(3) or "")[1],
                                    "duckduckgo"))
        return out

    def _searxng(self, q: str, k: int, lang: str) -> list[SearchResult]:
        if not self.settings.searxng_url:
            raise ValueError("falta searxng_url en configs/web.yaml")
        url = self.settings.searxng_url.rstrip("/") + "/search?" + urllib.parse.urlencode({"q": q, "format": "json"})
        data = json.loads(self._get_api(url, "buscar_web:searxng"))
        return [SearchResult(r.get("title", ""), r.get("url", ""), r.get("content", ""), "searxng")
                for r in data.get("results", [])]

    def _brave(self, q: str, k: int, lang: str) -> list[SearchResult]:
        """Brave Search API (opcional): clave del usuario en la variable HOPE_BRAVE_API_KEY, nunca en el repo."""
        key = os.environ.get("HOPE_BRAVE_API_KEY")
        if not key:
            raise ValueError("Brave Search necesita la variable de entorno HOPE_BRAVE_API_KEY")
        url = self.settings.url("brave") + "?" + urllib.parse.urlencode({"q": q, "count": k})
        data = json.loads(self.client.get(url, purpose="buscar_web:brave", respect_robots=False,
                                          headers={"X-Subscription-Token": key}).text)
        return [SearchResult(r.get("title", ""), r.get("url", ""), extract_text(r.get("description", ""))[1], "brave")
                for r in data.get("web", {}).get("results", [])]

    def _google_cse(self, q: str, k: int, lang: str) -> list[SearchResult]:
        """Google Programmable Search (opcional): HOPE_GOOGLE_CSE_KEY y HOPE_GOOGLE_CSE_CX en el entorno.
        Nunca se lee google.com directamente: sus términos lo prohíben y está en «prohibido»."""
        key, cx = os.environ.get("HOPE_GOOGLE_CSE_KEY"), os.environ.get("HOPE_GOOGLE_CSE_CX")
        if not key or not cx:
            raise ValueError("Google Programmable Search necesita HOPE_GOOGLE_CSE_KEY y HOPE_GOOGLE_CSE_CX")
        url = self.settings.url("google_cse") + "?" + urllib.parse.urlencode({"cx": cx, "q": q, "num": min(k, 10)})
        data = json.loads(self.client.get(url, purpose="buscar_web:google_cse", respect_robots=False,
                                          secret_query={"key": key}).text)
        return [SearchResult(r.get("title", ""), r.get("link", ""), r.get("snippet", ""), "google_cse")
                for r in data.get("items", [])]

    # --- leer ---------------------------------------------------------------------
    def _wikipedia_title(self, url: str) -> tuple[str, str] | None:
        tmpl = self.settings.url("wikipedia_pagina")
        rx = re.escape(tmpl).replace(re.escape("{lang}"), r"(?P<lang>[a-z\-]+)").replace(
            re.escape("{title}"), r"(?P<title>[^?#]+)")
        m = re.fullmatch(rx, url.split("#")[0])
        if not m:
            return None
        return (m.groupdict().get("lang") or self.settings.lang,
                urllib.parse.unquote(m.group("title")).replace("_", " "))

    def read(self, url: str) -> Page:
        wiki = self._wikipedia_title(url.strip())
        links: list = []
        if wiki:                                       # Wikipedia: texto plano por la API (más limpio)
            lang, title = wiki
            api = self.settings.url("wikipedia").format(lang=lang) + "?" + urllib.parse.urlencode(
                {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json",
                 "redirects": 1})
            r = self.client.get(api, purpose="leer_web:wikipedia", respect_robots=False)
            pages = json.loads(r.text).get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            raw_title, raw, declared = page.get("title", title), page.get("extract", ""), None
        else:
            r = self.client.get(url, purpose="leer_web")
            if "html" in r.content_type or r.text.lstrip()[:1] == "<":
                raw_title, raw, declared = extract_text(r.text)
                links = extract_links(r.text, r.url)
            else:
                raw_title, raw, declared = "", r.text, None
        text, removed = neutralize(raw)
        page = Page(url.strip(), neutralize(raw_title)[0], text, r.fetched, license_for(url, declared),
                    r.from_cache, r.truncated, removed, links)
        self._trace("leer", url=page.url, titulo=page.title, extracto=clip(text, 300))
        return page

    # --- texto para el runtime (⟦buscar_web⟧ / ⟦leer_web⟧) ---------------------------
    def search_text(self, query: str, k: int = 3) -> str:
        return format_results(self.search(query, k=k))

    def read_text(self, url: str) -> str:
        return format_page(self.read(url))

    # --- memoria: procedencia, grafo y contradicciones -----------------------------
    def remember(self, page: Page, memory, concept: str, fact: tuple[str, str] | None = None) -> dict:
        """Guarda lo leído como OBSERVADO con su URL; opcionalmente un dato «atributo = valor».

        ``memory`` es un ``KnowledgeBase`` (o un ``MemoryStore``). Si otra fuente ya dio otro valor
        para el mismo concepto y atributo, ambos quedan y se registra la contradicción."""
        from hope.epistemic import EpistemicStatus
        from hope.memory.store import Level
        store = getattr(memory, "store", memory)
        content = clip(page.text, 2000)
        for (oid,) in store.db.execute("SELECT id FROM engrams WHERE kind = 'web' AND description = ? AND "
                                       "content = ? AND archived = 0", (f"fuente: {page.url}", content)):
            store.touch(oid)
            store.commit()
            return {"id": oid, "nuevo": False, "contradicciones": []}
        eid = store.add_engram(kind="web", level=Level.SEMANTICA, concept=concept, content=content,
                               description=f"fuente: {page.url}", importance=0.5, confidence=0.5,
                               status=EpistemicStatus.OBSERVADO)
        store._history(eid, {"op": "procedencia", "url": page.url, "titulo": page.title, "recuperado": page.fetched,
                             "licencia": page.license, "instrucciones_eliminadas": page.removed})
        store.commit()
        graph = getattr(memory, "graph", None)
        if graph is None:
            from hope.memory.graph import KnowledgeGraph
            graph = KnowledgeGraph(store)
        graph.link(eid)
        contradictions = []
        if fact:
            r = store.remember_fact(concept, fact[0], fact[1], source=page.url)
            from hope.memory.store import Relation
            store.relate(r.engram.id, eid, Relation.DEPENDENCIA, confidence=0.9, provenance=page.url)
            contradictions = [{"id": oid, "valor": store.get(oid).content, "fuente": store.get(oid).description}
                              for oid in r.contradicts]
            store.commit()
        return {"id": eid, "nuevo": True, "contradicciones": contradictions}


# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------
SPECS = [
    ToolSpec("hte_web_search_v1", "buscar_web", "1.0", "investigación externa",
             "buscar información en internet", "Busca con Wikipedia, arXiv, DuckDuckGo o SearxNG (sin claves) y "
             "devuelve títulos, URL y fragmentos marcados como dato no confiable.",
             {"query": "str"}, {"resultados": "str"}, permissions=["network.access (nivel 3, configs/web.yaml)"],
             risks=["contenido no confiable (inyección de instrucciones): se neutraliza y se marca"],
             limitations=["los fragmentos se recortan", "resultados según el buscador"],
             usage_examples=["⟦buscar_web: ley de Ohm⟧", "⟦buscar_web: arxiv: state space models⟧"],
             do_not_use_when=["la respuesta se puede calcular o está en la memoria propia"]),
    ToolSpec("hte_web_read_v1", "leer_web", "1.0", "investigación externa",
             "leer una página web y citarla con procedencia", "Descarga una página (con caché, límites y "
             "robots.txt), extrae el texto sin scripts y lo devuelve marcado con su URL.",
             {"url": "str"}, {"texto": "str"}, permissions=["network.access (nivel 3, configs/web.yaml)"],
             risks=["contenido no confiable: nunca se ejecuta, se neutraliza y se marca"],
             limitations=["solo texto (HTML, texto plano, JSON, XML)", "se recorta"],
             usage_examples=["⟦leer_web: https://es.wikipedia.org/wiki/Ley_de_Ohm⟧"]),
]


def register(catalog, tools) -> None:
    """Registra las dos herramientas en el catálogo (cada uso pasa por ``tool.execute`` y ``network.access``)."""
    for spec, fn in zip(SPECS, (lambda query: tools.search_text(query), lambda url: tools.read_text(url))):
        if spec.id not in catalog:
            catalog.register(spec, fn)


def load(guardian, **kw) -> WebTools:
    return WebTools(WebClient(guardian, **kw))
