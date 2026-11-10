"""Código open source de GitHub para entrenar a Hope, por vías legales y filtrado (pedido del usuario).

**Fuentes, en orden de preferencia** (sección ``codigo_abierto`` de ``configs/fuentes_web.yaml``):

1. Conjuntos curados de Hugging Face, con licencia por archivo y la exclusión voluntaria de los
   autores ya aplicada («Am I in The Stack?»):

   * ``bigcode/the-stack-v2-dedup`` y ``bigcode/the-stack-dedup`` (BigCode). Son de **acceso
     restringido**: hay que entrar con la cuenta de Hugging Face a la ficha del conjunto, leer y
     aceptar sus términos, crear un token de lectura (https://huggingface.co/settings/tokens) y
     ponerlo en la variable de entorno ``HF_TOKEN`` (nunca en un archivo del repo). The Stack v2
     solo trae los identificadores: el contenido se baja de Software Heritage
     (``contenido_desde: swh_s3``), más lento; The Stack v1 dedup trae el código en el Parquet.
   * ``codeparrot/github-code-clean``: alternativa **sin acceso restringido** (campo ``license``).

2. Clonado directo de repos públicos de GitHub elegidos por tema y lenguaje (API de búsqueda, con
   token opcional en ``HOPE_GITHUB_TOKEN``; un pedido cada ``ritmo_s`` segundos, dentro de los
   límites de la API) con ``git clone --depth N --filter=blob:limit=…``. Sirve sobre todo por el
   **historial de commits**: los commits que arreglan algo se vuelven episodios
   «problema → corrección» (``hope.data.repos.history_docs``; los commits con coautoría de una IA
   se excluyen igual que allí). Los conjuntos de datos no tienen historial.

**Licencias.** Por defecto solo permisivas (MIT, Apache-2.0, BSD-2/3, ISC, Unlicense, CC0, Zlib;
MPL-2.0 opcional). GPL/AGPL/LGPL quedan **desactivadas** y se pueden activar
(``licencias.incluir_copyleft``): si usar código copyleft para entrenar un modelo obliga a algo es
una cuestión discutida y sin resolver. Sin licencia = excluido. Un encabezado
``SPDX-License-Identifier`` en el archivo manda sobre la licencia del repo. Cada archivo guarda
licencia, repo, ruta y commit (``provenance.jsonl``) y se genera ``ATRIBUCIONES.md``.

**Filtros.** Archivos generados, minificados, vendorizados, binarios, demasiado largos o con
líneas enormes, fuera; secretos y datos personales (claves de API, tokens, contraseñas, claves
privadas, emails, IP públicas) reemplazados por marcadores (``<CLAVE_API>``, ``<EMAIL>``…) con
patrones conocidos y entropía; calidad para código (que parsee: ``ast.parse`` en Python, ``bash -n``
en shell, llaves balanceadas en los lenguajes con llaves; comentarios, estructura, repetición);
deduplicación exacta y aproximada (MinHash + LSH en ``vistos.sqlite``), también contra el corpus
de código que ya existe; preferencia de lenguajes con pesos.

**Salida.** ``data/codigo/<fecha>/`` (``train.bin``, ``val.bin``, ``provenance.jsonl``,
``ATRIBUCIONES.md``) y la suma de todos los días en ``data/pre/codigo_abierto`` para el plan.
Corre sola dentro de ``python -m hope.data.web_sources --diario`` (con su propia cuota) o aparte::

    python -m hope.data.codigo_abierto --diario
    python -m hope.data.codigo_abierto --atribuciones data/pre/codigo_abierto
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import gzip
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.parse
import warnings
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from hope.data.corpus import EOD
from hope.data.web_corpus import CorpusWriter
from hope.tools.web import WebDenied, WebError

# ---------------------------------------------------------------------------
# Licencias (identificadores SPDX)
# ---------------------------------------------------------------------------
PERMISSIVE = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "CC0-1.0", "Zlib"]
WEAK_COPYLEFT = ["MPL-2.0"]
COPYLEFT = ["GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0", "AGPL-3.0"]

_ALIASES = {
    "mit": "MIT", "mit license": "MIT", "expat": "MIT", "mit-0": "MIT",
    "apache-2.0": "Apache-2.0", "apache 2.0": "Apache-2.0", "apache2": "Apache-2.0", "apache-2": "Apache-2.0",
    "apache license 2.0": "Apache-2.0", "apache license, version 2.0": "Apache-2.0", "apache 2": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause", "bsd 2-clause": "BSD-2-Clause", "simplified bsd": "BSD-2-Clause",
    "freebsd": "BSD-2-Clause", "bsd-2": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause", "bsd 3-clause": "BSD-3-Clause", "new bsd": "BSD-3-Clause",
    "modified bsd": "BSD-3-Clause", "bsd-3": "BSD-3-Clause", "bsd-3-clause-clear": "BSD-3-Clause",
    "isc": "ISC", "unlicense": "Unlicense", "the unlicense": "Unlicense",
    "cc0-1.0": "CC0-1.0", "cc0": "CC0-1.0", "cc0 1.0": "CC0-1.0", "zlib": "Zlib", "zlib license": "Zlib",
    "mpl-2.0": "MPL-2.0", "mozilla public license 2.0": "MPL-2.0",
    "gpl-2.0": "GPL-2.0", "gpl-3.0": "GPL-3.0", "gpl2": "GPL-2.0", "gpl3": "GPL-3.0",
    "lgpl-2.1": "LGPL-2.1", "lgpl-3.0": "LGPL-3.0", "agpl-3.0": "AGPL-3.0",
}


def spdx(name: str | None) -> str | None:
    """Normaliza un nombre de licencia a SPDX (``"mit"`` → ``MIT``, ``GPL-3.0-or-later`` → ``GPL-3.0``).
    Lo desconocido se devuelve tal cual (y no pasa el filtro); vacío o «sin aserción» → None."""
    if not name:
        return None
    t = str(name).strip()
    low = re.sub(r"\s+", " ", t.lower()).strip()
    if low in ("", "none", "null", "noassertion", "other", "no_license", "unknown"):
        return None
    low = re.sub(r"(-only|-or-later|\+)$", "", low)
    if low in _ALIASES:
        return _ALIASES[low]
    for canon in PERMISSIVE + WEAK_COPYLEFT + COPYLEFT:
        if low == canon.lower():
            return canon
    return t


SPDX_HEADER = re.compile(r"SPDX-License-Identifier:\s*([^\n*]+?)\s*(?:\*/|-->|$)", re.M)


class LicensePolicy:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.allowed = {spdx(x) for x in (cfg.get("permisivas") or PERMISSIVE)}
        if cfg.get("incluir_mpl", False):
            self.allowed |= set(WEAK_COPYLEFT)
        if cfg.get("incluir_copyleft", False):
            self.allowed |= set(COPYLEFT)

    def allows(self, expr: str | list | None) -> bool:
        """Expresión SPDX o lista de licencias (todas deben estar permitidas). ``A OR B``: alcanza una
        rama; ``A AND B``: todas. Sin licencia: no."""
        if isinstance(expr, (list, tuple)):
            items = [x for x in expr if spdx(x)]
            return bool(items) and all(self.allows(x) for x in items)
        if not expr or not str(expr).strip():
            return False
        e = str(expr).replace("(", " ").replace(")", " ")
        branches = re.split(r"\s+OR\s+", e, flags=re.I)
        return any(all(spdx(p) in self.allowed for p in re.split(r"\s+AND\s+", b.strip(), flags=re.I) if p.strip())
                   for b in branches if b.strip())


def detect_license_text(text: str) -> str | None:
    """Licencia a partir del texto de un archivo LICENSE/COPYING (los casos comunes)."""
    t = " ".join(text[:6000].split())
    low = t.lower()
    if "gnu affero general public license" in low:
        return "AGPL-3.0"
    if "gnu lesser general public license" in low:
        return "LGPL-2.1" if "version 2.1" in low else "LGPL-3.0"
    if "gnu general public license" in low:
        return "GPL-2.0" if "version 2," in low and "version 3" not in low[:300] else "GPL-3.0"
    if "mozilla public license" in low and "2.0" in low:
        return "MPL-2.0"
    if "apache license" in low and "version 2.0" in low:
        return "Apache-2.0"
    if "permission is hereby granted, free of charge" in low:
        return "MIT"
    if "this is free and unencumbered software released into the public domain" in low:
        return "Unlicense"
    if "cc0 1.0 universal" in low:
        return "CC0-1.0"
    if "altered source versions must be plainly marked" in low:
        return "Zlib"
    if "permission to use, copy, modify, and/or distribute this software for any purpose" in low:
        return "ISC"
    if "redistribution and use in source and binary forms" in low:
        return "BSD-3-Clause" if ("neither the name" in low or "names of its contributors" in low) else "BSD-2-Clause"
    return None


def file_license(text: str, repo_licenses) -> str | None:
    """Licencia de un archivo: su ``SPDX-License-Identifier`` si lo declara; si no, la del repo."""
    m = SPDX_HEADER.search(text[:4000])
    if m:
        return m.group(1).strip()
    if isinstance(repo_licenses, str):
        repo_licenses = [repo_licenses]
    lics = [spdx(x) for x in (repo_licenses or []) if spdx(x)]
    return " AND ".join(dict.fromkeys(lics)) or None


# ---------------------------------------------------------------------------
# Lenguajes
# ---------------------------------------------------------------------------
EXT_LANG = {
    ".py": "python", ".pyi": "python",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp", ".ipp": "cpp",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript",
    ".rs": "rust", ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".gcode": "gcode", ".gco": "gcode", ".ngc": "gcode",
    ".ino": "arduino",
    ".go": "go", ".cs": "csharp", ".lua": "lua", ".swift": "swift", ".scad": "openscad", ".rb": "ruby",
    ".php": "php", ".scala": "scala", ".m": "objc", ".jl": "julia", ".r": "r", ".dart": "dart",
}
LABEL_LANG = {"c++": "cpp", "cpp": "cpp", "c": "c", "python": "python", "java": "java", "kotlin": "kotlin",
              "javascript": "javascript", "typescript": "typescript", "rust": "rust", "shell": "shell",
              "bash": "shell", "g-code": "gcode", "gcode": "gcode", "arduino": "arduino", "go": "go",
              "c#": "csharp", "lua": "lua", "swift": "swift", "openscad": "openscad", "ruby": "ruby", "php": "php"}
BRACE_LANGS = {"c", "cpp", "java", "kotlin", "javascript", "typescript", "rust", "arduino", "go", "csharp",
               "swift", "scala", "dart", "php", "openscad"}
DEFAULT_WEIGHTS = {"python": 1.0, "cpp": 1.0, "c": 1.0, "arduino": 1.0, "gcode": 1.0, "rust": 0.8, "java": 0.6,
                   "kotlin": 0.7, "javascript": 0.6, "typescript": 0.7, "shell": 0.8, "otros": 0.0}


def detect_language(path: str, label: str | None = None) -> str | None:
    lang = EXT_LANG.get(Path(path or "").suffix.lower())
    if lang:
        return lang
    return LABEL_LANG.get(str(label or "").strip().lower())


def language_weight(lang: str, weights: dict) -> float:
    return float(weights.get(lang, weights.get("otros", 0.0)))


def sampled(key: str, weight: float) -> bool:
    """Muestreo determinista por hash: el mismo archivo siempre recibe la misma decisión."""
    if weight >= 1:
        return True
    return weight > 0 and int(key[:8], 16) / 0x1_0000_0000 < weight


# ---------------------------------------------------------------------------
# Archivos que no sirven: generados, minificados, vendorizados, binarios, gigantes
# ---------------------------------------------------------------------------
VENDORED = {"node_modules", "vendor", "vendors", "third_party", "third-party", "thirdparty", "external", "extern",
            "deps", "dist", "build", "out", "site-packages", ".venv", "venv", "__pycache__", "bower_components",
            "pods", ".git", "target", ".gradle", "jspm_packages", "_deps", "cmake-build-debug", "cmake-build-release"}
GENERATED_NAMES = re.compile(r"(\.min\.(js|css)$|-min\.js$|\.bundle\.js$|_pb2(_grpc)?\.py$|\.pb\.(go|cc|h)$|"
                             r"\.g\.(dart|cs)$|\.designer\.cs$|package-lock\.json$|yarn\.lock$|cargo\.lock$|"
                             r"poetry\.lock$|\.generated\.\w+$|_generated\.\w+$|moc_\w+\.cpp$|\.d\.ts$)", re.I)
GENERATED_MARKS = re.compile(r"(auto-?generated|automatically generated|do not (edit|modify)|@generated|"
                             r"this file (is|was) (auto-?)?generated|autogenerado|generado automáticamente|"
                             r"^[ \t]*(?:#|//|/?\*|--|;)[^\n]*\bgenerated (?:by|from|with)\b)", re.I | re.M)
HEX_LINE = re.compile(r"^\s*(0x[0-9a-fA-F]{1,8}\s*,\s*){6,}")


def file_reason(path: str, text: str, lang: str, cfg: dict | None = None) -> str | None:
    """Motivo para descartar el archivo antes de mirar su contenido en detalle (o None si sirve)."""
    cfg = cfg or {}
    parts = [p.lower() for p in Path(path or "").parts[:-1]]
    if any(p in VENDORED for p in parts):
        return "vendorizado"
    if GENERATED_NAMES.search(path or ""):
        return "generado"
    if "\x00" in text or text.count("\ufffd") > max(3, len(text) // 200):
        return "binario"
    size = len(text.encode("utf-8", "ignore"))
    if size > int(cfg.get("max_bytes_archivo", 100_000)):
        return "muy largo"
    lines = text.splitlines() or [""]
    nonempty = [line for line in lines if line.strip()]
    if len(lines) > int(cfg.get("max_lineas", 5000)):
        return "muy largo"
    if max(len(line) for line in lines) > int(cfg.get("max_largo_linea", 1000)):
        return "líneas enormes"
    if size < int(cfg.get("min_bytes_archivo", 100)) or len(nonempty) < 3:
        return "muy corto"
    if sum(len(line) for line in nonempty) / len(nonempty) > float(cfg.get("max_largo_medio", 120)):
        return "minificado"
    if GENERATED_MARKS.search(text[:2000]):
        return "generado"
    if lang != "gcode":
        alnum = sum(map(str.isalnum, text)) / max(len(text), 1)
        if alnum < float(cfg.get("min_alfanumerico", 0.25)):
            return "pocos caracteres alfanuméricos"
        if sum(bool(HEX_LINE.match(line)) for line in nonempty) / len(nonempty) > 0.3:
            return "datos embebidos"
    return None


# ---------------------------------------------------------------------------
# Secretos y datos personales → marcadores
# ---------------------------------------------------------------------------
SECRET_PATTERNS = [
    ("CLAVE_PRIVADA", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
                                 r"-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----")),
    ("CLAVE_API", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),                         # AWS
    ("TOKEN", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),              # GitHub
    ("TOKEN", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("TOKEN", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),                              # GitLab
    ("TOKEN", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),                         # Slack
    ("CLAVE_API", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),                             # Google
    ("CLAVE_API", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),                         # Anthropic
    ("CLAVE_API", re.compile(r"\bsk-(?:proj-|live-)?[A-Za-z0-9_-]{20,}\b")),             # OpenAI y similares
    ("CLAVE_API", re.compile(r"\b[rsp]k_(?:live|test)_[0-9A-Za-z]{16,}\b")),             # Stripe
    ("TOKEN", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),                                   # Hugging Face
    ("TOKEN", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("TOKEN", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b")),            # SendGrid
    ("TOKEN", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),  # JWT
    ("TOKEN", re.compile(r"\b\d{8,10}:AA[0-9A-Za-z_-]{30,}\b")),                         # Telegram
    ("TOKEN", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+")),
    ("TOKEN", re.compile(r"https://discord(?:app)?\.com/api/webhooks/[0-9]+/[A-Za-z0-9_-]+")),
]
URL_CREDENTIALS = re.compile(r"\b([a-z][a-z0-9+.-]*://)([^\s:/@'\"<>]+):([^\s@/'\"<>]+)@", re.I)
ASSIGNMENT = re.compile(
    r"(?i)\b([\w.-]*(?:passw(?:or)?d|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|auth[_-]?key|"
    r"client[_-]?secret|credentials?|private[_-]?key|contrase(?:ñ|n)a|clave)[\w.-]*)"
    r"(\s*[:=]>?\s*|\s*,\s*)([\"'])([^\"'\n]{4,200})\3")
PLACEHOLDER = re.compile(r"(^[<{$%]|[<>{}$]|^(x+|\*+|\.+|-+|_+)$|your|example|changeme|dummy|placeholder|"
                         r"secret$|password$|^token$|environ|getenv|process\.env|^none$|^null$|^test)", re.I)
QUOTED_BLOB = re.compile(r"([\"'])([A-Za-z0-9+/=_-]{32,})\1")
EMAIL = re.compile(r"(?<![\w.+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b")
EMAIL_KEEP_DOMAINS = re.compile(r"(^|\.)(example\.(com|org|net)|localhost|test|invalid|users\.noreply\.github\.com)$",
                                re.I)
EMAIL_NOT_DOMAINS = re.compile(r"\.(png|jpe?g|gif|svg|webp|js|ts|css|py|h|c|cpp|md|json)$", re.I)
IPV4 = re.compile(r"(?<![\w.])((?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})"
                  r"(?![\w]|\.\d)")


def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _assign_marker(key: str) -> str:
    k = key.lower()
    if re.search(r"pass|pwd|contrase", k):
        return "CONTRASEÑA"
    if "token" in k:
        return "TOKEN"
    if re.search(r"key|clave", k):
        return "CLAVE_API"
    return "SECRETO"


def scrub_secrets(text: str, *, emails: bool = True, ips: bool = True,
                  min_entropy: float = 4.3) -> tuple[str, Counter]:
    """Reemplaza secretos y datos personales por marcadores. Devuelve (texto, conteo por marcador)."""
    found: Counter = Counter()

    def sub(marker):
        def f(_m):
            found[marker] += 1
            return f"<{marker}>"
        return f

    for marker, pat in SECRET_PATTERNS:
        text = pat.sub(sub(marker), text)

    def creds(m):
        found["CONTRASEÑA"] += 1
        return f"{m.group(1)}<USUARIO>:<CONTRASEÑA>@"
    text = URL_CREDENTIALS.sub(creds, text)

    def assign(m):
        value = m.group(4)
        if PLACEHOLDER.search(value) or value.startswith("<") or " " in value.strip() and len(value) > 60:
            return m.group(0)
        marker = _assign_marker(m.group(1))
        found[marker] += 1
        return f"{m.group(1)}{m.group(2)}{m.group(3)}<{marker}>{m.group(3)}"
    text = ASSIGNMENT.sub(assign, text)

    def blob(m):
        v = m.group(2)
        classes = sum(bool(re.search(p, v)) for p in (r"[a-z]", r"[A-Z]", r"\d"))
        if classes == 3 and entropy(v) >= min_entropy:
            found["SECRETO"] += 1
            return f"{m.group(1)}<SECRETO>{m.group(1)}"
        return m.group(0)
    text = QUOTED_BLOB.sub(blob, text)

    if emails:
        def mail(m):
            local, dom = m.group(1), m.group(2)
            if EMAIL_KEEP_DOMAINS.search(dom) or EMAIL_NOT_DOMAINS.search(dom) or local.lower() == "git":
                return m.group(0)
            found["EMAIL"] += 1
            return "<EMAIL>"
        text = EMAIL.sub(mail, text)
    if ips:
        def ip(m):
            try:
                if not ipaddress.ip_address(m.group(1)).is_global:
                    return m.group(0)
            except ValueError:
                return m.group(0)
            found["IP"] += 1
            return "<IP>"
        text = IPV4.sub(ip, text)
    return text, found


# ---------------------------------------------------------------------------
# Calidad del código
# ---------------------------------------------------------------------------
STRUCTURE = re.compile(r"^[ \t]*(?:def |class |async def |fn |pub (?:fn|struct|enum|trait) |impl\b|struct |enum |"
                       r"trait |interface |function\b|fun |func |module |namespace |template[ \t]*<|export |"
                       r"public |private |protected |static |void |int |bool |float |double |char |auto |"
                       r"[A-Za-z_][\w-]*[ \t]*\(\)[ \t]*\{?[ \t]*$)", re.M)
GCODE_LINE = re.compile(r"^\s*(?:N\d+\s*)?(?:[GMTFSXYZEIJKRP]-?\d|;|\(|%|$)", re.I)


def braces_balanced(text: str, lang: str) -> bool:
    """Llaves, corchetes y paréntesis balanceados, salteando cadenas y comentarios (aproximado)."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    i, n = 0, len(text)
    quotes = "\"`" if lang == "rust" else "\"'`"
    while i < n:
        ch = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if lang in ("c", "cpp", "arduino") and ch == "#" and (i == 0 or text[i - 1] == "\n"):
            j = text.find("\n", i)                         # directivas del preprocesador
            i = n if j < 0 else j
            continue
        if ch in quotes:
            j = i + 1
            while j < n and text[j] != ch and not (ch != "`" and text[j] == "\n"):
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
        i += 1
    return not stack


def parses(text: str, lang: str) -> bool | None:
    """¿El código parsea? True/False, o None si no hay cómo comprobarlo para ese lenguaje."""
    if lang == "python":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(text)
            return True
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return False
    if lang == "shell" and shutil.which("bash"):
        try:
            r = subprocess.run(["bash", "-n"], input=text, capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return None
    if lang == "gcode":
        lines = [line for line in text.splitlines() if line.strip()]
        return sum(bool(GCODE_LINE.match(line)) for line in lines) / max(len(lines), 1) >= 0.8
    if lang in BRACE_LANGS:
        return braces_balanced(text, lang)
    return None


def _comment_lines(lines: list[str], lang: str) -> int:
    if lang in ("python", "shell"):
        marks = ("#",)
    elif lang == "gcode":
        marks = (";", "(")
    else:
        marks = ("//", "/*", "*", "*/")
    n = sum(line.lstrip().startswith(marks) for line in lines)
    if lang == "python":
        n += sum(line.strip().startswith(('"""', "'''")) for line in lines)
    return n


def code_quality(text: str, lang: str) -> tuple[float, list[str]]:
    """Nota 0–1 y motivos. Si no parsea: 0."""
    reasons = []
    ok = parses(text, lang)
    if ok is False:
        return 0.0, ["no parsea"]
    lines = [line for line in text.splitlines() if line.strip()]
    n = max(len(lines), 1)
    r = _comment_lines(lines, lang) / n
    if r < 0.02:
        s_com = 0.3
        reasons.append("sin comentarios")
    elif r > 0.6:
        s_com = 0.5
        reasons.append("casi todo comentarios")
    else:
        s_com = 1.0
    s_struct = 1.0 if (lang == "gcode" or STRUCTURE.search(text)) else 0.6
    avg = sum(len(line) for line in lines) / n
    s_len = 1.0 if 4 <= avg <= 90 else 0.4
    dup = 1 - len(set(lines)) / n
    s_dup = 1.0 if dup < 0.25 else 0.3
    if dup >= 0.25:
        reasons.append("líneas repetidas")
    raw = text.encode("utf-8", "ignore")
    if len(raw) > 2000 and len(zlib.compress(raw)) / len(raw) < 0.1:
        s_dup = min(s_dup, 0.2)
        reasons.append("muy repetitivo")
    score = 0.35 * s_com + 0.25 * s_struct + 0.2 * s_len + 0.2 * s_dup
    return round(score, 3), reasons


# ---------------------------------------------------------------------------
# Deduplicación exacta y aproximada (MinHash + LSH), persistente
# ---------------------------------------------------------------------------
TOKEN = re.compile(r"\w+|[^\w\s]")
HEADER = re.compile(r"^(⟦[^\n]*⟧|# archivo: [^\n]*)\n")


def exact_key(text: str) -> str:
    norm = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(norm.encode("utf-8", "ignore")).hexdigest()


def shingles(text: str, n: int = 5) -> np.ndarray:
    toks = TOKEN.findall(text)
    t = np.fromiter((zlib.crc32(x.encode("utf-8", "ignore")) for x in toks), dtype=np.uint64, count=len(toks))
    if len(t) < n:
        return np.unique(t) if len(t) else np.zeros(1, dtype=np.uint64)
    h = np.zeros(len(t) - n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n):
            h = h * np.uint64(1_000_003) + t[i:len(t) - n + 1 + i]
    return np.unique(h)


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 20261109):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2 ** 63, num_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2 ** 63, num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, text: str) -> np.ndarray:
        s = shingles(text)
        out = np.full(self.num_perm, np.iinfo(np.uint32).max, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for k in range(0, len(s), 4096):
                c = s[k:k + 4096, None]
                out = np.minimum(out, ((c * self.a[None, :] + self.b[None, :]) >> np.uint64(32)).min(axis=0))
        return out.astype(np.uint32)


def _find_byte(f, pos: int, byte: int, block: int = 1 << 20) -> int | None:
    while True:
        f.seek(pos)
        b = f.read(block)
        if not b:
            return None
        i = b.find(bytes([byte]))
        if i >= 0:
            return pos + i
        pos += len(b)


class Dedup:
    """Exacta (SHA-256 del texto sin espacios finales) y aproximada (MinHash, LSH por bandas) en SQLite."""

    def __init__(self, path: str | Path, *, threshold: float = 0.8, num_perm: int = 128, bands: int = 16):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS exactos (sha TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS firmas (id INTEGER PRIMARY KEY, sha TEXT, firma BLOB);
            CREATE TABLE IF NOT EXISTS bandas (banda INTEGER, clave INTEGER, id INTEGER);
            CREATE INDEX IF NOT EXISTS bandas_i ON bandas (banda, clave);
            CREATE TABLE IF NOT EXISTS indexado (ruta TEXT PRIMARY KEY, bytes INTEGER);
            CREATE TABLE IF NOT EXISTS repos (nombre TEXT PRIMARY KEY, estado TEXT);""")
        self.mh = MinHasher(num_perm)
        self.threshold, self.bands = threshold, bands
        self.rows = num_perm // bands

    def _band_keys(self, sig: np.ndarray) -> list[int]:
        return [int.from_bytes(hashlib.blake2b(sig[b * self.rows:(b + 1) * self.rows].tobytes(), digest_size=8)
                               .digest(), "big", signed=True) for b in range(self.bands)]

    def check(self, text: str) -> tuple[str | None, str, np.ndarray | None]:
        """(motivo o None, sha, firma). No agrega: llamar a ``add`` si se guarda."""
        sha = exact_key(text)
        if self.db.execute("SELECT 1 FROM exactos WHERE sha = ?", (sha,)).fetchone():
            return "duplicado", sha, None
        sig = self.mh.signature(text)
        ids = set()
        for b, k in enumerate(self._band_keys(sig)):
            ids.update(r[0] for r in self.db.execute("SELECT id FROM bandas WHERE banda = ? AND clave = ?", (b, k)))
        for i in ids:
            row = self.db.execute("SELECT firma FROM firmas WHERE id = ?", (i,)).fetchone()
            if row and float(np.mean(np.frombuffer(row[0], dtype=np.uint32) == sig)) >= self.threshold:
                return "casi_duplicado", sha, sig
        return None, sha, sig

    def add(self, sha: str, sig: np.ndarray | None, text: str | None = None) -> None:
        self.db.execute("INSERT OR IGNORE INTO exactos VALUES (?)", (sha,))
        if sig is None and text is not None:
            sig = self.mh.signature(text)
        if sig is not None:
            cur = self.db.execute("INSERT INTO firmas (sha, firma) VALUES (?, ?)", (sha, sig.tobytes()))
            self.db.executemany("INSERT INTO bandas VALUES (?, ?, ?)",
                                [(b, k, cur.lastrowid) for b, k in enumerate(self._band_keys(sig))])

    def commit(self) -> None:
        self.db.commit()

    def index_existing(self, dirs, *, max_doc: int = 200_000, chunk: int = 64 << 20) -> int:
        """Agrega al índice los documentos de ``train.bin``/``val.bin`` de otras fuentes de código (de
        forma incremental: recuerda hasta qué byte leyó). Saca el encabezado ``⟦repo: …⟧``."""
        n = 0
        for d in dirs:
            for name in ("train.bin", "val.bin"):
                p = Path(d) / name
                if not p.exists():
                    continue
                key = str(p.resolve())
                row = self.db.execute("SELECT bytes FROM indexado WHERE ruta = ?", (key,)).fetchone()
                start = row[0] if row else 0
                with open(p, "rb") as f:
                    while True:                                    # por tramos: el corpus puede pesar GB
                        f.seek(start)
                        data = f.read(chunk)
                        end = data.rfind(bytes([EOD]))
                        if end < 0:
                            if len(data) == chunk:                 # documento más largo que el tramo: saltarlo
                                nxt = _find_byte(f, start + chunk, EOD)
                                if nxt is None:
                                    break
                                start = nxt + 1
                                continue
                            break
                        for raw in data[:end].split(bytes([EOD])):
                            if not raw.strip() or len(raw) > max_doc:
                                continue
                            text = HEADER.sub("", raw.decode("utf-8", "ignore"), count=1)
                            sha = exact_key(text)
                            if not self.db.execute("SELECT 1 FROM exactos WHERE sha = ?", (sha,)).fetchone():
                                self.add(sha, None, text)
                                n += 1
                        start += end + 1
                        self.db.execute("INSERT OR REPLACE INTO indexado VALUES (?, ?)", (key, start))
                        self.db.commit()
        return n

    def repo_seen(self, name: str) -> bool:
        return bool(self.db.execute("SELECT 1 FROM repos WHERE nombre = ?", (name.lower(),)).fetchone())

    def mark_repo(self, name: str, estado: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO repos VALUES (?, ?)", (name.lower(), estado))
        self.db.commit()


# ---------------------------------------------------------------------------
# Tubería: licencia → lenguaje → filtros → secretos → calidad → dedup → escritura
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, cfg: dict, writer: CorpusWriter, dedup: Dedup):
        self.cfg, self.writer, self.dedup = cfg, writer, dedup
        self.policy = LicensePolicy(cfg.get("licencias"))
        self.weights = {**DEFAULT_WEIGHTS, **(cfg.get("pesos_lenguajes") or {})}
        self.filters = cfg.get("filtros") or {}
        self.min_quality = float(self.filters.get("calidad_minima", 0.6))
        self.counts: Counter = Counter()
        self.by_lang: Counter = Counter()
        self.secrets: Counter = Counter()
        self.bytes = 0

    def _write(self, text: str, sha: str, sig, meta: dict) -> str:
        r = self.writer.add(text, meta)
        if r == "guardado":
            self.dedup.add(sha, sig)
            self.bytes += len(text.encode("utf-8", "ignore")) + 1
        return r

    def add_file(self, text: str, meta: dict) -> str:
        """Un archivo de código. ``meta``: repo, ruta, licencias (repo), commit, lenguaje (etiqueta), fuente…"""
        st = self._add_file(text, meta)
        self.counts[st] += 1
        return st

    def _add_file(self, text: str, meta: dict) -> str:
        path = meta.get("ruta", "")
        lang = detect_language(path, meta.get("lenguaje"))
        if lang is None:
            return "lenguaje no soportado"
        lic = file_license(text, meta.get("licencias"))
        if not self.policy.allows(lic):
            return "licencia"
        reason = file_reason(path, text, lang, self.filters)
        if reason:
            return reason
        if not sampled(exact_key(text), language_weight(lang, self.weights)):
            return "muestreo por lenguaje"
        text, found = scrub_secrets(text, emails=self.filters.get("quitar_emails", True),
                                    ips=self.filters.get("quitar_ips", True))
        q, why = code_quality(text, lang)
        if q < self.min_quality:
            return "no parsea" if why == ["no parsea"] else "calidad"
        dup, sha, sig = self.dedup.check(text)
        if dup:
            return dup
        self.secrets.update(found)
        head = (f"⟦repo: {meta.get('repo', '?')}⟧ ⟦archivo: {path}⟧ ⟦lenguaje: {lang}⟧ ⟦licencia: {lic}⟧\n")
        out = {k: v for k, v in meta.items() if k not in ("licencias", "lenguaje")}
        out.update(tipo="archivo", lenguaje=lang, licencia=lic, calidad=q, secretos_quitados=sum(found.values()))
        r = self._write(head + text, sha, sig, out)
        if r == "guardado":
            self.by_lang[lang] += len(text.encode("utf-8", "ignore"))
        return r

    def add_episode(self, text: str, meta: dict) -> str:
        """Episodio «problema → corrección» de un commit. La licencia es la del repo."""
        st = self._add_episode(text, meta)
        self.counts["episodio: " + st] += 1
        return st

    def _add_episode(self, text: str, meta: dict) -> str:
        lic = file_license("", meta.get("licencias"))
        if not self.policy.allows(lic):
            return "licencia"
        m = re.search(r"^Archivos: (.*)$", text, re.M)
        files = [f.strip() for f in (m.group(1).split(",") if m else [])]
        code = [f for f in files if detect_language(f) and not any(p.lower() in VENDORED for p in Path(f).parts[:-1])
                and not GENERATED_NAMES.search(f)]
        if not code:
            return "sin código"
        langs = {detect_language(f) for f in code}
        if not any(sampled(exact_key(text), language_weight(x, self.weights)) for x in langs):
            return "muestreo por lenguaje"
        text, found = scrub_secrets(text, emails=self.filters.get("quitar_emails", True),
                                    ips=self.filters.get("quitar_ips", True))
        dup, sha, sig = self.dedup.check(text)
        if dup:
            return dup
        self.secrets.update(found)
        out = {k: v for k, v in meta.items() if k != "licencias"}
        out.update(tipo="episodio", licencia=lic, lenguajes=sorted(langs), secretos_quitados=sum(found.values()))
        return self._write(text, sha, sig, out)


# ---------------------------------------------------------------------------
# Atribuciones
# ---------------------------------------------------------------------------
def write_attributions(folder: str | Path) -> dict:
    """``ATRIBUCIONES.md`` y ``atribuciones.json`` desde ``provenance.jsonl``: repo, licencia, origen,
    commit y cuántos archivos y episodios se usaron. Las licencias permisivas piden conservar el aviso
    de copyright: esta lista es ese aviso (el modelo no redistribuye los archivos)."""
    folder = Path(folder)
    prov = folder / "provenance.jsonl"
    repos: dict[tuple, dict] = {}
    if prov.exists():
        with open(prov, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r.get("repo", "?"), r.get("licencia", "?"))
                a = repos.setdefault(key, {"repo": key[0], "licencia": key[1], "archivos": 0, "episodios": 0,
                                           "commits": set(), "origen": r.get("url_repo") or r.get("url", ""),
                                           "fuente": r.get("fuente", ""), "dataset": r.get("dataset", "")})
                a["episodios" if r.get("tipo") == "episodio" else "archivos"] += 1
                if r.get("commit") and len(a["commits"]) < 3:
                    a["commits"].add(r["commit"])
    rows = sorted(({**a, "commits": sorted(a["commits"])} for a in repos.values()),
                  key=lambda a: (a["repo"].lower(), a["licencia"]))
    lic_totals = Counter()
    for a in rows:
        lic_totals[a["licencia"]] += a["archivos"] + a["episodios"]
    (folder / "atribuciones.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Atribuciones del código abierto usado para entrenar a Hope", "",
          "Generado automáticamente desde `provenance.jsonl` (hope/data/codigo_abierto.py). Cada fila es un",
          "repositorio cuyo código o historial entró al corpus, con su licencia y su origen. Los derechos de",
          "autor pertenecen a sus autores; las licencias (MIT, Apache-2.0, BSD…) piden conservar este aviso.", "",
          "## Licencias", ""]
    md += [f"- {lic}: {n} documentos" for lic, n in lic_totals.most_common()]
    md += ["", "## Repositorios", "", "| Repositorio | Licencia | Archivos | Episodios | Commit | Origen |",
           "|---|---|---:|---:|---|---|"]
    for a in rows:
        origin = a["origen"] or (f"https://github.com/{a['repo']}" if "/" in a["repo"] else "")
        via = f" (vía {a['dataset']})" if a["dataset"] else ""
        md.append(f"| {a['repo']} | {a['licencia']} | {a['archivos']} | {a['episodios']} | "
                  f"{', '.join(c[:12] for c in a['commits'])} | {origin}{via} |")
    (folder / "ATRIBUCIONES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"repos": len(rows), "licencias": dict(lic_totals)}


# ---------------------------------------------------------------------------
# Recolectores
# ---------------------------------------------------------------------------
HF = "https://huggingface.co"
GITHUB_API = "https://api.github.com"
GH_LANG = {"cpp": "C++", "arduino": "C++", "gcode": "G-code", "shell": "Shell", "csharp": "C#"}


class Budget:
    def __init__(self, net_bytes: int, out_bytes: int):
        self.net, self.out = net_bytes, out_bytes

    def ok(self) -> bool:
        return self.net > 0 and self.out > 0


def git_clone(url: str, dest: Path, *, depth: int, blob_limit: str, timeout: float = 900) -> None:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    r = subprocess.run(["git", "clone", "--quiet", "--depth", str(depth), "--single-branch", "--no-tags",
                        f"--filter=blob:limit={blob_limit}", url, str(dest)],
                       capture_output=True, text=True, timeout=timeout, env=env, errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


@contextlib.contextmanager
def _no_lazy_fetch():
    """Con clon parcial, git bajaría en silencio los blobs que faltan: se prohíbe (fuera de la cuota)."""
    old = os.environ.get("GIT_NO_LAZY_FETCH")
    os.environ["GIT_NO_LAZY_FETCH"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GIT_NO_LAZY_FETCH", None)
        else:
            os.environ["GIT_NO_LAZY_FETCH"] = old


class Collectors:
    def __init__(self, client, root: Path, pipe: Pipeline, *, clone=git_clone, sleep=time.sleep):
        self.client, self.root, self.pipe, self.clone, self.sleep = client, root, pipe, clone, sleep

    def _pace(self, src: dict) -> None:
        settings = getattr(self.client, "settings", None)
        if settings is not None:
            settings.min_interval = max(settings.min_interval, float(src.get("ritmo_s", 3)))

    def _get_json(self, src: dict, url: str, headers: dict):
        self._pace(src)
        return json.loads(self.client.get(url, purpose=f"codigo:{src['nombre']}", respect_robots=False,
                                          headers=headers).text)

    # --- (a) conjuntos de Hugging Face --------------------------------------------
    def hf_codigo(self, src: dict, state: dict, budget: Budget):
        var = src.get("variable_token", "HF_TOKEN")
        token = os.environ.get(var)
        if src.get("requiere_token") and not token:
            state["aviso"] = (f"falta {var}: aceptá los términos de {src['dataset']} en Hugging Face y poné tu "
                              f"token de lectura en la variable de entorno {var}")
            return
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if "archivos" not in state:
            files = []
            for folder in src.get("carpetas") or ["data"]:
                url = f"{HF}/api/datasets/{src['dataset']}/tree/main/{urllib.parse.quote(folder)}"
                try:
                    items = self._get_json(src, url, headers)
                except WebError as e:
                    state.setdefault("avisos_carpetas", []).append(f"{folder}: {str(e)[:120]}")
                    continue
                files += sorted(it["path"] for it in items if it.get("type") == "file"
                                and it.get("path", "").endswith(".parquet"))
            if not files:
                state["aviso"] = "no se pudo listar los archivos del conjunto: " + "; ".join(
                    state.pop("avisos_carpetas", []))[:300]
                return                                             # se reintenta en la próxima corrida
            state.pop("avisos_carpetas", None)
            state["archivos"] = _interleave(files, src.get("carpetas") or ["data"])
        cols = {"contenido": "content", "repo": "repo_name", "ruta": "path", "licencias": "license",
                "lenguaje": "language", **(src.get("columnas") or {})}
        files = state["archivos"]
        i = state.get("archivo", 0)
        while i < len(files) and budget.ok():
            path = files[i]
            dest = self.root / "raw" / src["nombre"] / path
            self._pace(src)
            url = f"{HF}/datasets/{src['dataset']}/resolve/main/{urllib.parse.quote(path)}"
            r = self.client.download(url, dest, purpose=f"codigo:{src['nombre']}", max_bytes=budget.net,
                                     headers=headers)
            budget.net -= r["bytes"]
            if not r["completo"]:
                return                                             # sigue mañana
            yield from self._rows(src, dest, cols, state, budget, headers)
            if state.get("fila") is not None and state.get("fila", 0) > 0:
                return                                             # cuota de salida agotada a mitad de archivo
            if src.get("borrar_crudos", True):
                dest.unlink(missing_ok=True)
                dest.with_suffix(dest.suffix + ".completo").unlink(missing_ok=True)
            i += 1
            state.update(archivo=i, fila=0)

    def _rows(self, src, dest, cols, state, budget, headers):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(dest)
        names = set(pf.schema_arrow.names)
        wanted = [c for c in cols.values() if c in names]
        row0 = state.get("fila", 0)
        j = 0
        for batch in pf.iter_batches(batch_size=512, columns=wanted):
            data = batch.to_pydict()
            for k in range(batch.num_rows):
                if j < row0:
                    j += 1
                    continue
                if not budget.ok():
                    state["fila"] = j
                    return
                j += 1
                g = lambda key: data.get(cols.get(key), [None] * batch.num_rows)[k]   # noqa: E731
                if cols.get("tipo_licencia") in names and str(g("tipo_licencia")).lower() == "no_license":
                    self.pipe.counts["licencia"] += 1
                    continue
                text = g("contenido")
                if text is None and g("blob_id") and src.get("contenido_desde") == "swh_s3":
                    text = self._swh_blob(src, g("blob_id"), budget, data.get("src_encoding", [None] * 512)[k])
                if not text:
                    continue
                before = self.pipe.bytes
                meta = {"fuente": src["nombre"], "dataset": src["dataset"], "repo": g("repo") or "?",
                        "ruta": g("ruta") or "", "commit": g("commit"), "licencias": _as_list(g("licencias")),
                        "lenguaje": g("lenguaje"), "url_repo": f"https://github.com/{g('repo')}" if g("repo") else ""}
                yield text, meta, "archivo"
                budget.out -= self.pipe.bytes - before
        state["fila"] = 0

    def _swh_blob(self, src, blob_id, budget, encoding):
        url = f"https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
        tmp = self.root / "raw" / src["nombre"] / "_blob.gz"
        tmp.unlink(missing_ok=True)
        tmp.with_suffix(".gz.completo").unlink(missing_ok=True)
        try:
            r = self.client.download(url, tmp, purpose=f"codigo:{src['nombre']}", max_bytes=min(budget.net, 2_000_000),
                                     respect_robots=False)
        except WebError:
            return None
        budget.net -= r["bytes"]
        try:
            return gzip.decompress(tmp.read_bytes()).decode(encoding or "utf-8", "replace")
        except (OSError, LookupError, EOFError):
            return None
        finally:
            tmp.unlink(missing_ok=True)
            tmp.with_suffix(".gz.completo").unlink(missing_ok=True)

    # --- (b) repos de GitHub ------------------------------------------------------------
    def github(self, src: dict, state: dict, budget: Budget):
        var = src.get("variable_token", "HOPE_GITHUB_TOKEN")
        token = os.environ.get(var)
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        queries = github_queries(src)
        qi, page = state.get("consulta", 0), state.get("pagina", 1)
        done_today, max_repos = 0, int(src.get("max_repos_por_dia", 20))
        api = src.get("api", GITHUB_API).rstrip("/")
        dedup = self.pipe.dedup
        policy = self.pipe.policy
        while budget.ok() and qi < len(queries) and done_today < max_repos:
            q = {"q": queries[qi], "sort": src.get("orden", "stars"), "order": "desc", "per_page": 30, "page": page}
            data = self._get_json(src, f"{api}/search/repositories?{urllib.parse.urlencode(q)}", headers)
            items = data.get("items") or []
            last = not items or page * 30 >= min(1000, int(data.get("total_count", 0)))
            for it in items:
                if not budget.ok() or done_today >= max_repos:
                    break
                full = it.get("full_name", "")
                if not full or dedup.repo_seen(full):
                    continue
                lic = spdx((it.get("license") or {}).get("spdx_id"))
                if not policy.allows(lic):
                    self.pipe.counts["repo: licencia"] += 1
                    dedup.mark_repo(full, f"licencia {lic}")
                    continue
                if it.get("fork") or int(it.get("size", 0)) > int(src.get("max_repo_mb", 200)) * 1000:
                    self.pipe.counts["repo: fork o muy grande"] += 1
                    dedup.mark_repo(full, "omitido")
                    continue
                done_today += 1
                yield from self._repo(src, it, lic, budget)
                dedup.mark_repo(full, "procesado")
            if last:
                qi, page = qi + 1, 1
            else:
                page += 1
            state.update(consulta=qi, pagina=page)
        if qi >= len(queries):
            state["aviso"] = "terminaron las búsquedas configuradas (agregá temas o subí estrellas_min)"

    def _repo(self, src: dict, it: dict, lic: str, budget: Budget):
        from hope.data.repos import code_docs, history_docs
        full = it["full_name"]
        url = it.get("clone_url") or f"https://github.com/{full}.git"
        if url.startswith(("http://", "https://")):
            self.client.authorize_page(url, f"codigo:{src['nombre']}", respect_robots=False)
        dest = self.root / "clones" / full.replace("/", "__")
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.clone(url, dest, depth=int(src.get("profundidad", 200)), blob_limit=str(src.get("limite_blob", "256k")))
        except (RuntimeError, subprocess.TimeoutExpired, OSError):
            self.pipe.counts["repo: error al clonar"] += 1
            shutil.rmtree(dest, ignore_errors=True)
            return
        size = dir_size(dest / ".git")
        budget.net -= size
        guardian = getattr(self.client, "guardian", None)
        if guardian is not None and hasattr(guardian, "record_network_bytes"):
            guardian.record_network_bytes(urllib.parse.urlsplit(url).hostname or "github.com", size)
        # Cotejo con el archivo de licencia del clon: si dice otra cosa, se usa la más restrictiva (excluye).
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "LICENCE"):
            p = dest / name
            if p.is_file():
                found = detect_license_text(p.read_text(encoding="utf-8", errors="ignore"))
                if found and found != lic and not self.pipe.policy.allows(found):
                    lic = found
                break
        head = ""
        with contextlib.suppress(RuntimeError, subprocess.TimeoutExpired, OSError):
            head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        shallow = set((dest / ".git" / "shallow").read_text().split()) if (dest / ".git" / "shallow").exists() else set()
        base = {"fuente": src["nombre"], "repo": full, "licencias": [lic], "url_repo": it.get("html_url")
                or f"https://github.com/{full}", "estrellas": it.get("stargazers_count")}
        try:
            with _no_lazy_fetch():
                for text, meta in code_docs(dest, include_ai=False):
                    if not budget.ok():
                        break
                    body = text.split("\n", 1)[1] if "\n" in text else ""
                    before = self.pipe.bytes
                    yield body, {**base, "ruta": meta["ruta"], "commit": head}, "archivo"
                    budget.out -= self.pipe.bytes - before
                if src.get("episodios", True):
                    only_fix = src.get("solo_arreglos", True)
                    for text, meta in history_docs(dest, include_ai=False,
                                                   max_commits=int(src.get("profundidad", 200))):
                        if not budget.ok():
                            break
                        if meta["commit"] in shallow or (only_fix and meta["tipo"] != "arreglo"):
                            continue
                        text = text.replace(f"⟦repo: {dest.name}⟧", f"⟦repo: {full}⟧ ⟦licencia: {lic}⟧", 1)
                        before = self.pipe.bytes
                        yield text, {**base, "commit": meta["commit"], "tipo_commit": meta["tipo"]}, "episodio"
                        budget.out -= self.pipe.bytes - before
        except RuntimeError:
            self.pipe.counts["repo: error de git"] += 1
        finally:
            if not src.get("conservar_clones", False):
                shutil.rmtree(dest, ignore_errors=True)


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (ValueError, TypeError):
            pass
        return [v]
    return [str(x) for x in v]


def _interleave(files: list[str], folders: list[str]) -> list[str]:
    """Alterna archivos de cada carpeta (lenguaje) para que la cuota diaria no sea todo de uno solo."""
    groups = [[f for f in files if f.startswith(folder.rstrip("/") + "/")] for folder in folders]
    rest = [f for f in files if not any(f in g for g in groups)]
    out = []
    for k in range(max([len(g) for g in groups] + [0])):
        out += [g[k] for g in groups if k < len(g)]
    return out + rest


def github_queries(src: dict) -> list[str]:
    stars = int(src.get("estrellas_min", 50))
    out = []
    for b in src.get("busquedas") or []:
        base = b["consulta"] if isinstance(b, dict) else str(b)
        langs = (b.get("lenguajes") if isinstance(b, dict) else None) or src.get("lenguajes") or [None]
        for lang in langs:
            q = base
            if lang:
                gl = GH_LANG.get(lang, lang)
                q += f' language:"{gl}"' if " " in gl or "+" in gl or "#" in gl else f" language:{gl}"
            out.append(f"{q} stars:>={stars} fork:false archived:false")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Corrida diaria
# ---------------------------------------------------------------------------
def load_config(path: str | Path = "configs/fuentes_web.yaml") -> dict:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return d.get("codigo_abierto") or {}


def daily(cfg: dict, client, *, date: str | None = None, root: str | Path | None = None, sleep=time.sleep,
          clone=git_clone, force: bool = False) -> dict:
    """La cuota de hoy: recorre las fuentes activas en orden, filtra y escribe ``<carpeta>/<fecha>/``,
    suma todo a ``consolidado`` y regenera las atribuciones."""
    from hope.data.web_sources import _consolidate
    root = Path(root or cfg.get("carpeta", "data/codigo"))
    date = date or dt.date.today().isoformat()
    state_path = root / "estado.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("_ultima_corrida") == date and not force:
        return {"fecha": date, "aviso": "la cuota de hoy de código ya se usó"}
    q = cfg.get("cuota") or {}
    budget = Budget(int(float(q.get("red_mb_por_dia", 1000)) * 1e6), int(float(q.get("salida_mb_por_dia", 300)) * 1e6))
    consolidated = Path(cfg.get("consolidado", "data/pre/codigo_abierto"))
    day = root / date
    dedup = Dedup(root / "vistos.sqlite", threshold=float((cfg.get("filtros") or {}).get("umbral_casi_duplicado", 0.8)))
    indexed = dedup.index_existing([consolidated, *(cfg.get("existente") or [])])
    writer = CorpusWriter(day, existing=[consolidated], apply_rules=False, near_dup=None, min_chars=0,
                          val_percent=float(cfg.get("val_porcentaje", 1.0)))
    pipe = Pipeline(cfg, writer, dedup)
    col = Collectors(client, root, pipe, clone=clone, sleep=sleep)
    report = {"fecha": date, "indexados_del_corpus": indexed, "fuentes": {}}
    for i, src in enumerate(s for s in cfg.get("fuentes") or [] if s.get("activo", True)):
        if not budget.ok():
            break
        if i:
            sleep(float(cfg.get("pausa_entre_fuentes_s", 10)))
        st = state.setdefault(src["nombre"], {})
        fn = {"hf_codigo": col.hf_codigo, "github": col.github}.get(src.get("tipo"))
        rep = {"errores": []}
        if fn is None:
            rep["errores"].append(f"tipo desconocido: {src.get('tipo')}")
            report["fuentes"][src["nombre"]] = rep
            continue
        net0, before, counts0 = budget.net, pipe.bytes, Counter(pipe.counts)
        try:
            for text, meta, kind in fn(src, st, budget):
                (pipe.add_episode if kind == "episodio" else pipe.add_file)(text, meta)
                if sum(pipe.counts.values()) % 500 == 0:
                    dedup.commit()
        except WebDenied as e:
            rep["errores"].append(str(e)[:200])
            rep["pendiente"] = e.pending
        except (WebError, ValueError, OSError, RuntimeError) as e:
            rep["errores"].append(str(e)[:200])
        dedup.commit()
        rep.update(bytes_red=net0 - budget.net, bytes_guardados=pipe.bytes - before,
                   motivos=dict(pipe.counts - counts0), **({"aviso": st["aviso"]} if st.get("aviso") else {}))
        report["fuentes"][src["nombre"]] = rep
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    totals = writer.close({"por_lenguaje": dict(pipe.by_lang), "secretos": dict(pipe.secrets)})
    _consolidate(day, consolidated)
    report.update(bytes_train_dia=totals["bytes_train"], bytes_por_lenguaje=dict(pipe.by_lang),
                  secretos_quitados=dict(pipe.secrets), atribuciones=write_attributions(consolidated))
    write_attributions(day)
    state["_ultima_corrida"] = date
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    (day / "informe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Código open source (licencias permisivas, filtrado) para entrenar")
    ap.add_argument("--config", default="configs/fuentes_web.yaml")
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    ap.add_argument("--diario", action="store_true", help="traer la cuota de hoy (sin esperar a web_sources)")
    ap.add_argument("--atribuciones", default=None, help="regenerar ATRIBUCIONES.md de esta carpeta")
    ap.add_argument("--revisar", default=None, help="mostrar qué haría el filtro con un archivo local")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.atribuciones:
        out = write_attributions(args.atribuciones)
    elif args.revisar:
        p = Path(args.revisar)
        text = p.read_text(encoding="utf-8", errors="ignore")
        lang = detect_language(p.name)
        clean, found = scrub_secrets(text)
        out = {"lenguaje": lang, "licencia_en_archivo": file_license(text, []),
               "descarte": file_reason(p.name, text, lang or "", cfg.get("filtros")) if lang else "lenguaje",
               "calidad": code_quality(clean, lang) if lang else None, "secretos": dict(found)}
    elif args.diario:
        from guardian.cli import load as load_guardian
        from hope.tools.web import WebClient
        client = WebClient(load_guardian(args.guardian), requested_by="codigo_abierto")
        out = daily(cfg, client)
    else:
        out = {"fuentes": [{k: s.get(k) for k in ("nombre", "tipo", "activo", "dataset")} for s in
                           cfg.get("fuentes") or []], "licencias": sorted(LicensePolicy(cfg.get("licencias")).allowed)}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
