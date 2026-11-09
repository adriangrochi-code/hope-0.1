"""Herramientas de texto, formatos y código (docs/HERRAMIENTAS_HN1.md §3, catálogo ampliado).

* ``validate`` — JSON, YAML, XML o TOML con la posición exacta del error (línea y columna).
* ``regex_test`` — probar una expresión regular contra casos: coincidencia completa, búsquedas y grupos.
* ``java_run`` — compilar y ejecutar Java (javac/java) en el sandbox; pensado para probar
  fragmentos de mods de Minecraft sin el juego.
* ``git_read`` — log/show/blame/diff de un repositorio local, solo lectura.

``validate`` y ``regex_test`` se ejecutan en el sandbox desde el catálogo: un XML con entidades
recursivas o una regex con retroceso catastrófico solo agotan el tiempo de ese proceso.
"""

from __future__ import annotations

import json
import re
import shutil

from hope.tools.sandbox import run_command

MAX = 2000


# ---------------------------------------------------------------------------
# Validar formatos

def _pos(text: str, index: int) -> tuple[int, int]:
    line = text.count("\n", 0, index) + 1
    col = index - (text.rfind("\n", 0, index) + 1) + 1
    return line, col


def validate(fmt: str, text: str) -> dict:
    f = fmt.strip().lower()
    if len(text) > 4 << 20:
        raise ValueError("texto demasiado grande (> 4 MB)")
    if f == "json":
        try:
            v = json.loads(text)
            return {"valido": True, "formato": "JSON", "tipo": type(v).__name__}
        except json.JSONDecodeError as e:
            return {"valido": False, "formato": "JSON", "linea": e.lineno, "columna": e.colno, "error": e.msg}
    if f in ("yaml", "yml"):
        import yaml
        try:
            docs = list(yaml.safe_load_all(text))
            return {"valido": True, "formato": "YAML", "documentos": len(docs)}
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None) or getattr(e, "context_mark", None)
            out = {"valido": False, "formato": "YAML",
                   "error": " ".join(x for x in (getattr(e, "context", None), getattr(e, "problem", None)) if x)
                   or str(e).splitlines()[0]}
            if mark is not None:
                out.update(linea=mark.line + 1, columna=mark.column + 1)
            return out
    if f == "xml":
        import xml.etree.ElementTree as ET
        if re.search(r"<!ENTITY", text, re.I):
            return {"valido": False, "formato": "XML", "error": "declaraciones <!ENTITY> no permitidas (riesgo de "
                    "expansión exponencial); quitarlas para validar", "linea": _pos(text, re.search(
                        r"<!ENTITY", text, re.I).start())[0], "columna": _pos(text, re.search(
                            r"<!ENTITY", text, re.I).start())[1]}
        try:
            root = ET.fromstring(text)
            return {"valido": True, "formato": "XML", "raiz": root.tag, "elementos": sum(1 for _ in root.iter())}
        except ET.ParseError as e:
            line, col = e.position
            return {"valido": False, "formato": "XML", "linea": line, "columna": col + 1,
                    "error": re.sub(r":\s*line \d+, column \d+$", "", str(e))}
    if f == "toml":
        try:
            import tomllib
        except ImportError:                          # Python < 3.11
            try:
                import tomli as tomllib
            except ImportError:
                return {"valido": None, "formato": "TOML", "error": "tomllib no disponible (Python ≥ 3.11 o pip install tomli)"}
        try:
            v = tomllib.loads(text)
            return {"valido": True, "formato": "TOML", "claves": len(v)}
        except tomllib.TOMLDecodeError as e:
            m = re.search(r"\(at line (\d+), column (\d+)\)", str(e))
            out = {"valido": False, "formato": "TOML", "error": re.sub(r"\s*\(at (line|end of document).*\)$", "", str(e))}
            if m:
                out.update(linea=int(m.group(1)), columna=int(m.group(2)))
            elif str(e).endswith("(at end of document)"):
                out.update(linea=text.count("\n") + 1, columna=len(text.rsplit("\n", 1)[-1]) + 1)
            return out
    raise ValueError("formato: json | yaml | xml | toml")


def validate_text(r: dict) -> str:
    if r["valido"]:
        extra = {k: v for k, v in r.items() if k not in ("valido", "formato")}
        return f"{r['formato']} válido" + (" (" + ", ".join(f"{k} {v}" for k, v in extra.items()) + ")" if extra else "")
    if r["valido"] is None:
        return r["error"]
    where = f" en línea {r['linea']}, columna {r['columna']}" if "linea" in r else ""
    return f"{r['formato']} inválido{where}: {r['error']}"


# ---------------------------------------------------------------------------
# Expresiones regulares

def regex_test(pattern: str, cases: list[str], flags: str = "") -> dict:
    fl = 0
    for ch in flags.lower():
        fl |= {"i": re.I, "m": re.M, "s": re.S, "x": re.X}.get(ch, 0)
    if len(pattern) > 2000 or sum(len(c) for c in cases) > 200_000:
        raise ValueError("patrón o casos demasiado largos")
    try:
        rx = re.compile(pattern, fl)
    except re.error as e:
        return {"valido": False, "error": e.msg, "posicion": e.pos}
    out = []
    for c in cases[:50]:
        found = [{"texto": m.group(0), "inicio": m.start(), "grupos": list(m.groups()),
                  **({"nombrados": m.groupdict()} if m.groupdict() else {})} for _, m in zip(range(20), rx.finditer(c))]
        out.append({"caso": c, "completa": rx.fullmatch(c) is not None, "busquedas": found})
    return {"valido": True, "grupos": rx.groups, "casos": out}


def regex_text(r: dict) -> str:
    if not r["valido"]:
        return f"patrón inválido en la posición {r['posicion']}: {r['error']}"
    parts = []
    for c in r["casos"]:
        hits = ", ".join(f"«{h['texto']}»@{h['inicio']}" + (f" grupos {h['grupos']}" if h["grupos"] else "")
                         for h in c["busquedas"][:5])
        parts.append(f"«{c['caso'][:40]}»: {'coincide completo' if c['completa'] else 'no coincide completo'}"
                     + (f"; encuentra {hits}" if hits else "; sin coincidencias"))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Java

JAVA_FLAGS = ["-Xmx256m", "-Xss4m", "-XX:+UseSerialGC", "-XX:TieredStopAtLevel=1", "-XX:CICompilerCount=1",
              "-XX:ReservedCodeCacheSize=32m", "-XX:CompressedClassSpaceSize=64m", "-XX:MaxMetaspaceSize=128m",
              "-Xshare:auto"]


def java_available() -> bool:
    return shutil.which("javac") is not None and shutil.which("java") is not None


def _java_files(code: str) -> tuple[dict[str, str], str]:
    """Uno o varios archivos («// archivo: Foo.java» separa) y la clase con ``main``."""
    files: dict[str, str] = {}
    chunks = re.split(r"^\s*//\s*(?:archivo|file)\s*:\s*(\S+\.java)\s*$", code, flags=re.M)
    if len(chunks) > 1:
        for name, body in zip(chunks[1::2], chunks[2::2]):
            files[name] = body.strip() + "\n"
    else:
        m = re.search(r"public\s+(?:final\s+|abstract\s+)*(?:class|record|enum|interface)\s+(\w+)", code)
        files[(m.group(1) if m else "Main") + ".java"] = code
    main = None
    for name, body in files.items():
        if re.search(r"static\s+void\s+main\s*\(", body):
            pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", body, re.M)
            main = (pkg.group(1) + "." if pkg else "") + name[:-5]
            break
    if main is None:
        raise ValueError("ninguna clase tiene «public static void main(String[] args)»")
    for name in files:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"nombre de archivo inválido: {name}")
    return files, main


def java_run(code: str, stdin: str = "", timeout_s: float = 60.0) -> dict:
    """Compila con javac y ejecuta con java, en una carpeta temporal del sandbox."""
    if not java_available():
        return {"ok": False, "stage": "entorno",
                "errors": "Java no está instalado (falta javac/java). Instalá un JDK 17 o 21 (por ejemplo "
                          "Temurin; en Linux «sudo apt install openjdk-21-jdk-headless») y volvé a intentar."}
    files, main = _java_files(code)
    jflags = " ".join(f"-J{f}" for f in JAVA_FLAGS[:3])
    script = (f"mkdir -p out && javac {jflags} -encoding UTF-8 -Xlint:all -d out {' '.join(files)} 2> compile.txt; "
              "status=$?; if [ $status -ne 0 ]; then echo '__COMPILE_ERROR__'; cat compile.txt; exit 1; fi; "
              f"cat compile.txt 1>&2; java {' '.join(JAVA_FLAGS)} -cp out {main}")
    r = run_command(["/bin/sh", "-c", script], files=files, stdin=stdin, timeout_s=timeout_s, mem_mb=4096)
    if "__COMPILE_ERROR__" in r.stdout:
        return {"ok": False, "stage": "compilación", "errors": r.stdout.split("__COMPILE_ERROR__", 1)[1][-MAX:]}
    return {"ok": r.ok, "stage": "ejecución", "stdout": r.stdout[-MAX:], "stderr": r.stderr[-MAX:],
            "returncode": r.returncode, "timed_out": r.timed_out, "main": main}


def java_text(r: dict) -> str:
    if r["stage"] in ("entorno", "compilación"):
        return ("error de compilación: " if r["stage"] == "compilación" else "") + r["errors"].strip()[-500:]
    if r["timed_out"]:
        return "tiempo agotado"
    if not r["ok"]:
        return f"error de ejecución (código {r['returncode']}): " + (r["stderr"].strip().splitlines() or ["?"])[0][:300]
    return r["stdout"].strip()


# ---------------------------------------------------------------------------
# Git (solo lectura)

GIT_SAFE = ["-c", "core.fsmonitor=false", "-c", "core.pager=cat", "-c", "diff.external=", "-c", "core.hooksPath=/dev/null",
            "-c", "safe.directory=*", "-c", "protocol.allow=never", "-c", "core.sshCommand=false",
            "-c", "credential.helper=", "-c", "core.untrackedCache=false", "--no-pager"]
REV = re.compile(r"[\w./~^@{}:-]+")


def _safe_arg(a: str) -> str:
    if a.startswith("-") or not REV.fullmatch(a) or ".." in a and not re.fullmatch(r"[\w./~^@-]+\.\.\.?[\w./~^@-]+", a):
        raise ValueError(f"argumento no permitido: {a!r}")
    return a


def git_read(repo: str, sub: str, args: list[str], max_count: int = 20, timeout_s: float = 20.0) -> dict:
    """``repo`` ya autorizado por el guardián (lectura). Solo log/show/blame/diff/status, sin pager,
    sin diff externos ni textconv, sin hooks ni red; en el sandbox."""
    git = shutil.which("git")
    if git is None:
        return {"ok": False, "salida": "git no está instalado"}
    sub = sub.strip().lower()
    a = [_safe_arg(x) for x in args]
    if sub == "log":
        argv = ["log", f"--max-count={max(1, min(max_count, 200))}", "--date=short",
                "--format=%h %ad %an: %s", "--no-color", *a]
    elif sub == "show":
        argv = ["show", "--stat", "-p", "--no-color", "--no-ext-diff", "--no-textconv", "--format=%H%n%an <%ae>%n%ad%n%n%B",
                *(a or ["HEAD"])]
    elif sub == "blame":
        if not a:
            raise ValueError("blame necesita un archivo")
        argv = ["blame", "--date=short", "-w", *(a[:-1]), "--", a[-1]]
    elif sub == "diff":
        argv = ["diff", "--no-color", "--no-ext-diff", "--no-textconv", "--stat", "-p", *a]
    elif sub == "status":
        argv = ["status", "--short", "--branch", "--untracked-files=no"]
    else:
        raise ValueError("subcomando: log | show | blame | diff | status (solo lectura)")
    r = run_command([git, *GIT_SAFE, "-C", repo, *argv], timeout_s=timeout_s)
    out = (r.stdout if r.ok else (r.stderr or r.stdout)).strip()
    return {"ok": r.ok, "salida": out[-6000:], "comando": "git " + " ".join(argv)}
