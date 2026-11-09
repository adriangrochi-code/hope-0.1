"""Herramientas de programación para Hope Code (docs/HERRAMIENTAS_HN1.md §3, nivel 3).

Todo lo que ejecuta código corre en el sandbox (carpeta temporal, sin entorno,
con límites). La búsqueda en archivos pide permiso al guardián: dentro del
directorio de trabajo es libre; fuera, requiere aprobación humana.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import sys

from hope.tools.sandbox import run_command

MAX = 2000


def run_tests(code: str, tests: str, timeout_s: float = 60.0) -> dict:
    """Ejecuta ``tests`` (pytest) contra ``code`` guardado como ``solucion.py``."""
    r = run_command([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_solucion.py"],
                    files={"solucion.py": code, "test_solucion.py": tests}, timeout_s=timeout_s)
    tail = (r.stdout + r.stderr).strip().splitlines()
    summary = next((line for line in reversed(tail) if re.search(r"passed|failed|error", line)), "")
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", summary)) else 0
    return {"ok": r.ok, "passed": passed, "failed": failed, "summary": summary,
            "output": "\n".join(tail[-30:])[-MAX:], "timed_out": r.timed_out}


def lint(code: str) -> dict:
    """Análisis estático con ruff (errores y sospechas sin ejecutar el código)."""
    ruff = shutil.which("ruff") or os.path.expanduser("~/.local/bin/ruff")
    if not os.path.exists(ruff):
        return {"ok": False, "issues": [], "error": "ruff no disponible"}
    r = run_command([ruff, "check", "--output-format", "json", "--no-cache", "codigo.py"],
                    files={"codigo.py": code}, timeout_s=30)
    try:
        issues = [{"line": i["location"]["row"], "code": i["code"], "message": i["message"]}
                  for i in json.loads(r.stdout or "[]")]
    except json.JSONDecodeError:
        return {"ok": False, "issues": [], "error": (r.stderr or r.stdout)[-300:]}
    return {"ok": not issues, "issues": issues}


def compile_run(code: str, lang: str = "c", stdin: str = "", timeout_s: float = 20.0) -> dict:
    """Compila C/C++ con gcc/g++ (o clang) y ejecuta el binario, todo en el sandbox."""
    compiler = {"c": ["gcc", "-std=c11"], "cpp": ["g++", "-std=c++17"]}.get(lang)
    if compiler is None:
        raise ValueError("lenguaje: c | cpp")
    src = "main.c" if lang == "c" else "main.cpp"
    script = (f"{' '.join(compiler)} -O1 -Wall -o prog {src} 2> compile.txt; status=$?; "
              "if [ $status -ne 0 ]; then echo '__COMPILE_ERROR__'; cat compile.txt; exit 1; fi; "
              "cat compile.txt 1>&2; ./prog")
    r = run_command(["/bin/sh", "-c", script], files={src: code}, stdin=stdin, timeout_s=timeout_s)
    if "__COMPILE_ERROR__" in r.stdout:
        return {"ok": False, "stage": "compilación", "errors": r.stdout.split("__COMPILE_ERROR__", 1)[1][-MAX:]}
    return {"ok": r.ok, "stage": "ejecución", "stdout": r.stdout[-MAX:], "warnings": r.stderr[-MAX:],
            "returncode": r.returncode, "timed_out": r.timed_out}


def search_code(pattern: str, root: str, max_results: int = 30) -> list[dict]:
    """Busca un patrón (regex) en archivos de ``root`` con ripgrep. El permiso lo verifica el llamador."""
    rg = shutil.which("rg")
    if rg is None:
        raise RuntimeError("ripgrep no disponible")
    r = run_command([rg, "--json", "-m", "5", "-e", pattern, os.path.abspath(root)], timeout_s=30)
    out = []
    for line in r.stdout.splitlines():
        ev = json.loads(line)
        if ev.get("type") == "match":
            d = ev["data"]
            out.append({"path": d["path"]["text"], "line": d["line_number"], "text": d["lines"]["text"].rstrip()[:200]})
            if len(out) >= max_results:
                break
    return out


def diff(a: str, b: str, name: str = "archivo") -> str:
    """Diff unificado entre dos versiones de un texto (cambios revisables)."""
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                        fromfile=f"{name} (antes)", tofile=f"{name} (después)"))[-MAX:]
