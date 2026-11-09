"""Trazas de ejecución para entrenar (estilo *Code World Model*, Meta 2025).

Idea: un modelo que aprende **qué hace** el código, no solo cómo se ve, programa mejor. Se
ejecutan programas Python chicos en el sandbox con ``sys.settrace`` y se escribe, para cada
línea ejecutada, el **estado de las variables después de esa línea** y lo que imprimió. Todo sale
de una ejecución real: no hay nada que creerle a un maestro.

Dos tareas por programa:

* **traza** — código numerado + una línea por paso ``n| var=valor …`` (+ ``⇒ «salida»``) + salida;
* **predecir la salida** (la tarea inversa) — código y «¿Qué imprime?» → salida exacta.

Ejemplo de traza::

    ⟦traza⟧
    Programa:
    1| t = 0
    2| for i in range(3):
    3|     t += i
    4| print(t)
    Ejecución:
    1| t=0
    2| t=0 i=0
    3| t=0 i=0
    …
    4| t=3 i=2 ⇒ «3»
    Salida:
    3
    ⟦fin⟧

Fuentes de programas:

* ``generated_programs`` — generador procedural (bucles, listas, diccionarios, cadenas,
  funciones, recursión, excepciones) con parámetros al azar;
* ``corpus_programs`` — funciones con ejemplos *doctest* y scripts cortos autocontenidos del
  corpus de código **con licencia permisiva** (MIT, BSD, Apache, ISC, PSF, CC0…); la licencia
  queda en la procedencia. La rama «codigo-abierto» arma el corpus; acá se lee una carpeta;
* ``exercise_programs`` — las soluciones de referencia del banco de código
  (``hope/bench/codigo.py``) aplicadas a las entradas de sus pruebas.

**Determinismo:** cada lote se ejecuta **dos veces** en procesos distintos (con otra semilla de
hash) y solo se guardan los programas cuya traza coincide; los módulos no deterministas
(``random``, ``time``, ``os``…) se rechazan antes de ejecutar.

Uso::

    python -m hope.data.trazas --n 2000 --out data/trazas/trazas.jsonl [--corpus carpeta]
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from hope.tools.sandbox import run_python

SAFE_MODULES = frozenset({"math", "itertools", "collections", "functools", "string", "re", "heapq", "bisect",
                          "operator", "fractions", "decimal", "statistics", "dataclasses", "typing", "enum"})
BANNED_NAMES = frozenset({"open", "input", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                          "id", "hash", "breakpoint", "exit", "quit", "help", "memoryview", "setattr", "delattr",
                          "getattr"})

# El trazador corre DENTRO del sandbox (python -I -c). Lee los programas por la entrada estándar.
TRACER = r'''
import sys, json, io, types
data = json.loads(sys.stdin.read())
MAXS, MAXR = data["max_steps"], data["max_repr"]
SKIP = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)
def rep(v, depth=0):
    if isinstance(v, (set, frozenset)):
        try:
            items = sorted(v)
        except TypeError:
            items = list(v)
        r = ("{" + ", ".join(rep(x, 1) for x in items) + "}") if items else ("set()" if isinstance(v, set) else "frozenset()")
    else:
        try:
            r = repr(v)
        except Exception:
            r = "<?>"
    if " at 0x" in r:
        r = "<" + type(v).__name__ + ">"
    return r if len(r) <= MAXR else r[:MAXR - 1] + "…"
class Stop(Exception):
    pass
results = []
real_stdout = sys.stdout
for prog in data["programs"]:
    buf = io.StringIO()
    steps, pending, state = [], {}, {"n": 0, "out": 0, "trunc": False}
    def snap(frame):
        return {k: rep(v) for k, v in frame.f_locals.items()
                if not k.startswith("__") and not isinstance(v, SKIP)}
    def record(frame, line, extra=None):
        out = buf.getvalue()
        new, state["out"] = out[state["out"]:], len(out)
        s = {"linea": line, "vars": snap(frame), "salida": new,
             "func": frame.f_code.co_name if frame.f_code.co_name != "<module>" else ""}
        if extra:
            s.update(extra)
        steps.append(s)
        state["n"] += 1
        if state["n"] >= MAXS:
            state["trunc"] = True
            raise Stop()
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != "<programa>":
            return None
        if event == "line":
            if frame in pending:
                record(frame, pending[frame])
            pending[frame] = frame.f_lineno
        elif event == "return":
            if frame in pending:
                extra = {"retorna": rep(arg)} if frame.f_code.co_name != "<module>" else None
                record(frame, pending.pop(frame), extra)
        return tracer
    err = None
    g = {"__name__": "__main__", "__builtins__": __builtins__}
    sys.stdin = io.StringIO(prog.get("stdin", ""))
    sys.stdout = buf
    try:
        code = compile(prog["code"], "<programa>", "exec")
        sys.settrace(tracer)
        try:
            exec(code, g)
        finally:
            sys.settrace(None)
    except Stop:
        err = "pasos"
    except BaseException as e:
        err = type(e).__name__ + ": " + str(e)[:200]
    sys.stdout = real_stdout
    results.append({"steps": steps, "salida": buf.getvalue(), "error": err, "truncado": state["trunc"]})
print(json.dumps(results))
'''


@dataclass
class Trace:
    code: str
    steps: list[dict]
    output: str
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def check_program(code: str) -> str | None:
    """``None`` si se puede trazar; si no, el motivo (módulos no deterministas, E/S, etc.)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"sintaxis: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in SAFE_MODULES:
                    return f"módulo no permitido: {m}"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"nombre no permitido: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__init__", "__name__"):
            return f"atributo interno: {node.attr}"
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            return "global/nonlocal/async"
    return None


def _run(programs: list[dict], max_steps: int, max_repr: int, timeout_s: float) -> list[dict] | None:
    payload = json.dumps({"programs": programs, "max_steps": max_steps, "max_repr": max_repr})
    r = run_python(TRACER, stdin=payload, timeout_s=timeout_s, mem_mb=1024)
    if not r.ok:
        return None
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None


def trace_programs(codes: list[str], *, stdins: list[str] | None = None, max_steps: int = 120,
                   max_repr: int = 60, timeout_s: float = 30.0, verify_twice: bool = True,
                   metas: list[dict] | None = None) -> list[Trace | None]:
    """Traza cada programa (en un solo proceso del sandbox por lote). ``None`` para los que no se
    pudieron trazar: rechazados, con error, truncados por cantidad de pasos o no deterministas."""
    out: list[Trace | None] = [None] * len(codes)
    idx = [i for i, c in enumerate(codes) if check_program(c) is None]
    if not idx:
        return out
    progs = [{"code": codes[i], "stdin": (stdins or [""] * len(codes))[i]} for i in idx]
    res = _run(progs, max_steps, max_repr, timeout_s)
    if res is None and len(idx) > 1:                      # un programa rompió el lote: de a uno
        res = []
        for p in progs:
            one = _run([p], max_steps, max_repr, timeout_s / 2)
            res.append(one[0] if one else {"error": "sandbox", "steps": [], "salida": "", "truncado": False})
    if res is None:
        return out
    res2 = _run(progs, max_steps, max_repr, timeout_s) if verify_twice else res
    for k, i in enumerate(idx):
        a = res[k]
        b = res2[k] if res2 is not None and k < len(res2) else None
        if a.get("error") or a.get("truncado") or not a.get("steps"):
            continue
        if verify_twice and (b is None or b != a):
            continue                                      # no determinista (p. ej. orden de un set de cadenas)
        out[i] = Trace(codes[i], a["steps"], a["salida"], None, (metas or [{}] * len(codes))[i])
    return out


# ---------------------------------------------------------------------------
# Textos de entrenamiento
# ---------------------------------------------------------------------------
def numbered(code: str) -> str:
    return "\n".join(f"{i}| {line}" for i, line in enumerate(code.rstrip("\n").split("\n"), 1))


def _vars_text(v: dict, prev: dict | None, mode: str) -> str:
    items = v.items() if mode == "completo" or prev is None else [(k, x) for k, x in v.items() if prev.get(k) != x]
    return " ".join(f"{k}={x}" for k, x in items)


def trace_text(t: Trace, *, mode: str = "completo", max_steps_text: int = 80) -> str:
    """Texto de la tarea «traza». ``mode="cambios"`` escribe solo las variables que cambiaron."""
    lines = ["⟦traza⟧", "Programa:", numbered(t.code), "Ejecución:"]
    prev_by_func: dict[str, dict] = {}
    for s in t.steps[:max_steps_text]:
        f = s.get("func", "")
        txt = _vars_text(s["vars"], prev_by_func.get(f), mode)
        prev_by_func[f] = s["vars"]
        row = f"{s['linea']}|" + (f" [{f}]" if f else "") + (f" {txt}" if txt else "")
        if "retorna" in s:
            row += f" ← {s['retorna']}"
        if s["salida"]:
            row += f" ⇒ «{s['salida'].rstrip(chr(10))}»"
        lines.append(row)
    if len(t.steps) > max_steps_text:
        lines.append(f"… ({len(t.steps) - max_steps_text} pasos más)")
    lines += ["Salida:", t.output.rstrip("\n"), "⟦fin⟧"]
    return "\n".join(lines)


def predict_text(t: Trace) -> str:
    """La tarea inversa: el programa y su salida exacta (predecir qué imprime)."""
    return "\n".join(["⟦predecir_salida⟧", "Programa:", t.code.rstrip("\n"), "¿Qué imprime?", "Salida:",
                      t.output.rstrip("\n"), "⟦fin⟧"])


def predict_prompt(code: str) -> str:
    """El enunciado de la tarea inversa sin la respuesta (para evaluar)."""
    return "\n".join(["⟦predecir_salida⟧", "Programa:", code.rstrip("\n"), "¿Qué imprime?", "Salida:", ""])


# ---------------------------------------------------------------------------
# Programas generados
# ---------------------------------------------------------------------------
_NAMES = ["ana", "beto", "carla", "dani", "eva", "fede", "gabi", "hugo"]
_WORDS = ["sol", "mar", "casa", "luz", "perro", "gato", "árbol", "río", "nube", "pan"]


def _p_loop_sum(r: random.Random) -> str:
    n, k, op = r.randint(3, 7), r.randint(1, 5), r.choice(["+", "*"])
    start = 0 if op == "+" else 1
    expr = f"i * {k}" if op == "+" else f"(i % {k} + 1)"
    return f"t = {start}\nfor i in range({n}):\n    t {op}= {expr}\nprint(t)"


def _p_while(r: random.Random) -> str:
    n, d = r.randint(20, 200), r.choice([2, 3, 10])
    return f"n = {n}\npasos = 0\nwhile n > 0:\n    n //= {d}\n    pasos += 1\nprint(pasos, n)"


def _p_list(r: random.Random) -> str:
    xs = [r.randint(-9, 20) for _ in range(r.randint(4, 7))]
    f = r.choice(["x % 2 == 0", "x > 3", "x < 0 or x > 10"])
    return f"xs = {xs}\nys = []\nfor x in xs:\n    if {f}:\n        ys.append(x * 2)\nprint(ys, len(ys))"


def _p_comprehension(r: random.Random) -> str:
    n = r.randint(4, 9)
    e = r.choice(["i * i", "i + 1", "i % 3"])
    c = r.choice(["i % 2 == 1", "i > 2", "True"])
    return f"cuad = [{e} for i in range({n}) if {c}]\ntotal = sum(cuad)\nprint(cuad)\nprint(total)"


def _p_dict_count(r: random.Random) -> str:
    w = " ".join(r.choice(_WORDS[:5]) for _ in range(r.randint(4, 8)))
    return (f"texto = {w!r}\ncuenta = {{}}\nfor p in texto.split():\n    cuenta[p] = cuenta.get(p, 0) + 1\n"
            f"mas = max(cuenta, key=cuenta.get)\nprint(cuenta[mas], len(cuenta))")


def _p_string(r: random.Random) -> str:
    s = r.choice(_WORDS + _NAMES)
    op = r.choice(["s[::-1]", "s.upper()", "s[1:-1]", "s * 2", "s.replace('a', 'o')", "len(s)"])
    return f"s = {s!r}\nt = {op}\nprint(t)\nprint(t == s)"


def _p_function(r: random.Random) -> str:
    a, b = r.randint(1, 9), r.randint(1, 9)
    body = r.choice(["return x * x + y", "if x > y:\n        return x - y\n    return y - x", "return (x + y) // 2"])
    return f"def f(x, y):\n    {body}\n\nr = f({a}, {b})\nprint(r)\nprint(f(r, {a}))"


def _p_recursion(r: random.Random) -> str:
    n = r.randint(3, 6)
    if r.random() < 0.5:
        return f"def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\n\nprint(fact({n}))"
    return f"def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\nprint(fib({n}))"


def _p_nested(r: random.Random) -> str:
    n, m = r.randint(2, 4), r.randint(2, 3)
    return f"pares = []\nfor i in range({n}):\n    for j in range({m}):\n        if (i + j) % 2 == 0:\n            pares.append((i, j))\nprint(len(pares), pares[-1])"


def _p_exception(r: random.Random) -> str:
    xs = [r.randint(0, 3) for _ in range(4)]
    return (f"xs = {xs}\nok = 0\nfallas = 0\nfor x in xs:\n    try:\n        y = 12 // x\n        ok += y\n"
            f"    except ZeroDivisionError:\n        fallas += 1\nprint(ok, fallas)")


def _p_sort(r: random.Random) -> str:
    people = r.sample(_NAMES, 4)
    ages = [r.randint(18, 70) for _ in people]
    return (f"edades = {dict(zip(people, ages))}\norden = sorted(edades, key=lambda n: edades[n])\n"
            f"print(orden[0], orden[-1])\nprint(sum(edades.values()) // len(edades))")


def _p_swap(r: random.Random) -> str:
    a, b = r.randint(1, 50), r.randint(1, 50)
    return f"a = {a}\nb = {b}\nwhile b:\n    a, b = b, a % b\nprint(a)"


def _p_class(r: random.Random) -> str:
    n = r.randint(2, 5)
    return (f"class Contador:\n    def __init__(self):\n        self.n = 0\n\n    def sumar(self, k):\n"
            f"        self.n += k\n        return self.n\n\nc = Contador()\nfor k in range({n}):\n    c.sumar(k)\nprint(c.n)")


GENERATORS = [_p_loop_sum, _p_while, _p_list, _p_comprehension, _p_dict_count, _p_string, _p_function,
              _p_recursion, _p_nested, _p_exception, _p_sort, _p_swap, _p_class]


def generated_programs(n: int, *, seed: int = 0) -> list[tuple[str, dict]]:
    rng = random.Random(f"trazas:{seed}")
    out = []
    for i in range(n):
        g = GENERATORS[i % len(GENERATORS)] if i < len(GENERATORS) else rng.choice(GENERATORS)
        out.append((g(rng), {"fuente": "generador", "familia": g.__name__[3:]}))
    return out


# ---------------------------------------------------------------------------
# Corpus con licencia permisiva
# ---------------------------------------------------------------------------
PERMISSIVE = re.compile(r"\b(MIT License|Permission is hereby granted, free of charge|BSD|Apache License|"
                        r"ISC License|Python Software Foundation|CC0|Unlicense|public domain|zlib)", re.I)
SPDX = re.compile(r"SPDX-License-Identifier:\s*(MIT|BSD-\d-Clause|Apache-2\.0|ISC|PSF-2\.0|CC0-1\.0|Unlicense|0BSD)")


def license_of(path: Path) -> str | None:
    """Licencia permisiva de un archivo (SPDX en el encabezado o un LICENSE en una carpeta superior)."""
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:2000]
    except OSError:
        return None
    m = SPDX.search(head)
    if m:
        return m.group(1)
    for d in [path.parent, *path.parents][:6]:
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "LICENCE"):
            f = d / name
            if f.is_file():
                m = PERMISSIVE.search(f.read_text(encoding="utf-8", errors="ignore")[:3000])
                return m.group(1) if m else None
    return None


DOCTEST = re.compile(r"^\s*>>> (.+)$", re.M)


def programs_from_source(src: str, *, max_lines: int = 40) -> list[str]:
    """Programas autocontenidos a partir de un archivo: cada función con ejemplos doctest (la función
    más ``print`` de cada ejemplo) y el archivo entero si es un script corto sin dependencias."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    lines = src.split("\n")
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        doc = ast.get_docstring(node) or ""
        calls = [c.strip() for c in DOCTEST.findall(doc) if node.name + "(" in c][:4]
        if not calls:
            continue
        fsrc = "\n".join(lines[node.lineno - 1 - len(node.decorator_list):node.end_lineno])
        if fsrc.count("\n") > max_lines:
            continue
        prog = fsrc + "\n\n" + "\n".join(f"print(repr({c}))" for c in calls)
        if check_program(prog) is None:
            out.append(prog)
    if len(lines) <= max_lines and "print(" in src and check_program(src) is None \
            and not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body):
        out.append(src.rstrip("\n"))
    return out


def corpus_programs(root: str | Path, *, max_files: int = 5000,
                    require_license: bool = True) -> list[tuple[str, dict]]:
    out = []
    for i, p in enumerate(sorted(Path(root).rglob("*.py"))):
        if i >= max_files:
            break
        lic = license_of(p)
        if require_license and lic is None:
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for prog in programs_from_source(src):
            out.append((prog, {"fuente": "corpus", "ruta": str(p), "licencia": lic}))
    return out


def exercise_programs(problems: Iterable[dict], *, per_problem: int = 2) -> list[tuple[str, dict]]:
    """Soluciones de referencia del banco aplicadas a las entradas de sus pruebas."""
    out = []
    for p in problems:
        calls = [m.group(1) for m in (re.match(r"\s*assert\s+(.+?)\s*==", t) for t in p.get("pruebas", [])) if m]
        if not calls or not p.get("solucion"):
            continue
        prog = p["solucion"].rstrip("\n") + "\n\n" + "\n".join(f"print({c})" for c in calls[:per_problem])
        out.append((prog, {"fuente": "ejercicio", "id": p.get("id"), "licencia": p.get("licencia", "propia")}))
    return out


# ---------------------------------------------------------------------------
def build(programs: list[tuple[str, dict]], *, batch: int = 64, mode: str = "completo",
          max_steps: int = 120) -> list[dict]:
    """Traza los programas y devuelve registros con los dos textos de entrenamiento."""
    rows = []
    for s in range(0, len(programs), batch):
        chunk = programs[s:s + batch]
        traces = trace_programs([c for c, _ in chunk], metas=[m for _, m in chunk], max_steps=max_steps)
        for t in traces:
            if t is None:
                continue
            rows.append({"codigo": t.code, "salida": t.output, "pasos": len(t.steps), **t.meta,
                         "traza": trace_text(t, mode=mode), "prediccion": predict_text(t)})
    return rows


def texts(rows: list[dict], *, tasks: tuple[str, ...] = ("traza", "prediccion")) -> list[str]:
    return [r[k] for r in rows for k in tasks if k in r]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Trazas de ejecución de programas Python (código + estado + salida)")
    ap.add_argument("--n", type=int, default=500, help="programas generados")
    ap.add_argument("--corpus", default=None, help="carpeta con código de licencia permisiva")
    ap.add_argument("--ejercicios", action="store_true", help="incluir las soluciones del banco de código")
    ap.add_argument("--modo", default="completo", choices=["completo", "cambios"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    progs = generated_programs(a.n, seed=a.seed)
    if a.corpus:
        progs += corpus_programs(a.corpus)
    if a.ejercicios:
        from hope.bench.codigo import bank
        progs += exercise_programs(bank())
    rows = build(progs, mode=a.modo)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"programas": len(progs), "trazados": len(rows), "salida": a.out}
    print(json.dumps(stats, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
