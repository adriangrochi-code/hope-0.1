"""Herramientas de nivel 0–1 y lógica (docs/HERRAMIENTAS_HN1.md §3).

* ``fraction_calc`` — aritmética exacta con fracciones (evaluador propio, sin ``eval``).
* ``stats`` — estadística descriptiva de una lista de números.
* ``solve`` / ``derivative`` / ``integral`` / ``limit`` — SymPy, **en sandbox**.
* ``linalg`` — determinante, inversa, autovalores o solución de A·x = b (NumPy, en sandbox).
* ``logic`` — satisfacibilidad y modelos con z3 (en sandbox).

Lo que usa bibliotecas que interpretan texto (SymPy, z3) corre en el sandbox,
porque la entrada la escribe el modelo y no es confiable.
"""

from __future__ import annotations

import ast
import json
import operator
import statistics
from fractions import Fraction

from hope.tools.builtin import normalize_number_text
from hope.tools.sandbox import run_python

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def fraction_calc(expr: str) -> Fraction:
    tree = ast.parse(normalize_number_text(expr), mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return Fraction(str(n.value))
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            e = ev(n.right)
            if e.denominator != 1 or abs(e) > 64:
                raise ValueError("solo potencias enteras pequeñas")
            return ev(n.left) ** int(e)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError("expresión no permitida")
    return ev(tree)


def stats(numbers: str) -> dict:
    xs = [float(normalize_number_text(t)) for t in numbers.replace(";", " ").replace(",", " ").split() if t]
    if not xs:
        raise ValueError("no hay números")
    return {"n": len(xs), "media": statistics.fmean(xs), "mediana": statistics.median(xs),
            "desvio": statistics.pstdev(xs), "min": min(xs), "max": max(xs)}


_SYMPY_HEAD = (
    "import json, sys, sympy as sp\n"
    "from sympy.parsing.sympy_parser import parse_expr, standard_transformations, "
    "implicit_multiplication_application, convert_xor\n"
    "T = standard_transformations + (implicit_multiplication_application, convert_xor)\n"
    "P = lambda s: parse_expr(s, transformations=T)\n"
    "d = json.loads(sys.stdin.read())\n")


def _sandboxed(code: str, payload: dict, timeout_s: float = 30.0):
    res = run_python(_SYMPY_HEAD + code, stdin=json.dumps(payload), timeout_s=timeout_s)
    if not res.ok:
        raise ValueError((res.stderr.strip().splitlines() or ["error"])[-1][:200])
    return json.loads(res.stdout.strip().splitlines()[-1])


def solve(equations: str, variables: str = "") -> list[dict]:
    """``equations``: «2x + 3 = 7» o varias separadas por «;». Devuelve las soluciones."""
    code = (
        "eqs = []\n"
        "for e in d['eqs'].split(';'):\n"
        "    l, r = (e.split('=', 1) + ['0'])[:2]\n"
        "    eqs.append(sp.Eq(P(l), P(r)))\n"
        "vs = [sp.Symbol(v.strip()) for v in d['vars'].split(',') if v.strip()] or sorted(\n"
        "    set().union(*[e.free_symbols for e in eqs]), key=str)\n"
        "sol = sp.solve(eqs, vs, dict=True)\n"
        "print(json.dumps([{str(k): str(sp.nsimplify(v)) for k, v in s.items()} for s in sol]))\n")
    return _sandboxed(code, {"eqs": normalize_number_text(equations), "vars": variables})


def derivative(expr: str, var: str = "x", order: int = 1) -> str:
    return _sandboxed("print(json.dumps(str(sp.simplify(sp.diff(P(d['e']), sp.Symbol(d['v']), d['n'])))))\n",
                      {"e": normalize_number_text(expr), "v": var, "n": int(order)})


def integral(expr: str, var: str = "x", lower: str | None = None, upper: str | None = None) -> str:
    code = ("x = sp.Symbol(d['v'])\n"
            "r = sp.integrate(P(d['e']), (x, P(d['a']), P(d['b']))) if d['a'] is not None else sp.integrate(P(d['e']), x)\n"
            "print(json.dumps(str(sp.simplify(r))))\n")
    return _sandboxed(code, {"e": normalize_number_text(expr), "v": var, "a": lower, "b": upper})


def limit(expr: str, var: str = "x", to: str = "0") -> str:
    return _sandboxed("print(json.dumps(str(sp.limit(P(d['e']), sp.Symbol(d['v']), P(d['t'])))))\n",
                      {"e": normalize_number_text(expr), "v": var, "t": to})


def linalg(op: str, matrix: str, vector: str | None = None) -> object:
    """``op``: det | inv | eig | solve; ``matrix``: JSON, p. ej. "[[1,2],[3,4]]"."""
    code = (
        "import numpy as np\n"
        "A = np.array(json.loads(d['m']), dtype=float)\n"
        "op = d['op']\n"
        "if op == 'det': r = float(np.linalg.det(A))\n"
        "elif op == 'inv': r = np.linalg.inv(A).round(10).tolist()\n"
        "elif op == 'eig': r = sorted(np.linalg.eigvals(A).real.round(10).tolist())\n"
        "elif op == 'solve': r = np.linalg.solve(A, np.array(json.loads(d['b']), dtype=float)).round(10).tolist()\n"
        "else: raise ValueError('operación desconocida')\n"
        "print(json.dumps(r))\n")
    return _sandboxed(code, {"op": op, "m": matrix, "b": vector})


def logic(constraints: str, variables: str = "", kind: str = "Int") -> dict:
    """Restricciones separadas por «;» sobre variables enteras, reales o booleanas.

    Ej.: logic("x + y == 10; x - y == 2", "x,y") → {"sat": True, "modelo": {"x": "6", "y": "4"}}.
    Las expresiones se interpretan con z3 dentro del sandbox.
    """
    code = (
        "import z3, re\n"
        "names = [v.strip() for v in d['vars'].split(',') if v.strip()] or sorted(set(re.findall(r'\\b[a-zA-Z_]\\w*\\b', d['c'])) - {'And','Or','Not','Implies','True','False'})\n"
        "mk = {'Int': z3.Int, 'Real': z3.Real, 'Bool': z3.Bool}[d['k']]\n"
        "env = {n: mk(n) for n in names}\n"
        "env.update(And=z3.And, Or=z3.Or, Not=z3.Not, Implies=z3.Implies)\n"
        "s = z3.Solver()\n"
        "for c in d['c'].split(';'):\n"
        "    if c.strip(): s.add(eval(c, {'__builtins__': {}}, env))\n"
        "r = s.check()\n"
        "out = {'sat': r == z3.sat}\n"
        "if r == z3.sat:\n"
        "    m = s.model()\n"
        "    out['modelo'] = {n: str(m.eval(env[n], model_completion=True)) for n in names}\n"
        "print(json.dumps(out))\n")
    return _sandboxed(code, {"c": constraints, "vars": variables, "k": kind})
