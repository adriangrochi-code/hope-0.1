"""Herramientas de cálculo exacto **certificado** (ARQUITECTURA_HN1 §11.16 T3).

Cada función devuelve un :class:`~hope.api.resultado.Resultado` con el valor **y** una prueba o una
segunda verificación independiente:

* ``calc_cert``: aritmética de intervalos (mpmath.iv): el valor real está garantizado dentro de [a, b].
* ``equivalencia``: SymPy simplifica la diferencia **y** se evalúa en puntos al azar con mpmath.
* ``raiz``: bisección con cambio de signo comprobado por intervalos en los extremos.
* ``integral``: dos métodos de cuadratura distintos tienen que coincidir dentro de la cota de error.
* ``lp``: programación lineal con HiGHS (SciPy); el certificado es la solución dual y la brecha de dualidad.
* ``smt``: SMT-LIB2 con z3; el modelo encontrado se reevalúa contra cada aserción (y con cvc5 si está).
* ``grafo``: camino mínimo (certificado: potenciales que cumplen la desigualdad triangular) y flujo
  máximo (certificado: corte mínimo de igual capacidad).

**Seguridad:** nunca se usa ``eval`` sobre texto del modelo. Las expresiones se analizan con ``ast`` y solo
se aceptan números, nombres permitidos, operadores aritméticos y llamadas a funciones de una lista.
Hay límites de tamaño (largo de la expresión, exponentes, tamaño de grafos y problemas lineales).
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import random
import time
from collections.abc import Callable
from typing import Any

from hope.api.resultado import Resultado

MAX_EXPR = 2000
MAX_EXP = 10_000            # exponente constante máximo en una potencia
MAX_ARISTAS = 20_000
MAX_LP = 2_000              # variables o restricciones

FUNCIONES = ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "ln",
             "sqrt", "abs", "floor", "ceil")
CONSTANTES = ("pi", "e")


class ExpresionInvalida(ValueError):
    pass


# --------------------------------------------------------------------------- análisis seguro
def _normalizar(expr: str) -> str:
    s = expr.strip().replace("^", "**").replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    return s


def analizar(expr: str, variables: tuple[str, ...] = ()) -> ast.Expression:
    """AST validado: solo números, variables declaradas, constantes, operadores y funciones permitidas."""
    s = _normalizar(expr)
    if not s or len(s) > MAX_EXPR:
        raise ExpresionInvalida(f"expresión vacía o de más de {MAX_EXPR} caracteres")
    try:
        arbol = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise ExpresionInvalida(f"sintaxis inválida: {e.msg}") from None
    permitidos = set(FUNCIONES) | set(CONSTANTES) | set(variables)
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load,
                             ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)):
            continue
        if isinstance(nodo, ast.Constant):
            if not isinstance(nodo.value, (int, float)) or isinstance(nodo.value, bool):
                raise ExpresionInvalida("solo se aceptan números")
            continue
        if isinstance(nodo, ast.Name):
            if nodo.id not in permitidos:
                raise ExpresionInvalida(f"nombre no permitido: {nodo.id}")
            continue
        if isinstance(nodo, ast.Call):
            if not isinstance(nodo.func, ast.Name) or nodo.func.id not in FUNCIONES or nodo.keywords \
                    or len(nodo.args) != 1:
                raise ExpresionInvalida("solo funciones matemáticas de un argumento: " + ", ".join(FUNCIONES))
            continue
        raise ExpresionInvalida(f"construcción no permitida: {type(nodo).__name__}")
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.Pow):
            exp = nodo.right.operand if isinstance(nodo.right, ast.UnaryOp) else nodo.right
            if isinstance(exp, ast.Constant):
                if abs(exp.value) > MAX_EXP:
                    raise ExpresionInvalida(f"exponente demasiado grande (máximo {MAX_EXP})")
            elif not isinstance(exp, ast.Name):
                # Potencias anidadas o con expresiones en el exponente (10**10**10) pueden explotar.
                raise ExpresionInvalida("el exponente tiene que ser un número o una variable")
    return arbol


def _evaluar(arbol: ast.Expression, num: Callable[[str], Any], fn: dict[str, Callable], const: dict[str, Any],
             vars_: dict[str, Any]) -> Any:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return num(repr(n.value))
        if isinstance(n, ast.Name):
            return vars_[n.id] if n.id in vars_ else const[n.id]
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Call):
            return fn[n.func.id](ev(n.args[0]))
        a, b = ev(n.left), ev(n.right)
        op = n.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            return a / b
        if isinstance(op, ast.Mod):
            return a % b
        return a ** b
    return ev(arbol)


def _iv_ctx(dps: int):
    from mpmath import iv
    iv.dps = dps
    fn = {"sin": iv.sin, "cos": iv.cos, "tan": iv.tan, "exp": iv.exp, "log": iv.log, "ln": iv.log,
          "sqrt": iv.sqrt, "abs": abs}
    const = {"pi": iv.pi, "e": iv.e}
    return iv, fn, const


def _mp_ctx(dps: int):
    from mpmath import mp
    mp.dps = dps
    fn = {"sin": mp.sin, "cos": mp.cos, "tan": mp.tan, "asin": mp.asin, "acos": mp.acos, "atan": mp.atan,
          "sinh": mp.sinh, "cosh": mp.cosh, "tanh": mp.tanh, "exp": mp.exp, "log": mp.log, "ln": mp.log,
          "sqrt": mp.sqrt, "abs": abs, "floor": mp.floor, "ceil": mp.ceil}
    const = {"pi": mp.pi, "e": mp.e}
    return mp, fn, const


def _t(nombre: str, t0: float, **kw) -> Resultado:
    return Resultado(nombre, segundos=time.perf_counter() - t0, **kw)


# --------------------------------------------------------------------------- herramientas
def calc_cert(expr: str, cifras: int = 15) -> Resultado:
    """Evalúa con aritmética de intervalos: el valor real queda **garantizado** dentro de [a, b]."""
    t0 = time.perf_counter()
    cifras = max(3, min(int(cifras), 200))
    arbol = analizar(expr)
    no_iv = {n.func.id for n in ast.walk(arbol) if isinstance(n, ast.Call)} - {
        "sin", "cos", "tan", "exp", "log", "ln", "sqrt", "abs"}
    if no_iv:
        raise ExpresionInvalida("sin aritmética de intervalos para: " + ", ".join(sorted(no_iv)))
    iv, fn, const = _iv_ctx(cifras + 10)
    x = _evaluar(arbol, iv.mpf, fn, const, {})
    a, b = x.a, x.b
    from mpmath import mp, mpf
    mp.dps = cifras + 10
    medio = (mpf(a) + mpf(b)) / 2
    ancho = mpf(b) - mpf(a)
    rel = ancho / abs(medio) if medio != 0 else ancho
    ok_precision = rel <= mpf(10) ** (-cifras)
    valor = mp.nstr(medio, cifras)
    cert = {"metodo": "aritmética de intervalos (mpmath.iv)", "intervalo": [mp.nstr(mpf(a), cifras + 5),
            mp.nstr(mpf(b), cifras + 5)], "ancho_relativo": mp.nstr(rel, 3), "cifras_pedidas": cifras}
    return _t("calc_cert", t0, ok=True, salida=f"{valor} ∈ [{cert['intervalo'][0]}, {cert['intervalo'][1]}]",
              valor=valor, certificado=cert, verificado=bool(ok_precision),
              avisos=[] if ok_precision else ["el intervalo es más ancho que las cifras pedidas"])


def equivalencia(a: str, b: str, variables: str = "x", puntos: int = 20) -> Resultado:
    """¿``a`` y ``b`` son la misma expresión? SymPy simplifica a − b y se confirma en puntos al azar."""
    t0 = time.perf_counter()
    vs = tuple(v.strip() for v in variables.replace(";", ",").split(",") if v.strip())
    ta, tb = analizar(a, vs), analizar(b, vs)
    import sympy as sp
    syms = {v: sp.Symbol(v) for v in vs}
    sfn = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
           "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "exp": sp.exp, "log": sp.log, "ln": sp.log,
           "sqrt": sp.sqrt, "abs": sp.Abs, "floor": sp.floor, "ceil": sp.ceiling}
    sconst = {"pi": sp.pi, "e": sp.E}
    ea = _evaluar(ta, sp.Rational, sfn, sconst, syms)
    eb = _evaluar(tb, sp.Rational, sfn, sconst, syms)
    simb = sp.simplify(ea - eb) == 0
    mp, fn, const = _mp_ctx(40)
    rnd = random.Random(0)
    max_dif, probados = 0.0, 0
    for _ in range(puntos * 3):
        if probados >= puntos:
            break
        vals = {v: mp.mpf(rnd.uniform(0.1, 3.0)) for v in vs}
        try:
            va = _evaluar(ta, mp.mpf, fn, const, vals)
            vb = _evaluar(tb, mp.mpf, fn, const, vals)
        except (ZeroDivisionError, ValueError):
            continue
        if isinstance(va, mp.mpc) or isinstance(vb, mp.mpc):
            continue
        escala = max(1, abs(va), abs(vb))
        max_dif = max(max_dif, float(abs(va - vb) / escala))
        probados += 1
    numerico = probados > 0 and max_dif < 1e-25
    iguales = bool(simb or numerico)
    cert = {"sympy_simplify": bool(simb), "puntos_evaluados": probados, "max_diferencia_relativa": max_dif}
    verificado = bool(simb) == bool(numerico) and probados > 0
    salida = ("equivalentes" if iguales else "distintas") + (
        "" if verificado else " (SymPy y la evaluación numérica no coinciden: revisar)")
    return _t("equivalencia", t0, ok=True, salida=salida, valor=iguales, certificado=cert, verificado=verificado)


def raiz(f: str, a: float, b: float, variable: str = "x", tol_cifras: int = 20) -> Resultado:
    """Raíz de f en [a, b] por bisección, con el cambio de signo comprobado por intervalos."""
    t0 = time.perf_counter()
    arbol = analizar(f, (variable,))
    iv, fni, consti = _iv_ctx(tol_cifras + 10)
    mp, _fn, _const = _mp_ctx(tol_cifras + 10)

    def signo_cierto(x) -> int:
        y = _evaluar(arbol, iv.mpf, fni, consti, {variable: iv.mpf(x)})
        if y.a > 0:
            return 1
        if y.b < 0:
            return -1
        return 0

    lo, hi = mp.mpf(a), mp.mpf(b)
    sa, sb = signo_cierto(lo), signo_cierto(hi)
    if sa == 0 or sb == 0 or sa == sb:
        return _t("raiz", t0, ok=False, salida="no hay cambio de signo comprobado en [a, b]",
                  certificado={"signo_a": sa, "signo_b": sb}, verificado=False)
    tol = mp.mpf(10) ** (-tol_cifras)
    for _ in range(4 * tol_cifras + 200):
        if hi - lo <= tol * max(1, abs(lo)):
            break
        m = (lo + hi) / 2
        sm = signo_cierto(m)
        if sm == 0:
            break
        if sm == sa:
            lo = m
        else:
            hi = m
    s_lo, s_hi = signo_cierto(lo), signo_cierto(hi)
    verificado = s_lo != 0 and s_hi != 0 and s_lo != s_hi
    valor = mp.nstr((lo + hi) / 2, tol_cifras)
    cert = {"metodo": "bisección con signos certificados por intervalos",
            "intervalo": [mp.nstr(lo, tol_cifras + 3), mp.nstr(hi, tol_cifras + 3)], "signos": [s_lo, s_hi]}
    return _t("raiz", t0, ok=True, salida=f"x ≈ {valor} (raíz en [{cert['intervalo'][0]}, {cert['intervalo'][1]}])",
              valor=valor, certificado=cert, verificado=verificado)


def integral(f: str, a: str, b: str, variable: str = "x", cifras: int = 15) -> Resultado:
    """Integral definida con dos cuadraturas distintas (tanh-sinh y Gauss-Legendre) que tienen que coincidir."""
    t0 = time.perf_counter()
    arbol = analizar(f, (variable,))
    mp, fn, const = _mp_ctx(cifras + 10)
    lim = [_evaluar(analizar(x), mp.mpf, fn, const, {}) for x in (a, b)]

    def g(x):
        return _evaluar(arbol, mp.mpf, fn, const, {variable: x})

    v1, e1 = mp.quad(g, lim, method="tanh-sinh", error=True)
    v2, _e2 = mp.quad(g, lim, method="gauss-legendre", error=True)
    dif = abs(v1 - v2)
    escala = max(1, abs(v1))
    tol = mp.mpf(10) ** (-cifras)
    verificado = bool(dif <= tol * escala * 10 and e1 <= tol * escala * 10)
    cert = {"tanh_sinh": mp.nstr(v1, cifras + 3), "gauss_legendre": mp.nstr(v2, cifras + 3),
            "error_estimado": mp.nstr(e1, 3), "diferencia": mp.nstr(dif, 3)}
    valor = mp.nstr(v1, cifras)
    return _t("integral", t0, ok=True, salida=f"{valor} (dos métodos; diferencia {cert['diferencia']})",
              valor=valor, certificado=cert, verificado=verificado)


def lp(problema: dict | str) -> Resultado:
    """Programación lineal con HiGHS (SciPy). Entrada JSON:
    ``{"c": [...], "A_ub": [[...]], "b_ub": [...], "A_eq": [[...]], "b_eq": [...], "bounds": [[lo, hi], ...],
    "maximizar": false}``. Certificado: duales, brecha de dualidad y factibilidad primal comprobadas."""
    t0 = time.perf_counter()
    p = json.loads(problema) if isinstance(problema, str) else dict(problema)
    try:
        from scipy.optimize import linprog
    except ImportError:
        return _t("lp", t0, ok=False, salida="falta SciPy (HiGHS): instalá python3-scipy o python3-highspy")
    import numpy as np
    c = np.asarray(p["c"], dtype=float)
    if c.size > MAX_LP:
        raise ExpresionInvalida(f"más de {MAX_LP} variables")
    signo = -1.0 if p.get("maximizar") else 1.0
    A_ub = np.asarray(p["A_ub"], dtype=float) if p.get("A_ub") else None
    b_ub = np.asarray(p["b_ub"], dtype=float) if p.get("b_ub") else None
    A_eq = np.asarray(p["A_eq"], dtype=float) if p.get("A_eq") else None
    b_eq = np.asarray(p["b_eq"], dtype=float) if p.get("b_eq") else None
    for M in (A_ub, A_eq):
        if M is not None and M.shape[0] > MAX_LP:
            raise ExpresionInvalida(f"más de {MAX_LP} restricciones")
    bounds = [tuple(x) for x in p["bounds"]] if p.get("bounds") else [(0, None)] * c.size
    r = linprog(signo * c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if r.status != 0:
        return _t("lp", t0, ok=False, salida=f"sin óptimo: {r.message}", certificado={"estado": int(r.status)},
                  verificado=False)
    x = r.x
    tol = 1e-7
    fact = True
    if A_ub is not None:
        fact &= bool(np.all(A_ub @ x <= b_ub + tol * (1 + np.abs(b_ub))))
    if A_eq is not None:
        fact &= bool(np.allclose(A_eq @ x, b_eq, atol=tol * (1 + np.abs(b_eq).max())))
    for xi, (lo, hi) in zip(x, bounds):
        fact &= (lo is None or xi >= lo - tol) and (hi is None or xi <= hi + tol)
    dual = 0.0
    if A_ub is not None:
        dual += float(b_ub @ r.ineqlin.marginals)
    if A_eq is not None:
        dual += float(b_eq @ r.eqlin.marginals)
    lower = getattr(r, "lower", None)
    upper = getattr(r, "upper", None)
    if lower is not None:
        dual += float(sum(m * (lo or 0.0) for m, (lo, _) in zip(lower.marginals, bounds) if lo is not None))
    if upper is not None:
        dual += float(sum(m * hi for m, (_, hi) in zip(upper.marginals, bounds) if hi is not None))
    primal = float(signo * c @ x)
    brecha = abs(primal - dual) / max(1.0, abs(primal))
    verificado = bool(fact and brecha < 1e-6)
    objetivo = signo * primal
    cert = {"metodo": "HiGHS + dual", "objetivo_primal": objetivo, "objetivo_dual": signo * dual,
            "brecha_relativa": brecha, "factible": bool(fact)}
    return _t("lp", t0, ok=True, salida=f"óptimo {objetivo:.10g} en x = {[round(float(v), 10) for v in x]}",
              valor={"objetivo": objetivo, "x": [float(v) for v in x]}, certificado=cert, verificado=verificado)


def smt(smtlib: str) -> Resultado:
    """Restricciones en SMT-LIB2 (``(declare-const x Int) (assert (> x 2)) …``). Se resuelve con z3; si
    hay modelo, cada aserción se reevalúa contra él; si cvc5 está instalado, también tiene que coincidir."""
    t0 = time.perf_counter()
    if len(smtlib) > 20_000:
        raise ExpresionInvalida("SMT-LIB demasiado largo")
    import z3
    asertos = z3.parse_smt2_string(smtlib)
    s = z3.Solver()
    s.set("timeout", 10_000)
    s.add(asertos)
    res = s.check()
    cert: dict[str, Any] = {"z3": str(res)}
    verificado = None
    valor: Any = str(res)
    if res == z3.sat:
        m = s.model()
        evals = [z3.is_true(m.eval(a, model_completion=True)) for a in asertos]
        cert["asertos_comprobados"] = f"{sum(evals)}/{len(evals)}"
        verificado = all(evals)
        valor = {str(d.name()): str(m[d]) for d in m.decls()}
    try:
        import cvc5.pythonic as c5
        sc = c5.Solver()
        sc.add(c5.parse_smt2_string(smtlib))
        otro = str(sc.check())
        cert["cvc5"] = otro
        coincide = otro == str(res)
        verificado = coincide if verificado is None else (verificado and coincide)
    except Exception:                                       # noqa: BLE001 — cvc5 es opcional
        cert["cvc5"] = "no instalado"
    salida = ("satisfacible: " + ", ".join(f"{k} = {v}" for k, v in valor.items())) if isinstance(valor, dict) \
        else {"unsat": "insatisfacible", "unknown": "desconocido (tiempo agotado)"}.get(str(res), str(res))
    return _t("smt", t0, ok=str(res) != "unknown", salida=salida, valor=valor, certificado=cert,
              verificado=verificado)


def grafo(problema: dict | str) -> Resultado:
    """Problemas de grafos con certificado. Entrada JSON:

    * ``{"op": "camino", "aristas": [[u, v, peso], ...], "origen": u, "destino": v, "dirigido": false}``
      — certificado: potenciales (distancias) que cumplen d(v) ≤ d(u) + w en todas las aristas.
    * ``{"op": "flujo", "aristas": [[u, v, capacidad], ...], "origen": s, "destino": t}``
      — certificado: corte mínimo con la misma capacidad que el flujo.
    """
    t0 = time.perf_counter()
    p = json.loads(problema) if isinstance(problema, str) else dict(problema)
    import networkx as nx
    aristas = p["aristas"]
    if len(aristas) > MAX_ARISTAS:
        raise ExpresionInvalida(f"más de {MAX_ARISTAS} aristas")
    op = p.get("op", "camino")
    if op == "camino":
        G = nx.DiGraph() if p.get("dirigido") else nx.Graph()
        for u, v, w in aristas:
            if w < 0:
                raise ExpresionInvalida("pesos negativos no soportados (usar Bellman-Ford aparte)")
            G.add_edge(u, v, weight=w)
        s, t = p["origen"], p["destino"]
        try:
            dist = nx.single_source_dijkstra_path_length(G, s)
            camino = nx.dijkstra_path(G, s, t)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return _t("grafo", t0, ok=True, salida="sin camino", valor=None, verificado=None)
        largo = sum(G[a][b]["weight"] for a, b in itertools.pairwise(camino))
        violaciones = 0
        for u, v, d in G.edges(data=True):
            for x, y in ((u, v), (v, u)) if not G.is_directed() else ((u, v),):
                if x in dist and dist.get(y, math.inf) > dist[x] + d["weight"] + 1e-9:
                    violaciones += 1
        verificado = violaciones == 0 and abs(largo - dist[t]) < 1e-9
        cert = {"metodo": "potenciales de Dijkstra", "aristas_revisadas": G.number_of_edges(),
                "violaciones": violaciones, "largo_recalculado": largo}
        return _t("grafo", t0, ok=True, salida=f"camino {' → '.join(map(str, camino))} (largo {largo:g})",
                  valor={"camino": camino, "largo": largo}, certificado=cert, verificado=verificado)
    if op == "flujo":
        G = nx.DiGraph()
        for u, v, cap in aristas:
            G.add_edge(u, v, capacity=cap)
        s, t = p["origen"], p["destino"]
        valor_flujo, _ = nx.maximum_flow(G, s, t)
        corte, (S, T) = nx.minimum_cut(G, s, t)
        cap_corte = sum(G[u][v]["capacity"] for u in S for v in G[u] if v in T)
        verificado = abs(valor_flujo - corte) < 1e-9 and abs(cap_corte - corte) < 1e-9
        cert = {"metodo": "flujo máximo = corte mínimo", "corte": corte, "lado_origen": sorted(map(str, S))}
        return _t("grafo", t0, ok=True, salida=f"flujo máximo {valor_flujo:g} (corte mínimo {corte:g})",
                  valor=valor_flujo, certificado=cert, verificado=verificado)
    raise ExpresionInvalida("op desconocida: usar «camino» o «flujo»")
