"""Herramientas deterministas iniciales (Herramientas §6.2 y §14, fase 1).

* ``arith`` — aritmética decimal de precisión controlada con un evaluador propio
  (solo números y operadores; nada de ``eval``).
* ``symbolic_equal`` — ¿dos expresiones son equivalentes? (SymPy, **en sandbox**,
  porque el parser de SymPy usa ``eval`` y las expresiones vienen de afuera).
* ``units`` — cálculo con unidades y comprobación dimensional (pint).
* ``python_run`` — ejecuta código Python en el sandbox.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from decimal import Decimal, localcontext

from hope.tools.catalog import Catalog, ToolSpec
from hope.tools.sandbox import run_python

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def normalize_number_text(text: str) -> str:
    """'0,05' → '0.05'; '1.234,5' → '1234.5'; quita espacios de miles."""
    t = text.strip()
    if re.search(r"\d\.\d{3},\d", t):
        t = t.replace(".", "")
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    return t.replace("×", "*").replace("·", "*").replace("−", "-").replace("^", "**")


def arith(expr: str, precision: int = 50) -> Decimal:
    tree = ast.parse(normalize_number_text(expr), mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Decimal(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponente demasiado grande")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        raise ValueError(f"expresión no permitida: {ast.dump(node)[:60]}")

    with localcontext() as ctx:
        ctx.prec = precision
        return ev(tree)


def numbers_equal(a: Decimal | float | str, b: Decimal | float | str, rel: float = 1e-9) -> bool:
    try:
        x = Decimal(normalize_number_text(str(a)))
        y = Decimal(normalize_number_text(str(b)))
    except Exception:
        return False
    return abs(x - y) <= Decimal(str(rel)) * max(abs(x), abs(y), Decimal(1))


def symbolic_equal(lhs: str, rhs: str, timeout_s: float = 20.0) -> bool | None:
    """True / False, o None si no se pudo decidir (error, tiempo agotado)."""
    code = (
        "import json, sys, sympy\n"
        "from sympy.parsing.sympy_parser import parse_expr, standard_transformations, "
        "implicit_multiplication_application, convert_xor\n"
        "tr = standard_transformations + (implicit_multiplication_application, convert_xor)\n"
        "d = json.loads(sys.stdin.read())\n"
        "a, b = parse_expr(d['l'], transformations=tr), parse_expr(d['r'], transformations=tr)\n"
        "print(json.dumps(bool(sympy.simplify(a - b) == 0)))\n")
    res = run_python(code, timeout_s=timeout_s,
                     stdin=json.dumps({"l": normalize_number_text(lhs), "r": normalize_number_text(rhs)}))
    if not res.ok:
        return None
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


_UREG = None


_QTY_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-zµμ°][A-Za-zµμ°_]*(?:\*\*-?\d+)?)")


def _group_quantities(expr: str) -> str:
    """«10 V / 2 A» → «(10 V) / (2 A)». Sin esto pint lo lee como 10·V/2·A, porque la
    multiplicación implícita liga más débil que «/»."""
    t = normalize_number_text(expr).replace("Ω", "ohm").replace("Ω", "ohm")
    return _QTY_RE.sub(lambda m: f"({m.group(1)} {m.group(2)})", t)


def units(expr: str, expected: str | None = None) -> dict:
    """Evalúa una expresión con unidades; si hay ``expected``, compara magnitud y dimensión."""
    global _UREG
    import pint
    if _UREG is None:
        _UREG = pint.UnitRegistry()
    q = _UREG.parse_expression(_group_quantities(expr))
    out = {"value": str(q)}
    if expected is not None:
        e = _UREG.parse_expression(_group_quantities(expected))
        same_dim = getattr(q, "dimensionality", None) == getattr(e, "dimensionality", None)
        out["same_dimension"] = same_dim
        out["equal"] = bool(same_dim and abs(q.to(e.units).magnitude - e.magnitude)
                            <= 1e-9 * max(abs(e.magnitude), 1))
    return out


def python_run(code: str, timeout_s: float = 10.0) -> dict:
    r = run_python(code, timeout_s=timeout_s)
    return {"ok": r.ok, "stdout": r.stdout, "stderr": r.stderr[-2000:], "timed_out": r.timed_out}


def default_catalog(guardian=None) -> Catalog:
    cat = Catalog(guardian)
    cat.register(ToolSpec(
        "hte_arith_v1", "Aritmética decimal", "1.0", "matemática",
        "calcular aritmética exacta suma resta multiplicación división potencia",
        "Evalúa expresiones numéricas con Decimal de 50 dígitos.", {"expr": "str"}, {"valor": "Decimal"},
        units_and_formats="acepta coma decimal", expected_precision="50 dígitos significativos",
        limitations=["sin variables ni funciones"], do_not_use_when=["hay unidades: usar hte_units_v1"],
        tests=["tests/test_tools.py"]), arith)
    cat.register(ToolSpec(
        "hte_symbolic_v1", "Equivalencia simbólica", "1.0", "matemática",
        "comprobar si dos expresiones algebraicas son equivalentes álgebra simbólica",
        "SymPy simplify(lhs - rhs) == 0, en sandbox.", {"lhs": "str", "rhs": "str"}, {"igual": "bool|None"},
        execution="sandbox", dependencies=["sympy"], cost="medio",
        limitations=["simplify puede no decidir equivalencias difíciles (devuelve False)"],
        tests=["tests/test_tools.py"]), symbolic_equal)
    cat.register(ToolSpec(
        "hte_units_v1", "Unidades y dimensiones", "1.0", "física",
        "calcular con unidades conversión comprobar dimensiones tensión corriente resistencia",
        "Cálculo con unidades físicas (pint) y comparación dimensional.",
        {"expr": "str", "expected": "str?"}, {"value": "str", "equal": "bool"},
        dependencies=["pint"], tests=["tests/test_tools.py"]), units)
    cat.register(ToolSpec(
        "hte_python_v1", "Ejecutar Python", "1.0", "programación",
        "ejecutar código python programa probar salida",
        "Ejecuta código Python en un proceso aislado con límites.", {"code": "str"},
        {"ok": "bool", "stdout": "str"}, execution="sandbox",
        risks=["el sandbox no bloquea la red"], permissions=["sandbox"],
        tests=["tests/test_tools.py"]), python_run)
    from hope.tools import math_tools as mt
    for tid, name, domain, obj, fn, ex in (
        ("hte_fraction_v1", "Fracciones exactas", "matemática", "fracciones exactas proporciones probabilidad",
         mt.fraction_calc, "sandbox no; evaluador propio"),
        ("hte_stats_v1", "Estadística descriptiva", "matemática", "media mediana desvío estadística lista de números",
         mt.stats, "biblioteca estándar"),
        ("hte_solve_v1", "Resolver ecuaciones", "matemática", "resolver ecuaciones sistemas despejar incógnita",
         mt.solve, "SymPy en sandbox"),
        ("hte_derivative_v1", "Derivadas", "matemática", "derivar derivada cálculo", mt.derivative, "SymPy en sandbox"),
        ("hte_integral_v1", "Integrales", "matemática", "integrar integral definida indefinida cálculo",
         mt.integral, "SymPy en sandbox"),
        ("hte_limit_v1", "Límites", "matemática", "límite cálculo", mt.limit, "SymPy en sandbox"),
        ("hte_linalg_v1", "Álgebra lineal", "matemática", "matriz determinante inversa autovalores sistema lineal",
         mt.linalg, "NumPy en sandbox"),
        ("hte_logic_v1", "Lógica y restricciones (z3)", "lógica",
         "lógica satisfacibilidad restricciones enteros modelo contraejemplo", mt.logic, "z3 en sandbox"),
    ):
        cat.register(ToolSpec(tid, name, "1.0", domain, obj, f"{name} ({ex}).", {"texto": "str"}, {"resultado": "str"},
                              execution="sandbox" if "sandbox" in ex else "en proceso",
                              tests=["tests/test_toolcalls.py"]), fn)
    from hope.tools import code_tools as ct
    from hope.tools import eng_tools as et
    for tid, name, domain, obj, fn, ex in (
        ("hte_pytest_v1", "Ejecutar pruebas", "programación", "probar código pruebas pytest tests",
         ct.run_tests, "pytest en sandbox"),
        ("hte_lint_v1", "Análisis estático", "programación", "errores estáticos lint estilo ruff",
         ct.lint, "ruff en sandbox"),
        ("hte_compile_v1", "Compilar y ejecutar C/C++", "programación", "compilar c c++ gcc programa",
         ct.compile_run, "gcc/g++ en sandbox"),
        ("hte_search_v1", "Buscar en código", "programación", "buscar en código archivos ripgrep",
         ct.search_code, "ripgrep; lectura con permiso"),
        ("hte_diff_v1", "Diferencias entre versiones", "programación", "diff cambios entre versiones parche",
         ct.diff, "difflib"),
        ("hte_circuit_v1", "Simulación de circuitos lineales", "electrónica",
         "simular circuito tensiones corrientes nodos resistencias capacitores continua alterna",
         et.circuit, "análisis nodal modificado propio (NumPy)"),
        ("hte_filter_v1", "Respuesta de filtros", "electrónica", "filtro respuesta en frecuencia corte polos",
         et.filter_response, "NumPy"),
        ("hte_sql_v1", "Consultas SQL sobre CSV", "datos", "consultar tabla csv sql datos", et.sql_csv,
         "SQLite en sandbox"),
        ("hte_convert_v1", "Conversión de formatos", "datos", "convertir csv json yaml formato", et.convert,
         "biblioteca estándar"),
        ("hte_correlation_v1", "Correlaciones entre columnas", "datos",
         "correlación patrones relación entre datos columnas csv", et.correlations, "NumPy"),
    ):
        cat.register(ToolSpec(tid, name, "1.0", domain, obj, f"{name} ({ex}).", {"texto": "str"}, {"resultado": "str"},
                              execution="sandbox" if "sandbox" in ex else "en proceso",
                              tests=["tests/test_toolcalls.py"]), fn)
    from hope.tools.extended import register_extended
    register_extended(cat, guardian)                      # catálogo ampliado (HERRAMIENTAS_HN1 §8)
    from hope.tools.lenguaje import register_lenguaje
    register_lenguaje(cat)                                # lengua es/en (hope/tools/lenguaje.py)
    return cat
