"""Herramientas científicas y de razonamiento general (docs/HERRAMIENTAS_HN1.md §3, catálogo ampliado).

* ``bases`` — conversión entre binario, octal, decimal y hexadecimal y operaciones de bits.
* ``fit`` — regresión lineal, polinómica, exponencial, potencial o logarítmica con R².
* ``ode`` — ecuaciones diferenciales ordinarias (SciPy ``solve_ivp`` si está; si no, RK4 propio).
* ``constant`` — constantes físicas CODATA 2018 con unidades, incertidumbre y fuente.
* ``plot`` — gráfico simple a PNG (matplotlib si está; si no, un rasterizador propio).

Las expresiones (``ode``, ``plot``) se compilan con un evaluador propio sobre el AST: solo
números, variables declaradas, operadores aritméticos y funciones de ``math``; nada de ``eval``.
``ode`` y ``plot`` corren además en el sandbox (``call_function``) porque la entrada la escribe
el modelo y puede ser costosa.
"""

from __future__ import annotations

import ast
import base64
import math
import operator
import re
import struct
import zlib

import numpy as np

from hope.tools.builtin import normalize_number_text

# ---------------------------------------------------------------------------
# Bases y bits

_INT_BIN = {ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
            ast.LShift: operator.lshift, ast.RShift: operator.rshift, ast.Add: operator.add,
            ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}
_WORDS = {"and": "&", "or": "|", "xor": "^", "not": "~", "shl": "<<", "shr": ">>", "y": "&", "o": "|"}


def _prep_bits(expr: str) -> str:
    t = expr.strip()
    m = re.fullmatch(r"base\s*(\d{1,2})\s*:\s*([0-9a-zA-Z]+)", t)
    if m:                                                   # «base 7: 1236» → decimal
        return str(int(m.group(2), int(m.group(1))))
    t = re.sub(r"\b([0-9][0-9A-Fa-f]*|[A-Fa-f][0-9A-Fa-f]*)[hH]\b", r"0x\1", t)   # FFh, 0B000h (estilo BIOS/ASM)
    t = re.sub(r"\b([01]+)b\b", r"0b\1", t)
    t = re.sub(r"\b(and|or|xor|not|shl|shr)\b", lambda m: _WORDS[m.group(1)], t, flags=re.IGNORECASE)
    return t


def bit_eval(expr: str) -> int:
    tree = ast.parse(_prep_bits(expr), mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _INT_BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, (ast.LShift, ast.Pow)) and b > 4096:
                raise ValueError("desplazamiento o exponente demasiado grande")
            if isinstance(n.op, (ast.FloorDiv, ast.Mod)) and b == 0:
                raise ValueError("división por cero")
            return _INT_BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.Invert, ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return ~v if isinstance(n.op, ast.Invert) else (-v if isinstance(n.op, ast.USub) else v)
        raise ValueError("solo enteros (0x.., 0b.., 0o.., ..h) y operadores & | ^ ~ << >> + - * // %")

    return ev(tree)


def bases(expr: str, width: int | None = None) -> dict:
    """Valor entero en las cuatro bases; con ``width`` (bits) se enmascara en complemento a dos."""
    v = bit_eval(expr)
    if width is None and v < 0:
        width = max(8, 8 * math.ceil((v.bit_length() + 1) / 8))
    out = {"dec": v}
    if width is not None:
        if not 1 <= width <= 4096:
            raise ValueError("ancho entre 1 y 4096 bits")
        u = v & ((1 << width) - 1)
        out.update(dec=u, con_signo=u - (1 << width) if u >> (width - 1) else u, ancho=width)
        v = u
    nd = (out.get("ancho") or max(1, v.bit_length()) + 3) // 4
    out.update(hex="0x" + format(v, "X").zfill(nd if width else 1), oct="0o" + format(v, "o"),
               bin="0b" + format(v, "b").zfill(width or 1), bits=max(1, v.bit_length()), unos=bin(v).count("1"))
    return out


def bases_text(arg: str) -> str:
    expr, _, w = arg.partition("|")
    m = re.search(r"\d+", w)
    r = bases(expr, int(m.group()) if m else None)
    s = f"{r['dec']} = {r['hex']} = {r['oct']} = {r['bin']} ({r['unos']} unos"
    if "ancho" in r:
        s += f"; {r['ancho']} bits, con signo {r['con_signo']}"
    return s + ")"


# ---------------------------------------------------------------------------
# Ajuste de curvas

def parse_numbers(s: str) -> list[float]:
    s = s.strip().strip("[]")
    if re.search(r"\s", s):                                     # «1,5 2,5 3» → coma decimal
        toks = [t for t in re.split(r"[\s;]+", s) if t]
        toks = [t.rstrip(",") for t in toks]
        return [float(normalize_number_text(t)) for t in toks]
    return [float(t) for t in re.split(r"[,;]", s) if t.strip()]


def _r2(y, yhat) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 if ss_tot == 0 and ss_res == 0 else (1 - ss_res / ss_tot if ss_tot else 0.0)


def _g(v: float) -> str:
    return f"{v:.6g}"


def fit(model: str, xs: list[float], ys: list[float]) -> dict:
    """Ajuste por mínimos cuadrados. El R² se calcula siempre en el espacio original de y."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(x) != len(y):
        raise ValueError(f"x tiene {len(x)} valores e y tiene {len(y)}")
    if len(x) < 2 or len(x) > 1_000_000:
        raise ValueError("hacen falta entre 2 y 1 000 000 puntos")
    m = model.strip().lower()
    if m in ("auto", "mejor"):
        cands = []
        for cand in ("lineal", "exp", "potencia", "log", "poli 2"):
            try:
                cands.append(fit(cand, xs, ys))
            except ValueError:
                pass
        best = max(c["r2"] for c in cands)
        r = next(c for c in cands if c["r2"] >= best - 1e-9)    # ante empate, el más simple
        r["candidatos"] = {c["modelo"]: round(c["r2"], 6) for c in cands}
        return r
    if m in ("lineal", "recta", "linear"):
        a, b = np.polyfit(x, y, 1)
        yhat = a * x + b
        return {"modelo": "lineal", "coef": {"a": float(a), "b": float(b)}, "ecuacion": f"y = {_g(a)}·x + {_g(b)}",
                "r2": _r2(y, yhat), "n": len(x)}
    pm = re.fullmatch(r"(?:poli|polinomica|polinómica|poly)\s*(\d+)|cuadratica|cuadrática|cubica|cúbica", m)
    if pm:
        deg = int(pm.group(1)) if pm.group(1) else (2 if m.startswith("cuadr") else 3)
        if deg < 1 or deg > 12 or deg >= len(x):
            raise ValueError("grado entre 1 y 12 y menor que la cantidad de puntos")
        c = np.polyfit(x, y, deg)
        terms = " + ".join(f"{_g(ci)}·x^{deg - i}" if deg - i > 1 else (f"{_g(ci)}·x" if deg - i == 1 else _g(ci))
                           for i, ci in enumerate(c))
        return {"modelo": f"poli {deg}", "coef": {f"c{deg - i}": float(ci) for i, ci in enumerate(c)},
                "ecuacion": "y = " + terms, "r2": _r2(y, np.polyval(c, x)), "n": len(x)}
    if m in ("exp", "exponencial"):
        if np.any(y <= 0):
            raise ValueError("el ajuste exponencial necesita y > 0")
        b, lna = np.polyfit(x, np.log(y), 1)
        a = math.exp(lna)
        return {"modelo": "exp", "coef": {"a": a, "b": float(b)}, "ecuacion": f"y = {_g(a)}·e^({_g(b)}·x)",
                "r2": _r2(y, a * np.exp(b * x)), "n": len(x)}
    if m in ("potencia", "potencial", "power"):
        if np.any(y <= 0) or np.any(x <= 0):
            raise ValueError("el ajuste potencial necesita x > 0 e y > 0")
        b, lna = np.polyfit(np.log(x), np.log(y), 1)
        a = math.exp(lna)
        return {"modelo": "potencia", "coef": {"a": a, "b": float(b)}, "ecuacion": f"y = {_g(a)}·x^{_g(b)}",
                "r2": _r2(y, a * x ** b), "n": len(x)}
    if m in ("log", "logaritmica", "logarítmica"):
        if np.any(x <= 0):
            raise ValueError("el ajuste logarítmico necesita x > 0")
        a, b = np.polyfit(np.log(x), y, 1)
        return {"modelo": "log", "coef": {"a": float(a), "b": float(b)}, "ecuacion": f"y = {_g(a)}·ln(x) + {_g(b)}",
                "r2": _r2(y, a * np.log(x) + b), "n": len(x)}
    raise ValueError("modelo: lineal | poli N | exp | potencia | log | auto")


def fit_text(arg: str) -> str:
    parts = [p.strip() for p in arg.split("|")]
    if len(parts) == 2:                                          # «lineal | 1,2; 2,4; 3,6» → pares
        pairs = [p for p in re.split(r";|\n", parts[1]) if p.strip()]
        xy = [parse_numbers(p.replace(",", " ") if p.count(",") == 1 else p) for p in pairs]
        parts = [parts[0], " ".join(str(p[0]) for p in xy), " ".join(str(p[1]) for p in xy)]
    if len(parts) != 3:
        raise ValueError("formato: ⟦ajuste: lineal | x1 x2 … | y1 y2 …⟧")
    r = fit(parts[0], parse_numbers(parts[1]), parse_numbers(parts[2]))
    s = f"{r['ecuacion']} (R² = {r['r2']:.6g}, n = {r['n']}, modelo {r['modelo']})"
    if "candidatos" in r:
        s += "; R² de cada modelo: " + ", ".join(f"{k} {v}" for k, v in r["candidatos"].items())
    return s


# ---------------------------------------------------------------------------
# Expresiones seguras (para ode y plot)

_FUNCS = {n: getattr(math, n) for n in ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                                         "exp", "log", "log10", "sqrt", "floor", "ceil", "atan2", "hypot")}
_FUNCS.update(abs=abs, ln=math.log, sen=math.sin, min=min, max=max, sign=lambda v: math.copysign(1.0, v) if v else 0.0)
_CONSTS = {"pi": math.pi, "e": math.e}
_FBIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
         ast.Pow: operator.pow, ast.Mod: operator.mod}


def compile_expr(src: str, names: set[str]):
    """Compila una expresión a una función ``f(env) -> float`` sin ``eval``."""
    tree = ast.parse(normalize_number_text(src), mode="eval")

    def comp(n):
        if isinstance(n, ast.Expression):
            return comp(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            v = float(n.value)
            return lambda env: v
        if isinstance(n, ast.Name):
            if n.id in names:
                k = n.id
                return lambda env: env[k]
            if n.id in _CONSTS:
                v = _CONSTS[n.id]
                return lambda env: v
            raise ValueError(f"nombre desconocido: {n.id}")
        if isinstance(n, ast.BinOp) and type(n.op) in _FBIN:
            f, a, b = _FBIN[type(n.op)], comp(n.left), comp(n.right)
            return lambda env: f(a(env), b(env))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            a, neg = comp(n.operand), isinstance(n.op, ast.USub)
            return (lambda env: -a(env)) if neg else a
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            f, args = _FUNCS[n.func.id], [comp(a) for a in n.args]
            return lambda env: f(*(a(env) for a in args))
        raise ValueError("expresión no permitida (solo números, variables, + - * / ^ y funciones de math)")

    return comp(tree)


# ---------------------------------------------------------------------------
# Ecuaciones diferenciales

_PRIME = re.compile(r"([A-Za-z_]\w*)('+)")


def _dname(v: str, k: int) -> str:
    return v if k == 0 else f"{v}__d{k}"


def ode(equations: str, initial: str, span: str, points: int = 5, method: str = "auto") -> dict:
    """Resuelve y' = f(t, y) (sistemas y orden superior). Ejemplo: ``y'' = -y | y(0)=1, y'(0)=0 | 0..pi``."""
    eqs = [e.strip() for e in re.split(r"[;\n]", equations) if e.strip()]
    order: dict[str, int] = {}
    rhs_src: dict[str, str] = {}
    for e in eqs:
        m = re.fullmatch(r"([A-Za-z_]\w*)('+)\s*=\s*(.+)", e) or re.fullmatch(r"d([A-Za-z_]\w*)/dt\s*=\s*(.+)", e)
        if not m:
            raise ValueError(f"ecuación inválida «{e}» (formato: y' = …, y'' = … o dy/dt = …)")
        v, k, rhs = (m.group(1), len(m.group(2)), m.group(3)) if m.re.groups == 3 else (m.group(1), 1, m.group(2))
        if k > 6:
            raise ValueError("orden máximo 6")
        order[v], rhs_src[v] = k, rhs
    state = [_dname(v, j) for v, k in order.items() for j in range(k)]
    names = set(state) | {"t"}
    f = {v: compile_expr(_PRIME.sub(lambda m: _dname(m.group(1), len(m.group(2))), rhs), names)
         for v, rhs in rhs_src.items()}
    init, t0 = {}, None
    for it in re.split(r"[,;]", initial):
        if not it.strip():
            continue
        m = re.fullmatch(r"\s*([A-Za-z_]\w*)('*)\s*\(\s*([^)]+)\)\s*=\s*(.+?)\s*", it)
        if not m:
            raise ValueError(f"condición inicial inválida «{it.strip()}» (formato: y(0) = 1)")
        tt = float(compile_expr(m.group(3), set())({}))
        if t0 is not None and tt != t0:
            raise ValueError("todas las condiciones iniciales deben darse en el mismo t")
        t0 = tt
        init[_dname(m.group(1), len(m.group(2)))] = float(compile_expr(m.group(4), set())({}))
    missing = [s for s in state if s not in init]
    if missing:
        raise ValueError("faltan condiciones iniciales para: " + ", ".join(s.replace("__d", "'") for s in missing))
    s = span.split("=", 1)[-1].strip()
    m = re.fullmatch(r"(.+?)\s*(?:\.\.|\ba\b|,)\s*(.+)", s)
    if m:
        ta, tb = (float(compile_expr(g, set())({})) for g in m.groups())
    else:
        ta, tb = t0, float(compile_expr(re.sub(r"^hasta\s+", "", s), set())({}))
    if ta != t0:
        raise ValueError(f"el intervalo empieza en {ta:g} pero las condiciones iniciales están en t = {t0:g}")
    if not math.isfinite(tb) or tb == ta:
        raise ValueError("intervalo inválido")

    def deriv(t, y):
        env = dict(zip(state, y))
        env["t"] = t
        out = []
        for v, k in order.items():
            out += [env[_dname(v, j + 1)] for j in range(k - 1)] + [f[v](env)]
        return out

    y0 = [init[s] for s in state]
    ts = np.linspace(ta, tb, max(2, points))
    try:
        if method == "rk4":
            raise ImportError("RK4 pedido")
        from scipy.integrate import solve_ivp
        sol = solve_ivp(deriv, (ta, tb), y0, method="DOP853", rtol=1e-11, atol=1e-13, t_eval=ts)
        if not sol.success:
            raise ValueError(f"SciPy no pudo integrar: {sol.message}")
        ys, method = sol.y.T, "SciPy solve_ivp (DOP853, rtol 1e-11)"
    except ImportError:
        seg = max(1, len(ts) - 1)
        n = int(min(200_000, max(4000, abs(tb - ta) * 4000)))
        n = -(-n // seg) * seg                                  # los puntos pedidos caen justo en un paso
        h = (tb - ta) / n
        y, t = np.array(y0, float), ta
        rows, k_next = [y.copy()], 1
        mark = [round((tt - ta) / h) for tt in ts]
        for i in range(1, n + 1):
            k1 = np.array(deriv(t, y))
            k2 = np.array(deriv(t + h / 2, y + h / 2 * k1))
            k3 = np.array(deriv(t + h / 2, y + h / 2 * k2))
            k4 = np.array(deriv(t + h, y + h * k3))
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t = ta + i * h
            if not np.all(np.isfinite(y)):
                raise ValueError(f"la solución diverge cerca de t = {t:g}")
            while k_next < len(mark) and mark[k_next] == i:
                rows.append(y.copy())
                k_next += 1
        ys, method = np.array(rows), f"RK4 propio ({n} pasos)"
    show = [s for s in state if "__d" not in s] or state
    idx = [state.index(s) for s in show]
    return {"metodo": method, "t_final": float(tb),
            "final": {s: float(ys[-1][i]) for s, i in zip(show, idx)},
            "puntos": [{"t": float(t), **{s: float(row[i]) for s, i in zip(show, idx)}} for t, row in zip(ts, ys)]}


def ode_text(r: dict) -> str:
    fin = ", ".join(f"{k}({r['t_final']:g}) = {v:.10g}" for k, v in r["final"].items())
    pts = "; ".join(f"t={p['t']:.4g}: " + ", ".join(f"{k}={v:.6g}" for k, v in p.items() if k != "t")
                    for p in r["puntos"][:5])
    return f"{fin} [{r['metodo']}] · {pts}"


# ---------------------------------------------------------------------------
# Constantes (CODATA 2018, NIST; los valores «exactos» son definiciones del SI desde 2019)

CODATA = "CODATA 2018 (NIST, physics.nist.gov/constants)"
_C = [
    # símbolo, nombre, valor, incertidumbre estándar (0 = exacta), unidad, alias
    ("c", "velocidad de la luz en el vacío", 299792458.0, 0, "m/s", "luz celeridad speed of light"),
    ("h", "constante de Planck", 6.62607015e-34, 0, "J·s", "planck"),
    ("ħ", "constante de Planck reducida", 1.054571817e-34, 0, "J·s", "hbar planck reducida dirac"),
    ("e", "carga elemental", 1.602176634e-19, 0, "C", "carga del electrón elementary charge"),
    ("k_B", "constante de Boltzmann", 1.380649e-23, 0, "J/K", "boltzmann kb k"),
    ("N_A", "constante de Avogadro", 6.02214076e23, 0, "1/mol", "avogadro na"),
    ("R", "constante molar de los gases", 8.314462618, 0, "J/(mol·K)", "gases ideales gas constant"),
    ("F", "constante de Faraday", 96485.33212, 0, "C/mol", "faraday"),
    ("σ", "constante de Stefan-Boltzmann", 5.670374419e-8, 0, "W/(m²·K⁴)", "stefan boltzmann sigma"),
    ("b", "constante de desplazamiento de Wien", 2.897771955e-3, 0, "m·K", "wien"),
    ("G", "constante de gravitación universal", 6.67430e-11, 0.00015e-11, "m³/(kg·s²)", "gravitacion newton"),
    ("g_n", "gravedad estándar", 9.80665, 0, "m/s²", "gravedad g aceleracion estandar"),
    ("ε_0", "permitividad eléctrica del vacío", 8.8541878128e-12, 0.0000000013e-12, "F/m",
     "epsilon0 permitividad vacio e0"),
    ("μ_0", "permeabilidad magnética del vacío", 1.25663706212e-6, 0.00000000019e-6, "N/A²",
     "mu0 permeabilidad u0"),
    ("m_e", "masa del electrón", 9.1093837015e-31, 0.0000000028e-31, "kg", "electron me"),
    ("m_p", "masa del protón", 1.67262192369e-27, 0.00000000051e-27, "kg", "proton mp"),
    ("m_n", "masa del neutrón", 1.67492749804e-27, 0.00000000095e-27, "kg", "neutron mn"),
    ("m_u", "constante de masa atómica", 1.66053906660e-27, 0.00000000050e-27, "kg", "uma dalton u atomic mass"),
    ("α", "constante de estructura fina", 7.2973525693e-3, 0.0000000011e-3, "", "alfa estructura fina alpha"),
    ("R_∞", "constante de Rydberg", 10973731.568160, 0.000021, "1/m", "rydberg"),
    ("a_0", "radio de Bohr", 5.29177210903e-11, 0.00000000080e-11, "m", "bohr"),
    ("eV", "electronvoltio", 1.602176634e-19, 0, "J", "electronvolt electronvoltio ev"),
    ("atm", "atmósfera estándar", 101325.0, 0, "Pa", "atmosfera presion estandar"),
    ("T_0", "cero Celsius", 273.15, 0, "K", "celsius cero absoluto kelvin"),
    ("Φ_0", "cuanto de flujo magnético", 2.067833848e-15, 0, "Wb", "flujo magnetico fluxon phi0"),
    ("K_J", "constante de Josephson", 483597.8484e9, 0, "Hz/V", "josephson"),
    ("R_K", "constante de von Klitzing", 25812.80745, 0, "Ω", "klitzing hall cuantico"),
    ("Z_0", "impedancia característica del vacío", 376.730313668, 0.000000057, "Ω", "impedancia vacio z0"),
]


def _fold(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def constant(query: str) -> list[dict]:
    q = _fold(query.strip())
    if q in ("", "lista", "todas", "list"):
        rows = _C
    else:
        exact = [c for c in _C if _fold(c[0]) == q or q.replace(" ", "_") == _fold(c[0])]
        rows = exact or [c for c in _C if all(w in _fold(f"{c[0]} {c[1]} {c[5]}") for w in q.split())]
    return [{"simbolo": s, "nombre": n, "valor": v, "incertidumbre": u, "exacta": u == 0, "unidad": un,
             "fuente": CODATA} for s, n, v, u, un, _ in rows]


def constant_text(query: str) -> str:
    rows = constant(query)
    if not rows:
        return "no encontrada; ⟦constantes: lista⟧ muestra todas"
    if len(rows) > 6:
        return "constantes: " + ", ".join(f"{r['simbolo']} ({r['nombre']})" for r in rows)
    return "; ".join(f"{r['simbolo']} = {r['valor']:.12g} {r['unidad']}".rstrip() + " ("
                     + ("exacta" if r["exacta"] else f"± {r['incertidumbre']:.2g}") + f"; {r['nombre']})"
                     for r in rows) + f" · fuente: {CODATA}"


# ---------------------------------------------------------------------------
# Gráficos

def png_bytes(rgb: np.ndarray) -> bytes:
    """PNG RGB de 8 bits a partir de un arreglo (alto, ancho, 3), sin dependencias."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].astype(np.uint8).tobytes() for i in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


_FONT = {  # 3×5 para rótulos numéricos del rasterizador propio
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
    "8": "111101111101111", "9": "111101111001111", "-": "000000111000000", ".": "000000000000010",
    "e": "000111111100111", "+": "000010111010000"}
_COLORS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]


def _raster(series, xr, yr, w=640, h=480) -> bytes:
    img = np.full((h, w, 3), 255, np.uint8)
    L, R, T, B = 60, w - 20, 20, h - 40

    def px(x, y):
        return (L + (x - xr[0]) / (xr[1] - xr[0]) * (R - L), B - (y - yr[0]) / (yr[1] - yr[0]) * (B - T))

    def line(x0, y0, x1, y1, col):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).round().astype(int)
        ys = np.linspace(y0, y1, n).round().astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        img[ys[ok], xs[ok]] = col

    def text(s, x, y):
        for k, ch in enumerate(s):
            bits = _FONT.get(ch)
            if bits:
                for i, b in enumerate(bits):
                    if b == "1":
                        img[y + 2 * (i // 3):y + 2 * (i // 3) + 2, x + 8 * k + 2 * (i % 3):x + 8 * k + 2 * (i % 3) + 2] = 0

    line(L, B, R, B, (0, 0, 0))
    line(L, T, L, B, (0, 0, 0))
    if xr[0] < 0 < xr[1]:
        line(*px(0, yr[0]), *px(0, yr[1]), (200, 200, 200))
    if yr[0] < 0 < yr[1]:
        line(*px(xr[0], 0), *px(xr[1], 0), (200, 200, 200))
    for k, (xs, ys) in enumerate(series):
        col = _COLORS[k % len(_COLORS)]
        pts = [px(a, b) for a, b in zip(xs, ys) if math.isfinite(b)]
        for (a0, b0), (a1, b1) in zip(pts, pts[1:]):
            line(a0, b0, a1, b1, col)
    text(f"{xr[0]:.3g}", L, B + 8)
    text(f"{xr[1]:.3g}", R - 8 * len(f"{xr[1]:.3g}"), B + 8)
    text(f"{yr[0]:.3g}", 2, B - 10)
    text(f"{yr[1]:.3g}", 2, T)
    return png_bytes(img)


def plot(spec: str, title: str = "") -> dict:
    """``sin(x); cos(x) | -3..3`` (funciones de x) o ``datos | 1 2 3 | 1 4 9`` (puntos)."""
    parts = [p.strip() for p in spec.split("|")]
    if parts[0].lower() in ("datos", "data", "puntos"):
        if len(parts) < 3:
            raise ValueError("formato: ⟦graficar: datos | x1 x2 … | y1 y2 …⟧")
        xs = parse_numbers(parts[1])
        series = [(xs, parse_numbers(p)) for p in parts[2:]]
        if any(len(s[1]) != len(xs) for s in series):
            raise ValueError("x e y deben tener la misma cantidad de valores")
        labels = [f"serie {i + 1}" for i in range(len(series))]
    else:
        rng = parts[1] if len(parts) > 1 else "-10..10"
        m = re.fullmatch(r"\s*(?:x\s*=\s*)?(.+?)\s*\.\.\s*(.+?)\s*", rng)
        if not m:
            raise ValueError("rango: a..b (por ejemplo -3..3)")
        a, b = (float(compile_expr(g, set())({})) for g in m.groups())
        if not b > a:
            raise ValueError("el rango debe ser creciente")
        xs = list(np.linspace(a, b, 400))
        labels = [f.strip() for f in parts[0].split(";") if f.strip()][:5]
        series = []
        for lab in labels:
            fn = compile_expr(lab, {"x"})
            ys = []
            for x in xs:
                try:
                    v = float(fn({"x": x}))
                except (ValueError, ZeroDivisionError, OverflowError):
                    v = float("nan")
                ys.append(v if math.isfinite(v) else float("nan"))
            series.append((xs, ys))
    allx = [v for s in series for v in s[0]]
    ally = [v for s in series for v in s[1] if math.isfinite(v)]
    if not ally:
        raise ValueError("no hay valores finitos para graficar")
    xr = (min(allx), max(allx))
    yr = (min(ally), max(ally))
    if yr[0] == yr[1]:
        yr = (yr[0] - 1, yr[1] + 1)
    if xr[0] == xr[1]:
        xr = (xr[0] - 1, xr[1] + 1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import io

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        for (sx, sy), lab in zip(series, labels):
            ax.plot(sx, sy, marker="o" if len(sx) <= 50 and parts[0].lower() in ("datos", "data", "puntos") else None,
                    label=lab)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="0.7", lw=0.8)
        if len(series) > 1 or parts[0].lower() not in ("datos", "data", "puntos"):
            ax.legend()
        if title:
            ax.set_title(title)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        data, engine = buf.getvalue(), "matplotlib"
    except ImportError:
        data, engine = _raster(series, xr, yr), "rasterizador propio"
    return {"png_b64": base64.b64encode(data).decode(), "motor": engine, "series": labels,
            "x": [xr[0], xr[1]], "y": [yr[0], yr[1]]}
