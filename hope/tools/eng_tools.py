"""Herramientas de datos y de ingeniería (docs/HERRAMIENTAS_HN1.md §3, niveles 4–5).

* ``circuit`` — simulador propio de circuitos lineales por análisis nodal
  modificado (MNA), en continua y en alterna (fasores), con NumPy. Formato de
  netlist estilo SPICE:

      V1 in 0 5          fuente de tensión (V, o amplitud en AC)
      I1 a 0 0.01        fuente de corriente (A)
      R1 in out 220      resistencia (ohm)
      C1 out 0 1e-6      capacitor (F)
      L1 out 0 1e-3      inductor (H)
      .ac 1000           (opcional) frecuencia en Hz; sin esto, análisis de continua

  Devuelve tensiones de nodo (módulo y fase en AC) y corrientes de las fuentes.
  Es un verificador independiente de los cálculos de Hope y de HCRI.
* ``filter_response`` — respuesta en frecuencia de H(s) = num(s)/den(s): módulo
  en dB, fase y frecuencia de corte de −3 dB. Atajos ``rc_pasabajos R C`` y
  ``rc_pasaaltos R C``.
* ``sql_csv`` — consulta SQL (solo SELECT) sobre un CSV, en el sandbox.
* ``convert`` — conversión entre CSV, JSON y YAML.
* ``correlations`` — correlaciones entre columnas numéricas de un CSV (Pearson y Spearman),
  ordenadas por intensidad: el primer paso para encontrar patrones en datos.

Limitaciones: el simulador es lineal (sin diodos ni transistores) y de una sola
frecuencia por análisis. Para lo no lineal, ngspice si está instalado (pendiente).
"""

from __future__ import annotations

import csv
import io
import json
import math
import re

import numpy as np
import yaml

from hope.tools.sandbox import run_python

SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9}


def parse_value(tok: str) -> float:
    m = re.fullmatch(r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)(meg|[fpnuµmkg])?\w*", tok.strip().lower().replace(",", "."))
    if not m:
        raise ValueError(f"valor inválido: {tok}")
    return float(m.group(1)) * SI.get(m.group(2) or "", 1.0)


def circuit(netlist: str) -> dict:
    elems, freq = [], None
    for raw in netlist.replace(";", "\n").splitlines():
        line = raw.split("*")[0].strip()
        if not line:
            continue
        if line.lower().startswith(".ac"):
            freq = parse_value(line.split()[1])
            continue
        parts = line.split()
        if len(parts) != 4 or parts[0][0].upper() not in "VIRCL":
            raise ValueError(f"línea inválida: «{line}» (formato: NOMBRE nodo+ nodo- valor)")
        elems.append((parts[0], parts[1], parts[2], parse_value(parts[3])))
    nodes = sorted({n for _, a, b, _ in elems for n in (a, b)} - {"0", "gnd"})
    idx = {n: i for i, n in enumerate(nodes)}
    vsrc = [e for e in elems if e[0][0].upper() == "V"]
    n, m = len(nodes), len(vsrc)
    if n == 0:
        raise ValueError("el circuito no tiene nodos (además de tierra)")
    w = 2 * math.pi * freq if freq else 0.0
    A = np.zeros((n + m, n + m), dtype=complex)
    z = np.zeros(n + m, dtype=complex)

    def stamp(a, b, y):
        for p, q, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            if p in idx and q in idx:
                A[idx[p], idx[q]] += s * y

    for name, a, b, val in elems:
        k = name[0].upper()
        if k == "R":
            if val <= 0:
                raise ValueError(f"{name}: resistencia no positiva")
            stamp(a, b, 1 / val)
        elif k == "C":
            stamp(a, b, 1j * w * val)                     # en continua es circuito abierto
        elif k == "L":
            if w == 0:
                stamp(a, b, 1e9)                          # en continua, cortocircuito (conductancia muy alta)
            else:
                stamp(a, b, 1 / (1j * w * val))
        elif k == "I":                                    # corriente que entra al nodo b desde a
            if a in idx:
                z[idx[a]] -= val
            if b in idx:
                z[idx[b]] += val
    for j, (name, a, b, val) in enumerate(vsrc):
        r = n + j
        if a in idx:
            A[idx[a], r] += 1
            A[r, idx[a]] += 1
        if b in idx:
            A[idx[b], r] -= 1
            A[r, idx[b]] -= 1
        z[r] = val
    try:
        x = np.linalg.solve(A, z)
    except np.linalg.LinAlgError:
        raise ValueError("el circuito no tiene solución única (nodo flotante o fuentes en conflicto)") from None

    def fmt(c: complex) -> dict | float:
        if freq:
            return {"modulo": round(abs(c), 9), "fase_grados": round(math.degrees(math.atan2(c.imag, c.real)), 6)}
        return round(c.real, 9)
    return {"analisis": f"AC {freq:g} Hz" if freq else "DC",
            "tensiones": {nd: fmt(x[i]) for nd, i in idx.items()},
            "corrientes_fuentes": {vsrc[j][0]: fmt(-x[n + j]) for j in range(m)}}


def filter_response(spec: str, freqs: list[float] | None = None) -> dict:
    """``spec``: «num=[1] den=[1e-3, 1]» (coeficientes en potencias decrecientes de s) o
    «rc_pasabajos 1k 100n» / «rc_pasaaltos R C»."""
    s = spec.strip().lower()
    if s.startswith("rc_"):
        kind, r, c = s.split()[:3]
        rc = parse_value(r) * parse_value(c)
        num, den = ([1.0], [rc, 1.0]) if "bajos" in kind else ([rc, 0.0], [rc, 1.0])
    else:
        num = json.loads(re.search(r"num\s*=\s*(\[.*?\])", s).group(1))
        den = json.loads(re.search(r"den\s*=\s*(\[.*?\])", s).group(1))
    f = np.array(freqs) if freqs else np.logspace(0, 6, 61)
    jw = 1j * 2 * np.pi * f
    h = np.polyval(num, jw) / np.polyval(den, jw)
    mag = 20 * np.log10(np.abs(h) + 1e-300)
    # Frecuencia de corte con una grilla fina (cada 0,1 % en escala logarítmica).
    fine = np.logspace(-2, 9, 11001)
    hf = np.abs(np.polyval(num, 1j * 2 * np.pi * fine) / np.polyval(den, 1j * 2 * np.pi * fine))
    magf = 20 * np.log10(hf + 1e-300)
    ref = magf.max()
    below = np.where(magf <= ref - 3.0103)[0]
    fc = None
    if len(below):
        if "altos" in s:
            above = np.where(magf > ref - 3.0103)[0]
            fc = float(fine[above[0]]) if len(above) else None
        else:
            fc = float(fine[below[0]])
    poles = np.roots(den)
    return {"frecuencias_hz": f.round(6).tolist()[:12], "modulo_db": mag.round(4).tolist()[:12],
            "fase_grados": np.degrees(np.angle(h)).round(4).tolist()[:12],
            "corte_3db_hz_aprox": fc, "polos": [complex(p).real for p in poles],
            "estable": bool(np.all(poles.real < 0))}


def sql_csv(csv_text: str, query: str) -> list[dict]:
    if not re.match(r"^\s*select\b", query, re.I) or ";" in query.strip().rstrip(";"):
        raise ValueError("solo se permite una consulta SELECT")
    code = (
        "import csv, io, json, sqlite3, sys\n"
        "d = json.loads(sys.stdin.read())\n"
        "rows = list(csv.reader(io.StringIO(d['csv'])))\n"
        "head, data = rows[0], rows[1:]\n"
        "db = sqlite3.connect(':memory:')\n"
        "db.execute('CREATE TABLE t (' + ', '.join(f'\"{h}\"' for h in head) + ')')\n"
        "db.executemany('INSERT INTO t VALUES (' + ','.join('?' * len(head)) + ')', data)\n"
        "cur = db.execute(d['q'])\n"
        "cols = [c[0] for c in cur.description]\n"
        "print(json.dumps([dict(zip(cols, r)) for r in cur.fetchmany(200)]))\n")
    res = run_python(code, stdin=json.dumps({"csv": csv_text, "q": query.rstrip(";")}), timeout_s=20)
    if not res.ok:
        raise ValueError((res.stderr.strip().splitlines() or ["error"])[-1][:200])
    return json.loads(res.stdout.strip().splitlines()[-1])


def convert(text: str, src: str, dst: str) -> str:
    src, dst = src.lower(), dst.lower()
    if src == "csv":
        data = list(csv.DictReader(io.StringIO(text)))
    elif src == "json":
        data = json.loads(text)
    elif src == "yaml":
        data = yaml.safe_load(text)
    else:
        raise ValueError("formatos: csv, json, yaml")
    if dst == "json":
        return json.dumps(data, ensure_ascii=False, indent=2)
    if dst == "yaml":
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    if dst == "csv":
        rows = data if isinstance(data, list) else [data]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()
    raise ValueError("formatos: csv, json, yaml")


def correlations(csv_text: str, min_abs: float = 0.0) -> dict:
    """Correlaciones de Pearson (lineal) y Spearman (por rangos, monótona) entre columnas numéricas.

    Recordatorio que se devuelve junto al resultado: correlación no implica causalidad; con pocas
    filas las correlaciones altas pueden ser casualidad (se informa ``n``)."""
    rows = list(csv.DictReader(io.StringIO(csv_text.strip())))
    cols: dict[str, list[float]] = {}
    for name in (rows[0].keys() if rows else []):
        try:
            vals = [float(str(r[name]).replace(",", ".")) for r in rows]
        except (TypeError, ValueError):
            continue
        if len(set(vals)) > 1:
            cols[name] = vals
    names = list(cols)

    def ranks(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        vals = np.asarray(v)
        for x in np.unique(vals):                     # empates: rango promedio
            idx = vals == x
            r[idx] = r[idx].mean()
        return r
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = np.asarray(cols[a]), np.asarray(cols[b])
            p = float(np.corrcoef(x, y)[0, 1])
            sp = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
            m = max(abs(p), abs(sp))
            if m < min_abs:
                continue
            fuerza = "fuerte" if m >= 0.7 else "moderada" if m >= 0.4 else "débil"
            pairs.append({"a": a, "b": b, "pearson": round(p, 4), "spearman": round(sp, 4), "fuerza": fuerza})
    pairs.sort(key=lambda d: -max(abs(d["pearson"]), abs(d["spearman"])))
    return {"n": len(rows), "columnas": names, "pares": pairs,
            "nota": "correlación no implica causalidad; con n chico puede ser casualidad"}
