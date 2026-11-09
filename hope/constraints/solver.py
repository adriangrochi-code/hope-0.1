"""HCRI — análisis de un problema guiado por restricciones (HCRI §2.3–2.4, §3).

``analyze(problema)`` hace lo que HCRI §3 pide **antes** de resolver:

  1. identifica magnitudes y la incógnita;
  2. busca invariantes aplicables (solo si sus dimensiones están en el problema);
  3. comprueba que los datos sean compatibles entre sí (CCE): datos
     contradictorios, datos faltantes, valores fuera de rango;
  4. si los datos alcanzan, deriva el valor esperado de la incógnita con un
     solver determinista (SymPy, en sandbox).

La red no resuelve lo que el solver verifica mejor (HCRI §2.3): este resultado
se usa para guiar a Hope y para criticar su respuesta, no para reemplazar su
razonamiento.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from hope.constraints.model import (LIBRARY, RANGES, Invariant, Quantity, extract_quantities,
                                    unknown_dimension)
from hope.tools.sandbox import run_python

SI_FACTOR_CODE = """
import json, sys, sympy, pint
u = pint.UnitRegistry()
d = json.loads(sys.stdin.read())
out = {"solutions": [], "contradictions": []}
for inv in d["invariants"]:
    syms = {s: sympy.Symbol(s) for s in inv["symbols"]}
    eqs = [sympy.sympify(e, locals=syms) for e in inv["equations"]]
    known = {}
    for s, (dim, unit) in inv["symbols"].items():
        if dim in d["known"]:
            q = d["known"][dim]
            known[syms[s]] = sympy.nsimplify(u.Quantity(q["value"], q["unit"]).to(unit).magnitude, rational=True)
    unknown = [s for s, (dim, _) in inv["symbols"].items() if dim not in d["known"]]
    if not unknown:
        for e in eqs:
            r = e.subs(known)
            scale = max([abs(float(v)) for v in known.values()] + [1.0])
            if abs(float(r)) > 1e-6 * scale:
                out["contradictions"].append({"invariant": inv["id"], "residual": float(r)})
        continue
    if len(unknown) == 1 and inv["symbols"][unknown[0]][0] == d["target"]:
        sol = sympy.solve([e.subs(known) for e in eqs], syms[unknown[0]], dict=True)
        for s in sol:
            val = s.get(syms[unknown[0]])
            if val is not None and val.is_real:
                out["solutions"].append({"invariant": inv["id"], "value": float(val),
                                         "unit": inv["symbols"][unknown[0]][1]})
print(json.dumps(out))
"""


@dataclass
class Analysis:
    quantities: list[Quantity]
    target: str | None
    invariants: list[Invariant]
    expected: list[dict] = field(default_factory=list)          # {invariant, value, unit}
    contradictions: list[dict] = field(default_factory=list)
    out_of_range: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.contradictions or self.out_of_range:
            return "datos_contradictorios"
        if self.ambiguous:
            return "ambiguo"
        if self.expected:
            return "determinado"
        if self.target and self.missing:
            return "datos_faltantes"
        return "sin_restricciones_conocidas"

    def hint(self) -> str:
        """Texto breve con las restricciones aplicables, para guiar el razonamiento de Hope."""
        if not self.invariants:
            return ""
        laws = "; ".join(f"{inv.name}: {' = 0; '.join(inv.equations)} = 0 ({inv.domain})" for inv in self.invariants)
        return f"Restricciones: {laws}"


def analyze(problem: str, library: list[Invariant] = LIBRARY) -> Analysis:
    qs = extract_quantities(problem)
    target = unknown_dimension(problem)
    by_dim: dict[str, list[Quantity]] = {}
    for q in qs:
        by_dim.setdefault(q.dimension, []).append(q)
    ambiguous = [d for d, lst in by_dim.items() if len({(q.value, q.unit) for q in lst}) > 1]
    known = {d: {"value": lst[0].value, "unit": lst[0].unit} for d, lst in by_dim.items() if d not in ambiguous}
    if target in known:
        known.pop(target)            # lo que se pregunta no es dato aunque aparezca mencionado
    present = set(known) | ({target} if target else set())
    applicable = [inv for inv in library if inv.dimensions() <= present and len(inv.dimensions() & set(known)) >= 2]
    out_of_range = [f"{q.text}: {r.reason}" for q in qs for r in RANGES
                    if r.dimension == q.dimension and ((r.low is not None and q.value < r.low)
                                                       or (r.high is not None and q.value > r.high))]
    an = Analysis(qs, target, applicable, out_of_range=out_of_range, ambiguous=ambiguous)
    if applicable and not ambiguous:
        payload = {"known": known, "target": target,
                   "invariants": [{"id": i.id, "equations": i.equations, "symbols": i.symbols} for i in applicable]}
        res = run_python(SI_FACTOR_CODE, stdin=json.dumps(payload), timeout_s=30)
        if res.ok:
            data = json.loads(res.stdout.strip().splitlines()[-1])
            an.expected, an.contradictions = data["solutions"], data["contradictions"]
    if len(an.expected) > 1:
        vals = [e["value"] for e in an.expected]
        if max(vals) - min(vals) > 1e-6 * max(abs(v) for v in vals):
            an.contradictions.append({"invariant": "varias leyes",
                                      "detail": "las leyes aplicables dan valores distintos: los datos se contradicen",
                                      "values": {e["invariant"]: e["value"] for e in an.expected}})
    if target and not an.expected and not an.contradictions:
        # ¿Qué falta? Leyes que incluyen la incógnita y algún dato, pero no todos los necesarios.
        for inv in library:
            dims = inv.dimensions()
            if target in dims and dims & set(known):
                lacking = sorted(dims - present)
                if lacking:
                    an.missing.append(f"para usar «{inv.name}» falta: {', '.join(lacking)}")
        if not an.missing:
            an.missing = [f"no hay datos suficientes para determinar la {target}"]
    return an
