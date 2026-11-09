"""DCE — aprendizaje de restricciones nuevas (HCRI §2.5, §8, §9).

Una restricción descubierta o enseñada **empieza como hipótesis**. Sube de
estado solo con evidencia:

* se prueba con casos (ejemplos favorables y búsqueda de contraejemplos);
* un solo contraejemplo válido la refuta;
* con suficientes casos que la cumplen y ninguno que la contradiga queda
  ``RESPALDADO``; ``VALIDADO`` requiere además un verificador independiente.

Funcionar en algunos ejemplos no la vuelve verdadera (HCRI §8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from hope.constraints.model import Invariant
from hope.epistemic import EpistemicRecord, EpistemicStatus, Evidence
from hope.tools.sandbox import run_python

EVAL_CODE = """
import json, sys, sympy
d = json.loads(sys.stdin.read())
syms = {s: sympy.Symbol(s) for s in d["symbols"]}
eqs = [sympy.sympify(e, locals=syms) for e in d["equations"]]
out = []
for case in d["cases"]:
    vals = {syms[k]: sympy.nsimplify(v, rational=True) for k, v in case.items() if k in syms}
    res = [float(e.subs(vals)) for e in eqs]
    scale = max([abs(float(v)) for v in vals.values()] + [1.0])
    out.append(all(abs(r) <= 1e-6 * scale for r in res))
print(json.dumps(out))
"""


@dataclass
class ConstraintTest:
    invariant: Invariant
    record: EpistemicRecord = field(default_factory=lambda: EpistemicRecord(EpistemicStatus.HIPOTESIS))
    passed: int = 0
    counterexamples: list[dict] = field(default_factory=list)


def propose(name: str, equations: list[str], symbols: dict[str, tuple[str, str]], domain: str,
            provenance: str) -> ConstraintTest:
    inv = Invariant(f"inv_{name.lower().replace(' ', '_')}", name, equations, symbols, domain,
                    provenance=provenance, status=EpistemicStatus.HIPOTESIS)
    return ConstraintTest(inv)


def test_cases(ct: ConstraintTest, cases: list[dict], *, source: str, independent: bool = False,
               min_cases: int = 5) -> ConstraintTest:
    """Evalúa la restricción en casos con valores medidos o calculados por otra vía."""
    res = run_python(EVAL_CODE, stdin=json.dumps({"symbols": list(ct.invariant.symbols),
                                                  "equations": ct.invariant.equations, "cases": cases}),
                     timeout_s=30)
    if not res.ok:
        return ct
    results = json.loads(res.stdout.strip().splitlines()[-1])
    ct.passed += sum(results)
    ct.counterexamples += [c for c, ok in zip(cases, results) if not ok]
    if ct.counterexamples:
        ct.record.transition(EpistemicStatus.REFUTADO)
    elif ct.passed >= min_cases:
        target = EpistemicStatus.VALIDADO_CONDICIONAL if independent else EpistemicStatus.RESPALDADO
        if ct.record.status != target:
            ct.record.transition(target, Evidence(source, independent=independent,
                                                  description=f"{ct.passed} casos sin contraejemplos"))
    ct.invariant.status = ct.record.status
    return ct
