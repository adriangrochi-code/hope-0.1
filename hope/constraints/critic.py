"""CRE — motor de razonamiento crítico (glosario §7; HCRI §2.6, §3; ARS §6).

Revisa un razonamiento de Hope **mientras se produce** (no solo al final) y
devuelve una crítica concreta. No duplica a los especialistas: delega en HCRI
las restricciones y en las herramientas la aritmética y las unidades.

Comprueba:
  * cada paso ``expresión = valor`` con aritmética exacta (y con unidades si las tiene);
  * que el resultado tenga la dimensión de lo que se pregunta;
  * que el resultado cumpla los invariantes aplicables (lo derivado por HCRI);
  * datos contradictorios o faltantes en el enunciado → la respuesta correcta es
    señalarlo, no inventar un número (Carta art. 4; UAC);
  * repetición de pasos (bucles improductivos) y conclusiones sin pasos.

Cada observación distingue lo **comprobado** con herramienta de lo **sospechado**.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hope.constraints.model import UNITS, extract_quantities
from hope.constraints.solver import Analysis, analyze
from hope.tools.builtin import arith, numbers_equal, units

STEP_RE = re.compile(r"Paso\s*\d+\s*:\s*(.+?)\s*=\s*([^=()]+?)\s*(?:\((.*)\))?\s*$", re.IGNORECASE)
RESULT_RE = re.compile(r"Resultado:\s*(.+)")
NUMERIC_EXPR = re.compile(r"^[\d\s.,+\-*/×·^()]+$")


@dataclass
class Issue:
    kind: str            # aritmetica | unidades | dimension | restriccion | datos_contradictorios |
                         # datos_faltantes | repeticion | sin_resultado | sin_pasos
    detail: str
    step: int | None = None
    checked: bool = True  # True = comprobado con herramienta; False = sospecha
    severity: str = "error"


@dataclass
class Critique:
    verdict: str                          # ok | errores | incierto | debe_declarar_limitacion
    issues: list[Issue] = field(default_factory=list)
    analysis: Analysis | None = None
    checked_steps: int = 0

    def feedback(self) -> str:
        """Texto breve para que Hope corrija en la siguiente iteración (Loop Controller)."""
        if not self.issues:
            return ""
        lines = ["Revisión:"]
        for i in self.issues:
            where = f"paso {i.step}: " if i.step else ""
            lines.append(f"- {where}{i.detail}")
        return "\n".join(lines)


def _split_value(text: str) -> tuple[str, str]:
    """'11 V' → ('11', 'V'); '0,05' → ('0,05', '')."""
    m = re.match(r"\s*(-?[\d.,]+)\s*(.*)$", text.strip())
    return (m.group(1), m.group(2).strip().rstrip(".")) if m else (text.strip(), "")


def check_steps(text: str) -> tuple[list[Issue], int]:
    issues, checked = [], 0
    seen: dict[str, int] = {}
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        n = len(seen) + 1
        key = re.sub(r"\s+", "", lhs + "=" + rhs)
        if key in seen:
            issues.append(Issue("repeticion", f"repite el paso {seen[key]} sin aportar nada nuevo", n,
                                severity="aviso"))
            continue
        seen[key] = n
        value, unit = _split_value(rhs)
        try:
            if NUMERIC_EXPR.match(lhs) and not unit:
                checked += 1
                if not numbers_equal(arith(lhs), value):
                    issues.append(Issue("aritmetica", f"{lhs} no da {value}; da {arith(lhs).normalize()}", n))
            elif extract_quantities(lhs) and unit:
                checked += 1
                r = units(lhs.replace("·", "*").replace("×", "*"), f"{value} {unit}")
                if not r.get("same_dimension", True):
                    issues.append(Issue("unidades", f"{lhs} no tiene las unidades de «{unit}»", n))
                elif not r.get("equal"):
                    issues.append(Issue("aritmetica", f"{lhs} no da {value} {unit}; da {r['value']}", n))
        except Exception:
            pass                                    # paso simbólico o no interpretable: no se puede comprobar
    return issues, checked


def critique(problem: str, reasoning: str, answer: str | None = None,
             analysis: Analysis | None = None) -> Critique:
    an = analysis or analyze(problem)
    issues, checked = check_steps(reasoning)
    result = answer if answer is not None else (RESULT_RE.search(reasoning).group(1).strip()
                                                if RESULT_RE.search(reasoning) else None)

    if an.status == "datos_contradictorios":
        detail = "; ".join(c.get("detail", f"los datos no cumplen {c['invariant']}") for c in an.contradictions)
        detail = "; ".join(filter(None, [detail] + an.out_of_range))
        declares = result is not None and re.search(r"contradic|incompatib|inconsist|imposible", result, re.I)
        if not declares:
            issues.append(Issue("datos_contradictorios",
                                f"el enunciado tiene datos incompatibles ({detail}); hay que señalarlo en vez de dar un número"))
        return Critique("debe_declarar_limitacion" if not declares else "ok", issues, an, checked)

    if an.status == "datos_faltantes":
        declares = result is not None and re.search(r"falta|insuficient|no se puede|no alcanza|missing", result, re.I)
        if not declares:
            issues.append(Issue("datos_faltantes", "; ".join(an.missing) + "; hay que pedir el dato que falta",
                                checked=True))
        return Critique("debe_declarar_limitacion" if not declares else "ok", issues, an, checked)

    if result is None:
        issues.append(Issue("sin_resultado", "no llegó a un resultado (falta la línea «Resultado:»)", checked=False))
    else:
        if checked == 0 and "Paso" not in reasoning:
            issues.append(Issue("sin_pasos", "da un resultado sin mostrar cómo llegó", checked=False, severity="aviso"))
        value, unit = _split_value(result)
        unit_key = unit.lower()
        if an.target and unit and unit_key in UNITS and UNITS[unit_key][1] != an.target:
            issues.append(Issue("dimension", f"se pregunta una {an.target} pero el resultado está en «{unit}»"))
        elif an.expected:
            exp = an.expected[0]
            try:
                got = units(f"{value} {unit or exp['unit']}", f"{exp['value']} {exp['unit']}")
                if not got.get("equal"):
                    issues.append(Issue("restriccion",
                                        f"el resultado {result} no cumple {exp['invariant']}: debería dar "
                                        f"{exp['value']:g} {exp['unit']}"))
            except Exception:
                issues.append(Issue("restriccion", f"no se pudo interpretar el resultado «{result}»", checked=False))

    errors = [i for i in issues if i.severity == "error"]
    if errors:
        verdict = "errores"
    elif issues or (checked == 0 and not an.expected):
        verdict = "incierto"
    else:
        verdict = "ok"
    return Critique(verdict, issues, an, checked)
