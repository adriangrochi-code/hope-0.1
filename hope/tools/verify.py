"""Verificación independiente de episodios del maestro (Herramientas §4 etapa 6; Language Core §10).

Cada paso con campo ``verificacion`` se comprueba con una herramienta. El
resultado del episodio:

* algún paso falla → ``REFUTADO`` (no se entrena con él; se guarda como ejemplo de error para RRA);
* todos los verificables pasan y hay al menos uno → ``RESPALDADO``;
* no hay nada verificable → ``INFERIDO`` (se entrena con menos peso).

El maestro no es la autoridad: la herramienta lo es (Herramientas §18).
"""

from __future__ import annotations

from hope.epistemic import EpistemicStatus
from hope.teacher.episode import Episode
from hope.tools.builtin import arith, numbers_equal, normalize_number_text
from hope.tools.catalog import Catalog, ToolDenied

TRAINING_WEIGHT = {EpistemicStatus.RESPALDADO: 1.0, EpistemicStatus.INFERIDO: 0.3, EpistemicStatus.REFUTADO: 0.0}


def check_step(catalog: Catalog, check: dict) -> str:
    """Devuelve 'ok', 'falla' o 'no_verificable'."""
    kind = check.get("tipo")
    try:
        if kind == "aritmetica":
            value = catalog.call("hte_arith_v1", expr=str(check["expr"]))
            # «tolerancia» relativa explícita cuando el resultado se redondeó a propósito.
            rel = min(float(check.get("tolerancia", 1e-9)), 1e-3)
            return "ok" if numbers_equal(value, str(check["esperado"]), rel=rel) else "falla"
        if kind == "simbolico":
            eq = catalog.call("hte_symbolic_v1", lhs=str(check["lhs"]), rhs=str(check["rhs"]))
            return "no_verificable" if eq is None else ("ok" if eq else "falla")
        if kind == "unidades":
            r = catalog.call("hte_units_v1", expr=str(check["expr"]), expected=str(check["esperado"]))
            return "ok" if r.get("equal") else "falla"
        if kind == "circuito":
            r = catalog.call("hte_circuit_v1", netlist=str(check["netlist"]))
            v = r["tensiones"][str(check["nodo"])]
            v = v["modulo"] if isinstance(v, dict) else v
            return "ok" if numbers_equal(v, str(check["esperado"]), rel=1e-6) else "falla"
        if kind == "codigo":
            r = catalog.call("hte_python_v1", code=str(check["codigo"]))
            if not r["ok"]:
                return "falla"
            want = check.get("salida_esperada")
            return "ok" if want is None or r["stdout"].strip() == str(want).strip() else "falla"
    except ToolDenied:
        return "no_verificable"
    except Exception:
        return "no_verificable"          # expresión que la herramienta no entiende: no prueba nada
    return "no_verificable"


def verify_episode(catalog: Catalog, ep: Episode) -> Episode:
    results = []
    for step in ep.data.get("pasos", []):
        chk = step.get("verificacion")
        results.append({"n": step.get("n"), "resultado": check_step(catalog, chk) if chk else "no_verificable"})
    # El resultado final también se comprueba si es una expresión aritmética simple.
    final = str(ep.data.get("resultado", ""))
    last = ep.data["pasos"][-1].get("resultado_intermedio", "")
    if final and last and normalize_number_text(final).replace(" ", "") != normalize_number_text(str(last)).replace(" ", ""):
        try:
            same = numbers_equal(arith(final), arith(str(last)))
        except Exception:
            same = None
        if same is False:
            results.append({"n": "resultado", "resultado": "falla"})
    outcomes = [r["resultado"] for r in results]
    if "falla" in outcomes:
        status = EpistemicStatus.REFUTADO
    elif "ok" in outcomes:
        status = EpistemicStatus.RESPALDADO
    else:
        status = EpistemicStatus.INFERIDO
    ep.verification = {"steps": results, "status": status.value, "weight": TRAINING_WEIGHT[status]}
    return ep
