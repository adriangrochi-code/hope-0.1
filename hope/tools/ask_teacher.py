"""``⟦preguntar_maestro: …⟧`` — consultar al maestro configurado como una herramienta más.

* La consulta pasa por el guardián (``Teacher.ask`` → acción ``teacher.query``): un maestro local
  es nivel 1; uno remoto sale a la red y requiere aprobación humana.
* Se le pide al maestro una respuesta breve y, si se puede, **una comprobación** con una
  herramienta determinista: ``COMPROBACION: ⟦calc: 220 * 0,05⟧ = 11``. El runtime ejecuta esa
  llamada él mismo (catálogo + guardián) y compara: la respuesta queda **verificada**,
  **refutada** o **no verificada**. Nunca se le cree al maestro sin comprobar (Herramientas §18).
* El texto del maestro se trata como dato: va dentro de ``⟪maestro · estado⟫ … ⟪fin⟫`` y se
  neutraliza (no puede abrir llamadas ``⟦ ⟧`` ni simular resultados).
"""

from __future__ import annotations

import re
from typing import Callable

from hope.tools.builtin import normalize_number_text, numbers_equal

# Herramientas deterministas con las que el maestro puede pedir que se compruebe su respuesta.
VERIFIERS = ("calc", "fraccion", "unidades", "resolver", "derivar", "integrar", "limite", "matriz", "logica",
             "estad", "bases", "constantes", "resistencias", "ajuste", "ode", "hash", "regex", "validar", "python")

PROMPT = """Respondé la pregunta de forma breve y precisa, en español.
Si la respuesta se puede comprobar con una herramienta determinista, agregá UNA línea de comprobación
con la llamada exacta y el resultado que debería dar. Herramientas válidas: {tools}.
Ejemplos de comprobación:
COMPROBACION: ⟦calc: 220 * 0,05⟧ = 11
COMPROBACION: ⟦unidades: 5 V / (0,02 A) en ohm⟧ = 250 ohm
Formato de la respuesta:
RESPUESTA: <respuesta>
COMPROBACION: <opcional, una sola línea>

PREGUNTA: {question}"""

_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def _numbers(s: str) -> list[str]:
    return [normalize_number_text(x) for x in _NUM.findall(s)]


def same_result(got: str, expected: str) -> bool:
    g, e = got.strip(), expected.strip()
    if g.casefold() == e.casefold():
        return True
    ge, ee = _numbers(g), _numbers(e)
    if ee and ge and len(ee) == len(ge) and all(numbers_equal(a, b, rel=1e-6) for a, b in zip(ge, ee)):
        return True
    return len(e) >= 3 and e.casefold() in g.casefold()


def ask_teacher(teacher, question: str, run_tool: Callable[[str, str], str] | None = None) -> dict:
    if teacher is None:
        return {"estado": "sin maestro", "respuesta": "", "detalle": "no hay maestro configurado (configs/teacher.yaml)"}
    q = question.strip()
    if not q or len(q) > 4000:
        raise ValueError("pregunta vacía o demasiado larga (máximo 4000 caracteres)")
    reply = teacher.ask([{"role": "user", "content": PROMPT.format(tools=", ".join(VERIFIERS), question=q)}],
                        purpose="herramienta preguntar_maestro")
    text = reply.text.strip()
    m = re.search(r"RESPUESTA\s*:\s*(.+?)(?=\n\s*COMPROBACI[OÓ]N\s*:|\Z)", text, re.S | re.I)
    answer = (m.group(1) if m else re.sub(r"(?im)^\s*COMPROBACI[OÓ]N\s*:.*$", "", text)).strip()
    cm = re.search(r"COMPROBACI[OÓ]N\s*:\s*⟦\s*([\wáéíóúñ]+)\s*:\s*(.*?)\s*⟧\s*=\s*(.+)", text, re.I)
    out = {"respuesta": answer, "modelo": reply.model, "estado": "no verificada",
           "detalle": "el maestro no dio una comprobación ejecutable"}
    if cm is None:
        return out
    name, arg, expected = cm.group(1).lower(), cm.group(2), cm.group(3).strip().splitlines()[0].strip()
    out["comprobacion"] = {"herramienta": name, "entrada": arg, "esperado": expected}
    if name not in VERIFIERS or run_tool is None:
        out["detalle"] = f"la comprobación usa «{name}», que no es una herramienta determinista admitida"
        return out
    got = run_tool(name, arg)
    out["comprobacion"]["obtenido"] = got
    if got.startswith(("error", "denegado", "pendiente")):
        out["detalle"] = f"no se pudo ejecutar la comprobación: {got[:120]}"
        return out
    if not same_result(got, expected):
        out.update(estado="refutada", detalle=f"⟦{name}⟧ dio «{got[:80]}», el maestro esperaba «{expected[:80]}»")
        return out
    nums_exp = _numbers(expected)
    if nums_exp and not any(numbers_equal(a, b, rel=1e-6) for a in _numbers(answer) for b in nums_exp[:1]):
        out["detalle"] = "la comprobación es correcta, pero la respuesta no menciona el valor comprobado"
        return out
    out.update(estado="verificada", detalle=f"comprobado con ⟦{name}⟧")
    return out


def ask_teacher_text(r: dict) -> str:
    if r["estado"] == "sin maestro":
        return r["detalle"]
    from hope.tools.web import clip, neutralize
    body, _ = neutralize(r["respuesta"])
    tag = {"verificada": "VERIFICADA", "refutada": "REFUTADA", "no verificada": "NO VERIFICADA"}[r["estado"]]
    return f"⟪maestro · {tag}: {r['detalle'][:140]}⟫ {clip(body, 360)} ⟪fin⟫"
