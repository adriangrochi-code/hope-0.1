"""Episodios de enseñanza: pedido al maestro, lectura y texto de entrenamiento
(ARQUITECTURA_HN1 §10.1; HLA §12; Herramientas §4 etapa 4).

Un episodio es la explicación completa de un concepto con un problema
resuelto paso a paso, verificable. Formato JSON (las claves en español):

    tema, idioma, problema, datos[], supuestos[], plan, pasos[{n, operacion,
    resultado_intermedio, justificacion, verificacion?}], resultado,
    verificacion_inversa, alternativas_descartadas[], errores_frecuentes[],
    variaciones[{problema, respuesta}]

``verificacion`` (opcional por paso) dice cómo comprobarlo con una herramienta:
    {"tipo": "aritmetica", "expr": "...", "esperado": "..."}
    {"tipo": "simbolico",  "lhs": "...", "rhs": "..."}
    {"tipo": "unidades",   "expr": "220 ohm * 0.05 A", "esperado": "11 V"}
    {"tipo": "codigo",     "codigo": "...", "salida_esperada": "..."}

Las **variaciones** no se usan para entrenar: son los problemas nuevos con
los que se comprueba si Hope aprendió (nivel «aplicación» de ART, HLA §6).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

REQUIRED = ("problema", "pasos", "resultado")

SYSTEM_PROMPT = (
    "Sos el maestro de Hope, una red neuronal pequeña que está aprendiendo. Enseñás con "
    "explicaciones didácticas completas: datos, supuestos, plan, cada paso con su justificación, "
    "cómo se llega al resultado y cómo comprobarlo desde el resultado hacia atrás. Cuando un paso "
    "se pueda comprobar con una herramienta (aritmética, álgebra simbólica, unidades o código), "
    "incluí el campo 'verificacion'. No inventes datos: si algo es incierto, decilo en 'supuestos'. "
    "Respondé solo con un objeto JSON con el formato pedido."
)

FORMAT_HINT = (
    '{"tema": "...", "idioma": "es|en", "problema": "...", "datos": [{"valor": "...", "unidad": "..."}], '
    '"supuestos": ["..."], "plan": "...", "pasos": [{"n": 1, "operacion": "...", '
    '"resultado_intermedio": "...", "justificacion": "...", "verificacion": {"tipo": "aritmetica", '
    '"expr": "...", "esperado": "..."}}], "resultado": "...", "verificacion_inversa": "...", '
    '"alternativas_descartadas": ["..."], "errores_frecuentes": ["..."], '
    '"variaciones": [{"problema": "...", "respuesta": "..."}]}'
)


RULES_HINT = (
    "Reglas: 'resultado' es la respuesta final breve (un valor con su unidad, o una frase corta si no es "
    "numérica) y coincide con el 'resultado_intermedio' del último paso. Verificaciones posibles: "
    "aritmetica ('expr' solo con números y + - * / ** paréntesis, sin unidades ni letras; 'esperado' solo "
    "un número, con punto decimal), unidades ('expr' como \"2 A * 5 ohm\", 'esperado' como \"10 V\"), "
    "codigo ('codigo' Python que imprime con print, 'salida_esperada' exacta). "
)


@dataclass
class Episode:
    data: dict
    source_model: str = ""
    raw_thinking: str | None = None
    verification: dict = field(default_factory=dict)     # lo completa hope.tools.verify

    @property
    def topic(self) -> str:
        return str(self.data.get("tema", ""))

    @property
    def variations(self) -> list[dict]:
        return [v for v in self.data.get("variaciones", []) if "problema" in v and "respuesta" in v]

    @staticmethod
    def _tool_call(step: dict) -> str | None:
        """Paso verificable → llamada a herramienta (docs/HERRAMIENTAS_HN1.md §2)."""
        v = step.get("verificacion") or {}
        kind = v.get("tipo")
        if kind == "aritmetica" and v.get("expr") and v.get("esperado") is not None:
            return f"⟦calc: {v['expr']}⟧ → {v['esperado']}"
        if kind == "unidades" and v.get("expr") and v.get("esperado"):
            unit = str(v["esperado"]).split()[-1]
            return f"⟦unidades: {v['expr']} en {unit}⟧ → {v['esperado']}"
        if kind == "codigo" and v.get("codigo") and v.get("salida_esperada") is not None:
            return f"⟦python: {v['codigo']}⟧ → {v['salida_esperada']}"
        return None

    def training_text(self, tool_calls: bool = False) -> str:
        """Texto con el que Hope entrena: explicación completa, sin las variaciones de evaluación.

        Con ``tool_calls`` los pasos verificables se escriben como llamadas a herramientas
        con su resultado, para que Hope aprenda a usarlas mientras razona.
        """
        d = self.data
        lines = [f"Tema: {d.get('tema', '')}", f"Problema: {d['problema']}"]
        if d.get("datos"):
            lines.append("Datos: " + "; ".join(
                f"{x.get('valor', '')} {x.get('unidad', '')}".strip() for x in d["datos"]))
        if d.get("supuestos"):
            lines.append("Supuestos: " + "; ".join(map(str, d["supuestos"])))
        if d.get("plan"):
            lines.append(f"Plan: {d['plan']}")
        for p in d["pasos"]:
            call = self._tool_call(p) if tool_calls else None
            if call:
                lines.append(f"Paso {p.get('n', '')}: {call} ({p.get('justificacion', '')})")
            else:
                lines.append(f"Paso {p.get('n', '')}: {p.get('operacion', '')} = {p.get('resultado_intermedio', '')}"
                             f" ({p.get('justificacion', '')})")
        lines.append(f"Resultado: {d['resultado']}")
        if d.get("verificacion_inversa"):
            lines.append(f"Comprobación: {d['verificacion_inversa']}")
        if d.get("errores_frecuentes"):
            lines.append("Errores frecuentes: " + "; ".join(map(str, d["errores_frecuentes"])))
        return "\n".join(lines)


class EpisodeParseError(ValueError):
    pass


def _fix_closer(text: str, pos: int) -> str | None:
    """Si en ``pos`` hay un cierre equivocado (``)`` o ``}`` donde va ``]``, etc.), lo cambia por el correcto."""
    stack, in_str, esc = [], False, False
    for ch in text[:pos]:
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}" and stack:
            stack.pop()
    if pos < len(text) and stack and text[pos] in ")]}" and text[pos] != stack[-1]:
        return text[:pos] + stack[-1] + text[pos + 1:]
    return None


def _loads_lenient(candidate: str) -> tuple[dict, bool]:
    """``json.loads`` con dos reparaciones mínimas (errores de tipeo típicos de modelos chicos):
    texto sobrante después del objeto y un cierre de corchete equivocado. No toca el contenido."""
    repaired = False
    for _ in range(5):
        try:
            data, end = json.JSONDecoder().raw_decode(candidate)
            return data, repaired or bool(candidate[end:].strip())
        except json.JSONDecodeError as e:
            fixed = _fix_closer(candidate, e.pos)
            if fixed is None:
                raise
            candidate, repaired = fixed, True
    return json.loads(candidate), repaired


def parse_episode(text: str) -> dict:
    """Extrae el JSON de la respuesta (con o sin bloque ```json) y valida los campos mínimos."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        data, repaired = _loads_lenient(candidate)
    except json.JSONDecodeError as e:
        raise EpisodeParseError(f"JSON inválido: {e}") from e
    if not isinstance(data, dict):
        raise EpisodeParseError("se esperaba un objeto JSON")
    if repaired:
        data["reparacion_json"] = True
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise EpisodeParseError(f"faltan campos: {missing}")
    if not isinstance(data["pasos"], list) or not data["pasos"]:
        raise EpisodeParseError("'pasos' debe ser una lista no vacía")
    return data


def curriculum_request(topic: str, level: int, language: str = "es", *, avoid: list[str] | None = None,
                       focus: str = "") -> list[dict]:
    """Pedido de la fase de arranque: el maestro enseña por iniciativa propia.

    ``avoid``: problemas ya enseñados de este tema (para que no los repita).
    ``focus``: aspecto o contexto sugerido para variar los problemas.
    """
    extra = ""
    if focus:
        extra += f"ENFOQUE SUGERIDO: {focus}\n"
    if avoid:
        extra += "NO repitas estos problemas ya enseñados:\n" + "\n".join(f"- {a[:160]}" for a in avoid[-6:]) + "\n"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"TEMA: {topic}\nNIVEL: {level} (1 = básico)\nIDIOMA: {language}\n{extra}"
                f"Enseñá este tema con un problema nuevo resuelto paso a paso y dos o más variaciones "
                f"con su respuesta (para evaluar si aprendió). {RULES_HINT}Formato:\n{FORMAT_HINT}")}]


def question_request(topic: str, *, attempt: str, expected: str | None, known: list[str],
                     problem: str, language: str = "es", review: str = "") -> list[dict]:
    """Pregunta dirigida de Hope (HLA §14): qué intentó, qué sabe y qué falló.
    No pide repetir todo, sino aclarar la carencia concreta."""
    known_txt = "\n".join(f"- {k}" for k in known[:5]) or "- (nada relevante en memoria)"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"TEMA: {topic}\nIDIOMA: {language}\n"
                f"Hope intentó resolver: {problem}\nSu respuesta fue: {attempt!r}\n"
                + (f"La respuesta correcta es: {expected}\n" if expected else "")
                + (f"Lo que detectó su propia revisión (CRE/HCRI):\n{review}\n" if review else "")
                + f"Lo que Hope ya tiene en memoria:\n{known_txt}\n"
                "Explicá qué parte del razonamiento falló y enseñá el procedimiento correcto con un "
                f"problema nuevo del mismo tipo y variaciones con respuesta. Formato:\n{FORMAT_HINT}")}]
