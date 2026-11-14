"""Llamadas a herramientas en cualquier formato de modelo (§11.24 T4, `docs/API_HOPE.md`).

Cada familia de modelos pide herramientas de otra forma. ``hope.api`` entiende todas y responde en la
misma, así las herramientas sirven **sea cual sea el modelo base** (Hope, Qwen3-Coder, Qwen3.5, Qwen 4…):

* ``hope``: ``⟦nombre: entrada⟧`` → se responde `` → resultado``.
* ``hermes`` (Qwen2.5/Qwen3 y muchos otros): ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>`` →
  ``<tool_response>...</tool_response>``.
* ``qwen_xml`` (Qwen3-Coder): ``<tool_call><function=nombre><parameter=p>valor</parameter></function></tool_call>``.
* ``json``: un bloque ```json {"name": ..., "arguments": {...}}``` en modelos sin formato propio.
* ``openai``: mensajes de la API compatible con OpenAI con ``tool_calls`` (``desde_openai``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from hope.api.resultado import Resultado

SINTAXIS = ("hope", "hermes", "qwen_xml", "json", "openai")

_HOPE = re.compile(r"⟦\s*([\wáéíóúñ]+)\s*:\s*(.*?)\s*⟧", re.DOTALL)
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_XML = re.compile(r"<function=([\w.\-]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAM_XML = re.compile(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class Llamada:
    nombre: str
    argumentos: dict[str, Any]
    sintaxis: str
    inicio: int = 0
    fin: int = 0
    id: str | None = None
    errores: list[str] = field(default_factory=list)


def _args_json(texto: str) -> dict:
    try:
        v = json.loads(texto) if texto.strip() else {}
    except json.JSONDecodeError:
        return {"entrada": texto}
    return v if isinstance(v, dict) else {"entrada": v}


def _valor_xml(texto: str) -> Any:
    t = texto.strip()
    if t[:1] in "[{" or t in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", t):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
    return t


def extraer_llamadas(texto: str) -> list[Llamada]:
    """Todas las llamadas a herramientas en ``texto``, en el orden en que aparecen, sin importar la sintaxis."""
    out: list[Llamada] = []
    for m in _TOOL_CALL.finditer(texto):
        cuerpo = m.group(1)
        f = _FUNC_XML.search(cuerpo)
        if f:
            args = {p.group(1): _valor_xml(p.group(2)) for p in _PARAM_XML.finditer(f.group(2))}
            out.append(Llamada(f.group(1), args, "qwen_xml", m.start(), m.end()))
            continue
        try:
            d = json.loads(cuerpo)
        except json.JSONDecodeError:
            out.append(Llamada("", {}, "hermes", m.start(), m.end(), errores=["JSON inválido en <tool_call>"]))
            continue
        args = d.get("arguments", d.get("parameters", {}))
        if isinstance(args, str):
            args = _args_json(args)
        out.append(Llamada(str(d.get("name", "")), args or {}, "hermes", m.start(), m.end()))
    ocupados = [(c.inicio, c.fin) for c in out]
    for m in _HOPE.finditer(texto):
        if not any(a <= m.start() < b for a, b in ocupados):
            out.append(Llamada(m.group(1), {"entrada": m.group(2)}, "hope", m.start(), m.end()))
    if not out:
        for m in _JSON_BLOCK.finditer(texto):
            try:
                d = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and "name" in d:
                args = d.get("arguments", {})
                out.append(Llamada(str(d["name"]), _args_json(args) if isinstance(args, str) else args, "json",
                                   m.start(), m.end()))
    out.sort(key=lambda c: c.inicio)
    return out


def desde_openai(mensaje: dict) -> list[Llamada]:
    """Llamadas de un mensaje de la API compatible con OpenAI (``{"role": "assistant", "tool_calls": [...]}``)."""
    out = []
    for tc in mensaje.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        out.append(Llamada(fn.get("name", ""), _args_json(args) if isinstance(args, str) else dict(args),
                           "openai", id=tc.get("id")))
    return out


def formatear_resultado(llamada: Llamada, resultado: Resultado, max_chars: int = 1200) -> str | dict:
    """Respuesta en la misma sintaxis en la que llamó el modelo."""
    texto = resultado.texto(max_chars)
    if llamada.sintaxis == "hope":
        return " → " + texto
    if llamada.sintaxis == "openai":
        return {"role": "tool", "tool_call_id": llamada.id, "name": llamada.nombre, "content": json.dumps(
            {"ok": resultado.ok, "salida": texto, "verificado": resultado.verificado}, ensure_ascii=False)}
    cuerpo = json.dumps({"name": llamada.nombre, "ok": resultado.ok, "salida": texto,
                         "verificado": resultado.verificado, "certificado": resultado.certificado},
                        ensure_ascii=False, default=str)[:max_chars + 400]
    return f"<tool_response>\n{cuerpo}\n</tool_response>"


def especificaciones_openai(fichas) -> list[dict]:
    """Lista ``tools`` para la API compatible con OpenAI (y para plantillas de chat de Qwen)."""
    return [{"type": "function", "function": {"name": f.nombre, "description": f.descripcion,
                                              "parameters": f.parametros}} for f in fichas]


def instrucciones_texto(fichas, sintaxis: str = "hermes") -> str:
    """Bloque de sistema que explica las herramientas a un modelo sin soporte nativo de herramientas."""
    if sintaxis not in SINTAXIS:
        raise ValueError(f"sintaxis desconocida: {sintaxis}")
    lineas = ["Tenés estas herramientas. Usalas para todo cálculo, dato exacto o verificación en vez de adivinar."]
    for f in fichas:
        params = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in f.parametros.get("properties", {}).items())
        cert = " [certifica el resultado]" if f.certificada else ""
        lineas.append(f"- {f.nombre}({params}): {f.descripcion}{cert}")
    ejemplo = {
        "hope": "Para llamar: ⟦nombre: entrada⟧ y esperá el resultado.",
        "hermes": 'Para llamar: <tool_call>{"name": "nombre", "arguments": {...}}</tool_call> y esperá <tool_response>.',
        "qwen_xml": "Para llamar: <tool_call><function=nombre><parameter=p>valor</parameter></function></tool_call>.",
        "json": 'Para llamar: un bloque ```json {"name": "nombre", "arguments": {...}}``` y esperá el resultado.',
        "openai": "Usá el mecanismo de tool_calls de la API.",
    }[sintaxis]
    lineas.append(ejemplo)
    lineas.append("Si una herramienta devuelve [NO verificado], decilo y no lo des por cierto.")
    return "\n".join(lineas)
