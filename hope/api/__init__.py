"""Fachada estable de Hope (ARQUITECTURA_HN1 §11.24 T1; `docs/API_HOPE.md`).

Interfaz versionada para programas externos (TISMA Code, la app, scripts) y para **cualquier modelo base**:
herramientas con certificado, esquemas para la API compatible con OpenAI, lectura de llamadas en cualquier
sintaxis, bucle de herramientas y acceso al crítico. Lo que no está acá es interno y puede cambiar.

Versionado semántico: un cambio que rompe la interfaz sube el primer número.
"""

from hope.api.agente import Respuesta, responder
from hope.api.formatos import (
    SINTAXIS,
    Llamada,
    desde_openai,
    especificaciones_openai,
    extraer_llamadas,
    formatear_resultado,
    instrucciones_texto,
)
from hope.api.herramientas import Ficha, Herramientas
from hope.api.resultado import Resultado

VERSION = "0.1.0"


def criticar(problema: str, razonamiento: str, respuesta: str | None = None) -> dict:
    """Crítico CRE/HCRI sobre un razonamiento (pasos aritméticos, unidades, datos faltantes)."""
    from hope.constraints.critic import critique
    c = critique(problema, razonamiento, respuesta)
    return {"veredicto": c.verdict, "problemas": [i.detail for i in c.issues]}


__all__ = [
    "SINTAXIS",
    "VERSION",
    "Ficha",
    "Herramientas",
    "Llamada",
    "Respuesta",
    "Resultado",
    "criticar",
    "desde_openai",
    "especificaciones_openai",
    "extraer_llamadas",
    "formatear_resultado",
    "instrucciones_texto",
    "responder",
]
