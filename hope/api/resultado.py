"""Resultado común de toda herramienta de ``hope.api``: valor + certificado (§11.16 T3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Resultado:
    """Lo que devuelve cualquier herramienta, sea cual sea el modelo que la llamó.

    * ``salida``: texto corto para devolverle al modelo.
    * ``valor``: dato estructurado (número, lista, dict…) para programas.
    * ``certificado``: prueba o segunda verificación independiente (intervalo, dual factible,
      modelo comprobado, evaluación en puntos al azar…). ``None`` si la herramienta no certifica.
    * ``verificado``: ``True`` si el certificado se comprobó; ``False`` si se intentó y falló;
      ``None`` si la herramienta no certifica. El crítico CRE puede rechazar lo que no venga
      verificado.
    """

    herramienta: str
    ok: bool
    salida: str
    valor: Any = None
    certificado: dict | None = None
    verificado: bool | None = None
    segundos: float = 0.0
    avisos: list[str] = field(default_factory=list)

    def a_dict(self) -> dict:
        d = asdict(self)
        try:
            json.dumps(d["valor"])
        except (TypeError, ValueError):
            d["valor"] = str(d["valor"])
        return d

    def texto(self, max_chars: int = 600) -> str:
        """Salida para el modelo, con la marca de verificación."""
        marca = {True: " [verificado]", False: " [NO verificado]", None: ""}[self.verificado]
        return (self.salida + marca)[:max_chars]
