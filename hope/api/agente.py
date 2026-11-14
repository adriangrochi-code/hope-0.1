"""Bucle de herramientas para **cualquier** modelo base (§11.24, §11.30).

El modelo es solo una función ``generar(texto) -> texto`` (Hope, un Qwen por llama.cpp o `hope serve`, un
servidor compatible con OpenAI…). El bucle:

1. arma el prompt con las instrucciones de herramientas en la sintaxis del modelo;
2. genera;
3. si la salida trae llamadas (en cualquier sintaxis), las ejecuta y agrega los resultados;
4. repite hasta que no haya llamadas o se llegue al límite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from hope.api.herramientas import Herramientas


@dataclass
class Respuesta:
    texto: str
    transcripcion: str
    llamadas: list[dict] = field(default_factory=list)
    vueltas: int = 0
    todo_verificado: bool = True


def responder(generar: Callable[[str], str], pedido: str, herramientas: Herramientas | None = None, *,
              sintaxis: str = "hermes", max_vueltas: int = 6, herramientas_visibles: int = 10) -> Respuesta:
    h = herramientas or Herramientas()
    visibles = [f.nombre for f in h.buscar(pedido, k=herramientas_visibles)]
    transcripcion = h.instrucciones(sintaxis, visibles) + "\n\nPedido: " + pedido + "\n"
    llamadas: list[dict] = []
    salida = ""
    vueltas = 0
    for vueltas in range(1, max_vueltas + 1):
        salida = generar(transcripcion)
        transcripcion += salida
        ejecutadas = h.ejecutar_texto(salida)
        if not ejecutadas:
            break
        for ll, r, resp in ejecutadas:
            llamadas.append({"herramienta": ll.nombre, "argumentos": ll.argumentos, "sintaxis": ll.sintaxis,
                             "ok": r.ok, "verificado": r.verificado, "salida": r.salida[:300]})
            transcripcion += "\n" + (resp if isinstance(resp, str) else str(resp)) + "\n"
    todo_ok = all(c["verificado"] is not False and c["ok"] for c in llamadas)
    return Respuesta(salida, transcripcion, llamadas, vueltas, todo_ok)
