"""Registro de herramientas de ``hope.api``: fichas, esquemas, búsqueda y ejecución (§11.16, §11.24 T1).

Une dos fuentes en una sola interfaz:

1. **Cálculo exacto certificado** (``hope/api/exactas.py``): devuelve valor + certificado.
2. **El catálogo de Hope** (``hope/runtime/toolcalls.py``: calc, unidades, resolver, python en sandbox,
   pruebas, lint, compilar, circuito, gcode, stl, marlin, web…): se llaman con un argumento ``entrada``
   y pasan por el guardián como siempre.

Cualquier modelo base puede usarlas: ``especificaciones()`` da los esquemas para la API compatible con OpenAI,
``instrucciones()`` el texto para modelos sin soporte nativo, y ``ejecutar_texto()`` encuentra y ejecuta las
llamadas que haya en la salida del modelo, en cualquier sintaxis (``hope.api.formatos``).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hope.api import exactas, formatos
from hope.api.resultado import Resultado


@dataclass
class Ficha:
    nombre: str
    descripcion: str
    parametros: dict                      # JSON Schema de los argumentos
    funcion: Callable[..., Resultado] | None = None
    certificada: bool = False
    area: str = "general"
    ejemplo: str = ""
    palabras: str = ""                    # extra para la búsqueda


def _p(**props) -> dict:
    req = [k for k, v in props.items() if not v.pop("opcional", False)]
    return {"type": "object", "properties": props, "required": req}


def _s(desc: str, opcional: bool = False, tipo: str = "string") -> dict:
    d = {"type": tipo, "description": desc}
    if opcional:
        d["opcional"] = True
    return d


EXACTAS = [
    Ficha("calc_cert", "Calcula una expresión con aritmética de intervalos: el valor real queda garantizado "
          "dentro del intervalo que devuelve.", _p(expr=_s("expresión, p. ej. pi*e o sqrt(2)/3"),
                                                     cifras=_s("cifras significativas (15 por defecto)", True,
                                                               "integer")),
          exactas.calc_cert, True, "matemática", '{"expr": "pi*e", "cifras": 20}', "cuenta aritmética exacta"),
    Ficha("equivalencia", "Dice si dos expresiones algebraicas son iguales: SymPy simplifica la diferencia y "
          "se confirma evaluando en puntos al azar.",
          _p(a=_s("primera expresión"), b=_s("segunda expresión"),
             variables=_s("variables separadas por coma (x por defecto)", True)),
          exactas.equivalencia, True, "matemática", '{"a": "(x+1)^2", "b": "x^2+2*x+1"}',
          "álgebra simplificar identidad"),
    Ficha("raiz", "Encuentra una raíz de f(x)=0 en [a, b] por bisección, con el cambio de signo certificado.",
          _p(f=_s("función de x"), a=_s("extremo izquierdo", tipo="number"), b=_s("extremo derecho", tipo="number"),
             variable=_s("variable (x por defecto)", True)),
          exactas.raiz, True, "matemática", '{"f": "x^2-2", "a": 0, "b": 2}', "ecuación cero solución numérica"),
    Ficha("integral", "Integral definida con dos métodos de cuadratura que tienen que coincidir.",
          _p(f=_s("integrando"), a=_s("límite inferior"), b=_s("límite superior"),
             variable=_s("variable (x por defecto)", True)),
          exactas.integral, True, "matemática", '{"f": "exp(-x^2)", "a": "0", "b": "1"}', "área cálculo"),
    Ficha("lp", "Programación lineal con óptimo global (HiGHS) y certificado dual.",
          _p(problema=_s("JSON con c, A_ub, b_ub, A_eq, b_eq, bounds, maximizar", tipo="object")),
          exactas.lp, True, "optimización",
          '{"problema": {"c": [3, 2], "A_ub": [[1, 1]], "b_ub": [4], "maximizar": true}}',
          "optimizar maximizar minimizar restricciones recursos"),
    Ficha("smt", "Resuelve restricciones lógicas y aritméticas en SMT-LIB2 (z3) y comprueba el modelo.",
          _p(smtlib=_s("texto SMT-LIB2: (declare-const x Int) (assert ...)")),
          exactas.smt, True, "lógica", '{"smtlib": "(declare-const x Int)(assert (> x 2))"}',
          "satisfacible restricciones lógica sudoku planificación"),
    Ficha("grafo", "Camino mínimo (certificado por potenciales) o flujo máximo (certificado por corte mínimo).",
          _p(problema=_s('JSON: {"op": "camino"|"flujo", "aristas": [[u, v, w]], "origen": u, "destino": v}',
                         tipo="object")),
          exactas.grafo, True, "grafos", '{"problema": {"op": "camino", "aristas": [["a", "b", 1]], '
          '"origen": "a", "destino": "b"}}', "ruta red distancia flujo capacidad"),
]


@dataclass
class Herramientas:
    """Herramientas de Hope para cualquier modelo base.

    ``guardian``: el guardián que controla el catálogo de Hope (archivos, red, ejecución). Sin guardián,
    las herramientas del catálogo que lo necesitan se niegan.
    """

    guardian: Any = None
    incluir_catalogo: bool = True
    registro: list[dict] = field(default_factory=list)
    _fichas: dict[str, Ficha] = field(default_factory=dict, init=False)
    _runner: Any = field(default=None, init=False)

    def __post_init__(self):
        for f in EXACTAS:
            self._fichas[f.nombre] = f
        if self.incluir_catalogo:
            from hope.runtime.toolcalls import HELP
            for nombre, ayuda in HELP.items():
                if nombre in self._fichas:
                    continue
                _desc, _, ej = ayuda.partition(":")
                self._fichas[nombre] = Ficha(nombre, ayuda, _p(entrada=_s("entrada en el formato del ejemplo")),
                                             None, False, "catálogo", ej.strip())

    # --- descubrir ------------------------------------------------------------------------------
    def fichas(self, nombres: list[str] | None = None) -> list[Ficha]:
        return [self._fichas[n] for n in (nombres or self._fichas) if n in self._fichas]

    def buscar(self, consulta: str, k: int = 8) -> list[Ficha]:
        """Las ``k`` herramientas más relevantes para una consulta (cuando hay demasiadas para el prompt)."""
        q = set(re.findall(r"\w+", consulta.lower()))

        def puntaje(f: Ficha) -> float:
            t = set(re.findall(r"\w+", f"{f.nombre} {f.descripcion} {f.palabras} {f.area}".lower()))
            return len(q & t) + (0.5 if f.certificada else 0)
        return sorted(self._fichas.values(), key=puntaje, reverse=True)[:k]

    def especificaciones(self, nombres: list[str] | None = None) -> list[dict]:
        return formatos.especificaciones_openai(self.fichas(nombres))

    def instrucciones(self, sintaxis: str = "hermes", nombres: list[str] | None = None) -> str:
        return formatos.instrucciones_texto(self.fichas(nombres), sintaxis)

    # --- ejecutar -------------------------------------------------------------------------------
    def _catalogo(self):
        if self._runner is None:
            from hope.runtime.toolcalls import ToolRunner
            from hope.tools.builtin import default_catalog
            self._runner = ToolRunner(default_catalog(self.guardian))
        return self._runner

    def ejecutar(self, nombre: str, argumentos: dict | str | None = None) -> Resultado:
        t0 = time.perf_counter()
        f = self._fichas.get(nombre)
        if f is None:
            r = Resultado(nombre, False, f"herramienta desconocida «{nombre}»")
        else:
            args = argumentos if isinstance(argumentos, dict) else {}
            if isinstance(argumentos, str):
                primero = next(iter(f.parametros.get("properties", {})), "entrada")
                args = {primero: argumentos}
            try:
                if f.funcion is not None:
                    props = f.parametros.get("properties", {})
                    args = {k: v for k, v in args.items() if k in props}
                    faltan = [k for k in f.parametros.get("required", []) if k not in args]
                    if faltan:
                        raise ValueError("faltan argumentos: " + ", ".join(faltan))
                    r = f.funcion(**args)
                else:
                    entrada = args.get("entrada", next(iter(args.values()), "")) if args else ""
                    out = self._catalogo().run(nombre, str(entrada))
                    ok = not out.startswith(("error:", "denegado:"))
                    r = Resultado(nombre, ok, out)
            except Exception as e:                                # noqa: BLE001 — se informa al modelo
                r = Resultado(nombre, False, f"error: {str(e)[:200]}")
        r.segundos = r.segundos or time.perf_counter() - t0
        self.registro.append({"herramienta": nombre, "argumentos": argumentos, "ok": r.ok,
                              "verificado": r.verificado, "segundos": round(r.segundos, 4)})
        return r

    def ejecutar_texto(self, texto: str) -> list[tuple[formatos.Llamada, Resultado, str | dict]]:
        """Encuentra las llamadas en la salida de un modelo (cualquier sintaxis), las ejecuta y devuelve
        (llamada, resultado, respuesta para el modelo en su misma sintaxis)."""
        out = []
        for ll in formatos.extraer_llamadas(texto):
            r = Resultado(ll.nombre, False, "; ".join(ll.errores)) if ll.errores else \
                self.ejecutar(ll.nombre, ll.argumentos)
            out.append((ll, r, formatos.formatear_resultado(ll, r)))
        return out
