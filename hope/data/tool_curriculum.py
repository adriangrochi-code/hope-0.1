"""Currículo de uso de herramientas (docs/HERRAMIENTAS_HN1.md §8).

Objetivo del usuario: que Hope use las herramientas **lo mejor posible**: la correcta, en el
momento justo, rápido y verificado. Este módulo escribe textos de entrenamiento donde cada
llamada ``⟦nombre: entrada⟧ → resultado`` **se ejecutó de verdad** con el catálogo y el
guardián: el resultado insertado es el que devolvió la herramienta, no el que dice el maestro
o el generador. Un ejemplo entra solo si la respuesta final coincide con la respuesta calculada
por separado (exacta, con ``fractions``) o, en código, si pasa pruebas ocultas.

Categorías (cada una enseña una decisión distinta):

* ``calculo``, ``fraccion``, ``estadistica``, ``unidades``, ``ecuacion``, ``circuito``:
  **elegir** la herramienta adecuada para el tipo de problema (calc no da fracciones exactas,
  estad resume listas, unidades conserva las unidades, resolver despeja, circuito simula).
* ``cadena``: **encadenar** herramientas, cada una con el resultado de la anterior
  (circuito → calc → unidades).
* ``trivial``: **no usar** herramienta cuando la cuenta es de una cifra o el dato ya está en
  el enunciado: responder directo es más rápido y igual de exacto.
* ``error``: **manejar errores** de la herramienta: reintentar si no respondió, o cambiar de
  herramienta o corregir la entrada si el error dice que la entrada no sirve.
* ``definicion``: usar ``⟦nota⟧`` / ``⟦memoria⟧`` para definiciones y ``⟦anotar⟧`` para guardarlas.
* ``codigo``: escribir código con ``⟦pruebas⟧`` y corregirlo **hasta que pasen**.
* ``generador`` y ``maestro``: episodios completos de ``hope/data/generators.py`` y de la
  cosecha del maestro (``datos_maestro/episodios_verificados.jsonl``) reescritos con las
  llamadas ejecutadas; si una herramienta contradice al episodio, el episodio se descarta.

Una parte de los ejemplos lleva la línea de **herramientas sugeridas** que arma la memoria de
uso (TLM, ``ToolUsageMemory.context_hint``) con lo que de verdad rindió en temas parecidos
durante la construcción del currículo. Así Hope aprende a leerla.

Exportación (fuente ``herramientas`` de los planes de entrenamiento)::

    python -m hope.data.tool_curriculum --out data/herramientas --n 20000

escribe ``train.bin``/``val.bin`` (documentos UTF-8 separados por el byte 0), ``ejemplos.jsonl``
y ``manifest.json``. Excluye los problemas del banco sellado (``hope/eval/tool_bench.py``).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

from hope.runtime.toolcalls import ARROW, ToolCall, ToolRunner

FAULT_MESSAGE = "error: la herramienta no respondió a tiempo; reintentar"
EXECUTED_TOOLS = ("calc", "fraccion", "estad", "unidades", "resolver", "circuito", "python", "pruebas",
                  "nota", "memoria", "anotar")


# ---------------------------------------------------------------------------
# Runner con fallas inyectadas (para enseñar y medir la recuperación)
# ---------------------------------------------------------------------------
@dataclass
class FaultInjectingRunner(ToolRunner):
    """Las primeras ``fail_next`` llamadas devuelven ``FAULT_MESSAGE`` sin ejecutarse
    (simula una herramienta que no respondió). Se registran como fallidas e inyectadas."""
    fail_next: int = 0
    injected: list[int] = field(default_factory=list)          # índices de llamadas con falla inyectada

    def run(self, name: str, arg: str) -> str:
        if self.fail_next > 0:
            self.fail_next -= 1
            self.injected.append(len(self.calls))
            self.calls.append(ToolCall(name, arg, FAULT_MESSAGE, False, 0.0))
            return FAULT_MESSAGE
        return super().run(name, arg)


# ---------------------------------------------------------------------------
# Glosario para ⟦nota⟧ / ⟦memoria⟧
# ---------------------------------------------------------------------------
GLOSSARY = {
    "ley de Ohm": "V = I · R: la tensión es la corriente por la resistencia",
    "potencia eléctrica": "P = V · I: la potencia es la tensión por la corriente",
    "resistencia equivalente en serie": "se suman las resistencias: R = R1 + R2",
    "resistencia equivalente en paralelo": "R = R1 · R2 / (R1 + R2) para dos resistencias",
    "divisor de tensión": "Vout = Vin · R2 / (R1 + R2) sin carga",
    "primera ley de Kirchhoff": "la suma de corrientes que entran a un nodo es igual a la suma de las que salen",
    "segunda ley de Kirchhoff": "la suma de las tensiones en una malla cerrada es cero",
    "capacitor": "componente que guarda energía en un campo eléctrico; Q = C · V",
    "inductor": "componente que guarda energía en un campo magnético; V = L · dI/dt",
    "constante de tiempo RC": "tau = R · C: tiempo en que el capacitor llega al 63 % de la carga",
    "frecuencia de corte RC": "fc = 1 / (2 · pi · R · C)",
    "media aritmética": "la suma de los valores dividida por la cantidad de valores",
    "mediana": "el valor del medio de los datos ordenados",
    "desvío estándar": "la raíz de la varianza: mide cuánto se alejan los datos de la media",
    "porcentaje": "p % de N es p / 100 · N",
    "número primo": "entero mayor que 1 que solo es divisible por 1 y por sí mismo",
    "fracción irreducible": "fracción cuyo numerador y denominador no tienen divisores comunes salvo 1",
    "ecuación lineal": "ecuación de primer grado: a · x + b = c, con solución x = (c - b) / a",
    "pendiente de una recta": "cuánto sube y por cada unidad de x: (y2 - y1) / (x2 - x1)",
    "teorema de Pitágoras": "en un triángulo rectángulo, a² + b² = c²",
    "velocidad media": "distancia recorrida dividida por el tiempo empleado",
    "aceleración": "cambio de velocidad por unidad de tiempo",
    "segunda ley de Newton": "F = m · a: la fuerza es la masa por la aceleración",
    "energía cinética": "E = m · v² / 2",
    "densidad": "masa dividida por volumen",
    "implicación lógica": "p → q es falsa solo cuando p es verdadera y q es falsa",
    "contraejemplo": "un caso que cumple las hipótesis y no cumple la conclusión",
    "variable en programación": "un nombre que guarda un valor que puede cambiar",
    "bucle for": "repite un bloque de código para cada elemento de una secuencia",
    "función en programación": "bloque de código con nombre que recibe entradas y devuelve un resultado",
    "prueba unitaria": "código que comprueba automáticamente que una función da el resultado esperado",
    "complejidad lineal": "el tiempo crece en proporción al tamaño de la entrada: O(n)",
    "byte": "unidad de información de 8 bits",
    "hertz": "unidad de frecuencia: un ciclo por segundo",
    "watt": "unidad de potencia: un joule por segundo",
    "ohm": "unidad de resistencia: un volt por ampere",
}


def concept_split(concept: str) -> str:
    """Partición fija de conceptos: ~20 % solo para el banco (definiciones nunca vistas al entrenar)."""
    return "banco" if int(hashlib.sha256(concept.encode()).hexdigest()[:6], 16) % 5 == 0 else "curriculo"


def default_notebook():
    from hope.memory.knowledge import KnowledgeBase, Notebook
    nb = Notebook(KnowledgeBase())
    for concept, text in GLOSSARY.items():
        nb.write(concept, text, source="glosario", verified=True)
    return nb


# ---------------------------------------------------------------------------
# Ejemplo y escritor (cada llamada se ejecuta al escribirla)
# ---------------------------------------------------------------------------
@dataclass
class ToolExample:
    categoria: str
    tema: str
    problema: str
    respuesta: str
    herramientas_ok: list[str]                 # herramientas correctas para la primera llamada útil
    necesita_herramienta: bool
    segmentos: list[tuple[str, bool]] = field(default_factory=list)   # (texto, generado_por_hope)
    pista: str = ""                            # línea de herramientas sugeridas (TLM)
    falla_inyectada: int = 0                   # primeras llamadas que fallan a propósito
    cadena: list[str] = field(default_factory=list)                    # herramientas que deben usarse
    pruebas_ocultas: str = ""                  # código: pruebas que no ve Hope
    comparar: str = "numero"                   # numero | texto | contiene | codigo
    origen: str = "currículo"

    @property
    def prompt(self) -> str:
        return f"Tema: {self.tema}\nProblema: {self.problema}\n" + (f"{self.pista}\n" if self.pista else "")

    @property
    def completion(self) -> str:
        return "".join(s for s, _ in self.segmentos)

    @property
    def text(self) -> str:
        return self.prompt + self.completion

    @property
    def model_segments(self) -> list[str]:
        """Lo que genera Hope entre resultado y resultado (para reproducir la solución de referencia)."""
        return [s for s, gen in self.segmentos if gen]

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "ToolExample":
        d = dict(d)
        d["segmentos"] = [tuple(s) for s in d.get("segmentos", [])]
        return ToolExample(**d)


class Writer:
    """Arma la solución de referencia ejecutando cada llamada con el runner."""

    def __init__(self, runner: ToolRunner):
        self.runner, self.segs, self.buf = runner, [], ""

    def say(self, text: str) -> None:
        self.buf += text

    def call(self, name: str, arg: str) -> str:
        self.buf += f"⟦{name}: {arg}⟧"
        self.segs.append((self.buf, True))
        self.buf = ""
        out = self.runner.run(name, arg)
        self.segs.append((ARROW + out, False))
        return out

    def finish(self, result: str) -> list[tuple[str, bool]]:
        self.buf += f"\nResultado: {result}\n" if self.segs else f"Resultado: {result}\n"
        self.segs.append((self.buf, True))
        return self.segs


# ---------------------------------------------------------------------------
# Comparación de respuestas
# ---------------------------------------------------------------------------
def _norm_text(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\W+", " ", s).strip()


def _num_unit(s: str) -> tuple[Fraction, str] | None:
    from hope.tools.builtin import normalize_number_text
    m = re.match(r"\s*([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?(?:\s*/\s*\d+)?)\s*(.*)$", s.strip().rstrip("."))
    if not m:
        return None
    try:
        num = Fraction(normalize_number_text(m.group(1)).replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None
    return num, m.group(2).strip()


def answer_ok(given: str | None, expected: str, mode: str = "numero", rel: float = 1e-4) -> bool:
    """Número (con tolerancia relativa) y unidad iguales; o texto normalizado; o «contiene»."""
    if given is None:
        return False
    if mode == "contiene":
        return _norm_text(expected) in _norm_text(given)
    if mode == "numero":
        a, b = _num_unit(given), _num_unit(expected)
        if a and b:
            return a[1].lower() == b[1].lower() and abs(a[0] - b[0]) <= Fraction(rel) * max(abs(a[0]), abs(b[0]), 1)
    return _norm_text(given) == _norm_text(expected)


def extract_answer(text: str) -> str | None:
    m = re.search(r"Resultado:\s*(.+)", text)
    return m.group(1).strip() if m else None


def _fmt(x: Fraction | float, sig: int = 6) -> str:
    v = float(x)
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.{sig}g}"


def _nice(x: Fraction) -> str:
    return _fmt(x, 12)


# ---------------------------------------------------------------------------
# Constructores por categoría: (rng, writer) → ToolExample (sin segmentos; los pone el writer)
# ---------------------------------------------------------------------------
def b_calculo(rng: random.Random, w: Writer) -> ToolExample:
    kind = rng.choice(["mul", "mul", "div", "suma", "porcentaje", "mixta"])
    if kind == "porcentaje":
        p, n = rng.choice([5, 12, 15, 18, 21, 35, 45, 65]), rng.randint(11, 999) * 4
        prob, expr, val = f"¿Cuánto es el {p}% de {n}?", f"{p} / 100 * {n}", Fraction(p * n, 100)
        tema = "porcentajes"
    elif kind == "mixta":
        a, b, c = rng.randint(12, 999), rng.randint(12, 99), rng.randint(2, 99)
        prob, expr, val = f"¿Cuánto es {a} + {b} * {c}?", f"{a} + {b} * {c}", Fraction(a + b * c)
        tema = "aritmética: orden de las operaciones"
    else:
        a, b = rng.randint(12, 9999), rng.randint(12, 999)
        if kind == "div":
            a = a * b
        op = {"mul": "*", "div": "/", "suma": "+"}[kind]
        val = Fraction({"*": a * b, "/": a // b, "+": a + b}[op])
        prob, expr = f"¿Cuánto es {a} {op} {b}?", f"{a} {op} {b}"
        tema = "aritmética con números grandes"
    w.say("Paso 1: ")
    out = w.call("calc", expr)
    return ToolExample("calculo", tema, prob, out, ["calc"], True, w.finish(out), cadena=["calc"],
                       comparar="numero") if answer_ok(out, _nice(val)) else None


def b_fraccion(rng: random.Random, w: Writer) -> ToolExample:
    a, b = rng.randint(1, 9), rng.choice([3, 6, 7, 9, 11, 12])
    c, d = rng.randint(1, 9), rng.choice([4, 5, 6, 8, 10, 15])
    op = rng.choice(["+", "-", "*"])
    val = {"+": Fraction(a, b) + Fraction(c, d), "-": Fraction(a, b) - Fraction(c, d),
           "*": Fraction(a, b) * Fraction(c, d)}[op]
    prob = f"Expresá como fracción exacta: {a}/{b} {op} {c}/{d}"
    w.say("Paso 1 (fracción exacta, no decimal): ")
    out = w.call("fraccion", f"{a}/{b} {op} {c}/{d}")
    if Fraction(out) != val:
        return None
    return ToolExample("fraccion", "fracciones exactas", prob, out, ["fraccion"], True, w.finish(out),
                       cadena=["fraccion"], comparar="texto")


def b_estadistica(rng: random.Random, w: Writer) -> ToolExample:
    xs = [rng.randint(1, 99) for _ in range(rng.randint(4, 9))]
    what = rng.choice(["media", "mediana", "max", "min", "desvio"])
    name = {"media": "la media", "mediana": "la mediana", "max": "el máximo", "min": "el mínimo",
            "desvio": "el desvío estándar"}[what]
    import statistics
    val = {"media": statistics.mean(xs), "mediana": statistics.median(xs), "max": max(xs), "min": min(xs),
           "desvio": statistics.stdev(xs)}[what]
    data = ", ".join(map(str, xs))
    w.say("Paso 1: ")
    out = w.call("estad", " ".join(map(str, xs)))
    m = re.search(rf"\b{what}=([-\d.e+]+)", out)
    if not m or not answer_ok(m.group(1), _fmt(val), rel=1e-4):
        return None
    w.say(f"\nPaso 2: tomo {what}.")
    return ToolExample("estadistica", "estadística descriptiva", f"¿Cuál es {name} de {data}?", m.group(1), ["estad"],
                       True, w.finish(m.group(1)), cadena=["estad"])


def b_unidades(rng: random.Random, w: Writer) -> ToolExample:
    if rng.random() < 0.5:
        cases = [("mA", "A", Fraction(1, 1000)), ("kohm", "ohm", 1000), ("mV", "V", Fraction(1, 1000)),
                 ("km", "m", 1000), ("ms", "s", Fraction(1, 1000)), ("A", "mA", 1000), ("uF", "nF", 1000),
                 ("kW", "W", 1000), ("h", "min", 60), ("min", "s", 60)]
        a, b, f = rng.choice(cases)
        x = rng.randint(2, 9999)
        prob, arg, val, unit = f"¿Cuántos {b} son {x} {a}?", f"{x} {a} en {b}", x * Fraction(f), b
        tema, ok = "unidades del SI y conversiones", ["unidades"]
    else:
        r = rng.choice([47, 100, 150, 220, 330, 470, 680, 1000, 2200])
        i = Fraction(rng.choice([1, 2, 5, 10, 20, 25, 50]), 1000)
        if rng.random() < 0.5:
            v = r * i
            prob, arg, val, unit = (f"¿Qué corriente circula por un resistor de {r} ohm con {_nice(v)} V?",
                                    f"{_nice(v)} V / ({r} ohm) en A", i, "A")
        else:
            prob, arg, val, unit = (f"¿Qué tensión cae en un resistor de {r} ohm con {_nice(i)} A?",
                                    f"{r} ohm * {_nice(i)} A en V", r * i, "V")
        tema, ok = "ley de Ohm y potencia eléctrica", ["unidades", "calc"]
    w.say("Paso 1: ")
    out = w.call("unidades", arg)
    if not answer_ok(out, f"{_nice(val)} {unit}"):
        return None
    return ToolExample("unidades", tema, prob, out, ok, True, w.finish(out), cadena=["unidades"])


def b_ecuacion(rng: random.Random, w: Writer) -> ToolExample:
    if rng.random() < 0.6:
        a, x, b = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 11, 13]), rng.randint(-20, 30), rng.randint(-40, 40)
        c = a * x + b
        eq = f"{a}x + {b} = {c}".replace("+ -", "- ")
        prob, expected, ok = f"Resolver {eq}", f"x = {x}", ["resolver"]
    else:
        x, y = rng.randint(-9, 15), rng.randint(-9, 15)
        eq = f"x + y = {x + y}; x - y = {x - y}"
        prob, expected, ok = f"Resolver el sistema x + y = {x + y}, x - y = {x - y}", f"x = {x}, y = {y}", ["resolver",
                                                                                                          "logica"]
    w.say("Paso 1: ")
    out = w.call("resolver", eq)
    if _norm_text(out) != _norm_text(expected):
        return None
    return ToolExample("ecuacion", "ecuaciones lineales", prob, out, ok, True, w.finish(out), cadena=["resolver"],
                       comparar="texto")


def _divider(rng: random.Random):
    vin = rng.choice([3, 5, 6, 9, 12, 15, 18, 24, 48])
    values = [100, 150, 220, 330, 470, 680, 1000, 1500, 2200, 3300, 4700, 6800, 10000, 22000, 47000]
    r1, r2 = rng.choice(values), rng.choice(values)            # ~2000 circuitos distintos
    return vin, r1, r2, f"V1 in 0 {vin}; R1 in out {r1}; R2 out 0 {r2}"


def _node(out: str, node: str = "out") -> str | None:
    m = re.search(rf'"{node}": ([-\d.eE+]+)', out)
    return m.group(1) if m else None


def b_circuito(rng: random.Random, w: Writer) -> ToolExample:
    vin, r1, r2, net = _divider(rng)
    val = Fraction(vin * r2, r1 + r2)
    prob = (f"Un divisor con R1 = {r1} ohm (arriba) y R2 = {r2} ohm (abajo) se alimenta con {vin} V. "
            f"¿Qué tensión hay en la salida?")
    w.say("Paso 1 (simular el circuito): ")
    out = w.call("circuito", net)
    v = _node(out)
    if v is None:
        return None
    res = f"{_fmt(Fraction(v))} V"
    if not answer_ok(res, f"{_nice(val)} V"):
        return None
    return ToolExample("circuito", "divisores de tensión y corriente", prob, res, ["circuito", "calc"], True,
                       w.finish(res), cadena=["circuito"])


def b_cadena(rng: random.Random, w: Writer) -> ToolExample:
    """circuito → calc → unidades: cada paso usa el resultado del anterior."""
    vin, r1, r2, net = _divider(rng)
    power = rng.random() < 0.5
    if power:
        val, unit = Fraction(vin * r2, r1 + r2) ** 2 / r2 * 1000, "mW"
        prob = (f"Un divisor con R1 = {r1} ohm y R2 = {r2} ohm se alimenta con {vin} V. "
                f"¿Qué potencia disipa R2, en mW?")
    else:
        val, unit = Fraction(vin, r1 + r2) * 1000, "mA"
        prob = (f"Un divisor con R1 = {r1} ohm y R2 = {r2} ohm se alimenta con {vin} V. "
                f"¿Qué corriente circula por R2, en mA?")
    w.say("Paso 1 (simular): ")
    v = _node(w.call("circuito", net))
    if v is None:
        return None
    if power:
        w.say("\nPaso 2 (P = V² / R2): ")
        x = w.call("calc", f"{v} ** 2 / {r2}")
        w.say("\nPaso 3 (pasar a mW): ")
        out = w.call("unidades", f"{x} W en mW")
    else:
        w.say("\nPaso 2 (I = V / R2): ")
        x = w.call("calc", f"{v} / {r2}")
        w.say("\nPaso 3 (pasar a mA): ")
        out = w.call("unidades", f"{x} A en mA")
    if not answer_ok(out, f"{_nice(val)} {unit}"):
        return None
    return ToolExample("cadena", "circuitos: simular, calcular y convertir", prob, out, ["circuito"], True,
                       w.finish(out), cadena=["circuito", "calc", "unidades"])


def b_trivial(rng: random.Random, w: Writer) -> ToolExample:
    kind = rng.choice(["cifra", "cifra", "cifra", "dato", "identidad"])
    if kind == "cifra":
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        op = rng.choice(["+", "*", "-"])
        if op == "-" and b > a:
            a, b = b, a
        val = {"+": a + b, "*": a * b, "-": a - b}[op]
        prob, tema = f"¿Cuánto es {a} {op} {b}?", "aritmética de una cifra"
    elif kind == "identidad":
        a = rng.randint(10, 99999)
        op, b = rng.choice([("*", 1), ("+", 0), ("-", 0), ("/", 1)])
        val, prob, tema = a, f"¿Cuánto es {a} {op} {b}?", "aritmética: elemento neutro"
    else:
        r = rng.choice([47, 100, 220, 330, 470, 1000])
        v = rng.choice([5, 9, 12])
        if rng.random() < 0.5:
            prob, val = f"Un resistor de {r} ohm tiene aplicada una tensión de {v} V. ¿Cuánto vale la resistencia?", f"{r} ohm"
        else:
            prob, val = f"Un resistor de {r} ohm tiene aplicada una tensión de {v} V. ¿Qué tensión tiene aplicada?", f"{v} V"
        tema = "leer los datos del enunciado"
    res = str(val)
    return ToolExample("trivial", tema, prob, res, [], False, w.finish(res))


def b_error(rng: random.Random, w: Writer) -> ToolExample:
    """Errores reales (entrada que la herramienta rechaza) o inyectados (la herramienta no respondió)."""
    kind = rng.choice(["timeout", "timeout", "unidades_en_calc", "python_sintaxis"])
    if kind == "timeout":
        base = rng.choice([b_calculo, b_unidades, b_fraccion, b_ecuacion])
        inner = FaultInjectingRunner(w.runner.catalog, memory=w.runner.memory, fail_next=1)
        w2 = Writer(inner)
        ex = base(rng, _RetryWriter(w2))
        if ex is None:
            return None
        ex.categoria, ex.falla_inyectada, ex.segmentos = "error", 1, w2.segs
        return ex
    if kind == "unidades_en_calc":
        r, i = rng.choice([100, 220, 470, 1000]), Fraction(rng.choice([2, 5, 10, 20]), 1000)
        v = r * i
        prob = f"¿Qué corriente circula por un resistor de {r} ohm con {_nice(v)} V?"
        w.say("Paso 1: ")
        err = w.call("calc", f"{_nice(v)} V / {r} ohm")
        if not err.startswith("error"):
            return None
        w.say("\nLa calculadora no acepta unidades; uso unidades: ")
        out = w.call("unidades", f"{_nice(v)} V / ({r} ohm) en A")
        if not answer_ok(out, f"{_nice(i)} A"):
            return None
        return ToolExample("error", "ley de Ohm y potencia eléctrica", prob, out, ["unidades", "calc"], True,
                           w.finish(out), cadena=["unidades"])
    n = rng.randint(5, 40)
    val = sum(i * i for i in range(1, n + 1))
    prob = f"Usando Python, ¿cuánto suma i² para i de 1 a {n}?"
    w.say("Paso 1: ")
    err = w.call("python", f"print(sum(i * i for i in range(1, {n} + 1))")
    if not err.startswith("error"):
        return None
    w.say("\nFalta cerrar un paréntesis; corrijo: ")
    out = w.call("python", f"print(sum(i * i for i in range(1, {n} + 1)))")
    if out != str(val):
        return None
    return ToolExample("error", "Python: corregir errores", prob, out, ["python", "calc"], True, w.finish(out),
                       cadena=["python"])


class _RetryWriter:
    """Envuelve un Writer: si una llamada devuelve la falla inyectada, la reintenta igual."""

    def __init__(self, w: Writer):
        self.w, self.runner = w, w.runner

    def say(self, text: str) -> None:
        self.w.say(text)

    def call(self, name: str, arg: str) -> str:
        out = self.w.call(name, arg)
        while out == FAULT_MESSAGE:
            self.w.say("\nReintento: ")
            out = self.w.call(name, arg)
        return out

    def finish(self, result: str):
        return self.w.finish(result)


def b_definicion(rng: random.Random, w: Writer, *, split: str = "curriculo") -> ToolExample:
    concepts = [c for c in GLOSSARY if concept_split(c) == split]
    c = rng.choice(concepts)
    tool = rng.choice(["nota", "nota", "nota", "memoria"])
    q = rng.choice(["¿Qué es {c}?", "¿Qué dice {c}?", "Definí: {c}", "¿Qué significa {c}?"]).format(c=c)
    w.say("Paso 1 (consultar el cuaderno): ")
    out = w.call(tool, c)
    if _norm_text(out) != _norm_text(GLOSSARY[c]):
        return None
    return ToolExample("definicion", "definiciones y notas", q, GLOSSARY[c], ["nota", "memoria"], True,
                       w.finish(out), cadena=[tool], comparar="contiene")


def b_anotar(rng: random.Random, w: Writer) -> ToolExample:
    x = rng.randint(2, 999)
    c = rng.choice(["clave del experimento", "valor medido", "resistencia de prueba", "código del lote"]) + f" {x}"
    t = f"{rng.randint(1, 9999)} {rng.choice(['ohm', 'V', 'mA', 'unidades'])}"
    w.say("Paso 1 (guardar en el cuaderno): ")
    out = w.call("anotar", f"{c} | {t}")
    if out != "anotado":
        return None
    w.say("\nPaso 2 (comprobar): ")
    back = w.call("nota", c)
    if back.strip() != t:
        return None
    return ToolExample("definicion", "definiciones y notas", f"Anotá que «{c}» es {t} y comprobalo.", t,
                       ["anotar"], True, w.finish(t), cadena=["anotar", "nota"], comparar="texto")


CODE_FAMILIES = ("lineal", "contar", "multiplos", "repetir")


def _code_task(rng: random.Random) -> dict:
    fam = rng.choice(CODE_FAMILIES)
    if fam == "lineal":
        a, b = rng.randint(2, 9), rng.randint(1, 20)
        f = lambda x: a * x + b  # noqa: E731
        xs, hx = rng.sample(range(-5, 10), 2), rng.sample(range(10, 30), 2)
        spec = f"Escribí una función `f(x)` que devuelva {a}·x + {b}."
        good = f"def f(x):\n    return {a} * x + {b}"
        bad = f"def f(x):\n    return {a} * x - {b}"
        name, call = "f", "f({})"
    elif fam == "contar":
        ch = rng.choice("aeiou")
        f = lambda s: s.lower().count(ch)  # noqa: E731
        words = ["Banana", "elefante", "ARROZ", "murciélago", "oso", "Uruguay", "pipa", "Ecuador", "iglú", "hola"]
        xs, hx = rng.sample(words, 2), rng.sample(words, 2)
        spec = f"Escribí una función `contar(s)` que cuente cuántas «{ch}» tiene el texto (mayúsculas incluidas)."
        good = f"def contar(s):\n    return s.lower().count({ch!r})"
        bad = f"def contar(s):\n    return s.count({ch!r})"
        name, call = "contar", "contar({!r})"
    elif fam == "multiplos":
        k = rng.randint(2, 5)
        f = lambda xs: sum(x for x in xs if x % k == 0)  # noqa: E731
        xs = [[rng.randint(0, 20) for _ in range(5)] for _ in range(2)]
        hx = [[rng.randint(0, 30) for _ in range(6)] for _ in range(2)]
        spec = f"Escribí una función `suma_multiplos(xs)` que sume los elementos de la lista que son múltiplos de {k}."
        good = f"def suma_multiplos(xs):\n    return sum(x for x in xs if x % {k} == 0)"
        bad = f"def suma_multiplos(xs):\n    return sum(x for x in xs if x % {k} == 1)"
        name, call = "suma_multiplos", "suma_multiplos({})"
    else:
        n = rng.randint(2, 4)
        f = lambda s: (s + "-") * (n - 1) + s  # noqa: E731
        words = ["ja", "eco", "si", "no", "tic", "ola"]
        xs, hx = rng.sample(words, 2), rng.sample(words, 2)
        spec = f"Escribí una función `repetir(s)` que repita el texto {n} veces separado por guiones."
        good = f"def repetir(s):\n    return '-'.join([s] * {n})"
        bad = f"def repetir(s):\n    return '-'.join([s] * {n - 1})"
        name, call = "repetir", "repetir({!r})"

    def tests(cases):
        body = "\n".join(f"    assert {call.format(c)} == {f(c)!r}" for c in cases)
        return f"from solucion import {name}\n\ndef test_{name}():\n{body}"
    examples = "; ".join(f"{call.format(c)} = {f(c)!r}" for c in xs)
    return {"spec": f"{spec} Ejemplos: {examples}.", "good": good, "bad": bad, "tests": tests(xs),
            "hidden": tests(hx + xs)}


def b_codigo(rng: random.Random, w: Writer) -> ToolExample:
    t = _code_task(rng)
    first_bad = rng.random() < 0.4
    w.say("Paso 1 (escribir la función y probarla con los ejemplos): ")
    out = w.call("pruebas", f"{t['bad'] if first_bad else t['good']} ||| {t['tests']}")
    if first_bad:
        if not out.startswith("0 pasaron"):
            return None
        w.say("\nFalla; corrijo y vuelvo a probar: ")
        out = w.call("pruebas", f"{t['good']} ||| {t['tests']}")
    if not out.startswith("1 pasaron, 0 fallaron"):
        return None
    return ToolExample("codigo", "programación con pruebas", t["spec"], "pasan las pruebas", ["pruebas"], True,
                       w.finish("pasan las pruebas"), cadena=["pruebas"], pruebas_ocultas=t["hidden"],
                       comparar="codigo")


BUILDERS: dict[str, Callable] = {
    "calculo": b_calculo, "fraccion": b_fraccion, "estadistica": b_estadistica, "unidades": b_unidades,
    "ecuacion": b_ecuacion, "circuito": b_circuito, "cadena": b_cadena, "trivial": b_trivial, "error": b_error,
    "definicion": b_definicion, "anotar": b_anotar, "codigo": b_codigo,
}

# Mezcla del currículo (proporciones). Lo trivial pesa bastante: «no usar» también se aprende.
MIX = {"calculo": 0.13, "fraccion": 0.06, "estadistica": 0.06, "unidades": 0.11, "ecuacion": 0.07,
       "circuito": 0.05, "cadena": 0.09, "trivial": 0.14, "error": 0.09, "definicion": 0.07, "anotar": 0.02,
       "codigo": 0.06, "generador": 0.05}


# ---------------------------------------------------------------------------
# Episodios (generadores y maestro) con las llamadas ejecutadas de verdad
# ---------------------------------------------------------------------------
def episode_example(data: dict, runner: ToolRunner, origin: str) -> ToolExample | None:
    """Reescribe un episodio con cada paso verificable como llamada **ejecutada**.

    Si el resultado real no coincide con el ``esperado`` del episodio, se descarta entero."""
    from hope.teacher.episode import Episode
    w = Writer(runner)
    d = data
    head = []
    if d.get("datos"):
        head.append("Datos: " + "; ".join(f"{x.get('valor', '')} {x.get('unidad', '')}".strip()
                                          for x in d["datos"] if isinstance(x, dict)))
    if d.get("plan"):
        head.append(f"Plan: {d['plan']}")
    w.say("\n".join(head) + ("\n" if head else ""))
    used = []
    for i, p in enumerate(d["pasos"]):
        if not isinstance(p, dict):
            return None
        call = Episode._tool_call(p)
        prefix = f"Paso {p.get('n', i + 1)}: "
        v = p.get("verificacion") or {}
        if call is None and v.get("tipo") == "circuito" and v.get("netlist"):
            w.say(prefix)
            node = _node(w.call("circuito", v["netlist"]), str(v.get("nodo", "out")))
            if node is None or not answer_ok(node, str(v.get("esperado", "")), rel=1e-6):
                return None
            used.append("circuito")
            w.say(f" ({p.get('justificacion', '')})\n")
            continue
        if call is None:
            w.say(f"{prefix}{p.get('operacion', '')} = {p.get('resultado_intermedio', '')} "
                  f"({p.get('justificacion', '')})\n")
            continue
        m = re.match(r"⟦(\w+): (.*)⟧ → (.*)$", call, re.DOTALL)
        name, arg, expected = m.group(1), m.group(2), m.group(3)
        w.say(prefix)
        out = w.call(name, arg)
        good = (out.strip() == expected.strip() if name == "python"
                else answer_ok(out, expected, rel=1e-6) or answer_ok(out, expected + " " + "", rel=1e-6))
        if not good:
            return None
        used.append(name)
        w.say(f" ({p.get('justificacion', '')})\n")
    res = str(d["resultado"])
    w.buf = w.buf.rstrip("\n")
    segs = w.finish(res)
    if not used:
        return None
    return ToolExample(origin, str(d.get("tema", "")), str(d["problema"]), res, [used[0]], True, segs,
                       cadena=used, comparar="numero", origen=origin)


def generator_examples(n: int, seed, runner: ToolRunner) -> list[ToolExample]:
    from hope.data.generators import FAMILIES, generate
    out = []
    fams = [f for f in FAMILIES if f != "datos_faltantes"]
    per = max(1, n // len(fams) + 1)
    for fam in fams:
        for level in (1, 2):
            for ep in generate(fam, per, seed=f"{seed}:{fam}", level=level):
                ex = episode_example(ep.data, runner, "generador")
                if ex is not None:
                    out.append(ex)
    random.Random(f"{seed}:gen").shuffle(out)
    return out[:n]


def teacher_examples(path: str | Path, runner: ToolRunner) -> list[ToolExample]:
    """Episodios RESPALDADO de la cosecha del maestro (``hope.teacher.harvest``)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        try:
            ex = episode_example(r["episodio"], runner, "maestro")
        except Exception:                   # episodio con formato raro: no entra
            ex = None
        if ex is not None:
            out.append(ex)
    return out


# ---------------------------------------------------------------------------
# Construcción del currículo
# ---------------------------------------------------------------------------
def make_runner(catalog=None, notebook=None, tlm=None) -> ToolRunner:
    from hope.tools.builtin import default_catalog
    return ToolRunner(catalog or default_catalog(), memory=notebook if notebook is not None else default_notebook(),
                      tlm=tlm)


def build_examples(n: int, *, seed="curriculo", mix: dict | None = None, runner: ToolRunner | None = None,
                   exclude: set[str] | None = None, tlm=None, hint_fraction: float = 0.4,
                   teacher_path: str | Path | None = None, definition_split: str = "curriculo",
                   log=None) -> list[ToolExample]:
    """``n`` ejemplos verificados según ``mix``. Las llamadas se ejecutan y alimentan la TLM;
    ``hint_fraction`` de los ejemplos lleva la línea de herramientas sugeridas por la TLM."""
    from hope.tools.tlm import ToolUsageMemory
    mix = dict(mix or MIX)
    runner = runner or make_runner()
    tlm = tlm if tlm is not None else ToolUsageMemory()
    exclude = exclude or set()
    rng = random.Random(f"tool_curriculum:{seed}")
    total = sum(mix.values())
    quotas = {k: int(round(n * v / total)) for k, v in mix.items()}
    out: list[ToolExample] = []
    if quotas.get("generador"):
        gens = generator_examples(quotas.pop("generador"), seed, runner)
        out += [g for g in gens if g.problema not in exclude]
    else:
        quotas.pop("generador", None)
    if teacher_path:
        out += [t for t in teacher_examples(teacher_path, runner) if t.problema not in exclude]
    seen = set(exclude)
    for cat, q in quotas.items():
        made, tries = 0, 0
        while made < q and tries < q * 6 + 20:
            tries += 1
            k0 = len(runner.calls)
            w = Writer(runner)
            ex = (BUILDERS[cat](rng, w, split=definition_split) if cat == "definicion" else BUILDERS[cat](rng, w))
            if ex is None or ex.problema in seen:
                continue
            seen.add(ex.problema)
            new_calls = runner.calls[k0:]
            if new_calls:
                tlm.record(new_calls, ex.tema, True)
            elif not ex.necesita_herramienta:
                tlm.record_direct(ex.tema, True)
            if rng.random() < hint_fraction:
                ex.pista = tlm.context_hint(ex.tema, ex.problema)
            out.append(ex)
            made += 1
        if log:
            log(f"{cat}: {made}/{q}")
    rng.shuffle(out)
    return out


def bench_problems() -> set[str]:
    """Problemas del banco sellado: el currículo nunca los incluye."""
    from hope.eval.tool_bench import DEFAULT_BENCH, build_bench, load_bench
    if Path(DEFAULT_BENCH).exists():
        return {it.problema for it in load_bench(DEFAULT_BENCH, check=False)}
    return {it.problema for it in build_bench()}


def export(out_dir: str | Path, n: int, *, seed="curriculo", val_percent: float = 3.0,
           teacher_path: str | Path | None = None, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exclude = bench_problems()
    from hope.tools.tlm import ToolUsageMemory
    tlm = ToolUsageMemory(out / "tlm_curriculo.json")
    exs = build_examples(n, seed=seed, exclude=exclude, teacher_path=teacher_path, tlm=tlm, log=log)
    tlm.save()
    files = {"train": open(out / "train.bin", "wb"), "val": open(out / "val.bin", "wb")}
    stats = {"train": 0, "val": 0, "bytes": 0, "por_categoria": {}, "excluidos_banco": len(exclude)}
    with open(out / "ejemplos.jsonl", "w", encoding="utf-8") as js:
        for ex in exs:
            text = ex.text
            h = hashlib.sha256(text.encode()).hexdigest()
            split = "val" if int(h[:8], 16) % 10_000 < val_percent * 100 else "train"
            files[split].write(text.encode("utf-8") + b"\x00")
            stats[split] += 1
            stats["bytes"] += len(text.encode())
            stats["por_categoria"][ex.categoria] = stats["por_categoria"].get(ex.categoria, 0) + 1
            js.write(json.dumps({"split": split, **ex.to_json()}, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    assert not ({e.problema for e in exs} & exclude), "un problema del banco se coló en el currículo"
    (out / "manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> dict:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", default="curriculo")
    ap.add_argument("--val-percent", type=float, default=3.0)
    ap.add_argument("--maestro", default="datos_maestro/episodios_verificados.jsonl",
                    help="episodios verificados de la cosecha del maestro (si existe)")
    args = ap.parse_args(argv)
    stats = export(args.out, args.n, seed=args.seed, val_percent=args.val_percent, teacher_path=args.maestro)
    print(json.dumps(stats, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
