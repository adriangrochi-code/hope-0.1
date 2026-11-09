"""HCRI — representación de magnitudes, restricciones e invariantes (HCRI §2.1–2.2, §4, §9).

* ``extract_quantities`` — reconoce magnitudes con unidad en un texto
  (CRE de HCRI: «Constraint Recognition», §2.1). Usa un mapa explícito de
  unidades en español e inglés para no confundir, por ejemplo, el artículo «a»
  con amperios.
* ``Invariant`` — propiedad que debe cumplirse dentro de un dominio (IRM, §2.2):
  ecuaciones, dominio de validez, excepciones, procedencia y estado epistémico.
  Una ley no se aplica fuera de su dominio.
* ``LIBRARY`` — invariantes iniciales: ley de Ohm, potencia eléctrica, rangos
  físicos (resistencias, tiempos y masas no negativos, probabilidades en [0, 1]).
  Se amplía con lo que Hope aprende (``dynamic.py``), empezando como hipótesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hope.epistemic import EpistemicStatus

# Unidad escrita → (unidad pint, dimensión que representa)
UNITS: dict[str, tuple[str, str]] = {
    "v": ("volt", "tension"), "volt": ("volt", "tension"), "volts": ("volt", "tension"),
    "voltio": ("volt", "tension"), "voltios": ("volt", "tension"), "mv": ("millivolt", "tension"),
    "kv": ("kilovolt", "tension"),
    "a": ("ampere", "corriente"), "amp": ("ampere", "corriente"), "amperio": ("ampere", "corriente"),
    "amperios": ("ampere", "corriente"), "ampere": ("ampere", "corriente"), "amperes": ("ampere", "corriente"),
    "ma": ("milliampere", "corriente"),
    "ohm": ("ohm", "resistencia"), "ohms": ("ohm", "resistencia"), "ohmio": ("ohm", "resistencia"),
    "ohmios": ("ohm", "resistencia"), "ω": ("ohm", "resistencia"), "kohm": ("kiloohm", "resistencia"),
    "kω": ("kiloohm", "resistencia"),
    "w": ("watt", "potencia"), "watt": ("watt", "potencia"), "watts": ("watt", "potencia"),
    "vatio": ("watt", "potencia"), "vatios": ("watt", "potencia"), "mw": ("milliwatt", "potencia"),
    "kw": ("kilowatt", "potencia"),
    "j": ("joule", "energia"), "joule": ("joule", "energia"), "joules": ("joule", "energia"),
    "julio": ("joule", "energia"), "julios": ("joule", "energia"),
    "s": ("second", "tiempo"), "seg": ("second", "tiempo"), "segundo": ("second", "tiempo"),
    "segundos": ("second", "tiempo"), "ms": ("millisecond", "tiempo"),
    "kg": ("kilogram", "masa"), "g": ("gram", "masa"),
    "m": ("meter", "longitud"), "metro": ("meter", "longitud"), "metros": ("meter", "longitud"),
    "cm": ("centimeter", "longitud"), "km": ("kilometer", "longitud"),
    "hz": ("hertz", "frecuencia"),
}
# Mayúsculas que importan: «A» es amperio, «a» suele ser un artículo.
CASE_SENSITIVE = {"a": "A", "m": None, "s": None, "g": None, "v": None, "w": None, "j": None}

NUM = r"(-?\d+(?:[.,]\d+)?)"
UNIT_TOKEN = r"([A-Za-zΩω]+)"
QTY_RE = re.compile(NUM + r"\s*" + UNIT_TOKEN + r"(?![A-Za-zÁÉÍÓÚáéíóúñ])")

# Palabras de la pregunta → dimensión de la incógnita
UNKNOWN_WORDS = {
    "tension": ["tensión", "tension", "voltaje", "voltage", "caída", "caida"],
    "corriente": ["corriente", "current", "intensidad"],
    "resistencia": ["resistencia", "resistance"],
    "potencia": ["potencia", "power"],
    "energia": ["energía", "energia", "energy"],
}


@dataclass
class Quantity:
    value: float
    unit: str            # unidad pint
    dimension: str       # tension, corriente, ...
    text: str
    start: int


def extract_quantities(text: str) -> list[Quantity]:
    out = []
    for m in QTY_RE.finditer(text):
        raw = m.group(2)
        key = raw.lower()
        if key not in UNITS:
            continue
        need = CASE_SENSITIVE.get(key, "")
        if need and raw != need:
            continue
        if key in ("m", "s", "g") and raw != key:
            continue
        unit, dim = UNITS[key]
        out.append(Quantity(float(m.group(1).replace(",", ".")), unit, dim, m.group(0), m.start()))
    return out


def unknown_dimension(question: str) -> str | None:
    """Dimensión de lo que se pregunta (la primera palabra clave después de «qué/cuál/cuánta»)."""
    q = question.lower()
    m = re.search(r"(qué|que|cuál|cual|cuánta|cuanta|cuánto|cuanto|what|which|how much)\s+(.{0,40})", q)
    window = m.group(2) if m else q
    best, pos = None, len(window) + 1
    for dim, words in UNKNOWN_WORDS.items():
        for w in words:
            i = window.find(w)
            if 0 <= i < pos:
                best, pos = dim, i
    return best


@dataclass
class Invariant:
    """Una propiedad que debe mantenerse dentro de un dominio (HCRI §2.2)."""

    id: str
    name: str
    equations: list[str]                      # en símbolos SymPy: "V - I*R"
    symbols: dict[str, tuple[str, str]]       # símbolo → (dimensión, unidad SI)
    domain: str
    conditions: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    provenance: str = "biblioteca inicial"
    status: EpistemicStatus = EpistemicStatus.VALIDADO_CONDICIONAL
    verification: str = "comparación numérica con herramienta"

    def dimensions(self) -> set[str]:
        return {d for d, _ in self.symbols.values()}


@dataclass
class RangeConstraint:
    dimension: str
    low: float | None
    high: float | None
    reason: str


LIBRARY: list[Invariant] = [
    Invariant("inv_ohm", "Ley de Ohm", ["V - I*R"],
              {"V": ("tension", "volt"), "I": ("corriente", "ampere"), "R": ("resistencia", "ohm")},
              domain="conductores óhmicos en corriente continua o valores instantáneos",
              conditions=["resistor óhmico (lineal)", "temperatura aproximadamente constante"],
              exceptions=["diodos, transistores y otros componentes no lineales"]),
    Invariant("inv_potencia_vi", "Potencia eléctrica", ["P - V*I"],
              {"P": ("potencia", "watt"), "V": ("tension", "volt"), "I": ("corriente", "ampere")},
              domain="corriente continua; en alterna, potencia instantánea o con factor de potencia",
              exceptions=["alterna con desfase: usar P = V·I·cos(φ)"]),
    Invariant("inv_potencia_ir", "Potencia disipada en una resistencia", ["P - I**2*R"],
              {"P": ("potencia", "watt"), "I": ("corriente", "ampere"), "R": ("resistencia", "ohm")},
              domain="resistor óhmico"),
    Invariant("inv_potencia_vr", "Potencia disipada en una resistencia (por tensión)", ["P - V**2/R"],
              {"P": ("potencia", "watt"), "V": ("tension", "volt"), "R": ("resistencia", "ohm")},
              domain="resistor óhmico"),
    Invariant("inv_energia", "Energía y potencia", ["E - P*t"],
              {"E": ("energia", "joule"), "P": ("potencia", "watt"), "t": ("tiempo", "second")},
              domain="potencia constante durante el intervalo"),
]

RANGES = [
    RangeConstraint("resistencia", 0, None, "una resistencia pasiva no es negativa"),
    RangeConstraint("tiempo", 0, None, "un intervalo de tiempo no es negativo"),
    RangeConstraint("masa", 0, None, "la masa no es negativa"),
    RangeConstraint("longitud", 0, None, "una longitud no es negativa"),
    RangeConstraint("frecuencia", 0, None, "una frecuencia no es negativa"),
]
