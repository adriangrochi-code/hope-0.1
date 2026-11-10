"""Ejercicios de lengua con respuesta exacta (español e inglés) para entrenar y para el refuerzo.

Pedido del usuario: generadores de ejercicios verificables —conjugación con voseo, concordancia,
tildes, puntuación, preposiciones, orden de palabras, plurales, sinónimos y corregir una oración—
con el mismo formato de episodio que ``hope/data/generators.py`` (se registran en ``FAMILIES``, así
``generate("conjugacion", …)`` y ``problems_from_generators`` los usan sin cambios).

Cada ejercicio tiene **una** respuesta correcta (las listas de palabras están elegidas para eso) y
``respuesta_ok`` la compara normalizando mayúsculas y espacios, pero **no** tildes ni signos cuando
eso es lo que se evalúa.
"""

from __future__ import annotations

import random
import re
import unicodedata


def _ep(tema: str, lang: str, problema: str, resultado: str, pasos: list[str], *, plan: str = "",
        errores: list[str] | None = None, tipo: str = "") -> dict:
    return {"tema": tema, "idioma": lang, "problema": problema, "datos": [], "supuestos": [], "plan": plan,
            "pasos": [{"n": i + 1, "operacion": p, "resultado_intermedio": "", "justificacion": p}
                      for i, p in enumerate(pasos)],
            "resultado": resultado, "verificacion_inversa": "", "errores_frecuentes": errores or [],
            "variaciones": [], "origen": "generador", "tipo_lengua": tipo}


# --- conjugación (con voseo) --------------------------------------------------------------------
_REG = ["cantar", "hablar", "caminar", "trabajar", "estudiar", "comer", "beber", "aprender", "vivir", "escribir",
        "abrir", "subir", "mirar", "comprar", "vender", "correr", "partir", "llegar", "buscar", "tomar"]
_IRR = ["ser", "estar", "ir", "tener", "hacer", "poder", "decir", "venir", "querer", "saber", "poner", "salir",
        "pensar", "dormir", "pedir", "jugar", "volver"]
_TIEMPOS_N = {1: ["presente"], 2: ["presente", "preterito", "imperfecto"],
              3: ["presente", "preterito", "imperfecto", "futuro", "condicional", "subjuntivo", "imperativo"]}
_NOMBRE_T = {"presente": "presente", "preterito": "pretérito perfecto simple", "imperfecto": "pretérito imperfecto",
             "futuro": "futuro", "condicional": "condicional", "subjuntivo": "presente de subjuntivo",
             "imperativo": "imperativo"}


def conjugacion(rng: random.Random, level: int, lang: str) -> dict:
    from hope.tools.lenguaje import PERSONAS, _EN_IRR, conjugar_en, conjugar_es
    if lang == "en":
        v = rng.choice(list(_EN_IRR) if level > 1 else ["walk", "play", "watch", "study", "stop", "carry", "jump"])
        forma = rng.choice(["past", "participle", "he/she/it", "gerund"])
        r = conjugar_en(v)[forma]
        nombre = {"past": "simple past", "participle": "past participle", "he/she/it": "third person singular",
                  "gerund": "-ing form"}[forma]
        return _ep("lengua: conjugación (en)", "en", f"Give the {nombre} of «{v}».", r, [f"{v} → {r}"],
                   tipo="conjugacion")
    v = rng.choice(_REG if level == 1 else _REG + _IRR)
    t = rng.choice(_TIEMPOS_N[min(level, 3)])
    formas = conjugar_es(v, t)
    persona = rng.choice(["vos", "vos", "tú"] if t == "imperativo" else [p for p in PERSONAS if p != "vosotros"] + ["vos"])
    r = formas[persona]
    q = f"Conjugá «{v}» en {_NOMBRE_T[t]} para «{persona}»."
    err = ["usar la forma de tú en lugar de la de vos (p. ej. «tienes» por «tenés»)"] if persona == "vos" else []
    return _ep("lengua: conjugación", "es", q, r, [f"{v}, {_NOMBRE_T[t]}, {persona} → {r}"], errores=err,
               tipo="conjugacion")


# --- concordancia ---------------------------------------------------------------------------------
_SUST = [("casa", "f"), ("libro", "m"), ("ciudad", "f"), ("río", "m"), ("mesa", "f"), ("árbol", "m"),
         ("canción", "f"), ("tren", "m"), ("montaña", "f"), ("puerto", "m"), ("calle", "f"), ("problema", "m"),
         ("mano", "f"), ("día", "m"), ("flor", "f"), ("mapa", "m")]
_ADJ = ["blanco", "nuevo", "antiguo", "pequeño", "largo", "rojo", "bonito", "oscuro", "alto", "viejo"]
_ADJ_INV = ["grande", "verde", "azul", "útil", "fácil"]


def _plural_es(w: str) -> str:
    irregulares = {"lápiz": "lápices", "luz": "luces", "voz": "voces", "pez": "peces", "camión": "camiones",
                   "canción": "canciones", "examen": "exámenes", "joven": "jóvenes", "tren": "trenes",
                   "árbol": "árboles", "crisis": "crisis", "lunes": "lunes", "virus": "virus", "régimen": "regímenes",
                   "carácter": "caracteres", "club": "clubes", "sofá": "sofás", "jabalí": "jabalíes",
                   "mamá": "mamás", "café": "cafés", "flor": "flores", "ciudad": "ciudades", "país": "países",
                   "inglés": "ingleses", "autobús": "autobuses", "reloj": "relojes", "rey": "reyes", "ley": "leyes"}
    if w in irregulares:
        return irregulares[w]
    if w[-1] in "aeiouáéó":
        return w + "s"
    if w.endswith("z"):
        return w[:-1] + "ces"
    if w.endswith(("s", "x")) and len(w) > 3:
        return w
    return w + "es"


def _adj(a: str, g: str, plural: bool) -> str:
    if a in _ADJ_INV:
        return (a + "s" if a[-1] == "e" else (a + "es" if not a.endswith("l") else a[:-0] + "es")) if plural else a
    base = a[:-1] + ("a" if g == "f" else "o")
    return base + "s" if plural else base


def concordancia(rng: random.Random, level: int, lang: str) -> dict:
    if lang == "en":
        n = rng.choice(["house", "car", "book", "city", "river", "child", "mouse", "box"])
        plur = {"child": "children", "mouse": "mice", "city": "cities", "box": "boxes"}.get(n, n + "s")
        det = rng.choice(["this", "these"])
        verb = "is" if det == "this" else "are"
        r = f"{det} {n if det == 'this' else plur} {verb}"
        return _ep("lengua: concordancia (en)", "en",
                   f"Complete with the right forms: «{det.capitalize()} ___ (={n}) ___ (=be) old.»",
                   f"{n if det == 'this' else plur} {verb}", [r], tipo="concordancia")
    s, g = rng.choice(_SUST)
    a = rng.choice(_ADJ + (_ADJ_INV if level > 1 else []))
    plural = rng.random() < 0.5
    art = {("m", False): "el", ("f", False): "la", ("m", True): "los", ("f", True): "las"}[(g, plural)]
    sus = _plural_es(s) if plural else s
    verbo = "son" if plural else "es"
    r = _adj(a, g, plural)
    if level >= 3:
        q = f"Completá con el verbo «ser» y el adjetivo «{a}» concordados: «{art.capitalize()} {sus} ___ ___.»"
        return _ep("lengua: concordancia", "es", q, f"{verbo} {r}", [f"{sus}: {g}, {'pl' if plural else 'sg'}"],
                   tipo="concordancia")
    q = f"Completá con «{a}» concordado: «{art.capitalize()} {sus} {verbo} ___.»"
    return _ep("lengua: concordancia", "es", q, r, [f"«{sus}» es {'femenino' if g == 'f' else 'masculino'} "
                                                     f"{'plural' if plural else 'singular'} → {r}"],
               errores=["concordar con el adjetivo en masculino por defecto"], tipo="concordancia")


# --- tildes y puntuación -----------------------------------------------------------------------------
_TILDES = ["árbol", "canción", "música", "también", "rápido", "fácil", "camión", "lápiz", "teléfono", "información",
           "número", "página", "último", "jamás", "después", "corazón", "pájaro", "máquina", "médico", "difícil",
           "sábado", "miércoles", "matemáticas", "lección", "ratón", "azúcar", "fútbol", "cámara", "público", "océano"]
_TILDES_EN = [("cafe", "café")]


def _sin(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tildes(rng: random.Random, level: int, lang: str) -> dict:
    w = rng.choice(_TILDES)
    sil = None
    try:
        from hope.tools.lenguaje import silabas_es
        sil = "-".join(silabas_es(w))
    except Exception:                                          # noqa: BLE001
        pass
    q = f"Escribí con la tilde correcta: «{_sin(w)}»."
    return _ep("lengua: tildes", "es", q, w, [f"sílabas: {sil}" if sil else w, f"→ {w}"], tipo="tildes")


_PREGUNTAS = [("como estas", "¿Cómo estás?"), ("que hora es", "¿Qué hora es?"), ("donde vivis", "¿Dónde vivís?"),
              ("cuando llega el tren", "¿Cuándo llega el tren?"), ("que lindo dia", "¡Qué lindo día!"),
              ("quien vino", "¿Quién vino?"), ("cuanto cuesta", "¿Cuánto cuesta?"), ("que suerte", "¡Qué suerte!")]
_COMAS = [("Juan mi vecino trabaja en el puerto", "Juan, mi vecino, trabaja en el puerto."),
          ("Compré pan leche y queso", "Compré pan, leche y queso."),
          ("Sin embargo no llegó", "Sin embargo, no llegó."),
          ("Por ejemplo el río Paraná es muy largo", "Por ejemplo, el río Paraná es muy largo.")]


def puntuacion(rng: random.Random, level: int, lang: str) -> dict:
    if lang == "en":
        src, r = rng.choice([("however it rained", "However, it rained."), ("i bought milk eggs and bread",
                                                                          "I bought milk, eggs and bread.")])
        return _ep("lengua: puntuación (en)", "en", f"Add capitals and punctuation: «{src}»", r, [r], tipo="puntuacion")
    src, r = rng.choice(_PREGUNTAS if level == 1 else _PREGUNTAS + _COMAS)
    return _ep("lengua: puntuación", "es", f"Escribí con mayúscula, tildes y signos de puntuación: «{src}»", r, [r],
               errores=["olvidar el signo de apertura («¿», «¡»)"], tipo="puntuacion")


# --- preposiciones, orden, plurales, sinónimos, corregir --------------------------------------------
_PREP = [("Pienso ___ vos todos los días.", "en"), ("Depende ___ ti.", "de"), ("Soñé ___ el mar.", "con"),
         ("Llegamos ___ Rosario de noche.", "a"), ("Se enamoró ___ la ciudad.", "de"), ("Confío ___ vos.", "en"),
         ("Voy ___ pie.", "a"), ("Trabaja ___ la mañana.", "por"), ("Es fácil ___ entender.", "de"),
         ("Me acuerdo ___ aquel verano.", "de"), ("Insistió ___ venir.", "en"), ("Se casó ___ su novia.", "con")]
_PREP_EN = [("She is interested ___ music.", "in"), ("It depends ___ the weather.", "on"),
            ("I'm afraid ___ dogs.", "of"), ("We arrived ___ the airport.", "at"), ("He is good ___ math.", "at")]


def preposiciones(rng: random.Random, level: int, lang: str) -> dict:
    frase, r = rng.choice(_PREP_EN if lang == "en" else _PREP)
    q = (f"Fill in the preposition: «{frase}»" if lang == "en" else f"Completá con la preposición: «{frase}»")
    return _ep("lengua: preposiciones", lang, q, r, [frase.replace("___", r)], tipo="preposiciones")


_ORACIONES = ["El tren llega a Rosario", "Mi hermana estudia música", "Los chicos juegan en la plaza",
              "La biblioteca cierra a las ocho", "Mañana vamos al río", "El puerto exporta soja"]
_ORACIONES_EN = ["The train arrives at noon", "My sister studies music", "We go to the river tomorrow"]


def orden(rng: random.Random, level: int, lang: str) -> dict:
    s = rng.choice(_ORACIONES_EN if lang == "en" else _ORACIONES)
    ws = s.split()
    mez = ws[:]
    while mez == ws:
        rng.shuffle(mez)
    q = (f"Put the words in order: {' / '.join(w.lower() if i else w for i, w in enumerate(mez))}" if lang == "en" else
         f"Ordená las palabras para formar una oración: {' / '.join(mez)}")
    return _ep("lengua: orden", lang, q, s + ".", [s], tipo="orden")


_PLUR = ["casa", "ciudad", "lápiz", "camión", "examen", "crisis", "luz", "flor", "café", "rey", "reloj", "país",
         "joven", "lunes", "pez", "árbol", "canción", "autobús"]
_PLUR_EN = [("mouse", "mice"), ("child", "children"), ("city", "cities"), ("box", "boxes"), ("knife", "knives"),
            ("man", "men"), ("tooth", "teeth"), ("bus", "buses"), ("story", "stories"), ("sheep", "sheep")]


def plurales(rng: random.Random, level: int, lang: str) -> dict:
    if lang == "en":
        w, p = rng.choice(_PLUR_EN)
        return _ep("lengua: plurales (en)", "en", f"Write the plural of «{w}».", p, [f"{w} → {p}"], tipo="plurales")
    w = rng.choice(_PLUR)
    p = _plural_es(w)
    return _ep("lengua: plurales", "es", f"Escribí el plural de «{w}».", p, [f"{w} → {p}"],
               errores=["olvidar el cambio de tilde («camión» → «camiones», «examen» → «exámenes»)"], tipo="plurales")


_SINON = [("rápido", "veloz"), ("comenzar", "empezar"), ("bonito", "lindo"), ("hablar", "conversar"),
          ("antiguo", "viejo"), ("feliz", "contento"), ("enorme", "gigante"), ("terminar", "finalizar")]
_DISTR = ["mesa", "lento", "oscuro", "dormir", "triste", "pequeño", "abrir", "nuevo"]
_SINON_EN = [("fast", "quick"), ("begin", "start"), ("big", "large"), ("happy", "glad")]


def sinonimos(rng: random.Random, level: int, lang: str) -> dict:
    pares = _SINON_EN if lang == "en" else _SINON
    w, s = rng.choice(pares)
    opciones = [s] + rng.sample([d for d in (["slow", "small", "sad", "close"] if lang == "en" else _DISTR)
                                 if d != s], 2)
    rng.shuffle(opciones)
    q = (f"Which is a synonym of «{w}»: {', '.join(opciones)}?" if lang == "en" else
         f"¿Cuál es sinónimo de «{w}»: {', '.join(opciones)}?")
    return _ep("lengua: sinónimos", lang, q, s, [f"{w} ≈ {s}"], tipo="sinonimos")


_CORRECTAS = ["Las casas blancas son antiguas.", "El tren llega tarde.", "Los chicos juegan en la plaza.",
              "La ciudad es muy antigua.", "Mi hermana tiene dos libros nuevos.", "El puerto exporta soja."]


def corregir(rng: random.Random, level: int, lang: str) -> dict:
    """Una oración correcta con un error de concordancia introducido (siempre distinto del original)."""
    from hope.runtime.lengua import MUTACIONES
    opciones = []
    for s in _CORRECTAS:
        for tipo in ("concordancia_genero", "concordancia_numero", "concordancia_adjetivo"):
            m = MUTACIONES[tipo](s, random.Random(rng.random()), [])
            if m and m != s:
                opciones.append((s, m, tipo))
    s, m, tipo = rng.choice(opciones)
    return _ep("lengua: corregir una oración", "es", f"Corregí la oración: «{m}»", s, [f"error de {tipo}: {m} → {s}"],
               tipo="corregir")


FAMILIAS_LENGUA = {
    "conjugacion": conjugacion, "concordancia": concordancia, "tildes": tildes, "puntuacion": puntuacion,
    "preposiciones": preposiciones, "orden": orden, "plurales": plurales, "sinonimos": sinonimos,
    "corregir": corregir,
}
PALABRAS_TEMA = {"conjugacion": ["conjuga", "verbo", "voseo"], "concordancia": ["concordancia"],
                 "tildes": ["tilde", "acentua"], "puntuacion": ["puntuaci"], "preposiciones": ["preposici"],
                 "orden": ["orden de palabras"], "plurales": ["plural"], "sinonimos": ["sinónim", "sinonim"],
                 "corregir": ["corregir"]}


def _norm(s: str, exacto: bool) -> str:
    s = " ".join(s.strip().split())
    s = s.strip("«»\"' ")
    return s if exacto else s.lower().rstrip(".")


def respuesta_ok(respuesta: str | None, esperado: str, tipo: str = "") -> bool:
    """Compara respuestas de lengua: en tildes y puntuación cuentan tildes, signos y mayúsculas; en el
    resto se ignoran mayúsculas y el punto final (nunca las tildes)."""
    if respuesta is None:
        return False
    exacto = tipo in ("tildes", "puntuacion", "corregir")
    r = re.sub(r"^(?:resultado|respuesta|answer)\s*:\s*", "", respuesta.strip(), flags=re.I)
    return _norm(r, exacto) == _norm(esperado, exacto)
