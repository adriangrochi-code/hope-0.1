"""Lenguaje natural con el enfoque de código: pruebas, mutaciones, tipos, diffs, FIM y trazas.

Pedido de la sesión principal (del usuario): aplicar al texto lo que se hace con código, en
español e inglés.

1. **«Pruebas» para texto** (``extraer_requisitos``): antes de escribir, el pedido se convierte en
   requisitos verificables — puntos a cubrir, largo, registro (formal/informal), variante
   (es-AR con voseo / neutro), idioma, términos obligatorios y sin contradicciones — y la respuesta
   se valida contra esa lista (``validar``). Cada requisito es también un ``Verificador`` (marca el
   tramo que falla), así el bucle de escritura de ``escritura.py`` lo usa para decidir qué reescribir.
2. **Mutaciones** (``MUTACIONES``, ``puntaje_mutaciones``): se introducen errores conocidos en textos
   buenos (quitar un punto, repetir, romper concordancia, cambiar de variante, contradecir un número,
   acortar…) y se mide qué fracción detectan las comprobaciones, y cuántas falsas alarmas dan sobre
   el texto sin mutar. Es la prueba de que las pruebas sirven.
3. **«Chequeo de tipos» del idioma** (``Concordancia``): concordancia de género y número
   (determinante/adjetivo ↔ sustantivo, predicativo ↔ sujeto), número y persona (sujeto ↔ verbo) y
   tiempos verbales dominantes, con la morfología de spaCy (``es_core_news_sm`` / ``en_core_web_sm``).
   Sin spaCy, el verificador no marca nada y lo dice (``disponible``).
4. **Edición por diff** (``Edicion``, ``aplicar_ediciones``): el bucle de revisión expresa cada cambio
   como buscar/reemplazar **exacto** de una oración marcada; lo demás queda intacto byte a byte
   (``preservado`` mide cuánto).
5. **Rellenar el medio (FIM) para texto** (``formato_fim``, ``ejemplos_fim``, ``rellenar``): formato de
   datos prefijo–sufijo–medio con centinelas ``⟦prefijo⟧ ⟦sufijo⟧ ⟦medio⟧ ⟦fin⟧`` para entrenar, y un
   modo de generación que completa o reescribe un fragmento dentro de un texto: ``modo="fim"`` (para
   un modelo entrenado con el formato) o ``modo="puntaje"`` (sirve ya: candidatos desde el prefijo,
   elegidos por la log-probabilidad del candidato **y del sufijo**).
6. **«Trazas» del lenguaje** (``traza``, ``datos_trazas``): ejemplos de entrenamiento con el texto y su
   análisis (lema, categoría, función y concordancia de cada palabra), como las trazas de ejecución
   en código: Hope ve la estructura de la oración junto a la oración.
"""

from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache

from hope.runtime.escritura import (VACIAS, ContextoRevision, Marca, oraciones, palabras)

# ---------------------------------------------------------------------------
# spaCy (opcional)
# ---------------------------------------------------------------------------
_MODELOS = {"es": "es_core_news_sm", "en": "en_core_web_sm"}


@lru_cache(maxsize=4)
def nlp(idioma: str = "es"):
    """Modelo de spaCy del idioma, o None si no está instalado."""
    try:
        import spacy
        return spacy.load(_MODELOS[idioma], disable=["ner"])
    except Exception:                                      # noqa: BLE001
        return None


def idioma_de(texto: str) -> str:
    w = palabras(texto[:3000])
    es = sum(x in ("de", "la", "el", "que", "los", "las", "del", "y", "es", "una") for x in w)
    en = sum(x in ("the", "of", "and", "is", "to", "was", "which", "a", "in", "that") for x in w)
    return "en" if en > es else "es"


# ---------------------------------------------------------------------------
# 1. Requisitos verificables
# ---------------------------------------------------------------------------
_VOS_PAL = {"vos", "sos", "tenés", "podés", "querés", "sabés", "hacés", "decís", "venís", "vivís", "estás", "andás",
            "mirá", "decime", "fijate", "acordate", "contame", "hacé", "tené", "vení", "decí", "escribí", "usá",
            "pensá", "probá", "buscá", "elegí", "mencioná", "explicá", "seguí"}
_TU_PAL = {"tú", "tienes", "puedes", "quieres", "sabes", "haces", "dices", "vienes", "vives", "eres", "mira",
           "dime", "fíjate", "acuérdate", "cuéntame", "haz", "ten", "ven", "di", "escribe", "usa", "piensa",
           "prueba", "busca", "elige", "contigo", "ti"}
_USTED_PAL = {"usted", "ustedes", "tiene", "puede"}       # «tiene/puede» solo cuentan junto a «usted»
_INFORMAL = {"che", "re", "onda", "copado", "genial", "boludo", "laburo", "guita", "chabón", "posta", "joya"}


def _tokens(texto: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", texto)]


def marcadores_variante(texto: str) -> dict:
    t = _tokens(texto)
    return {"voseo": [w for w in t if w in _VOS_PAL], "tuteo": [w for w in t if w in _TU_PAL - {"ten", "di", "ven",
                                                                                                  "mira", "usa", "ti"}],
            "usted": [w for w in t if w in ("usted", "ustedes")], "informal": [w for w in t if w in _INFORMAL]}


@dataclass
class Resultado:
    requisito: str
    ok: bool
    detalle: str = ""


@dataclass
class Requisito:
    """Un requisito verificable. ``tipo``: punto | termino | largo | registro | variante | idioma |
    sin_contradicciones."""
    tipo: str
    valor: object = None
    descripcion: str = ""

    def verificar(self, texto: str) -> tuple[Resultado, list[Marca]]:
        f = getattr(self, f"_v_{self.tipo}")
        ok, det, marcas = f(texto)
        for m in marcas:
            m.verificador = "requisitos"
        return Resultado(self.descripcion or f"{self.tipo}: {self.valor}", ok, det), marcas

    # --- comprobaciones ---------------------------------------------------------------
    def _v_punto(self, texto):
        raices = [w[:5] for w in palabras(str(self.valor)) if w not in VACIAS] or [str(self.valor).lower()[:5]]
        hay = {w[:5] for w in palabras(texto)}
        falta = [r for r in raices if r not in hay]
        ok = not falta
        return ok, "" if ok else f"no cubre «{self.valor}»", [] if ok else [
            Marca(-1, -1, "requisito", f"falta el punto «{self.valor}»", 1.0)]

    def _v_termino(self, texto):
        ok = str(self.valor).lower() in texto.lower()
        return ok, "" if ok else f"falta el término «{self.valor}»", [] if ok else [
            Marca(-1, -1, "requisito", f"falta el término obligatorio «{self.valor}»", 1.0)]

    def _v_largo(self, texto):
        lo, hi, unidad = self.valor
        n = len(palabras(texto)) if unidad == "palabras" else len(texto)
        marcas = []
        if hi is not None and n > hi:
            # el tramo sobrante, desde la oración donde se pasa del máximo
            cuenta, corte = 0, len(texto)
            for a, b in oraciones(texto):
                cuenta += len(palabras(texto[a:b])) if unidad == "palabras" else b - a
                if cuenta > hi:
                    corte = a
                    break
            marcas.append(Marca(corte, len(texto), "requisito", f"largo {n} > {hi} {unidad}", 1.0))
        if lo is not None and n < lo:
            marcas.append(Marca(-1, -1, "requisito", f"largo {n} < {lo} {unidad}", 1.0))
        return not marcas, "; ".join(m.detalle for m in marcas), marcas

    def _v_variante(self, texto):
        mk = marcadores_variante(texto)
        malo = mk["tuteo"] if self.valor == "es-AR" else mk["voseo"] if self.valor == "neutro" else []
        marcas = []
        for w in set(malo):
            for m in re.finditer(rf"\b{re.escape(w)}\b", texto, flags=re.I):
                marcas.append(Marca(m.start(), m.end(), "variante", f"«{w}» no es {self.valor}", 1.0))
        return not malo, "" if not malo else f"formas de otra variante: {', '.join(sorted(set(malo)))}", marcas

    def _v_registro(self, texto):
        mk = marcadores_variante(texto)
        if self.valor == "formal":
            malo = mk["voseo"] + mk["tuteo"] + mk["informal"]
        else:
            malo = mk["usted"]
        marcas = [Marca(m.start(), m.end(), "registro", f"«{m.group()}» no es registro {self.valor}", 1.0)
                  for w in set(malo) for m in re.finditer(rf"\b{re.escape(w)}\b", texto, flags=re.I)]
        return not malo, "" if not malo else f"marcas de otro registro: {', '.join(sorted(set(malo)))}", marcas

    def _v_idioma(self, texto):
        ok = not texto.strip() or idioma_de(texto) == self.valor
        return ok, "" if ok else f"no está en {self.valor}", [] if ok else [
            Marca(0, len(texto), "requisito", f"idioma distinto de {self.valor}", 2.0)]

    def _v_sin_contradicciones(self, texto):
        marcas = contradicciones(texto)
        return not marcas, "; ".join(m.detalle for m in marcas), marcas


_NUM = r"\d+(?:[.,]\d+)?"


def contradicciones(texto: str) -> list[Marca]:
    """Contradicciones simples: el mismo sujeto con dos valores numéricos distintos para la misma
    relación («X tiene 3 lunas» … «X tiene 5 lunas»), y «X es Y» junto a «X no es Y»."""
    marcas = []
    vistos: dict[tuple, tuple[str, int]] = {}
    pat = re.compile(rf"\b([\wáéíóúñ]+) (tiene|mide|pesa|has|is|es|fue|was|cuenta con|"
                     rf"tenía|had) (?:unos |unas |about |de |cerca de )?({_NUM}) ([\wáéíóúñ]+)", flags=re.I)
    for m in pat.finditer(texto):
        # clave: el núcleo del sujeto (la palabra justo antes del verbo), la relación y la unidad
        key = (m.group(1).lower(), m.group(2).lower(), m.group(4).lower()[:5])
        val = m.group(3).replace(",", ".")
        if key in vistos and vistos[key][0] != val:
            marcas.append(Marca(m.start(), m.end(), "contradiccion",
                                f"{m.group(1)} {m.group(2)} {vistos[key][0]} y también {val} {m.group(4)}", 2.0))
        vistos.setdefault(key, (val, m.start()))
    afirm = {}
    afirmaciones = [(m, m.group(1), bool(m.group(2)), m.group(4)) for m in re.finditer(
        r"\b([\wáéíóúñ]+) (no )?(es|son|fue|era) ([\wáéíóúñ]+)", texto, flags=re.I)]
    afirmaciones += [(m, m.group(1), bool(m.group(3)), m.group(4)) for m in re.finditer(
        r"\b(\w+) (is|are|was|were) (not )?(\w+)", texto, flags=re.I)]
    for m, suj, neg, pred in sorted(afirmaciones, key=lambda x: x[0].start()):
        if suj.lower() in VACIAS or pred.lower() in VACIAS:
            continue
        key = (suj.lower(), pred.lower())
        if key in afirm and afirm[key] != neg:
            marcas.append(Marca(m.start(), m.end(), "contradiccion", f"«{m.group()}» contradice lo dicho antes", 2.0))
        afirm.setdefault(key, neg)
    return marcas


_LISTA = r"([^.;:\n]+)"


def _partir_lista(s: str) -> list[str]:
    s = re.sub(r"\b(y|e|and|o|or)\b", ",", s)
    return [x.strip(" «»\"'") for x in s.split(",") if x.strip(" «»\"'") and len(x.strip()) > 1]


def extraer_requisitos(pedido: str, *, idioma: str | None = None, siempre: bool = True) -> list[Requisito]:
    """Convierte el pedido (español o inglés) en requisitos verificables. Reconoce:

    * puntos: «mencioná/incluí/explicá/hablá de/cubrí X, Y y Z», «mention/include/cover/explain X, Y and Z»;
    * términos obligatorios: lo que va entre comillas («…» o "…") y «usá el término X» / «use the term X»;
    * largo: «en N palabras», «como máximo/a lo sumo N palabras», «al menos N palabras», «N caracteres»,
      «in N words», «at most/at least N words»;
    * registro: «formal», «informal», «de usted»; variante: «rioplatense», «argentino», «es-AR»,
      «de vos», «voseo» → es-AR; «neutro», «sin voseo» → neutro;
    * idioma: «en inglés» / «in English» → en; «en español» / «in Spanish» → es.
    Con ``siempre`` se agrega «sin contradicciones»."""
    p = pedido
    low = p.lower()
    reqs: list[Requisito] = []
    verbos = (r"(?:mencion[áa]r?|menciona|inclu[íi]r?|incluye|explic[áa]r?|explica|habl[áa] de|hablar de|cubr[íi]r?|"
              r"cubre|nombr[áa]|tra[ée]r? |mention|include|cover|explain|discuss|talk about)")
    for m in re.finditer(verbos + r"\s+(?:los siguientes puntos:?\s*|the following:?\s*|que\s+)?" + _LISTA, p, flags=re.I):
        for x in _partir_lista(m.group(1)):
            if len(x.split()) <= 5:
                reqs.append(Requisito("punto", x, f"cubre «{x}»"))
    for m in re.finditer(r"«([^»]+)»|\"([^\"]+)\"|“([^”]+)”", p):
        t = next(g for g in m.groups() if g)
        reqs.append(Requisito("termino", t, f"usa el término «{t}»"))
    for m in re.finditer(r"(?:us[áa]r?|use|utiliz[áa]r?) (?:el término|la palabra|the term|the word) ([\wáéíóúñ-]+)",
                         p, flags=re.I):
        reqs.append(Requisito("termino", m.group(1), f"usa el término «{m.group(1)}»"))
    unidad_re = r"(palabras|words|caracteres|characters)"
    lo = hi = None
    unidad = "palabras"
    if m := re.search(rf"(?:como máximo|a lo sumo|no más de|menos de|at most|no more than|under|max(?:imum)?) (\d+) "
                      rf"{unidad_re}", low):
        hi, unidad = int(m.group(1)), m.group(2)
    if m := re.search(rf"(?:al menos|como mínimo|más de|at least|over|min(?:imum)?) (\d+) {unidad_re}", low):
        lo, unidad = int(m.group(1)), m.group(2)
    if lo is None and hi is None and (m := re.search(rf"\b(?:en|de|in|of|about|unas?|around) (\d+) {unidad_re}", low)):
        n, unidad = int(m.group(1)), m.group(2)
        lo, hi = int(n * 0.7), int(n * 1.3)
    if lo is not None or hi is not None:
        u = "palabras" if unidad in ("palabras", "words") else "caracteres"
        reqs.append(Requisito("largo", (lo, hi, u), f"largo entre {lo or 0} y {hi or '∞'} {u}"))
    if re.search(r"\b(rioplatense|argentino|es-ar|de vos|voseo|con vos)\b", low):
        reqs.append(Requisito("variante", "es-AR", "variante es-AR (voseo)"))
    elif re.search(r"\b(neutro|neutral|sin voseo|español estándar)\b", low):
        reqs.append(Requisito("variante", "neutro", "español neutro (sin voseo)"))
    if re.search(r"\b(informal|coloquial|casual)\b", low):
        reqs.append(Requisito("registro", "informal", "registro informal"))
    elif re.search(r"\b(formal|de usted|académico|formally)\b", low):
        reqs.append(Requisito("registro", "formal", "registro formal"))
    if re.search(r"\b(en inglés|in english)\b", low):
        reqs.append(Requisito("idioma", "en", "en inglés"))
    elif re.search(r"\b(en español|in spanish)\b", low):
        reqs.append(Requisito("idioma", "es", "en español"))
    elif idioma:
        reqs.append(Requisito("idioma", idioma, f"en {idioma}"))
    if siempre:
        reqs.append(Requisito("sin_contradicciones", None, "sin contradicciones"))
    vistos, out = set(), []
    for r in reqs:
        k = (r.tipo, str(r.valor).lower())
        if k not in vistos:
            vistos.add(k)
            out.append(r)
    return out


@dataclass
class Validacion:
    resultados: list[Resultado]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.resultados)

    @property
    def tasa(self) -> float:
        return sum(r.ok for r in self.resultados) / max(len(self.resultados), 1)

    def informe(self) -> str:
        return "\n".join(f"{'✔' if r.ok else '✘'} {r.requisito}" + (f" — {r.detalle}" if r.detalle else "")
                         for r in self.resultados)


def validar(texto: str, requisitos: list[Requisito]) -> Validacion:
    return Validacion([r.verificar(texto)[0] for r in requisitos])


class VerificadorRequisitos:
    """Los requisitos como verificador del bucle de escritura."""
    nombre = "requisitos"

    def __init__(self, requisitos: list[Requisito]):
        self.requisitos = requisitos

    def revisar(self, texto: str, ctx: ContextoRevision) -> list[Marca]:
        out = []
        for r in self.requisitos:
            out += r.verificar(texto)[1]
        return out


# ---------------------------------------------------------------------------
# 3. Concordancia («chequeo de tipos» del idioma)
# ---------------------------------------------------------------------------
def _f(tok, feat: str) -> str | None:
    v = tok.morph.get(feat)
    return v[0] if v else None


def _choca(a, b, feat: str) -> bool:
    x, y = _f(a, feat), _f(b, feat)
    return x is not None and y is not None and x != y


_A_TONICA = {"agua", "águila", "alma", "arma", "hacha", "área", "aula", "hambre", "ala", "ama", "ancla", "arpa",
             "asa", "aura", "ave", "hada", "haba", "habla", "acta", "alga", "ansia", "asma", "aria", "alba"}


def _a_tonica(w: str) -> bool:
    return w in _A_TONICA or w.startswith(("á", "há"))


class LexicoGenero:
    """Género de cada sustantivo según el corpus: cuántas veces va con «el/los/un/este…» frente a
    «la/las/una/esta…». Hace falta porque spaCy deduce el género del sustantivo **del propio
    artículo**: en «el terminal» etiqueta «terminal» como masculino y no ve el error."""
    _M = {"el", "los", "un", "unos", "este", "estos", "ese", "esos", "aquel", "del", "al"}
    _F = {"la", "las", "una", "unas", "esta", "estas", "esa", "esas", "aquella"}

    def __init__(self, cuentas: dict[str, list[int]], minimo: int = 3, margen: float = 0.9):
        self.c, self.minimo, self.margen = cuentas, minimo, margen

    @classmethod
    def desde_corpus(cls, textos: list[str], **kw) -> "LexicoGenero":
        c: dict[str, list[int]] = {}
        for t in textos:
            ws = re.findall(r"[a-záéíóúüñ]+", t.lower())
            for a, b in zip(ws, ws[1:]):
                if a in cls._M or a in cls._F:
                    c.setdefault(b, [0, 0])[0 if a in cls._M else 1] += 1
        return cls(c, **kw)

    def genero(self, palabra: str) -> str | None:
        """"Masc" / "Fem" si el corpus es claro (≥ ``minimo`` apariciones y ≥ ``margen`` de un lado)."""
        m, f = self.c.get(palabra.lower(), (0, 0))
        n = m + f
        if n < self.minimo:
            return None
        return "Masc" if m / n >= self.margen else "Fem" if f / n >= self.margen else None


class Concordancia:
    """Verificador de concordancia con spaCy (español e inglés).

    * género y número: determinante/adjetivo → su sustantivo (``det``, ``amod``) y predicativo/
      participio pasivo → sujeto (``nsubj``/``nsubj:pass`` con ``cop`` o auxiliar pasivo);
    * número y persona: sujeto → verbo conjugado (o su auxiliar/cópula conjugados);
    * tiempos: oraciones cuyo verbo principal está en un tiempo distinto del dominante del texto (≥ 70 %),
      con severidad baja (puede ser legítimo).
    Las palabras sin rasgo morfológico no se comparan (spaCy no siempre los da)."""
    nombre = "concordancia"

    def __init__(self, idioma: str | None = None, tiempos: bool = True, sev_tiempo: float = 0.3,
                 genero: LexicoGenero | None = None):
        """``genero``: léxico del corpus para el género de los sustantivos (ver ``LexicoGenero``)."""
        self.idioma, self.tiempos, self.sev_tiempo, self.lex = idioma, tiempos, sev_tiempo, genero

    def disponible(self, idioma: str = "es") -> bool:
        return nlp(idioma) is not None

    def errores(self, texto: str) -> list[Marca]:
        lang = self.idioma or idioma_de(texto)
        model = nlp(lang)
        if model is None or not texto.strip():
            return []
        doc = model(texto)
        marcas = []

        def add(tok_a, tok_b, det):
            a = min(tok_a.idx, tok_b.idx)
            b = max(tok_a.idx + len(tok_a), tok_b.idx + len(tok_b))
            marcas.append(Marca(a, b, "concordancia", det, 1.0, self.nombre))

        for t in doc:
            h = t.head
            if not t.is_alpha or not h.is_alpha:
                continue                                   # signos, plantillas wiki, números
            coord = any(c.dep_ == "conj" for c in h.children) or h.dep_ == "conj"
            if t.dep_ in ("det", "amod") and h.pos_ in ("NOUN", "PROPN") and t.pos_ in ("DET", "ADJ"):
                cerca = abs(t.i - h.i) <= 3 and not any(x.is_punct or x.pos_ == "CCONJ"
                                                        for x in doc[min(t.i, h.i) + 1:max(t.i, h.i)])
                if t.pos_ == "ADJ" and (not cerca or coord):
                    continue                               # adjetivo lejano o sobre sustantivos coordinados
                if lang == "es" and t.lower_ in ("el", "un", "algún", "ningún") and _a_tonica(h.lower_):
                    continue                               # «el agua», «un águila»: femeninos con a tónica
                lex = self.lex.genero(h.text) if (self.lex is not None and lang == "es" and h.pos_ == "NOUN") else None
                if lex is not None and _f(t, "Gender") not in (None, lex):
                    add(t, h, f"«{t.text} {h.text}»: «{h.text}» es {'masculino' if lex == 'Masc' else 'femenino'}")
                    continue
                for feat in (("Gender", "Number") if lang == "es" else ("Number",)):
                    if _choca(t, h, feat):
                        add(t, h, f"«{t.text} {h.text}»: {feat.lower()} distinto ({_f(t, feat)}/{_f(h, feat)})")
                        break
            if t.dep_ in ("nsubj", "nsubj:pass", "nsubjpass") and t.pos_ in ("NOUN", "PROPN", "PRON"):
                if any(c.dep_ == "conj" for c in t.children):
                    continue                               # sujeto coordinado: plural aunque cada uno sea singular
                verbos = [h] + [c for c in h.children if c.dep_ in ("aux", "aux:pass", "cop", "auxpass")]
                fin = [v for v in verbos if _f(v, "VerbForm") == "Fin" and v.i > t.i]   # sujeto antepuesto
                for v in fin:
                    if _choca(t, v, "Number") or (t.pos_ == "PRON" and _choca(t, v, "Person")):
                        add(t, v, f"«{t.text} … {v.text}»: el verbo no concuerda con el sujeto")
                        break
                # predicativo (adjetivo con cópula) o participio pasivo ↔ sujeto (español)
                if lang == "es" and t.pos_ != "PRON" and h.i > t.i and (
                        (h.pos_ == "ADJ" and any(c.dep_ == "cop" for c in h.children)) or
                        (h.pos_ == "VERB" and _f(h, "VerbForm") == "Part" and t.dep_ == "nsubj:pass")):
                    for feat in ("Gender", "Number"):
                        if _choca(t, h, feat):
                            add(t, h, f"«{t.text} … {h.text}»: {feat.lower()} del predicativo distinto")
                            break
        if self.tiempos:
            raices = []
            for s in doc.sents:
                r = s.root
                v = r if _f(r, "VerbForm") == "Fin" else next((c for c in r.children if _f(c, "VerbForm") == "Fin"),
                                                              None)
                if v is not None and _f(v, "Tense"):
                    raices.append((s, _f(v, "Tense")))
            if len(raices) >= 3:
                from collections import Counter
                dom, n = Counter(t for _, t in raices).most_common(1)[0]
                if n / len(raices) >= 0.7:
                    for s, tense in raices:
                        if tense != dom:
                            marcas.append(Marca(s.start_char, s.end_char, "tiempo_verbal",
                                                f"tiempo {tense} entre oraciones en {dom}", self.sev_tiempo,
                                                self.nombre))
        return marcas

    def revisar(self, texto: str, ctx: ContextoRevision) -> list[Marca]:
        return self.errores(texto)


# ---------------------------------------------------------------------------
# 2. Mutaciones
# ---------------------------------------------------------------------------
def _cambiar_palabra(texto: str, rng: random.Random, pares: dict[str, str]) -> str | None:
    cands = [m for m in re.finditer(r"\b(" + "|".join(map(re.escape, pares)) + r")\b", texto)]
    if not cands:
        return None
    m = rng.choice(cands)
    return texto[:m.start()] + pares[m.group()] + texto[m.end():]


_GENERO = {"la": "el", "el": "la", "las": "los", "los": "las", "una": "un", "un": "una", "esta": "este",
           "este": "esta", "The": "The"}
_NUMERO_V = {"es": "son", "son": "es", "fue": "fueron", "fueron": "fue", "tiene": "tienen", "tienen": "tiene",
             "está": "están", "están": "está", "was": "were", "were": "was", "has": "have", "have": "has",
             "is": "are", "are": "is"}
_PERSONA = {"es": "soy", "fue": "fuimos", "tiene": "tenemos", "está": "estamos", "son": "somos", "is": "am"}


def _m_genero(texto, rng, reqs):
    pares = {k: v for k, v in _GENERO.items() if k != "The"}
    return _cambiar_palabra(texto, rng, pares)


def _m_adjetivo(texto, rng, reqs):
    cands = [m for m in re.finditer(r"\b(?:la|las|el|los) [a-záéíóúñ]+ ([a-záéíóúñ]{3,}(?:o|a|os|as))\b", texto)]
    if not cands:
        return None
    m = rng.choice(cands)
    adj = m.group(1)
    flip = {"o": "a", "a": "o", "os": "as", "as": "os"}
    suf = "os" if adj.endswith("os") else "as" if adj.endswith("as") else adj[-1]
    nuevo = adj[:-len(suf)] + flip[suf]
    a = m.start(1)
    return texto[:a] + nuevo + texto[a + len(adj):]


def _m_numero(texto, rng, reqs):
    return _cambiar_palabra(texto, rng, _NUMERO_V)


def _m_persona(texto, rng, reqs):
    return _cambiar_palabra(texto, rng, _PERSONA)


def _m_repetir(texto, rng, reqs):
    sp = [s for s in oraciones(texto) if s[1] - s[0] > 30]
    if not sp:
        return None
    a, b = rng.choice(sp)
    return texto[:b] + " " + texto[a:b] + texto[b:]


def _m_quitar_punto(texto, rng, reqs):
    puntos = [r for r in reqs if r.tipo in ("punto", "termino")]
    for r in rng.sample(puntos, len(puntos)):
        raiz = [w[:5] for w in palabras(str(r.valor)) if w not in VACIAS]
        sp = [(a, b) for a, b in oraciones(texto) if any(w[:5] in raiz for w in palabras(texto[a:b]))]
        if sp:
            out = texto
            for a, b in reversed(sp):
                out = out[:a] + out[b:]
            return " ".join(out.split())
    return None


def _m_voseo(texto, rng, reqs):
    if idioma_de(texto) == "en":
        return None
    var = next((r.valor for r in reqs if r.tipo == "variante"), "neutro")
    frase = " Si querés saber más, fijate en los registros." if var != "es-AR" else \
        " Si quieres saber más, fíjate en los registros."
    sp = oraciones(texto)
    if not sp:
        return None
    a, b = rng.choice(sp)
    return texto[:b] + frase + texto[b:]


def _m_contradiccion(texto, rng, reqs):
    pat = re.compile(rf"\b([\wáéíóúñ]+) (tiene|mide|pesa|has|es|fue|was|cuenta con) "
                     rf"(?:unos |unas |about |de |cerca de )?({_NUM}) ([\wáéíóúñ]+)", flags=re.I)
    ms = list(pat.finditer(texto))
    if not ms:
        sp = oraciones(texto)
        m = next((x for x in re.finditer(r"\b([\wáéíóúñ]{3,}) (es|is) ([\wáéíóúñ]{4,})", texto)
                  if x.group(1).lower() not in VACIAS and x.group(3).lower() not in VACIAS), None)
        if not m or not sp:
            return None
        neg = f" {m.group(1)} no {m.group(2)} {m.group(3)}." if m.group(2) == "es" else \
            f" {m.group(1)} is not {m.group(3)}."
        return texto + neg
    m = rng.choice(ms)
    n = m.group(3).replace(",", ".")
    otro = str(int(float(n)) * 2 + 1)
    return texto + f" {m.group(1).capitalize()} {m.group(2)} {otro} {m.group(4)}."


def _m_acortar(texto, rng, reqs):
    if not any(r.tipo == "largo" for r in reqs):
        return None
    w = texto.split()
    return " ".join(w[:max(3, len(w) // 4)])


def _m_alargar(texto, rng, reqs):
    if not any(r.tipo == "largo" and r.valor[1] for r in reqs):
        return None
    return texto + " " + texto


def _m_fuera_de_tema(texto, rng, reqs):
    sp = oraciones(texto)
    if not sp:
        return None
    frase = (" Los pingüinos emperador bucean hasta quinientos metros buscando calamares y peces pequeños."
             if idioma_de(texto) == "es" else " Emperor penguins dive five hundred metres to catch squid and fish.")
    a, b = rng.choice(sp)
    return texto[:b] + frase + texto[b:]


def _m_palabra_rota(texto, rng, reqs):
    ws = [m for m in re.finditer(r"\b[a-záéíóúñ]{6,}\b", texto)]
    if not ws:
        return None
    m = rng.choice(ws)
    w = list(m.group())
    i = rng.randrange(1, len(w) - 2)
    w[i], w[i + 1] = w[i + 1], w[i]
    w.insert(rng.randrange(1, len(w)), rng.choice("qxzk"))
    return texto[:m.start()] + "".join(w) + texto[m.end():]


def _m_idioma(texto, rng, reqs):
    if not any(r.tipo == "idioma" for r in reqs):
        return None
    return ("The history of the region is long and was shaped by trade, wars and the arrival of new people."
            if idioma_de(texto) == "es" else
            "La historia de la región es larga y estuvo marcada por el comercio, las guerras y la llegada de gente.")


MUTACIONES = {
    "concordancia_genero": _m_genero, "concordancia_adjetivo": _m_adjetivo, "concordancia_numero": _m_numero,
    "persona_verbal": _m_persona, "repetir_oracion": _m_repetir, "quitar_punto": _m_quitar_punto,
    "otra_variante": _m_voseo, "contradiccion": _m_contradiccion, "acortar": _m_acortar, "alargar": _m_alargar,
    "fuera_de_tema": _m_fuera_de_tema, "palabra_rota": _m_palabra_rota, "otro_idioma": _m_idioma,
}


@dataclass
class PuntajeMutaciones:
    por_tipo: dict = field(default_factory=dict)          # tipo → {"aplicadas", "detectadas"}
    falsas_alarmas: int = 0                               # textos originales con alguna marca/falla
    originales: int = 0

    @property
    def deteccion(self) -> float:
        a = sum(v["aplicadas"] for v in self.por_tipo.values())
        return sum(v["detectadas"] for v in self.por_tipo.values()) / max(a, 1)

    @property
    def tasa_falsas_alarmas(self) -> float:
        return self.falsas_alarmas / max(self.originales, 1)


def _puntaje_fallas(texto, verificadores, reqs, ctx) -> tuple[float, set]:
    from hope.runtime.escritura import penalidad, revisar
    pen = penalidad(revisar(texto, verificadores, ctx)) if verificadores else 0.0
    fallas = {i for i, r in enumerate(validar(texto, reqs).resultados) if not r.ok} if reqs else set()
    return pen, fallas


def puntaje_mutaciones(casos: list[tuple[str, list[Requisito]]], verificadores: list, *, seed: int = 0,
                       tipos: list[str] | None = None, ctx_fn=None) -> PuntajeMutaciones:
    """Para cada (texto bueno, requisitos) aplica cada mutación y cuenta si las comprobaciones la
    detectan: una mutación se detecta si sube la penalidad de los verificadores o falla un requisito
    que en el original pasaba. Falsa alarma: el original ya falla algún requisito o tiene marcas de
    severidad ≥ 1 de concordancia/requisitos."""
    rng = random.Random(seed)
    out = PuntajeMutaciones()
    for texto, reqs in casos:
        ctx = ctx_fn(texto) if ctx_fn else ContextoRevision()
        base_pen, base_fallas = _puntaje_fallas(texto, verificadores, reqs, ctx)
        out.originales += 1
        from hope.runtime.escritura import revisar
        alarmas = [m for m in revisar(texto, verificadores, ctx) if m.tipo in ("concordancia", "requisito",
                                                                               "contradiccion", "variante")]
        if base_fallas or alarmas:
            out.falsas_alarmas += 1
        for tipo, fn in MUTACIONES.items():
            if tipos and tipo not in tipos:
                continue
            mut = fn(texto, rng, reqs)
            if mut is None or mut == texto:
                continue
            pen, fallas = _puntaje_fallas(mut, verificadores, reqs, ctx)
            d = out.por_tipo.setdefault(tipo, {"aplicadas": 0, "detectadas": 0})
            d["aplicadas"] += 1
            # detectada: sube la penalidad o falla un requisito que antes pasaba (no alcanza con contar
            # fallas: la mutación puede arreglar otro requisito sin querer, p. ej. el largo)
            d["detectadas"] += int(pen > base_pen + 1e-9 or bool(fallas - base_fallas))
    return out


# ---------------------------------------------------------------------------
# 4. Edición por diff
# ---------------------------------------------------------------------------
@dataclass
class Edicion:
    buscar: str
    reemplazar: str
    motivo: str = ""
    aplicada: bool = False


def edicion_unica(texto: str, a: int, b: int, reemplazo: str, motivo: str = "") -> Edicion:
    """Edición buscar/reemplazar del tramo ``[a, b)`` con el contexto mínimo para que ``buscar`` sea
    único en el texto (se agregan caracteres de antes y, si hace falta, de después)."""
    pre = post = 0
    while True:
        s = texto[a - pre:b + post]
        if texto.count(s) == 1 or (a - pre <= 0 and b + post >= len(texto)):
            return Edicion(s, texto[a - pre:a] + reemplazo + texto[b:b + post], motivo)
        if a - pre > 0:
            pre = min(a, pre + 8)
        else:
            post = min(len(texto) - b, post + 8)


def aplicar_ediciones(texto: str, ediciones: list[Edicion]) -> str:
    """Aplica cada edición si ``buscar`` aparece **exactamente una vez**; si no, la deja sin aplicar
    (``aplicada = False``) en lugar de adivinar dónde iba. Lo demás queda intacto."""
    for e in ediciones:
        if e.buscar and texto.count(e.buscar) == 1:
            texto = texto.replace(e.buscar, e.reemplazar, 1)
            e.aplicada = True
        else:
            e.aplicada = False
    return texto


def preservado(antes: str, despues: str) -> float:
    """Fracción de las oraciones del texto original que siguen idénticas en el nuevo."""
    a = [antes[x:y] for x, y in oraciones(antes)]
    b = [despues[x:y] for x, y in oraciones(despues)]
    if not a:
        return 1.0
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks()) / len(a)


def diff_texto(antes: str, despues: str) -> str:
    a = [antes[x:y] for x, y in oraciones(antes)]
    b = [despues[x:y] for x, y in oraciones(despues)]
    return "\n".join(difflib.unified_diff(a, b, "antes", "después", lineterm="", n=0))


# ---------------------------------------------------------------------------
# 5. Rellenar el medio (FIM) para texto
# ---------------------------------------------------------------------------
FIM_PRE, FIM_SUF, FIM_MED, FIM_FIN = "⟦prefijo⟧", "⟦sufijo⟧", "⟦medio⟧", "⟦fin⟧"


def formato_fim(prefijo: str, sufijo: str, medio: str | None = None) -> str:
    """Formato PSM: ``⟦prefijo⟧P⟦sufijo⟧S⟦medio⟧M⟦fin⟧``. Sin ``medio`` es el prompt de generación."""
    base = f"{FIM_PRE}{prefijo}{FIM_SUF}{sufijo}{FIM_MED}"
    return base if medio is None else f"{base}{medio}{FIM_FIN}"


def ejemplos_fim(textos: list[str], *, por_texto: int = 2, seed: int = 0, frac_fim: float = 1.0) -> list[str]:
    """Ejemplos de entrenamiento: el «medio» es una oración (a veces dos) del texto; el resto queda de
    prefijo y sufijo. ``frac_fim`` < 1 deja una parte de los textos en orden normal (como en código,
    para no olvidar la generación de izquierda a derecha)."""
    rng = random.Random(seed)
    out = []
    for t in textos:
        sp = oraciones(t)
        if len(sp) < 3:
            continue
        for _ in range(por_texto):
            if rng.random() > frac_fim:
                out.append(t)
                continue
            i = rng.randrange(1, len(sp) - 1)
            j = min(len(sp) - 1, i + (1 if rng.random() < 0.3 else 0))
            a, b = sp[i][0], sp[j][1]
            out.append(formato_fim(t[:a], t[b:], t[a:b]))
    return out


@dataclass
class Relleno:
    texto: str
    candidatos: list[str]
    puntajes: list[float]
    modo: str


def rellenar(model, prefijo: str, sufijo: str, *, modo: str = "puntaje", n: int = 6, max_new: int = 160,
             temperature: float = 0.7, largo_sufijo: int = 96, oraciones_max: int = 1,
             incluir_voraz: bool = True) -> Relleno:
    """Completa o reescribe el fragmento entre ``prefijo`` y ``sufijo``.

    * ``"fim"``: genera desde ``formato_fim(prefijo, sufijo)`` hasta ``⟦fin⟧`` (necesita un modelo
      entrenado con ``ejemplos_fim``).
    * ``"puntaje"``: ``n`` candidatos desde el prefijo (en lote), cortados en ``oraciones_max``
      oraciones, y gana el de mayor log-probabilidad media del candidato **más** los primeros
      ``largo_sufijo`` bytes del sufijo: el que mejor encaja con lo que sigue."""
    from hope.core.batched import generate_batch
    if modo == "fim":
        outs = generate_batch(model, [formato_fim(prefijo, sufijo)] * n, max_new,
                              temperature=temperature, stop_seq=FIM_FIN.encode())
        if incluir_voraz:
            outs = generate_batch(model, [formato_fim(prefijo, sufijo)], max_new, stop_seq=FIM_FIN.encode()) + outs[1:]
        cands = [o.decode("utf-8", errors="ignore").split(FIM_FIN)[0].strip() for o in outs]
    else:
        outs = []
        if incluir_voraz:
            outs += generate_batch(model, [prefijo], max_new)
        outs += generate_batch(model, [prefijo] * (n - len(outs)), max_new, temperature=temperature)
        cands = [_cortar(o.decode("utf-8", errors="ignore"), oraciones_max) for o in outs]
    cands = [c for c in dict.fromkeys(cands) if c.strip()] or [""]
    from hope.runtime.escritura import Escritor
    esc = Escritor.__new__(Escritor)
    esc.model = model
    sep = "" if not cands[0] or prefijo.endswith((" ", "\n")) else " "
    scores = esc._logprob(prefijo + sep, [c + (" " if sufijo and not sufijo[0].isspace() else "") + sufijo[:largo_sufijo]
                                         for c in cands])
    best = max(range(len(cands)), key=lambda i: scores[i])
    return Relleno(cands[best], cands, scores, modo)


def _cortar(t: str, n: int) -> str:
    sp = oraciones(t)
    if not sp:
        return t.strip()
    return t[:sp[min(n, len(sp)) - 1][1]].strip()


# ---------------------------------------------------------------------------
# 6. Trazas del lenguaje
# ---------------------------------------------------------------------------
_POS_ES = {"DET": "det", "NOUN": "sust", "PROPN": "nombre", "VERB": "verbo", "AUX": "aux", "ADJ": "adj",
           "ADV": "adv", "ADP": "prep", "PRON": "pron", "CCONJ": "conj", "SCONJ": "conj.sub", "NUM": "núm",
           "PUNCT": "punt", "PART": "part", "INTJ": "interj", "SYM": "símb", "X": "otro", "SPACE": "esp"}
_FEATS = {"Gender": {"Masc": "m", "Fem": "f"}, "Number": {"Sing": "sg", "Plur": "pl"},
          "Person": {"1": "1", "2": "2", "3": "3"},
          "Tense": {"Pres": "pres", "Past": "pas", "Imp": "imp", "Fut": "fut"}}


def _rasgos(t) -> str:
    out = []
    for f, m in _FEATS.items():
        v = _f(t, f)
        if v in m:
            out.append(m[v])
    return ".".join(out)


def traza(texto: str, idioma: str | None = None, *, max_oraciones: int = 4) -> str | None:
    """Texto + análisis por palabra (como una traza de ejecución): ``palabra(lema) categoría función→núcleo
    rasgos``, y la verificación de concordancia de cada oración. None si no hay spaCy."""
    lang = idioma or idioma_de(texto)
    model = nlp(lang)
    if model is None:
        return None
    doc = model(texto)
    conc = Concordancia(lang, tiempos=False)
    partes = []
    for k, s in enumerate(doc.sents):
        if k >= max_oraciones:
            break
        lin = [f"Oración: {s.text.strip()}"]
        for t in s:
            if t.is_space or t.pos_ == "PUNCT":
                continue
            lema = f"({t.lemma_})" if t.lemma_.lower() != t.text.lower() else ""
            dep = t.dep_ if t.head is t else f"{t.dep_}→{t.head.text}"
            r = _rasgos(t)
            lin.append(f"  {t.text}{lema} {_POS_ES.get(t.pos_, t.pos_)} {dep}" + (f" [{r}]" if r else ""))
        errs = conc.errores(s.text)
        lin.append("  concordancia: " + ("bien" if not errs else "; ".join(e.detalle for e in errs)))
        partes.append("\n".join(lin))
    return "⟦análisis⟧\n" + "\n".join(partes) + "\n⟦fin⟧"


def datos_trazas(textos: list[str], *, max_oraciones: int = 3, con_texto: bool = True) -> list[str]:
    """Ejemplos de entrenamiento: el pasaje y, a continuación, su traza lingüística."""
    out = []
    for t in textos:
        tr = traza(t, max_oraciones=max_oraciones)
        if tr is None:
            break
        out.append(f"{t.strip()}\n{tr}" if con_texto else tr)
    return out


def escribir_datos(carpeta, ejemplos: list[str], val_pct: float = 5.0) -> dict:
    """Escribe ``train.bin``/``val.bin`` (bytes UTF-8 separados por el byte 0, como ``hope.data.corpus``)."""
    import hashlib
    from pathlib import Path
    from hope.data.corpus import EOD, split_of
    d = Path(carpeta)
    d.mkdir(parents=True, exist_ok=True)
    bufs = {"train": bytearray(), "val": bytearray()}
    for e in ejemplos:
        sha = hashlib.sha256(e.encode()).hexdigest()
        bufs[split_of(sha, val_pct)] += e.encode("utf-8") + bytes([EOD])
    for split, b in bufs.items():
        (d / f"{split}.bin").write_bytes(bytes(b))
    return {k: len(v) for k, v in bufs.items()}
