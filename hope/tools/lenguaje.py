"""Herramientas de lengua para español e inglés (catálogo HTE; ``⟦nombre: texto⟧``).

Pedido del usuario: herramientas de ortografía, gramática y estilo con **dependencias opcionales y
licencias abiertas** (nada de la RAE). Cada herramienta usa la mejor fuente instalada y, si no hay
ninguna, un respaldo propio; el resultado siempre dice qué fuente usó (``fuente``), así nunca se
confunde «no encontré errores» con «no pude revisar».

| Herramienta | Fuente preferida (licencia) | Respaldo sin dependencias |
|---|---|---|
| ``ortografia`` | pyspellchecker (MIT; listas de frecuencia de OpenSubtitles) | vocabulario de wordfreq o del corpus local |
| ``gramatica`` | LanguageTool **local** vía language_tool_python (LGPL) | reglas propias + concordancia (spaCy) |
| ``diccionario`` | volcado local de Wiktionary de kaikki.org (CC BY-SA), ``HOPE_WIKTIONARY`` | lema (spaCy) y frecuencia |
| ``sinonimos`` | Open Multilingual Wordnet vía NLTK (licencias abiertas por idioma) | no disponible |
| ``conjugar`` | propio (regulares + 20 irregulares frecuentes, con voseo; inglés con irregulares) | — |
| ``analizar`` / ``concordancias`` / ``entidades`` | spaCy (MIT; modelos es CC BY-SA/GPL, en MIT) | no disponible |
| ``idioma``, ``variante``, ``registro`` | propio (marcadores léxicos) | — |
| ``legibilidad`` | propio: Fernández-Huerta e INFLESZ (es), Flesch (en) | — |
| ``silabas`` | silabeo propio (es) y pyphen (LGPL/MPL; diccionarios de LibreOffice) | propio |
| ``repeticiones``, ``fluidez`` | propio; KenLM (LGPL) si hay un modelo en ``HOPE_KENLM`` | n-gramas de bytes del corpus |
| ``coherencia`` | NLI chico abierto vía transformers si está (``HOPE_NLI``) | contradicciones + cohesión léxica |
| ``tildes_puntuacion`` | propio + el corrector ortográfico | propio |
| ``frecuencia`` | wordfreq (Apache-2.0; datos CC BY-SA) | corpus local |
| ``parafrasear`` | Argos Translate local (MIT), ida y vuelta por el inglés | no disponible |

Nada usa la red: los modelos y diccionarios tienen que estar instalados de antes.
"""

from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache

from hope.tools.catalog import Catalog, ToolSpec

_PAL = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:['’][A-Za-z]+)?")


def _palabras(texto: str):
    return [(m.group(), m.start(), m.end()) for m in _PAL.finditer(texto)]


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _lang(texto: str, idioma: str | None) -> str:
    if idioma in ("es", "en"):
        return idioma
    from hope.runtime.lengua import idioma_de
    return idioma_de(texto)


# ---------------------------------------------------------------------------
# Fuentes opcionales (perezosas)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _spell(idioma: str):
    try:
        from spellchecker import SpellChecker
        return SpellChecker(language=idioma)
    except Exception:                                          # noqa: BLE001
        return None


def _zipf(palabra: str, idioma: str) -> float | None:
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(palabra, idioma)
    except Exception:                                          # noqa: BLE001
        return None


@lru_cache(maxsize=1)
def _corpus_vocab() -> dict[str, int]:
    from collections import Counter
    from hope.runtime.escritura import corpus_local, palabras
    try:
        return dict(Counter(w for r in corpus_local(espanol=False) for w in palabras(r["texto"])))
    except Exception:                                          # noqa: BLE001
        return {}


@lru_cache(maxsize=2)
def _ngramas(idioma: str):
    from hope.runtime.escritura import ModeloNgramas, corpus_local
    from hope.runtime.lengua import idioma_de
    textos = [r["texto"] for r in corpus_local(espanol=False) if idioma_de(r["texto"]) == idioma]
    return ModeloNgramas(5).entrenar(textos) if textos else None


@lru_cache(maxsize=2)
def _languagetool(idioma: str):
    """LanguageTool local (nunca la API pública). None si no está instalado o no tiene el servidor."""
    if os.environ.get("HOPE_SIN_LANGUAGETOOL"):
        return None
    try:
        import language_tool_python
        cache = os.environ.get("LTP_PATH") or os.path.expanduser("~/.cache/language_tool_python")
        if not os.path.isdir(cache) or not os.listdir(cache):
            return None                                        # descargarlo necesitaría red: no se intenta
        return language_tool_python.LanguageTool("es" if idioma == "es" else "en-US")
    except Exception:                                          # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Ortografía, tildes y puntuación
# ---------------------------------------------------------------------------
def ortografia(texto: str, idioma: str | None = None, max_sugerencias: int = 3) -> dict:
    lang = _lang(texto, idioma)
    sp = _spell(lang)
    voc = None if sp is not None else _corpus_vocab()
    fuente = "pyspellchecker" if sp is not None else ("wordfreq" if _zipf("casa", "es") else "corpus local")
    errores = []
    toks = _palabras(texto)
    for i, (w, a, b) in enumerate(toks):
        lw = w.lower()
        if len(lw) < 3 or (w[0].isupper() and a > 0 and texto[max(0, a - 2):a].strip() not in (".", "!", "?", "¿", "¡")):
            continue                                            # nombres propios a mitad de oración
        sug = []
        if sp is not None:
            if lw in sp:
                continue
            cand = sp.candidates(lw) or set()
            zw = _zipf(lw, lang) or 0.0
            # La misma palabra con otra tilde es un error de tilde si esa forma es bastante más
            # frecuente («arbol» 3,5 → «árbol» 4,4); si las dos son comunes («esta»/«está»), no se
            # decide acá. Sin variante con tilde, una palabra frecuente (marcas, préstamos) se acepta.
            tilde = [c for c in cand if c != lw and _sin_tildes(c) == _sin_tildes(lw)
                     and (_zipf(c, lang) or 0.0) > zw + 0.5]
            if not tilde and zw >= 3.0:
                continue
            sug = tilde + sorted(set(cand) - set(tilde), key=lambda c: -sp.word_usage_frequency(c))
            sug = sug[:max_sugerencias]
        elif fuente == "wordfreq":
            if (_zipf(lw, lang) or 0) > 0:
                continue
        elif voc.get(lw, 0) >= 2:
            continue
        errores.append({"palabra": w, "inicio": a, "fin": b, "sugerencias": sug,
                        "tilde": any(_sin_tildes(s) == _sin_tildes(lw) and s != lw for s in sug)})
    return {"idioma": lang, "fuente": fuente, "errores": errores}


def tildes_puntuacion(texto: str, idioma: str | None = None) -> dict:
    lang = _lang(texto, idioma)
    out = []

    def add(a, b, det, sug=""):
        out.append({"inicio": a, "fin": b, "detalle": det, "sugerencia": sug})
    if lang == "es":
        for abre, cierra in (("¿", "?"), ("¡", "!")):
            for m in re.finditer(r"[^.!?¿¡\n]*[" + re.escape(cierra) + r"]", texto):
                s = m.group()
                if abre not in s:
                    add(m.start(), m.end(), f"falta «{abre}» de apertura")
    for m in re.finditer(r"\s+([,.;:!?])", texto):
        add(m.start(), m.end(), f"espacio antes de «{m.group(1)}»", m.group(1))
    for m in re.finditer(r"([,;:])(?=[A-Za-zÁÉÍÓÚÑáéíóúñ])", texto):
        add(m.start(), m.end() + 1, f"falta espacio después de «{m.group(1)}»", m.group(1) + " ")
    for m in re.finditer(r"(?<![.\d])\.(?=[A-ZÁÉÍÓÚÑ][a-záéíóúñ])", texto):
        add(m.start(), m.end() + 1, "falta espacio después del punto", ". ")
    for m in re.finditer(r"([.!?])\s+([a-záéíóúñ])", texto):
        if not re.search(r"\b(?:etc|p\. ej|Sr|Sra|Dr|e\.g|i\.e|aprox)\.$", texto[:m.start() + 1]):
            add(m.start(2), m.start(2) + 1, "minúscula después de punto", m.group(2).upper())
    for m in re.finditer(r"  +", texto):
        add(m.start(), m.end(), "espacios dobles", " ")
    for m in re.finditer(r"([,;:])\1+|\.{4,}", texto):
        add(m.start(), m.end(), "signo repetido")
    if lang == "es":
        for e in ortografia(texto, "es")["errores"]:
            if e["tilde"]:
                s = next(x for x in e["sugerencias"] if _sin_tildes(x) == _sin_tildes(e["palabra"].lower()))
                add(e["inicio"], e["fin"], f"tilde: «{e['palabra']}» → «{s}»", s)
    return {"idioma": lang, "fuente": "reglas propias + corrector", "problemas": out}


# ---------------------------------------------------------------------------
# Gramática
# ---------------------------------------------------------------------------
_REGLAS_ES = [
    (r"\b(a) (el)\b", "contracción: «a el» → «al»", "al"),
    (r"\b(de) (el)\b", "contracción: «de el» → «del»", "del"),
    (r"\b(hubieron|habían|habrán|hubieran) (\w+s)\b(?! (sido|estado|hecho|dicho|visto|tenido|llegado|ido))",
     "«haber» impersonal no va en plural («hubo/había…»)", None),
    (r"\bmas sin embargo\b", "redundancia: «mas sin embargo»", "sin embargo"),
    (r"\b(\w+) \1\b", "palabra repetida", None),
    (r"\bdeber de (\w+r)\b", "obligación: «deber + infinitivo» (sin «de»)", None),
    (r"\bpienso de que|creo de que|opino de que\b", "dequeísmo", None),
]
_REGLAS_EN = [
    (r"\b(a) ([aeiouAEIOU]\w+)", "«a» antes de vocal: «an»", None),
    (r"\b(an) ([b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]\w+)", "«an» antes de consonante: «a»", None),
    (r"\b(could|should|would|must) of\b", "«could of» → «could have»", None),
    (r"\b(\w+) \1\b", "repeated word", None),
    (r"\b(he|she|it) (don't)\b", "«doesn't»", None),
]


def gramatica(texto: str, idioma: str | None = None) -> dict:
    lang = _lang(texto, idioma)
    lt = _languagetool(lang)
    if lt is not None:
        ms = lt.check(texto)
        return {"idioma": lang, "fuente": "LanguageTool local",
                "errores": [{"inicio": m.offset, "fin": m.offset + m.errorLength, "detalle": m.message,
                             "regla": m.ruleId, "sugerencias": m.replacements[:3]} for m in ms]}
    errores = []
    for pat, det, sug in (_REGLAS_ES if lang == "es" else _REGLAS_EN):
        for m in re.finditer(pat, texto, flags=re.I):
            if det in ("palabra repetida", "repeated word") and m.group(1).lower() in ("que", "had", "that", "nos"):
                continue
            errores.append({"inicio": m.start(), "fin": m.end(), "detalle": det, "regla": "propia",
                            "sugerencias": [sug] if sug else []})
    from hope.runtime.lengua import Concordancia, nlp
    fuente = "reglas propias"
    if nlp(lang) is not None:
        fuente += " + concordancia (spaCy)"
        for e in Concordancia(lang, tiempos=False, genero=_lexico_genero() if lang == "es" else None).errores(texto):
            errores.append({"inicio": e.inicio, "fin": e.fin, "detalle": e.detalle, "regla": "concordancia",
                            "sugerencias": []})
    return {"idioma": lang, "fuente": fuente, "errores": sorted(errores, key=lambda e: e["inicio"])}


@lru_cache(maxsize=1)
def _lexico_genero():
    from hope.runtime.escritura import corpus_local
    from hope.runtime.lengua import LexicoGenero
    return LexicoGenero.desde_corpus([r["texto"] for r in corpus_local()])


def concordancias(texto: str, idioma: str | None = None) -> dict:
    from hope.runtime.lengua import Concordancia, nlp
    lang = _lang(texto, idioma)
    if nlp(lang) is None:
        return {"idioma": lang, "fuente": "no disponible (falta spaCy)", "errores": []}
    es = Concordancia(lang, genero=_lexico_genero() if lang == "es" else None).errores(texto)
    return {"idioma": lang, "fuente": "spaCy + léxico de género del corpus",
            "errores": [{"inicio": e.inicio, "fin": e.fin, "tipo": e.tipo, "detalle": e.detalle} for e in es]}


# ---------------------------------------------------------------------------
# Análisis, entidades, idioma, variante, registro
# ---------------------------------------------------------------------------
def analizar(texto: str, idioma: str | None = None) -> dict:
    from hope.runtime.lengua import traza
    t = traza(texto, _lang(texto, idioma), max_oraciones=6)
    return {"fuente": "spaCy" if t else "no disponible (falta spaCy)", "analisis": t or ""}


@lru_cache(maxsize=2)
def _nlp_ner(idioma: str):
    try:
        import spacy
        from hope.runtime.lengua import _MODELOS
        return spacy.load(_MODELOS[idioma])
    except Exception:                                          # noqa: BLE001
        return None


def entidades(texto: str, idioma: str | None = None) -> dict:
    lang = _lang(texto, idioma)
    n = _nlp_ner(lang)
    if n is None:
        return {"fuente": "no disponible (falta spaCy)", "entidades": []}
    return {"fuente": "spaCy", "entidades": [{"texto": e.text, "tipo": e.label_, "inicio": e.start_char,
                                              "fin": e.end_char} for e in n(texto).ents]}


def idioma(texto: str) -> dict:
    from hope.runtime.escritura import palabras
    w = palabras(texto[:5000])
    es = sum(x in ("de", "la", "el", "que", "los", "las", "del", "y", "es", "una", "en", "por") for x in w)
    en = sum(x in ("the", "of", "and", "is", "to", "was", "which", "a", "in", "that", "for") for x in w)
    tot = es + en
    if tot == 0:
        return {"idioma": "desconocido", "confianza": 0.0}
    lang = "es" if es >= en else "en"
    return {"idioma": lang, "confianza": round(max(es, en) / tot, 3), "marcadores": {"es": es, "en": en}}


_US_UK = {"color": "colour", "favor": "favour", "honor": "honour", "labor": "labour", "center": "centre",
          "theater": "theatre", "meter": "metre", "liter": "litre", "analyze": "analyse", "organize": "organise",
          "realize": "realise", "recognize": "recognise", "defense": "defence", "license": "licence",
          "traveled": "travelled", "gray": "grey", "catalog": "catalogue", "dialog": "dialogue",
          "program": "programme", "aluminum": "aluminium", "tire": "tyre", "check": "cheque"}


def variante(texto: str, idioma: str | None = None) -> dict:
    lang = _lang(texto, idioma)
    if lang == "en":
        toks = [w.lower() for w, _, _ in _palabras(texto)]
        us = [w for w in toks if w in _US_UK]
        uk = [w for w in toks if w in _US_UK.values() or re.fullmatch(r"\w+is(e|ed|ing|ation)", w) and
              w.replace("is", "iz", 1) in {k.replace("ize", "iz") for k in _US_UK}]
        v = "en-US" if len(us) > len(uk) else "en-GB" if uk else "sin marcas"
        return {"idioma": "en", "variante": v, "us": us, "uk": uk}
    from hope.runtime.lengua import marcadores_variante
    mk = marcadores_variante(texto)
    v = "es-AR" if mk["voseo"] and not mk["tuteo"] else "mezcla" if mk["voseo"] and mk["tuteo"] else \
        "neutro (tuteo)" if mk["tuteo"] else "neutro (usted)" if mk["usted"] else "sin marcas"
    return {"idioma": "es", "variante": v, **{k: v2 for k, v2 in mk.items() if k != "informal"}}


_COLOQ_EN = {"gonna", "wanna", "gotta", "kinda", "yeah", "cool", "dude", "stuff", "ok", "okay", "guys"}


def registro(texto: str, idioma: str | None = None) -> dict:
    lang = _lang(texto, idioma)
    toks = [w.lower() for w, _, _ in _palabras(texto)]
    if lang == "en":
        contr = re.findall(r"\b\w+n't\b|\b\w+'(?:re|ll|ve|d|m)\b", texto, flags=re.I)
        col = [w for w in toks if w in _COLOQ_EN]
        exc = texto.count("!")
        puntos = len(contr) + 2 * len(col) + exc
        return {"idioma": "en", "registro": "informal" if puntos >= 2 else "formal", "contracciones": contr,
                "coloquialismos": col}
    from hope.runtime.lengua import marcadores_variante
    mk = marcadores_variante(texto)
    inf = len(mk["voseo"]) + len(mk["tuteo"]) + 2 * len(mk["informal"]) + texto.count("!") // 2
    form = len(mk["usted"]) + sum(w in ("asimismo", "no obstante", "por consiguiente", "cabe", "dicho") for w in toks)
    r = "informal" if inf > form else "formal" if form > inf or not inf else "neutro"
    return {"idioma": "es", "registro": r, "marcas_informales": mk["voseo"] + mk["tuteo"] + mk["informal"],
            "marcas_formales": mk["usted"]}


# ---------------------------------------------------------------------------
# Sílabas y legibilidad
# ---------------------------------------------------------------------------
_V = "aeiouáéíóúü"
_FUERTES = "aeoáéíóú"
_INSEP = {"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "kl", "kr", "pl", "pr", "tl", "tr", "ch", "ll", "rr"}


def silabas_es(palabra: str) -> list[str]:
    """Silabeo del español por reglas (consonantes entre vocales, grupos inseparables, hiatos con
    vocal fuerte o débil tónica). Cubre la gran mayoría de las palabras; los casos dudosos (p. ej.
    prefijos como «sub-») siguen la regla general."""
    w = palabra.lower()
    if not w:
        return []
    es_v = [c in _V for c in w]
    # núcleos: grupos de vocales, partidos en hiatos
    sil_lim = []                                          # índices donde empieza una sílaba nueva
    i = 0
    n = len(w)
    # 1) separar vocales en hiato
    def hiato(a, b):
        return (a in _FUERTES and b in _FUERTES) or a in "íú" or b in "íú"
    # 2) recorrer consonantes entre núcleos
    nucleos = []
    while i < n:
        if es_v[i]:
            j = i
            while j + 1 < n and es_v[j + 1] and not hiato(w[j], w[j + 1]):
                j += 1
            nucleos.append((i, j))
            i = j + 1
        else:
            i += 1
    if len(nucleos) <= 1:
        return [palabra]
    for (a1, b1), (a2, _b2) in zip(nucleos, nucleos[1:]):
        cons = w[b1 + 1:a2]
        k = len(cons)
        if k == 0:
            corte = a2
        elif k == 1:
            corte = a2 - 1
        elif k == 2:
            corte = a2 - 2 if cons in _INSEP else a2 - 1
        elif k == 3:
            corte = a2 - 2 if cons[1:] in _INSEP else a2 - 1
        else:
            corte = a2 - 2 if cons[2:] in _INSEP else a2 - 2
        if w[corte - 1:corte + 1] in ("ch", "ll", "rr") and corte - 1 > b1:
            corte -= 1
        sil_lim.append(corte)
    out, prev = [], 0
    for c in sil_lim:
        out.append(palabra[prev:c])
        prev = c
    out.append(palabra[prev:])
    return [s for s in out if s]


def _silabas_en(palabra: str) -> int:
    try:
        import pyphen
        return max(1, len(pyphen.Pyphen(lang="en_US").inserted(palabra).split("-")))
    except Exception:                                          # noqa: BLE001
        w = palabra.lower()
        n = len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and not w.endswith("le") else 0)
        return max(1, n)


def silabas(texto: str, idioma: str | None = None) -> dict:
    lang = _lang(texto, idioma)
    out = []
    for w, _, _ in _palabras(texto):
        if lang == "es":
            out.append({"palabra": w, "silabas": silabas_es(w)})
        else:
            try:
                import pyphen
                parts = pyphen.Pyphen(lang="en_US").inserted(w).split("-")
            except Exception:                                  # noqa: BLE001
                parts = [w]
            out.append({"palabra": w, "silabas": parts, "cuenta": _silabas_en(w)})
    return {"idioma": lang, "fuente": "reglas propias" if lang == "es" else "pyphen", "palabras": out}


def legibilidad(texto: str, idioma: str | None = None) -> dict:
    from hope.runtime.escritura import oraciones
    lang = _lang(texto, idioma)
    ws = [w for w, _, _ in _palabras(texto)]
    n_or = max(len(oraciones(texto)), 1)
    if not ws:
        return {"idioma": lang, "palabras": 0}
    syl = sum(len(silabas_es(w)) if lang == "es" else _silabas_en(w) for w in ws)
    p, s = len(ws), syl
    if lang == "es":
        fh = 206.84 - 0.60 * (100 * s / p) - 1.02 * (100 * n_or / p)
        infl = 206.835 - 62.3 * s / p - p / n_or
        nivel = ("muy difícil" if infl < 40 else "algo difícil" if infl < 55 else "normal" if infl < 65 else
                 "bastante fácil" if infl < 80 else "muy fácil")
        return {"idioma": "es", "fernandez_huerta": round(fh, 1), "inflesz": round(infl, 1), "nivel": nivel,
                "palabras": p, "oraciones": n_or, "silabas_por_palabra": round(s / p, 2)}
    fre = 206.835 - 1.015 * p / n_or - 84.6 * s / p
    fk = 0.39 * p / n_or + 11.8 * s / p - 15.59
    return {"idioma": "en", "flesch": round(fre, 1), "flesch_kincaid_grado": round(fk, 1), "palabras": p,
            "oraciones": n_or, "silabas_por_palabra": round(s / p, 2)}


# ---------------------------------------------------------------------------
# Repeticiones, fluidez, coherencia, frecuencia
# ---------------------------------------------------------------------------
def repeticiones(texto: str, idioma: str | None = None, ventana: int = 40) -> dict:
    from collections import Counter
    from hope.runtime.escritura import VACIAS, ContextoRevision, Repeticiones, tasa_repeticion
    ms = Repeticiones().revisar(texto, ContextoRevision())
    toks = [(w.lower(), a) for w, a, _ in _palabras(texto)]
    cerca = []
    for i, (w, a) in enumerate(toks):
        if w in VACIAS or len(w) < 5:
            continue
        for v, b in toks[i + 1:i + 1 + ventana // 4]:
            if v == w:
                cerca.append({"palabra": w, "inicio": a, "otra": b})
                break
    return {"rep4": round(tasa_repeticion(texto, 4), 3), "bucles": [m.detalle for m in ms if m.severidad >= 2],
            "ngramas_repetidos": [m.detalle for m in ms if m.severidad < 2][:20],
            "palabras_cerca": cerca[:20],
            "mas_frecuentes": Counter(w for w, _ in toks if w not in VACIAS and len(w) > 3).most_common(5)}


def fluidez(texto: str, idioma: str | None = None) -> dict:
    from hope.runtime.escritura import oraciones
    lang = _lang(texto, idioma)
    km = os.environ.get("HOPE_KENLM")
    if km and os.path.exists(km):
        try:
            import kenlm
            lm = kenlm.Model(km)
            ors = [(texto[a:b], lm.perplexity(texto[a:b])) for a, b in oraciones(texto)]
            return {"idioma": lang, "fuente": "KenLM", "oraciones": [{"texto": t, "perplejidad": round(p, 1)}
                                                                     for t, p in ors]}
        except Exception:                                      # noqa: BLE001
            pass
    ng = _ngramas(lang)
    if ng is None:
        return {"idioma": lang, "fuente": "no disponible (no hay corpus local)", "oraciones": []}
    ors = [{"texto": texto[a:b], "bits_por_byte": round(ng.bits_por_byte(texto[a:b]), 2)} for a, b in oraciones(texto)]
    total = ng.bits_por_byte(texto) if texto else 0.0
    return {"idioma": lang, "fuente": "5-gramas de bytes del corpus local", "bits_por_byte": round(total, 3),
            "oraciones": ors, "peor": max(ors, key=lambda o: o["bits_por_byte"])["texto"] if ors else ""}


def coherencia(texto: str, idioma: str | None = None) -> dict:
    from hope.runtime.escritura import VACIAS, oraciones, palabras
    from hope.runtime.lengua import contradicciones
    lang = _lang(texto, idioma)
    sents = [texto[a:b] for a, b in oraciones(texto)]
    nli = os.environ.get("HOPE_NLI")
    pares = []
    if nli:
        try:
            from transformers import pipeline
            clf = pipeline("text-classification", model=nli, local_files_only=True)
            for a, b in zip(sents, sents[1:]):
                r = clf({"text": a, "text_pair": b})[0] if isinstance(clf({"text": a, "text_pair": b}), list) else \
                    clf({"text": a, "text_pair": b})
                if "contra" in r["label"].lower():
                    pares.append({"a": a, "b": b, "puntaje": round(r["score"], 3)})
            fuente = f"NLI local ({nli})"
        except Exception:                                      # noqa: BLE001
            nli = None
    if not nli:
        fuente = "contradicciones explícitas + cohesión léxica"
        pares = [{"detalle": m.detalle} for m in contradicciones(texto)]
    cont = [{w[:5] for w in palabras(s) if w not in VACIAS and len(w) > 3} for s in sents]
    coh = [len(a & b) / max(min(len(a), len(b)), 1) for a, b in zip(cont, cont[1:])]
    sueltas = [sents[i + 1] for i, c in enumerate(coh) if c == 0 and len(cont[i + 1]) >= 4]
    return {"idioma": lang, "fuente": fuente, "contradicciones": pares,
            "cohesion_media": round(sum(coh) / len(coh), 3) if coh else None,
            "oraciones_sin_relacion": sueltas[:5]}


def frecuencia(texto: str, idioma: str | None = None) -> dict:
    lang = _lang(texto, idioma)
    ws = list(dict.fromkeys(w.lower() for w, _, _ in _palabras(texto)))[:50]
    if _zipf("casa", "es"):
        return {"idioma": lang, "fuente": "wordfreq (Zipf: 7 = muy común, 3 = rara, 0 = no registrada)",
                "palabras": {w: _zipf(w, lang) for w in ws}}
    voc = _corpus_vocab()
    return {"idioma": lang, "fuente": "corpus local (apariciones)", "palabras": {w: voc.get(w, 0) for w in ws}}


# ---------------------------------------------------------------------------
# Conjugación (español con voseo; inglés)
# ---------------------------------------------------------------------------
PERSONAS = ("yo", "tú", "vos", "él/ella/usted", "nosotros", "vosotros", "ellos/ustedes")
_TERM = {
    "ar": {"presente": ["o", "as", "ás", "a", "amos", "áis", "an"],
           "preterito": ["é", "aste", "aste", "ó", "amos", "asteis", "aron"],
           "imperfecto": ["aba", "abas", "abas", "aba", "ábamos", "abais", "aban"],
           "subjuntivo": ["e", "es", "es", "e", "emos", "éis", "en"]},
    "er": {"presente": ["o", "es", "és", "e", "emos", "éis", "en"],
           "preterito": ["í", "iste", "iste", "ió", "imos", "isteis", "ieron"],
           "imperfecto": ["ía", "ías", "ías", "ía", "íamos", "íais", "ían"],
           "subjuntivo": ["a", "as", "as", "a", "amos", "áis", "an"]},
    "ir": {"presente": ["o", "es", "ís", "e", "imos", "ís", "en"],
           "preterito": ["í", "iste", "iste", "ió", "imos", "isteis", "ieron"],
           "imperfecto": ["ía", "ías", "ías", "ía", "íamos", "íais", "ían"],
           "subjuntivo": ["a", "as", "as", "a", "amos", "áis", "an"]},
}
_FUT = ["é", "ás", "ás", "á", "emos", "éis", "án"]
_COND = ["ía", "ías", "ías", "ía", "íamos", "íais", "ían"]
# Irregulares frecuentes: formas completas por tiempo (orden de PERSONAS) y raíz de futuro.
_IRR = {
    "ser": {"presente": "soy eres sos es somos sois son", "preterito": "fui fuiste fuiste fue fuimos fuisteis fueron",
            "imperfecto": "era eras eras era éramos erais eran", "subjuntivo": "sea seas seas sea seamos seáis sean",
            "futuro_raiz": "ser", "imperativo": "sé sé"},
    "estar": {"presente": "estoy estás estás está estamos estáis están",
              "preterito": "estuve estuviste estuviste estuvo estuvimos estuvisteis estuvieron",
              "subjuntivo": "esté estés estés esté estemos estéis estén", "imperativo": "está está"},
    "ir": {"presente": "voy vas vas va vamos vais van", "preterito": "fui fuiste fuiste fue fuimos fuisteis fueron",
           "imperfecto": "iba ibas ibas iba íbamos ibais iban", "subjuntivo": "vaya vayas vayas vaya vayamos vayáis vayan",
           "imperativo": "ve andá"},
    "haber": {"presente": "he has has ha hemos habéis han",
              "preterito": "hube hubiste hubiste hubo hubimos hubisteis hubieron",
              "subjuntivo": "haya hayas hayas haya hayamos hayáis hayan", "futuro_raiz": "habr"},
    "tener": {"presente": "tengo tienes tenés tiene tenemos tenéis tienen",
              "preterito": "tuve tuviste tuviste tuvo tuvimos tuvisteis tuvieron",
              "subjuntivo": "tenga tengas tengas tenga tengamos tengáis tengan", "futuro_raiz": "tendr",
              "imperativo": "ten tené"},
    "hacer": {"presente": "hago haces hacés hace hacemos hacéis hacen",
              "preterito": "hice hiciste hiciste hizo hicimos hicisteis hicieron",
              "subjuntivo": "haga hagas hagas haga hagamos hagáis hagan", "futuro_raiz": "har",
              "imperativo": "haz hacé"},
    "poder": {"presente": "puedo puedes podés puede podemos podéis pueden",
              "preterito": "pude pudiste pudiste pudo pudimos pudisteis pudieron",
              "subjuntivo": "pueda puedas puedas pueda podamos podáis puedan", "futuro_raiz": "podr"},
    "decir": {"presente": "digo dices decís dice decimos decís dicen",
              "preterito": "dije dijiste dijiste dijo dijimos dijisteis dijeron",
              "subjuntivo": "diga digas digas diga digamos digáis digan", "futuro_raiz": "dir",
              "imperativo": "di decí"},
    "venir": {"presente": "vengo vienes venís viene venimos venís vienen",
              "preterito": "vine viniste viniste vino vinimos vinisteis vinieron",
              "subjuntivo": "venga vengas vengas venga vengamos vengáis vengan", "futuro_raiz": "vendr",
              "imperativo": "ven vení"},
    "querer": {"presente": "quiero quieres querés quiere queremos queréis quieren",
               "preterito": "quise quisiste quisiste quiso quisimos quisisteis quisieron",
               "subjuntivo": "quiera quieras quieras quiera queramos queráis quieran", "futuro_raiz": "querr"},
    "saber": {"presente": "sé sabes sabés sabe sabemos sabéis saben",
              "preterito": "supe supiste supiste supo supimos supisteis supieron",
              "subjuntivo": "sepa sepas sepas sepa sepamos sepáis sepan", "futuro_raiz": "sabr"},
    "poner": {"presente": "pongo pones ponés pone ponemos ponéis ponen",
              "preterito": "puse pusiste pusiste puso pusimos pusisteis pusieron",
              "subjuntivo": "ponga pongas pongas ponga pongamos pongáis pongan", "futuro_raiz": "pondr",
              "imperativo": "pon poné"},
    "salir": {"presente": "salgo sales salís sale salimos salís salen",
              "subjuntivo": "salga salgas salgas salga salgamos salgáis salgan", "futuro_raiz": "saldr",
              "imperativo": "sal salí"},
    "dar": {"presente": "doy das das da damos dais dan", "preterito": "di diste diste dio dimos disteis dieron",
            "subjuntivo": "dé des des dé demos deis den"},
    "ver": {"presente": "veo ves ves ve vemos veis ven", "imperfecto": "veía veías veías veía veíamos veíais veían",
            "preterito": "vi viste viste vio vimos visteis vieron", "subjuntivo": "vea veas veas vea veamos veáis vean"},
    "pensar": {"presente": "pienso piensas pensás piensa pensamos pensáis piensan",
               "subjuntivo": "piense pienses pienses piense pensemos penséis piensen", "imperativo": "piensa pensá"},
    "dormir": {"presente": "duermo duermes dormís duerme dormimos dormís duermen",
               "subjuntivo": "duerma duermas duermas duerma durmamos durmáis duerman", "imperativo": "duerme dormí"},
    "pedir": {"presente": "pido pides pedís pide pedimos pedís piden",
              "subjuntivo": "pida pidas pidas pida pidamos pidáis pidan", "imperativo": "pide pedí"},
    "jugar": {"presente": "juego juegas jugás juega jugamos jugáis juegan", "imperativo": "juega jugá"},
    "volver": {"presente": "vuelvo vuelves volvés vuelve volvemos volvéis vuelven", "imperativo": "vuelve volvé"},
}
TIEMPOS = ("presente", "preterito", "imperfecto", "futuro", "condicional", "subjuntivo", "imperativo")


def conjugar_es(verbo: str, tiempo: str = "presente") -> dict[str, str]:
    """Conjugación del verbo en el tiempo pedido, con las personas de ``PERSONAS`` (incluye vos).
    ``imperativo`` da solo tú y vos."""
    v = verbo.lower().strip()
    if v[-2:] not in _TERM:
        raise ValueError(f"no es un infinitivo en -ar, -er o -ir: {verbo}")
    raiz, grupo = v[:-2], v[-2:]
    irr = _IRR.get(v, {})
    if tiempo == "imperativo":
        if "imperativo" in irr:
            tu, vos = irr["imperativo"].split()
        else:
            tu = raiz + {"ar": "a", "er": "e", "ir": "e"}[grupo]
            vos = raiz + {"ar": "á", "er": "é", "ir": "í"}[grupo]
        return {"tú": tu, "vos": vos}
    if tiempo in ("futuro", "condicional"):
        base = irr.get("futuro_raiz", v)
        return dict(zip(PERSONAS, (base + t for t in (_FUT if tiempo == "futuro" else _COND))))
    if tiempo not in _TERM[grupo]:
        raise ValueError(f"tiempo desconocido: {tiempo} (opciones: {', '.join(TIEMPOS)})")
    if tiempo in irr:
        return dict(zip(PERSONAS, irr[tiempo].split()))
    formas = [raiz + t for t in _TERM[grupo][tiempo]]
    # cambios ortográficos frecuentes (-car/-gar/-zar en pretérito «yo» y subjuntivo)
    if grupo == "ar" and tiempo in ("preterito", "subjuntivo"):
        for suf, rep in (("c", "qu"), ("g", "gu"), ("z", "c")):
            if raiz.endswith(suf):
                r2 = raiz[:-1] + rep
                formas = [r2 + t if (tiempo == "subjuntivo" or i == 0) else f for i, (f, t) in
                          enumerate(zip(formas, _TERM[grupo][tiempo]))]
                break
    return dict(zip(PERSONAS, formas))


_EN_IRR = {"be": ("is", "was", "been"), "have": ("has", "had", "had"), "do": ("does", "did", "done"),
           "go": ("goes", "went", "gone"), "say": ("says", "said", "said"), "make": ("makes", "made", "made"),
           "take": ("takes", "took", "taken"), "come": ("comes", "came", "come"), "see": ("sees", "saw", "seen"),
           "know": ("knows", "knew", "known"), "get": ("gets", "got", "gotten"), "give": ("gives", "gave", "given"),
           "find": ("finds", "found", "found"), "think": ("thinks", "thought", "thought"),
           "tell": ("tells", "told", "told"), "become": ("becomes", "became", "become"),
           "leave": ("leaves", "left", "left"), "feel": ("feels", "felt", "felt"), "bring": ("brings", "brought",
                                                                                          "brought"),
           "begin": ("begins", "began", "begun"), "keep": ("keeps", "kept", "kept"), "write": ("writes", "wrote",
                                                                                            "written"),
           "run": ("runs", "ran", "run"), "eat": ("eats", "ate", "eaten"), "buy": ("buys", "bought", "bought")}


def conjugar_en(verb: str) -> dict[str, str]:
    v = verb.lower().strip()
    if v in _EN_IRR:
        s3, past, part = _EN_IRR[v]
    else:
        s3 = v + "es" if re.search(r"(s|x|z|ch|sh)$", v) else v[:-1] + "ies" if re.search(r"[^aeiou]y$", v) else v + "s"
        if v.endswith("e"):
            past = v + "d"
        elif re.search(r"[^aeiou]y$", v):
            past = v[:-1] + "ied"
        elif re.fullmatch(r"[^aeiou]*[aeiou][^aeiouwxy]", v):
            past = v + v[-1] + "ed"
        else:
            past = v + "ed"
        part = past
    ger = (v[:-1] + "ing" if v.endswith("e") and not v.endswith("ee") else
           v + v[-1] + "ing" if re.fullmatch(r"[^aeiou]*[aeiou][^aeiouwxy]", v) else v + "ing")
    if v == "be":
        return {"base": "be", "I": "am", "you/we/they": "are", "he/she/it": "is", "past": "was/were",
                "participle": "been", "gerund": "being"}
    return {"base": v, "he/she/it": s3, "past": past, "participle": part, "gerund": ger}


def conjugar(texto: str) -> dict:
    """``verbo [tiempo]`` en español (tiempos: presente, pretérito, imperfecto, futuro, condicional,
    subjuntivo, imperativo) o un verbo en inglés (``en: verb``)."""
    t = texto.strip()
    if t.lower().startswith("en:"):
        return {"idioma": "en", "formas": conjugar_en(t[3:].strip())}
    partes = t.split()
    verbo = partes[0]
    tiempo = _sin_tildes(partes[1].lower()) if len(partes) > 1 else "presente"
    if verbo.lower()[-2:] not in _TERM:
        return {"idioma": "en", "formas": conjugar_en(verbo)}
    return {"idioma": "es", "verbo": verbo, "tiempo": tiempo, "formas": conjugar_es(verbo, tiempo)}


# ---------------------------------------------------------------------------
# Diccionario, sinónimos, paráfrasis (dependen de datos locales)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _wiktionary() -> dict:
    p = os.environ.get("HOPE_WIKTIONARY")
    if not p or not os.path.exists(p):
        return {}
    import json
    d: dict = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            w = r.get("word")
            glosses = [g for s in r.get("senses", []) for g in s.get("glosses", [])][:3]
            if w and glosses:
                d.setdefault((w.lower(), r.get("lang_code", "")), []).append(
                    {"categoria": r.get("pos"), "definiciones": glosses})
    return d


def diccionario(texto: str, idioma: str | None = None) -> dict:
    w = texto.strip().split()[0] if texto.strip() else ""
    lang = idioma if idioma in ("es", "en") else _lang(w, None)
    wk = _wiktionary()
    out = {"palabra": w, "idioma": lang}
    if wk:
        out.update(fuente="Wiktionary (volcado local, CC BY-SA)", entradas=wk.get((w.lower(), lang), []))
    else:
        out["fuente"] = "sin volcado de Wiktionary (definir HOPE_WIKTIONARY con un JSONL de kaikki.org)"
    from hope.runtime.lengua import nlp
    n = nlp(lang)
    if n is not None and w:
        t = n(w)[0]
        out["lema"], out["categoria_gramatical"], out["rasgos"] = t.lemma_, t.pos_, str(t.morph)
    z = _zipf(w.lower(), lang)
    if z is not None:
        out["frecuencia_zipf"] = z
    return out


def sinonimos(texto: str, idioma: str | None = None) -> dict:
    w = texto.strip().split()[0] if texto.strip() else ""
    lang = idioma if idioma in ("es", "en") else _lang(w, None)
    try:
        from nltk.corpus import wordnet as wn
        code = "spa" if lang == "es" else "eng"
        lem = {l.replace("_", " ") for s in wn.synsets(w, lang=code) for l in s.lemma_names(code)} - {w}
        return {"palabra": w, "fuente": "Open Multilingual Wordnet (NLTK)", "sinonimos": sorted(lem)[:20]}
    except Exception as e:                                     # noqa: BLE001
        return {"palabra": w, "fuente": f"no disponible (instalar nltk y los datos «wordnet» y «omw-1.4»: {type(e).__name__})",
                "sinonimos": []}


def parafrasear(texto: str, idioma: str | None = None) -> dict:
    lang = _lang(texto, idioma)
    try:
        import argostranslate.translate as at
        pivote = "en" if lang == "es" else "es"
        ida = at.translate(texto, lang, pivote)
        return {"fuente": f"Argos Translate local ({lang}→{pivote}→{lang})", "parafrasis": at.translate(ida, pivote, lang)}
    except Exception as e:                                     # noqa: BLE001
        return {"fuente": f"no disponible (instalar argostranslate y los paquetes es↔en: {type(e).__name__})",
                "parafrasis": ""}


# ---------------------------------------------------------------------------
# Evaluación combinada (filtro de calidad, crítica, recompensa, explicaciones del maestro)
# ---------------------------------------------------------------------------
def evaluar_lengua(texto: str, idioma: str | None = None, *, gramatica_completa: bool = True) -> dict:
    """Errores de lengua cada 100 palabras (ortografía, gramática/concordancia, tildes y puntuación,
    bucles) y un ``puntaje`` en [0, 1] (1 = sin errores; baja 0,1 por cada error cada 100 palabras).
    Es lo que usan ``quality.rules(lengua=…)``, ``critique(lengua=True)``, ``verified_reward(lengua_peso=…)``
    y el filtro de explicaciones del tutor."""
    lang = _lang(texto, idioma)
    n = max(len(_palabras(texto)), 1)
    orto = ortografia(texto, lang)["errores"]
    gram = gramatica(texto, lang)["errores"] if gramatica_completa else []
    tp = tildes_puntuacion(texto, lang)["problemas"]
    tp = [p for p in tp if not p["detalle"].startswith("tilde:")]          # ya cuentan en ortografía
    rep = repeticiones(texto, lang)
    cuentas = {"ortografia": len(orto), "gramatica": len(gram), "puntuacion": len(tp), "bucles": len(rep["bucles"])}
    por100 = 100 * sum(cuentas.values()) / n
    detalles = ([f"ortografía: {e['palabra']}" for e in orto[:3]] + [f"gramática: {e['detalle']}" for e in gram[:3]]
                + [p["detalle"] for p in tp[:2]] + [f"repetición: {b}" for b in rep["bucles"][:1]])
    return {"idioma": lang, "palabras": n, "cuentas": cuentas, "errores_cada_100": round(por100, 2),
            "puntaje": round(max(0.0, 1.0 - 0.1 * por100), 3), "detalles": detalles}


class VerificadorLengua:
    """Una herramienta de lengua como ``Verificador`` del bucle de escritura (``hope/runtime/escritura.py``):
    ``ortografia``, ``gramatica`` o ``tildes_puntuacion`` marcan los tramos con error."""

    def __init__(self, herramienta: str = "ortografia", idioma: str | None = None, severidad: float = 1.0):
        self.nombre, self.idioma, self.sev = herramienta, idioma, severidad

    def revisar(self, texto: str, ctx=None) -> list:
        from hope.runtime.escritura import Marca
        r = FUNCIONES[self.nombre](texto, self.idioma)
        items = r.get("errores", r.get("problemas", []))
        out = []
        for e in items:
            det = e.get("detalle") or f"«{e.get('palabra')}» → {'/'.join(e.get('sugerencias', [])) or '?'}"
            out.append(Marca(int(e["inicio"]), int(e["fin"]), self.nombre, det, self.sev, self.nombre))
        return out


def verificadores_lengua(idioma: str | None = None) -> list:
    return [VerificadorLengua("ortografia", idioma, 0.7), VerificadorLengua("gramatica", idioma, 1.0),
            VerificadorLengua("tildes_puntuacion", idioma, 0.5)]


# ---------------------------------------------------------------------------
# Catálogo y llamadas ⟦nombre: texto⟧
# ---------------------------------------------------------------------------
FUNCIONES = {
    "ortografia": ortografia, "gramatica": gramatica, "diccionario": diccionario, "sinonimos": sinonimos,
    "conjugar": conjugar, "analizar": analizar, "idioma": idioma, "legibilidad": legibilidad,
    "repeticiones": repeticiones, "fluidez": fluidez, "coherencia": coherencia, "variante": variante,
    "tildes_puntuacion": tildes_puntuacion, "frecuencia": frecuencia, "concordancias": concordancias,
    "entidades": entidades, "registro": registro, "parafrasear": parafrasear, "silabas": silabas,
}
HELP = {
    "ortografia": "ortografía es/en: ⟦ortografia: Esto es un exemplo⟧",
    "gramatica": "gramática (LanguageTool local o reglas + concordancia): ⟦gramatica: Las casa es lindo⟧",
    "diccionario": "lema, categoría y definición (Wiktionary local): ⟦diccionario: ferrocarril⟧",
    "sinonimos": "sinónimos (Open Multilingual Wordnet): ⟦sinonimos: rápido⟧",
    "conjugar": "conjugar con voseo: ⟦conjugar: tener presente⟧, ⟦conjugar: cantar imperativo⟧, ⟦conjugar: en: write⟧",
    "analizar": "análisis por palabra (lema, función, rasgos): ⟦analizar: Las casas blancas⟧",
    "idioma": "idioma del texto: ⟦idioma: The river is long⟧",
    "legibilidad": "legibilidad (INFLESZ / Flesch): ⟦legibilidad: texto⟧",
    "repeticiones": "repeticiones y bucles: ⟦repeticiones: texto⟧",
    "fluidez": "fluidez por oración (n-gramas del corpus): ⟦fluidez: texto⟧",
    "coherencia": "contradicciones y cohesión: ⟦coherencia: texto⟧",
    "variante": "variante (es-AR, neutro, en-US, en-GB): ⟦variante: Vos tenés razón⟧",
    "tildes_puntuacion": "tildes y puntuación: ⟦tildes_puntuacion: Como estas?⟧",
    "frecuencia": "frecuencia de uso de cada palabra: ⟦frecuencia: ferrocarril tren⟧",
    "concordancias": "concordancia de género, número y persona: ⟦concordancias: Los trenes llega⟧",
    "entidades": "personas, lugares y organizaciones: ⟦entidades: Rosario está en Santa Fe⟧",
    "registro": "registro formal/informal: ⟦registro: che, mirá esto⟧",
    "parafrasear": "parafrasear (Argos local): ⟦parafrasear: texto⟧",
    "silabas": "sílabas: ⟦silabas: ferrocarril⟧",
}


def resumen(nombre: str, r: dict) -> str:
    """Texto corto para insertar en la generación después de ``⟦nombre: …⟧ →``."""
    if nombre in ("ortografia",):
        es = r["errores"]
        return "sin errores" if not es else "; ".join(
            f"{e['palabra']}→{'/'.join(e['sugerencias']) or '?'}" for e in es[:6])
    if nombre in ("gramatica", "concordancias"):
        es = r["errores"]
        return "sin errores" if not es else "; ".join(e["detalle"] for e in es[:5])
    if nombre == "tildes_puntuacion":
        ps = r["problemas"]
        return "sin problemas" if not ps else "; ".join(p["detalle"] for p in ps[:5])
    if nombre == "conjugar":
        return ", ".join(f"{k}: {v}" for k, v in r["formas"].items())
    if nombre == "silabas":
        return " · ".join("-".join(p["silabas"]) for p in r["palabras"][:12])
    if nombre == "legibilidad":
        return ", ".join(f"{k} {v}" for k, v in r.items() if k != "idioma")
    if nombre == "repeticiones":
        return f"rep-4 {r['rep4']}" + (f"; bucles: {'; '.join(r['bucles'][:2])}" if r["bucles"] else "")
    if nombre == "fluidez":
        return f"{r.get('bits_por_byte', '?')} bits/byte; peor: «{r.get('peor', '')[:80]}»"
    if nombre == "analizar":
        return r["analisis"][:500]
    import json
    return json.dumps(r, ensure_ascii=False, default=str)[:500]


def register_lenguaje(cat: Catalog) -> None:
    for nombre, fn in FUNCIONES.items():
        cat.register(ToolSpec(
            f"hte_lengua_{nombre}_v1", nombre.replace("_", " ").capitalize(), "1.0", "lengua",
            HELP[nombre].split(":")[0] + " español inglés texto", HELP[nombre], {"texto": "str"}, {"resultado": "dict"},
            dependencies=["opcionales: pyspellchecker, language_tool_python, spacy, pyphen, wordfreq, nltk, "
                          "argostranslate, kenlm, transformers"],
            limitations=["sin red: diccionarios y modelos instalados de antes", "el resultado dice qué fuente usó"],
            permissions=["tool.execute"], tests=["tests/test_herramientas_lengua.py"], usage_examples=[HELP[nombre]],
            provenance="herramientas de lengua (hope/tools/lenguaje.py)"),
            lambda texto="", _fn=fn, **kw: _fn(texto, **kw) if kw else _fn(texto))


def dispatch(runner, nombre: str, arg: str) -> str:
    r = runner.catalog.call(f"hte_lengua_{nombre}_v1", texto=arg)
    return resumen(nombre, r)
