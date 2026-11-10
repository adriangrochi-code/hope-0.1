"""Herramientas de lengua es/en (hope/tools/lenguaje.py), ejercicios y conexiones (sin red)."""

import pytest

from hope.tools import lenguaje as L

SPACY = pytest.mark.skipif(__import__("hope.runtime.lengua", fromlist=["x"]).nlp("es") is None,
                           reason="spaCy no instalado")


def test_conjugacion_con_voseo_y_en_ingles():
    assert L.conjugar_es("tener")["vos"] == "tenés" and L.conjugar_es("tener")["tú"] == "tienes"
    assert L.conjugar_es("cantar", "imperativo") == {"tú": "canta", "vos": "cantá"}
    assert L.conjugar_es("decir", "imperativo")["vos"] == "decí"
    assert L.conjugar_es("buscar", "preterito")["yo"] == "busqué"
    assert L.conjugar_es("vivir", "futuro")["nosotros"] == "viviremos"
    assert L.conjugar_es("poner", "condicional")["yo"] == "pondría"
    assert L.conjugar_es("ser")["vos"] == "sos"
    assert L.conjugar_en("write")["past"] == "wrote" and L.conjugar_en("stop")["past"] == "stopped"
    assert L.conjugar_en("study")["he/she/it"] == "studies"


def test_silabas_y_legibilidad():
    assert L.silabas_es("ferrocarril") == ["fe", "rro", "ca", "rril"]
    assert L.silabas_es("construcción") == ["cons", "truc", "ción"]
    assert L.silabas_es("aéreo") == ["a", "é", "re", "o"]
    assert L.silabas_es("leía") == ["le", "í", "a"]
    assert L.silabas_es("hablar") == ["ha", "blar"]
    fac = L.legibilidad("El río es largo. Tiene agua.", "es")["inflesz"]
    dif = L.legibilidad("La consolidación institucional contemporánea presupone transformaciones "
                        "administrativas extraordinariamente complejas.", "es")["inflesz"]
    assert fac > dif
    assert "flesch" in L.legibilidad("The river is long. It has water.", "en")


def test_variante_registro_idioma_repeticiones():
    assert L.variante("Vos tenés razón.")["variante"] == "es-AR"
    assert L.variante("Tú tienes razón.")["variante"].startswith("neutro")
    assert L.variante("The colour of the centre.", "en")["variante"] == "en-GB"
    assert L.variante("The color of the center.", "en")["variante"] == "en-US"
    assert L.registro("che, mirá esto, está re copado")["registro"] == "informal"
    assert L.registro("I'm gonna do it, it's cool!", "en")["registro"] == "informal"
    assert L.idioma("The river is long and the city is old")["idioma"] == "en"
    r = L.repeticiones("de la ciudad de la ciudad de la ciudad de la ciudad")
    assert r["rep4"] > 0.5 and r["bucles"]


def test_tildes_puntuacion_y_coherencia():
    ps = [p["detalle"] for p in L.tildes_puntuacion("Como estas? Bien ,gracias. que bueno", "es")["problemas"]]
    assert any("¿" in p for p in ps) and any("espacio antes" in p for p in ps) and any("minúscula" in p for p in ps)
    c = L.coherencia("Marte tiene 2 lunas. Marte tiene 3 lunas.")
    assert c["contradicciones"]


def test_ortografia_con_o_sin_corrector():
    r = L.ortografia("Esto es un exemplo con informacion del arbol. Él está acá.", "es")
    malas = {e["palabra"] for e in r["errores"]}
    if r["fuente"] == "pyspellchecker":
        assert {"exemplo", "informacion", "arbol"} <= malas and "está" not in malas
        assert any(e["tilde"] for e in r["errores"] if e["palabra"] == "arbol")
    else:
        assert "fuente" in r                                 # respaldo declarado


@SPACY
def test_gramatica_y_concordancias():
    g = L.gramatica("Fui a el cine y y volví. Las casa es lindo.", "es")
    det = " ".join(e["detalle"] for e in g["errores"])
    assert "al" in det and ("repetida" in det) and g["fuente"]
    assert L.concordancias("Los trenes llega tarde.")["errores"]
    assert not L.concordancias("Los trenes llegan tarde.")["errores"]
    assert L.gramatica("I could of gone. She bought a apple.", "en")["errores"]


def test_catalogo_y_llamadas_en_el_texto():
    from hope.runtime.toolcalls import HELP, ToolRunner
    from hope.tools.builtin import default_catalog
    r = ToolRunner(default_catalog())
    assert set(L.FUNCIONES) <= set(HELP)
    assert "tenés" in r.run("conjugar", "tener presente")
    assert "fe-rro-ca-rril" in r.run("silabas", "ferrocarril")
    for n in ("diccionario", "sinonimos", "parafrasear", "frecuencia", "fluidez", "entidades", "analizar"):
        assert isinstance(r.run(n, "ferrocarril"), str)        # sin dependencia: lo dice, no se rompe


def test_evaluar_lengua_y_conexiones():
    bueno = L.evaluar_lengua("El puerto de Rosario está sobre el río Paraná y exporta soja.")
    malo = L.evaluar_lengua("de la ciudad de la ciudad de la ciudad de la ciudad. informacion del arbol")
    assert bueno["puntaje"] > malo["puntaje"]
    from hope.data.quality import rules
    texto = "La informacion del arbol es util. El exemplo del camion es facil y muy rapido para todos."
    assert rules(texto, min_chars=10).keep                             # sin «lengua»: como siempre
    r = rules(texto, min_chars=10, lengua=2.0)
    assert not r.keep and any("lengua" in x for x in r.reasons)
    assert rules("El puerto de Rosario está sobre el río Paraná y exporta soja.", min_chars=10, lengua=2.0).keep
    from hope.constraints.critic import critique
    c = critique("¿Cuánto es 2 + 2?", "Paso 1: 2 + 2 = 4\nResultado: 4 informacion", "4", lengua=True)
    assert any(i.kind == "lengua" for i in c.issues) and c.verdict != "errores"
    from hope.train.rlvr import default_extract, verified_reward
    p = {"problema": "¿Cuánto es 2 + 2?", "respuesta": "4"}
    eq = lambda a, b: a == b                                   # noqa: E731
    r0 = verified_reward(p, "Paso 1: 2 + 2 = 4\nResultado: 4", default_extract, eq)
    r1 = verified_reward(p, "Paso 1: 2 + 2 = 4 la informacion del arbol\nResultado: 4", default_extract, eq,
                         lengua_peso=0.5)
    assert r1 < r0
    from hope.teacher.tutor import TutorConfig
    assert TutorConfig().filtro_lengua is None
    vs = L.verificadores_lengua("es")
    from hope.runtime.escritura import revisar
    assert revisar("La informacion del arbol.", vs)


def test_ejercicios_de_lengua_con_respuesta_exacta():
    from hope.data.ejercicios_lengua import FAMILIAS_LENGUA, respuesta_ok
    from hope.data.generators import FAMILIES, generate
    assert set(FAMILIAS_LENGUA) <= set(FAMILIES)
    for fam in FAMILIAS_LENGUA:
        for lang in ("es", "en"):
            for e in generate(fam, 5, seed=3, level=3, lang=lang):
                d = e.data
                assert d["problema"] and d["resultado"]
                assert respuesta_ok(d["resultado"], d["resultado"], d.get("tipo_lengua", ""))
    vos = [e.data for e in generate("conjugacion", 40, seed=1, level=2) if "«vos»" in e.data["problema"]]
    assert vos
    for d in generate("corregir", 10, seed=0):
        assert d.data["problema"].split("«")[1].rstrip("»") != d.data["resultado"]
    assert respuesta_ok("Resultado: Tenés", "tenés", "conjugacion")
    assert not respuesta_ok("¿como estás?", "¿Cómo estás?", "puntuacion")
    from hope.train.rlvr import problems_from_generators
    assert problems_from_generators(["tildes", "plurales"], 2)
