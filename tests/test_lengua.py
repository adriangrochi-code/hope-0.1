"""Lenguaje con el enfoque de código: requisitos, mutaciones, concordancia, diffs, FIM y trazas."""

import pytest
import torch

from hope.core.hn1 import HN1Model, preset
from hope.runtime.escritura import Contenido, Escritor, Repeticiones
from hope.runtime.lengua import (FIM_FIN, FIM_MED, FIM_PRE, FIM_SUF, Concordancia, Edicion, VerificadorRequisitos,
                                 aplicar_ediciones, contradicciones, datos_trazas, diff_texto, edicion_unica,
                                 ejemplos_fim, escribir_datos, extraer_requisitos, formato_fim, marcadores_variante,
                                 nlp, preservado, puntaje_mutaciones, rellenar, traza, validar)

PEDIDO = ("Escribí en español neutro, en unas 40 palabras, sobre el puerto de Rosario. Mencioná el río Paraná, "
          "la soja y el ferrocarril. Usá el término «terminal portuaria».")
BUENO = ("El puerto de Rosario está sobre el río Paraná y es uno de los más activos del país. La terminal portuaria "
         "exporta soja y otros granos que llegan en camiones y por ferrocarril desde la región pampeana. Rosario "
         "tiene 5 terminales privadas a lo largo de la costa.")
SPACY = pytest.mark.skipif(nlp("es") is None, reason="spaCy es_core_news_sm no instalado")


def test_requisitos_desde_el_pedido_es_y_en():
    reqs = extraer_requisitos(PEDIDO)
    tipos = {(r.tipo, str(r.valor)) for r in reqs}
    assert {("punto", "el río Paraná"), ("punto", "la soja"), ("punto", "el ferrocarril"),
            ("termino", "terminal portuaria"), ("variante", "neutro"), ("idioma", "es")} <= tipos
    assert any(r.tipo == "largo" and r.valor == (28, 52, "palabras") for r in reqs)
    assert validar(BUENO, reqs).ok, validar(BUENO, reqs).informe()
    en = extraer_requisitos("Write at most 30 words in English, formally. Mention the river, trade and railways.")
    assert {r.tipo for r in en} >= {"punto", "largo", "registro", "idioma", "sin_contradicciones"}
    malo = validar("Vos sabés que el puerto tiene 3 grúas. El puerto tiene 5 grúas.", reqs)
    assert not malo.ok and "✘" in malo.informe()
    falla = {r.requisito for r in malo.resultados if not r.ok}
    assert any("neutro" in f for f in falla) and any("contradicciones" in f for f in falla)


def test_variante_registro_y_contradicciones():
    mk = marcadores_variante("Si querés, fijate vos; tú tienes razón; usted puede.")
    assert mk["voseo"] and mk["tuteo"] and mk["usted"]
    assert contradicciones("Marte tiene 2 lunas. Marte tiene 3 lunas.")
    assert contradicciones("El agua es potable. El agua no es potable.")
    assert not contradicciones("Marte tiene 2 lunas. Júpiter tiene 95 lunas.")
    ar = extraer_requisitos("Contalo de vos, en rioplatense.")
    assert not validar("Tú tienes que verlo.", ar).ok and validar("Vos tenés que verlo.", ar).ok


@SPACY
def test_concordancia_es_y_en():
    c = Concordancia()
    assert not c.errores(BUENO)
    tipos = [m.detalle for m in c.errores("Las casas blanco fue construido por los vecinos.")]
    assert any("blanco" in t for t in tipos) and any("fue" in t for t in tipos)
    assert c.errores("La ciudad es antiguo.")
    assert c.errores("Los trenes llega cada día.")
    if nlp("en") is not None:
        assert len(Concordancia().errores("The trains arrives every day. These house is old.")) == 2
        assert not Concordancia().errores("The trains arrive every day. This house is old.")


@SPACY
def test_mutaciones_se_detectan_y_el_original_no_da_falsas_alarmas():
    reqs = extraer_requisitos(PEDIDO)
    esq_ctx = lambda t: __import__("hope.runtime.escritura", fromlist=["x"]).ContextoRevision(  # noqa: E731
        pedido=PEDIDO, esquema=__import__("hope.runtime.escritura", fromlist=["x"]).Esquema(
            ["puerto", "rosario", "paraná", "soja", "ferrocarril"]))
    from hope.runtime.escritura import corpus_local
    from hope.runtime.lengua import LexicoGenero
    lex = LexicoGenero.desde_corpus([r["texto"] for r in corpus_local(particion="almacen")] + [BUENO])
    assert lex.genero("ciudad") == "Fem" and lex.genero("río") == "Masc"
    vs = [Repeticiones(), Contenido(min_palabras=5, marcar_faltantes=False), Concordancia(genero=lex)]
    p = puntaje_mutaciones([(BUENO, reqs)], vs, ctx_fn=esq_ctx)
    assert p.falsas_alarmas == 0
    det = {k: v["detectadas"] for k, v in p.por_tipo.items()}
    for tipo in ("concordancia_genero", "repetir_oracion", "quitar_punto", "otra_variante", "contradiccion",
                 "acortar", "alargar", "fuera_de_tema"):
        assert det.get(tipo) == 1, (tipo, p.por_tipo)
    assert p.deteccion >= 0.75


def test_edicion_por_diff_conserva_el_resto():
    t = "Uno dos. La ciudad la ciudad la ciudad. Tres cuatro. La ciudad la ciudad la ciudad."
    a = t.rfind("La ciudad")
    ed = edicion_unica(t, a, len(t), "Fin.")
    assert t.count(ed.buscar) == 1
    nuevo = aplicar_ediciones(t, [ed])
    assert ed.aplicada and nuevo == t[:a] + "Fin."
    assert preservado(t, nuevo) == pytest.approx(3 / 4)
    amb = Edicion("La ciudad", "X")                       # aparece varias veces: no se adivina
    assert aplicar_ediciones(t, [amb]) == t and not amb.aplicada
    assert "+Fin." in diff_texto(t, nuevo)


def _gen_falso():
    def gen(prompts, max_new, temp):
        if temp == 0.0:
            return [b"El puerto exporta soja. el puerto el puerto el puerto el puerto el puerto el puerto. Fin."]
        return [b"Llega por ferrocarril desde el campo. Sigue."] * len(prompts)
    return gen


def test_escritor_con_requisitos_y_ediciones():
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny")).eval()
    e = Escritor(m, verificadores=[Repeticiones()], generador=_gen_falso(), requisitos=True, max_rondas=2)
    inf = e.escribir("Sobre el puerto: ", pedido="Mencioná la soja y el ferrocarril.", largo=60)
    assert {str(r.valor) for r in inf.requisitos} >= {"la soja", "el ferrocarril"}
    assert inf.validacion is not None and inf.validacion.ok, inf.validacion.informe()
    assert inf.ediciones and all(isinstance(x, Edicion) for x in inf.ediciones)
    assert inf.texto.startswith("El puerto exporta soja.")              # lo no marcado quedó igual
    assert e.verificadores == [e.verificadores[0]] and not any(isinstance(v, VerificadorRequisitos)
                                                               for v in e.verificadores)


def test_fim_formato_datos_y_relleno():
    s = formato_fim("A ", " C", "B")
    assert s == f"{FIM_PRE}A {FIM_SUF} C{FIM_MED}B{FIM_FIN}"
    ej = ejemplos_fim([BUENO], por_texto=3)
    assert len(ej) == 3 and all(e.startswith(FIM_PRE) and e.endswith(FIM_FIN) for e in ej)
    for e in ej:                                          # el medio vuelve a armar el texto original
        pre, rest = e[len(FIM_PRE):].split(FIM_SUF)
        suf, rest = rest.split(FIM_MED)
        med = rest[:-len(FIM_FIN)]
        assert pre + med + suf == BUENO
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny")).eval()
    r = rellenar(m, "El puerto de Rosario ", " Rosario tiene terminales.", n=3, max_new=30)
    assert r.modo == "puntaje" and r.texto in r.candidatos and len(r.puntajes) == len(r.candidatos)
    r2 = rellenar(m, "El puerto ", " Fin.", modo="fim", n=2, max_new=20)
    assert r2.modo == "fim"


@SPACY
def test_trazas_del_lenguaje(tmp_path):
    t = traza("Las casas blancas fueron construidas por los vecinos.")
    assert t.startswith("⟦análisis⟧") and "casas(casa) sust nsubj→construidas [f.pl]" in t
    assert "concordancia: bien" in t
    t2 = traza("La ciudad es antiguo.")
    assert "concordancia: «ciudad" in t2
    datos = datos_trazas([BUENO, "Otra oración corta sobre el río."])
    assert len(datos) == 2 and BUENO in datos[0]
    n = escribir_datos(tmp_path / "trazas", datos)
    assert n["train"] + n["val"] > 0 and (tmp_path / "trazas" / "train.bin").exists()
