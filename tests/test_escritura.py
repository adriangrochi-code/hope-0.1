"""Escribir mejor sin más parámetros (hope/runtime/escritura.py, hope/core/paso.py, contrastiva, knn)."""

import math
from dataclasses import replace

import pytest
import torch

from hope.core.contrastiva import ProcesadorContrastivo, aprendiz_mismo_modelo
from hope.core.hn1 import HN1Model, preset, stable_loop
from hope.core.knn import AlmacenKNN, ProcesadorKNN, logprobs_knn, mezclar
from hope.core.paso import ComputoAdaptativo, Paso, generar, logprobs_paso
from hope.runtime.escritura import (Contenido, ContextoRevision, ConversacionConResumen, Escritor, Esquema,
                                    EjemplosEstilo, Marca, ModeloNgramas, Repeticiones, ResumenContinuo, adaptar,
                                    agente_escritor, chrf, distinct_n, elegir_mbr, mejor_de_n, oraciones,
                                    planificar, resumir_extractivo, revisar, tasa_repeticion)

BASE = preset("hn1-tiny")
PROMPTS = ["hola mundo, esto es una prueba. ", "x = 1\ny = 2\nz = x + y\n" * 3, "superextraordinariamente"]
TEXTO = ("La ciudad tiene un puerto antiguo sobre el río. El puerto fue el centro del comercio durante "
         "dos siglos. Hoy la ciudad conserva el casco viejo y muchas iglesias de piedra. ") * 3


def _wake(m):
    with torch.no_grad():
        for mod in m.modules():
            out = getattr(mod, "out", None)
            if isinstance(out, torch.nn.Linear) and out.weight.abs().sum() == 0:
                torch.nn.init.normal_(out.weight, std=0.05)
        for z in m.regions:
            z.router.weight.normal_(0, 1.0)
        if m.memory is not None:
            m.memory.gate.fill_(0.5)
        torch.nn.init.normal_(m.uncertainty.weight, std=0.5)
    return m


def _model(cfg=BASE, seed=0):
    torch.manual_seed(seed)
    return _wake(HN1Model(cfg).eval())


# ---------------------------------------------------------------------------
# Generación paso a paso y cómputo adaptativo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [BASE, replace(BASE, halting="convergencia"),
                                 stable_loop(BASE, mean_iters=2.0, train_max=4, infer_max=6, tol=1e-3)])
def test_generar_sin_opciones_es_igual_a_generate(cfg):
    m = _model(cfg)
    for p in PROMPTS:
        assert generar(m, p, 40) == m.generate(p, 40)
        assert generar(m, p, 30, n_iters=1) == m.generate(p, 30, n_iters=1)


def test_computo_adaptativo_respeta_umbral_y_profundidad_entrenada():
    m = _model()
    p = PROMPTS[0]
    # umbral infinito: nunca profundiza → igual a generar con las iteraciones de base
    nunca = generar(m, p, 40, adaptativo=ComputoAdaptativo(umbral=math.inf, iters_base=1), informe=True)
    assert nunca.texto == m.generate(p, 40, n_iters=1)
    assert nunca.stats.parches_profundos == 0 and nunca.stats.parches > 0
    # umbral −∞: todos los parches generados usan la profundidad entrenada (y nunca más que ella)
    ca = ComputoAdaptativo(umbral=-math.inf, iters_base=1, iters_profundo=99, ignorar_parada=False)
    assert ca.profundo(m) == m.cfg.max_iters
    siempre = generar(m, p, 40, adaptativo=ca, informe=True)
    assert siempre.stats.parches_profundos == siempre.stats.parches > 0
    ref = m.generate(p, 40, n_iters=m.cfg.max_iters)
    assert siempre.texto == ref
    # con un umbral intermedio, solo una parte de los parches profundiza
    sig = []
    paso = Paso(m, adaptativo=ComputoAdaptativo(umbral=0.0, iters_base=1))
    paso.feed(list(p.encode()))
    for b in m.generate(p, 60):
        sig.append(paso.unc)
        paso.feed([b])
    med = sorted(sig)[len(sig) // 2]
    mitad = generar(m, p, 60, adaptativo=ComputoAdaptativo(umbral=med, iters_base=1), informe=True)
    assert 0 <= mitad.stats.parches_profundos <= mitad.stats.parches


def test_contexto_profundo_es_el_de_n_iters_completo():
    """El contexto profundo de un parche es exactamente el de ``GlobalCache`` con esa profundidad."""
    m = _model()
    seq = list("una prueba de contexto profundo con varios parches ".encode())
    from hope.core.hn1 import GlobalCache
    for ignorar in (False, True):
        paso = Paso(m, adaptativo=ComputoAdaptativo(umbral=-math.inf, iters_base=1, ignorar_parada=ignorar))
        paso.feed(seq[:5])
        for b in seq[5:]:
            paso.feed([b])
        assert paso.deep.full_depth == ignorar
        ref = GlobalCache(m, None, m.cfg.max_iters, full_depth=ignorar).upto(paso.patch_states, paso.cur_pid)
        got = paso.deep.upto(paso.patch_states, paso.cur_pid)
        assert torch.allclose(ref, got, atol=1e-5)
        with torch.no_grad():                              # y lo mismo que la parte global sin caché
            full = m._global(torch.stack(paso.patch_states[:paso.cur_pid], 1), None, m.cfg.max_iters, ignorar)[0]
        assert torch.allclose(full, got, atol=1e-4)
    # con un núcleo ACT, por defecto el modo profundo ignora la parada (la de CPU colapsó a 1 iteración)
    assert ComputoAdaptativo().completo(m) and not ComputoAdaptativo().completo(_model(replace(BASE, halting="convergencia")))


def test_logprobs_paso_coincide_con_forward():
    m = _model()
    ctx, txt = "hola mundo, ", "esto es una prueba."
    lp, _ = logprobs_paso(m, ctx, txt)
    x = torch.tensor([list((ctx + txt).encode())])
    with torch.no_grad():
        ref = torch.log_softmax(m(x[:, :-1])["logits"][0].float(), -1)
    t = list(txt.encode())
    want = [float(ref[len(ctx) - 1 + i, b]) for i, b in enumerate(t)]
    assert max(abs(a - b) for a, b in zip(lp, want)) < 1e-3


# ---------------------------------------------------------------------------
# Contrastiva
# ---------------------------------------------------------------------------
def test_contrastiva_restriccion_de_plausibilidad_y_beta_cero():
    exp, apr = _model(seed=0), _model(seed=1)
    p = PROMPTS[0]
    assert generar(exp, p, 30, procesadores=[ProcesadorContrastivo(apr, beta=0.0)]) == exp.generate(p, 30)
    proc = ProcesadorContrastivo(apr, alpha=0.1)
    paso = Paso(exp)
    paso.feed(list(p.encode()))
    proc.iniciar(list(p.encode()))
    for _ in range(20):
        logp = torch.log_softmax(paso.logits, -1)
        out = proc(paso, logp)
        b = int(out.argmax())
        assert logp[b] >= logp.max() + math.log(0.1) - 1e-6        # siempre dentro de la cabeza plausible
        assert torch.isinf(out[logp < logp.max() + math.log(0.1) - 1e-6]).all()
        proc.avanzar(b)
        paso.feed([b])
    # el propio modelo con menos iteraciones como aprendiz: corre y no usa otra copia de pesos
    yo = aprendiz_mismo_modelo(exp, n_iters=1)
    assert yo.aprendiz is exp
    assert len(generar(exp, p, 20, procesadores=[yo])) > 0


# ---------------------------------------------------------------------------
# kNN-LM
# ---------------------------------------------------------------------------
def test_knn_memoriza_el_corpus_y_lambda_cero_no_cambia_nada():
    m = _model()
    st = AlmacenKNN.construir(m, [TEXTO], dim=16, ventana=64)
    assert st.info.claves > 100 and st.keys.shape[1] == 16 and st.memoria_mb() > 0
    ctx, txt = TEXTO[:120], TEXTO[120:220]
    r = logprobs_knn(m, st, ctx, txt, lams=(0.0, 0.5))
    lp, _ = logprobs_paso(m, ctx, txt)
    assert max(abs(a - b) for a, b in zip(r[0.0], lp)) < 1e-3
    assert sum(r[0.5]) > sum(r[0.0])                                # el texto está en el almacén
    p = PROMPTS[0]
    assert generar(m, p, 30, procesadores=[ProcesadorKNN(st, lam=0.0)]) == m.generate(p, 30)
    # índice aproximado: el vecino más cercano coincide casi siempre con el exacto
    q = st.keys[:50] + 0.01 * torch.randn(50, 16)
    d_ex, v_ex = st.buscar(q, 1)
    st.indexar(nlist=8)
    d_ap, v_ap = st.buscar(q, 1, nprobe=4)
    assert (v_ex[:, 0] == v_ap[:, 0]).float().mean() > 0.8
    lm = torch.log_softmax(torch.randn(256), -1)
    assert torch.allclose(mezclar(lm, lm, 0.3), lm, atol=1e-5)


# ---------------------------------------------------------------------------
# Métricas, verificadores y MBR
# ---------------------------------------------------------------------------
def test_metricas_y_verificadores():
    bucle = "de la ciudad de la ciudad de la ciudad de la ciudad de la ciudad"
    assert tasa_repeticion(bucle) > 0.5 and tasa_repeticion("una frase sin nada repetido acá") == 0.0
    assert distinct_n(["a b c d", "a b c d"], 2) == 0.5
    assert chrf("hola", "hola") == pytest.approx(1.0) and chrf("hola", "xyzw") == 0.0
    ms = Repeticiones().revisar(bucle, ContextoRevision())
    assert ms and all(m.tipo == "repeticion" for m in ms)
    ng = ModeloNgramas(4).entrenar([TEXTO])
    assert ng.bits_por_byte("El puerto fue el centro") < ng.bits_por_byte("Xq zzv kjw pplq")
    ctx = ContextoRevision(tema="puerto", esquema=Esquema(["puerto", "comercio", "iglesias"]))
    marcas = Contenido(min_palabras=3).revisar("Los gatos duermen mucho durante todo el día. El puerto creció.", ctx)
    assert any(m.inicio == 0 for m in marcas)                       # la oración fuera de tema
    assert any(m.inicio < 0 and "comercio" in m.detalle for m in marcas)
    # verificador externo (protocolo de la otra sesión): una función que devuelve tuplas o diccionarios
    ext = adaptar(lambda t: [(0, 3, "ortografía dudosa")] + [{"inicio": 4, "fin": 5, "tipo": "gramatica",
                                                            "detalle": "concordancia"}], "lengua")
    out = revisar("Los gato duermen.", [ext])
    assert [m.tipo for m in out] == ["lengua", "gramatica"]
    roto = adaptar(lambda t: 1 / 0, "roto")
    assert revisar("x", [roto])[0].tipo == "error"                  # un verificador roto no frena nada
    s = "Uno. Dos.\nTres"
    assert [s[a:b] for a, b in oraciones(s)] == ["Uno.", "Dos.", "Tres"]


def test_mbr_elige_el_consenso_y_castiga_lo_marcado():
    cands = ["el puerto de la ciudad es antiguo", "el puerto de la ciudad es muy antiguo",
             "zzzz qqq", "el puerto de la ciudad es antiguo y grande"]
    el = elegir_mbr(cands)
    assert el.indice != 2
    malo = "el puerto de la ciudad de la ciudad de la ciudad de la ciudad es antiguo"
    el2 = elegir_mbr([malo, malo, cands[0], cands[1]], [Repeticiones()], peso=5.0)
    assert el2.texto != malo
    m = _model()
    e = mejor_de_n(m, PROMPTS[0], n=3, max_new=20)
    assert len(e.utilidad) == 3 and 0 <= e.indice < 3


# ---------------------------------------------------------------------------
# Bucle de escritura
# ---------------------------------------------------------------------------
class _Reloj:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _gen_falso(reloj, costo=0.1):
    """Generador determinista: el borrador repite; las alternativas son oraciones limpias distintas."""
    n = {"i": 0}

    def gen(prompts, max_new, temp):
        reloj.t += costo
        out = []
        for _p in prompts:
            if temp == 0.0:
                out.append("El puerto es antiguo. de la ciudad de la ciudad de la ciudad de la ciudad. Fin.".encode())
            else:
                n["i"] += 1
                out.append(f"El comercio creció en el siglo {n['i']}. Sigue.".encode())
        return out
    return gen


def test_escritor_reescribe_solo_lo_marcado_y_respeta_el_presupuesto():
    from hope.runtime.budget import SpeedTarget
    m = _model()
    reloj = _Reloj()
    e = Escritor(m, verificadores=[Repeticiones()], generador=_gen_falso(reloj), max_rondas=3, clock=reloj)
    inf = e.escribir("Sobre el puerto: ", tema="puerto", largo=90)
    assert "de la ciudad de la ciudad" in inf.borrador
    assert "de la ciudad de la ciudad" not in inf.texto
    assert inf.texto.startswith("El puerto es antiguo.") and inf.texto.endswith("Fin.")   # lo no marcado queda
    assert inf.rondas[-1].penalidad < inf.rondas[0].penalidad and inf.mejoro
    # presupuesto mínimo: no se empieza ninguna reescritura (nunca llega tarde)
    reloj2 = _Reloj()
    e2 = Escritor(m, verificadores=[Repeticiones()], generador=_gen_falso(reloj2, costo=1.0), clock=reloj2,
                  objetivo=SpeedTarget(tokens_per_s=1e6))
    inf2 = e2.escribir("Sobre el puerto: ", tema="puerto", largo=100)
    assert inf2.texto == inf2.borrador and inf2.motivo == "presupuesto de tiempo cumplido"
    # con modelo real (sin generador falso) corre de punta a punta
    inf3 = Escritor(m, n_candidatos=3, max_rondas=1).escribir("hola mundo, ", largo=40)
    assert isinstance(inf3.texto, str) and inf3.rondas[0].fase == "borrador"


def test_agente_escritor_dentro_de_budgeted_responder():
    from hope.runtime.budget import BudgetedResponder, SpeedTarget
    m = _model()
    reloj = _Reloj()
    ag = agente_escritor(Escritor(m, verificadores=[Repeticiones()], generador=_gen_falso(reloj), clock=reloj),
                         largo=80)
    rep = BudgetedResponder([ag]).respond("puerto", "Escribí sobre el puerto: ", SpeedTarget(tokens_per_s=1e9))
    assert rep.agent == "escritor" and "de la ciudad de la ciudad" not in rep.text


def test_ejemplos_de_estilo_y_planificar():
    est = EjemplosEstilo([TEXTO.replace(". ", ".\n\n", 1), "Los gatos son animales domésticos que duermen "
                          "muchas horas al día y cazan ratones en los graneros de las casas de campo."],
                         max_chars=300)
    assert "puerto" in est.buscar("el puerto y el comercio")[0]
    assert est.anteponer("Escribí: ", "gatos domésticos").startswith("Los gatos")
    esq = planificar("puerto", "Contame del puerto antiguo", est)
    assert "puerto" in esq.palabras_clave and esq.guia


# ---------------------------------------------------------------------------
# Resumen continuo y conversación que no olvida
# ---------------------------------------------------------------------------
def test_resumen_continuo_en_el_cuaderno_y_conversacion_compactada():
    from hope.memory.knowledge import KnowledgeBase, Notebook
    nb = Notebook(KnowledgeBase())
    res = ResumenContinuo(nb, "c1", max_chars=200, recientes=1)
    assert resumir_extractivo(["El puerto es antiguo. El puerto tiene grúas. Los gatos duermen."], 40) \
        .startswith("El puerto")
    m = _model()
    conv = ConversacionConResumen(m, res, max_context=256)
    for i in range(8):
        conv.say(f"Usuario: hablemos del puerto número {i}, que tiene grúas.\nHope: ", 24)
    assert conv.compactaciones >= 1
    assert len(conv.history) <= 256
    assert res.texto() and "puerto" in res.texto()
    nota = nb.lookup(res.concepto)
    assert nota is not None and "puerto" in nota.text
    assert bytes(conv.history).startswith("Resumen de la conversación".encode()) or conv.compactaciones == 0
    # sin superar la ventana da lo mismo que Conversation
    from hope.core.batched import Conversation
    a, b = Conversation(m), ConversacionConResumen(m, ResumenContinuo(None, "c2"))
    assert a.say("hola mundo, ", 20) == b.say("hola mundo, ", 20)


def test_charla_con_resumen(tmp_path):
    from hope.app.charla import Charla
    ch = Charla(tmp_path / "salida", tmp_path / "datos", tmp_path)
    ch.model = _model()
    r = ch.decir("c", "hola, ¿qué tal el puerto?", max_bytes=20, herramientas=False, contexto=False, resumen=True)
    assert r["ok"] and "resumen" in r["traza"]
    r2 = ch.decir("c", "otra pregunta", max_bytes=20, herramientas=False, contexto=False, resumen=False)
    assert r2["ok"] and "resumen" not in r2["traza"]
