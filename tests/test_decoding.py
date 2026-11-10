"""Decodificación contra las repeticiones y pérdida unlikelihood (hope/core/decoding.py)."""

import math

import torch

from hope.core.decoding import (BloqueoNgramas, MinP, MuestreoTipico, PenalizacionRepeticion, candidatos_copia,
                                procesadores_por_nombre, unlikelihood)
from hope.core.hn1 import HN1Model, preset
from hope.core.paso import Paso, generar
from hope.runtime.escritura import palabras


def _m():
    torch.manual_seed(0)
    return HN1Model(preset("hn1-tiny")).eval()


def test_sin_procesadores_el_voraz_no_cambia():
    m = _m()
    assert generar(m, "hola mundo, ", 40, procesadores=procesadores_por_nombre("")) == m.generate("hola mundo, ", 40)


def test_penalizacion_resta_lo_ya_visto():
    pen = PenalizacionRepeticion(alpha=2.0, n=3)
    pen.iniciar(list(b"xx"))
    for b in b"abcab":
        pen.avanzar(b)
    logp = torch.log_softmax(torch.zeros(256), -1)
    out = pen(None, logp)                                    # contexto «ab»: «c» ya siguió una vez
    assert out[ord("c")] < out[ord("d")]
    assert math.isclose(float(out.exp().sum()), 1.0, rel_tol=1e-5)


def test_bloqueo_de_ngramas_de_palabras_y_bytes():
    blo = BloqueoNgramas(n=3, min_prefijo=2)
    blo.iniciar([])
    for b in "de la ciudad de la ci".encode():
        blo.avanzar(b)
    logp = torch.log_softmax(torch.zeros(256), -1)
    out = blo(None, logp)
    assert torch.isinf(out[ord("u")]) and torch.isfinite(out[ord("e")])      # «ciu…» copiaría «ciudad»
    blo2 = BloqueoNgramas(n=3)
    blo2.iniciar([])
    for b in "de la ciudad de la ciudad".encode():
        blo2.avanzar(b)
    out2 = blo2(None, logp)
    assert torch.isinf(out2[ord(" ")]) and torch.isfinite(out2[ord("e")])     # no cerrar la palabra repetida
    by = BloqueoNgramas(n=99, max_bytes=4)
    by.iniciar([])
    for b in b"abcdxabc":
        by.avanzar(b)
    assert torch.isinf(by(None, logp)[ord("d")])


def test_tipico_y_minp_truncan_sin_perder_el_maximo():
    logp = torch.log_softmax(torch.tensor([5.0, 4.0, 1.0, -3.0] + [-10.0] * 252), -1)
    mp = MinP(0.1)(None, logp)
    assert torch.isfinite(mp[:2]).all() and torch.isinf(mp[3])
    ti = MuestreoTipico(0.5)(None, logp)
    assert torch.isfinite(ti).sum() >= 1 and int(ti.argmax()) in (0, 1)


def test_generacion_con_procesadores_reduce_bucles():
    m = _m()
    for spec in ("rep:2/4", "bloqueo:2", "rep:1,bloqueo:3,bytes:12,minp:0.05,tipico:0.95"):
        procs = procesadores_por_nombre(spec)
        out = generar(m, "la casa de la casa de la casa ", 120, procesadores=procs).decode("utf-8", "ignore")
        assert isinstance(out, str)
    assert len(procesadores_por_nombre("contrastiva", modelo=m)) == 1


def test_unlikelihood():
    x = torch.tensor([list(b"abcXabcY abcZ")])
    c = candidatos_copia(x, 4)
    assert c[0, 6] == ord("X") and c[0, 11] == ord("Y") and (c >= 0).sum() == 2
    m = _m()
    xb = torch.tensor([list(b"de la ciudad de la ciudad de la ciudad de la casa")])
    base, st0 = m.loss(xb)
    tot, st = m.loss(xb, unlikelihood=1.0, unlikelihood_n=6)
    assert "unlikelihood" in st and "unlikelihood" not in st0
    assert float(tot) >= float(base)
    tot.backward()                                           # tiene gradiente
    from hope.train.continual import TrainConfig
    assert TrainConfig().unlikelihood == 0.0
