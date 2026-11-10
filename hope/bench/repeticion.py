"""Repetición y diversidad antes y después de la antirrepetición (``hope/core/decoding.py``), en CPU.

Uso::

    python -m hope.bench.repeticion --salida docs/resultados/repeticion_datos.json

1. **Decodificación**: los mismos 16 casos de ``hope.bench.lenguaje`` con cada estrategia, en los dos
   checkpoints. Métricas: rep-4, distinct-2, bits/byte de 5-gramas del corpus, palabras desconocidas,
   errores de lengua cada 100 palabras (``evaluar_lengua``: ortografía, gramática/concordancia,
   puntuación, bucles) y segundos por respuesta.
2. **Unlikelihood** (``--pasos``): desde ``hn1-cpu-mini``, dos ramas con los mismos pasos y datos
   (corpus de almacén): solo entropía cruzada y entropía cruzada + unlikelihood. Se mide el voraz
   (rep-4, distinct-2) y el BPB en evaluación.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
import zlib
from pathlib import Path

import torch

from hope.core.decoding import procesadores_por_nombre
from hope.core.paso import generar
from hope.runtime.escritura import Lexico, ModeloNgramas, corpus_local, distinct_n, palabras, pasajes, tasa_repeticion
from hope.tools.lenguaje import evaluar_lengua
from hope.train.run import load_light

LN2 = math.log(2)
ESTRATEGIAS = [
    ("voraz", "", 0.0),
    ("voraz + penalización suave (rep:1/8)", "rep:1/8", 0.0),
    ("voraz + penalización fuerte (rep:2/6)", "rep:2/6", 0.0),
    ("voraz + bloqueo 3-gramas de palabras", "bloqueo:3", 0.0),
    ("voraz + bloqueo 4-gramas", "bloqueo:4", 0.0),
    ("voraz + penalización + bloqueo 4", "rep:1/8,bloqueo:4", 0.0),
    ("muestreo T=0,8 top-p 0,95 (referencia)", "", 0.8),
    ("típico τ=0,9, T=0,8", "tipico:0.9", 0.8),
    ("min-p 0,05, T=0,8", "minp:0.05", 0.8),
    ("min-p 0,1 + penalización, T=0,7", "minp:0.1,rep:1/8", 0.7),
]


def medir(salidas: list[str], segs: list[float], ng, lex) -> dict:
    desc, err = [], []
    for s in salidas:
        ws = [w for w in palabras(s) if len(w) >= 4]
        desc.append(sum(w not in lex.voc for w in ws) / max(len(ws), 1))
        err.append(evaluar_lengua(s, "es")["errores_cada_100"] if s.strip() else 0.0)
    return {"rep4": statistics.mean(tasa_repeticion(s) for s in salidas), "distinct2": distinct_n(salidas, 2),
            "ng_bits": statistics.mean(ng.bits_por_byte(s) for s in salidas if s) if any(salidas) else None,
            "desconocidas": statistics.mean(desc), "errores_lengua_cada_100": statistics.mean(err),
            "segundos": statistics.mean(segs), "ejemplo": salidas[0][:160]}


def decodificacion(ruta: str, cs, ng, lex, largo: int, log) -> dict:
    m = load_light(ruta)
    out = {"modelo": ruta, "estrategias": {}}
    for nombre, spec, temp in ESTRATEGIAS:
        sal, segs = [], []
        for _t, p, _r in cs:
            t = time.perf_counter()
            o = generar(m, p, largo, temperature=temp, top_p=0.95 if temp and not spec else None,
                        procesadores=procesadores_por_nombre(spec), seed=zlib.crc32(p.encode()) % 1000)
            segs.append(time.perf_counter() - t)
            sal.append(o.decode("utf-8", errors="ignore"))
        row = medir(sal, segs, ng, lex)
        out["estrategias"][nombre] = row
        log(f"  [{Path(ruta).stem}] {nombre}: rep4 {row['rep4']:.3f} · dist2 {row['distinct2']:.3f} · ng "
            f"{row['ng_bits']:.2f} · desc {row['desconocidas']:.2f} · err/100 {row['errores_lengua_cada_100']:.1f}"
            f" · {row['segundos']:.2f} s")
        log(f"      «{row['ejemplo'][:120]}»")
    return out


def unlikelihood_ab(ruta: str, alm: list[str], ev: list[str], cs, ng, lex, pasos: int, largo: int, log) -> dict:
    base = load_light(ruta)
    textos = [t.encode("utf-8") for t in alm]
    ev_txt = [p for d in ev for p in pasajes(d, 600, 200)[:1]][:24]

    def bpb(m):
        tot = n = 0.0
        with torch.no_grad():
            for t in ev_txt:
                x = torch.tensor([list(t.encode("utf-8"))])
                tot += float(torch.nn.functional.cross_entropy(m(x[:, :-1])["logits"][0].float(), x[0, 1:],
                                                               reduction="sum"))
                n += x.shape[1] - 1
        return tot / n / LN2

    def voraz(m):
        sal = [generar(m, p, largo).decode("utf-8", errors="ignore") for _t, p, _r in cs]
        return {"rep4": statistics.mean(tasa_repeticion(s) for s in sal), "distinct2": distinct_n(sal, 2),
                "ejemplo": sal[0][:160]}

    out = {"pasos": pasos, "inicial": {"bpb": bpb(base), **voraz(base)}}
    log(f"  UL inicial: {out['inicial']}")
    for nombre, w in (("solo entropía cruzada", 0.0), ("+ unlikelihood (peso 1, n=8)", 1.0)):
        torch.manual_seed(0)
        m = copy.deepcopy(base).train()
        opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0.0)
        rng = random.Random(5)
        t0 = time.perf_counter()
        uls = []
        for _ in range(pasos):
            xs = []
            for _ in range(8):
                b = rng.choice(textos)
                i = rng.randrange(0, max(len(b) - 385, 1))
                b = b[i:i + 385]
                xs.append(list(b) + [0] * (385 - len(b)))
            loss, st = m.loss(torch.tensor(xs), unlikelihood=w, unlikelihood_n=8)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if "unlikelihood" in st:
                uls.append(st["unlikelihood"])
        m.eval()
        row = {"segundos_entreno": time.perf_counter() - t0, "bpb": bpb(m), **voraz(m)}
        if uls:
            row["unlikelihood_media_final"] = statistics.mean(uls[-50:])
        out[nombre] = row
        log(f"  UL {nombre}: {row}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelos", nargs="+", default=["checkpoints/hn1-cpu-mini.pt", "checkpoints/hn1-25m-cpu.pt"])
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--largo", type=int, default=200)
    ap.add_argument("--pasos", type=int, default=300)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--salida", default="docs/resultados/repeticion_datos.json")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.hilos)
    from hope.bench.lenguaje import casos

    def log(s):
        print(s, flush=True)
    cs = casos(a.prompts, a.largo)
    alm = [r["texto"] for r in corpus_local(particion="almacen")]
    ev = [r["texto"] for r in corpus_local(particion="eval")]
    ng, lex = ModeloNgramas(5).entrenar(alm), Lexico.desde_corpus(alm)
    out = {"casos": len(cs), "decodificacion": []}
    for ruta in a.modelos:
        out["decodificacion"].append(decodificacion(ruta, cs, ng, lex, a.largo, log))
        Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.pasos:
        out["unlikelihood"] = unlikelihood_ab(a.modelos[0], alm, ev, cs, ng, lex, a.pasos, a.largo, log)
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"listo → {a.salida}")


if __name__ == "__main__":
    main()
