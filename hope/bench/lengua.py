"""Medición del enfoque «como código» para lenguaje (``hope/runtime/lengua.py``), en CPU.

Uso::

    python -m hope.bench.lengua --salida docs/resultados/lengua_datos.json

Partes (misma partición de evaluación que ``hope.bench.lenguaje``):

A. **Mutaciones** sobre pasajes **buenos** de evaluación (español e inglés): qué fracción de cada error
   conocido detectan las comprobaciones, con distintos conjuntos de verificadores, y cuántas falsas
   alarmas dan sobre el texto sin tocar.
B. **Requisitos y concordancia sobre lo generado**: por cada caso se arma un pedido con requisitos
   alcanzables (largo de la referencia, dos palabras de la referencia, español neutro, sin
   contradicciones) y se mide qué fracción cumple cada técnica, los errores de concordancia cada 100
   palabras y, en el bucle, cuánto del borrador se conserva (edición por diff).
C. **Rellenar el medio**: se quita una oración de un pasaje y se rellena con el voraz desde el prefijo o
   con ``rellenar(modo="puntaje")``; chrF contra la oración original y log-prob del sufijo.
D. **Ajuste fino corto y controlado** (``--pasos``): desde el mismo checkpoint, tres ramas con los mismos
   pasos — solo texto, texto + trazas lingüísticas, texto + FIM — y se mide BPB en evaluación,
   concordancia de lo generado y relleno en modo ``fim``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from hope.runtime.escritura import (VACIAS, Contenido, ContextoRevision, Escritor, Esquema, Fluidez, Lexico,
                                    ModeloNgramas, Repeticiones, chrf, corpus_local, mejor_de_n, oraciones,
                                    palabras, pasajes)
from hope.runtime.lengua import (Concordancia, LexicoGenero, VerificadorRequisitos, datos_trazas, ejemplos_fim,
                                 extraer_requisitos, idioma_de, preservado, puntaje_mutaciones, rellenar, validar)
from hope.train.run import load_light

LN2 = math.log(2)


def _claves(texto: str, k: int, rng: random.Random) -> list[str]:
    ws = [w for w in dict.fromkeys(palabras(texto)) if w not in VACIAS and len(w) >= 6]
    return rng.sample(ws, min(k, len(ws)))


def pedido_para(texto: str, titulo: str, rng: random.Random, k: int = 3) -> str:
    n = len(palabras(texto))
    kws = _claves(texto, k, rng)
    if idioma_de(texto) == "en":
        return f"Write about {titulo} in English, in about {n} words. Mention {', '.join(kws)}."
    return f"Escribí sobre {titulo} en español neutro, en unas {n} palabras. Mencioná {', '.join(kws)}."


def _ctx_de(pedido: str):
    reqs = extraer_requisitos(pedido)
    claves = [str(r.valor) for r in reqs if r.tipo == "punto"]
    return ContextoRevision(pedido=pedido, esquema=Esquema(claves))


# ---------------------------------------------------------------------------
def parte_a(log, ng, lexico, lexg, n: int = 16) -> dict:
    rng = random.Random(0)
    out = {}
    for lang, esp in (("es", True), ("en", False)):
        docs = [r for r in corpus_local(particion="eval", espanol=esp) if idioma_de(r["texto"]) == lang]
        casos, pedidos = [], {}
        for r in docs:
            ps = pasajes(r["texto"], max_chars=600, min_chars=250)
            if ps:
                ped = pedido_para(ps[0], r["titulo"].split(",")[0], rng)
                pedidos[ps[0]] = ped
                casos.append((ps[0], extraer_requisitos(ped)))
            if len(casos) >= n:
                break
        conjuntos = {
            "solo requisitos": [],
            "básicos (repetición, fluidez, léxico, contenido)": [Repeticiones(), Fluidez(ng), lexico,
                                                                 Contenido(marcar_faltantes=False)],
            "concordancia (spaCy + léxico de género)": [Concordancia(genero=lexg)],
            "todos": [Repeticiones(), Fluidez(ng), lexico, Contenido(marcar_faltantes=False), Concordancia(genero=lexg)],
        }
        if lang == "en":                                   # fluidez y léxico con el corpus en inglés
            alm_en = [r["texto"] for r in corpus_local(particion="almacen", espanol=False)
                      if idioma_de(r["texto"]) == "en"]
            ng_en, lex_en = ModeloNgramas(5).entrenar(alm_en), Lexico.desde_corpus(alm_en)
            conjuntos = {k: [Fluidez(ng_en) if isinstance(v, Fluidez) else lex_en if isinstance(v, Lexico) else v
                             for v in vs] for k, vs in conjuntos.items()}
        res = {}
        for nombre, vs in conjuntos.items():
            reqs_casos = casos if nombre != "concordancia (spaCy + léxico de género)" else [(t, []) for t, _ in casos]
            p = puntaje_mutaciones(reqs_casos, vs, ctx_fn=lambda t: _ctx_de(pedidos.get(t, "")))
            res[nombre] = {"deteccion": p.deteccion, "falsas_alarmas": p.tasa_falsas_alarmas,
                           "por_tipo": {k: v["detectadas"] / max(v["aplicadas"], 1) for k, v in p.por_tipo.items()},
                           "aplicadas": {k: v["aplicadas"] for k, v in p.por_tipo.items()}}
            log(f"  A[{lang}] {nombre}: detección {p.deteccion:.2f} · falsas alarmas {p.tasa_falsas_alarmas:.2f}")
        out[lang] = {"casos": len(casos), "conjuntos": res}
    return out


# ---------------------------------------------------------------------------
def casos_b(n: int, largo: int) -> list[tuple[str, str, str]]:
    from hope.bench.lenguaje import casos
    return casos(n, largo)


def parte_b(m, log, cs, ng, lexico, lexg, largo: int) -> dict:
    rng = random.Random(2)
    conc = Concordancia(genero=lexg)
    basicos = [Repeticiones(), Fluidez(ng), lexico, Contenido()]
    tecnicas = {}
    filas = {}
    for tema, p, ref in cs:
        n = len(palabras(ref))
        kws = _claves(ref, 2, rng)
        filas[p] = f"Seguí escribiendo en español neutro, en unas {n} palabras. Mencioná {' y '.join(kws)}."

    def run(nombre, fn):
        val, conc_err, words, segs, pres, eds = [], 0, 0, [], [], [0, 0]
        for tema, p, ref in cs:
            reqs = extraer_requisitos(filas[p])
            t = time.perf_counter()
            r = fn(tema, p, filas[p], reqs)
            segs.append(time.perf_counter() - t)
            texto = r if isinstance(r, str) else r.texto
            if not isinstance(r, str):
                pres.append(preservado(r.borrador, r.texto))
                eds[0] += sum(e.aplicada for e in r.ediciones)
                eds[1] += len(r.ediciones)
            v = validar(texto, reqs)
            val.append(v)
            conc_err += len([e for e in conc.errores(texto) if e.tipo == "concordancia"])
            words += len(palabras(texto))
        row = {"requisitos_cumplidos": statistics.mean(v.tasa for v in val),
               "casos_que_cumplen_todo": sum(v.ok for v in val) / len(val),
               "concordancia_cada_100_palabras": 100 * conc_err / max(words, 1),
               "segundos": statistics.mean(segs)}
        por = {}
        for v in val:
            for x in v.resultados:
                k = x.requisito.split(" ")[0] if not x.requisito.startswith("cubre") else "cubre"
                por.setdefault(k, []).append(x.ok)
        row["por_requisito"] = {k: sum(v) / len(v) for k, v in por.items()}
        if pres:
            row["preservado"] = statistics.mean(pres)
            row["ediciones_aplicadas"] = f"{eds[0]}/{eds[1]}"
        tecnicas[nombre] = row
        log(f"  B {nombre}: requisitos {row['requisitos_cumplidos']:.2f} · todo {row['casos_que_cumplen_todo']:.2f}"
            f" · concordancia/100 {row['concordancia_cada_100_palabras']:.2f} · {row['segundos']:.2f} s"
            + (f" · preservado {row['preservado']:.2f}" if pres else ""))

    from hope.core.paso import generar
    run("voraz", lambda tema, p, ped, reqs: generar(m, p, largo).decode("utf-8", errors="ignore"))
    run("MBR N=4 + requisitos + concordancia", lambda tema, p, ped, reqs: mejor_de_n(
        m, p, 4, largo, temperature=0.6, verificadores=basicos + [conc, VerificadorRequisitos(reqs)]).texto)
    e1 = Escritor(m, verificadores=basicos, n_candidatos=4, temperatura=0.6, max_rondas=2)
    run("bucle (verificadores básicos)", lambda tema, p, ped, reqs: e1.escribir(p, tema=tema, largo=largo, pedido=ped))
    e2 = Escritor(m, verificadores=basicos + [conc], n_candidatos=4, temperatura=0.6, max_rondas=2, requisitos=True)
    run("bucle + requisitos + concordancia", lambda tema, p, ped, reqs: e2.escribir(p, tema=tema, largo=largo,
                                                                                    pedido=ped))
    return tecnicas


# ---------------------------------------------------------------------------
def partir_medio(texto: str) -> tuple[str, str, str] | None:
    sp = oraciones(texto)
    if len(sp) < 3:
        return None
    i = len(sp) // 2
    a, b = sp[i]
    return texto[:a], texto[a:b], texto[b:]


def parte_c(m, log, docs: list[str], modos=("voraz", "puntaje"), n: int = 12) -> dict:
    from hope.core.batched import generate_batch
    trios = [t for t in (partir_medio(p) for d in docs for p in pasajes(d, 700, 300)[:1]) if t][:n]
    out = {}
    for modo in modos:
        cs, lps, segs = [], [], []
        for pre, med, suf in trios:
            t = time.perf_counter()
            if modo == "voraz":
                g = generate_batch(m, [pre], 160)[0].decode("utf-8", errors="ignore")
                sp = oraciones(g)
                txt = g[:sp[0][1]].strip() if sp else g.strip()
            else:
                txt = rellenar(m, pre, suf, modo=modo, n=6, max_new=160).texto
            segs.append(time.perf_counter() - t)
            cs.append(chrf(txt, med))
            esc = Escritor.__new__(Escritor)
            esc.model = m
            lps.append(esc._logprob(pre + txt + " ", [suf[:96]])[0])
        out[modo] = {"chrf": statistics.mean(cs), "logprob_sufijo_nats": statistics.mean(lps),
                     "segundos": statistics.mean(segs), "casos": len(trios)}
        log(f"  C {modo}: chrF {out[modo]['chrf']:.3f} · log-prob del sufijo {out[modo]['logprob_sufijo_nats']:.3f}"
            f" · {out[modo]['segundos']:.2f} s")
    return out


# ---------------------------------------------------------------------------
def _lotes(fuentes: list[list[bytes]], pesos: list[float], seq: int, batch: int, rng: random.Random):
    while True:
        xs = []
        for _ in range(batch):
            src = rng.choices(fuentes, weights=pesos)[0]
            b = rng.choice(src)
            if len(b) > seq + 1:
                i = rng.randrange(0, len(b) - seq - 1)
                b = b[i:i + seq + 1]
            xs.append(list(b) + [0] * (seq + 1 - len(b)))
        yield torch.tensor(xs)


def parte_d(ruta: str, log, alm: list[str], ev_docs: list[str], cs, lexg, pasos: int, largo: int) -> dict:
    rng = random.Random(3)
    texto = [t.encode("utf-8") for t in alm]
    t0 = time.perf_counter()
    trz = [x.encode("utf-8") for x in datos_trazas([p for t in alm for p in pasajes(t, 500, 150)[:2]],
                                                   max_oraciones=2)]
    t_trz = time.perf_counter() - t0
    fim = [x.encode("utf-8") for x in ejemplos_fim([p for t in alm for p in pasajes(t, 700, 200)[:2]], por_texto=2)]
    log(f"  D datos: {len(trz)} trazas ({sum(map(len, trz)) / 1e3:.0f} kB, {t_trz:.0f} s con spaCy) · "
        f"{len(fim)} ejemplos FIM ({sum(map(len, fim)) / 1e3:.0f} kB)")
    ramas = {"solo texto": ([texto], [1.0]), "texto + trazas": ([texto, trz], [0.5, 0.5]),
             "texto + FIM": ([texto, fim], [0.5, 0.5])}
    base = load_light(ruta)
    ev_txt = [p for d in ev_docs for p in pasajes(d, 600, 200)[:1]][:24]
    conc = Concordancia(genero=lexg)

    def bpb(m) -> float:
        m.eval()
        tot = n = 0.0
        with torch.no_grad():
            for t in ev_txt:
                x = torch.tensor([list(t.encode("utf-8"))])
                ce = torch.nn.functional.cross_entropy(m(x[:, :-1])["logits"][0].float(), x[0, 1:], reduction="sum")
                tot += float(ce)
                n += x.shape[1] - 1
        return tot / n / LN2

    def conc_gen(m) -> float:
        from hope.core.paso import generar
        err = w = 0
        for _tema, p, _ref in cs[:12]:
            g = generar(m, p, largo).decode("utf-8", errors="ignore")
            err += len([e for e in conc.errores(g) if e.tipo == "concordancia"])
            w += len(palabras(g))
        return 100 * err / max(w, 1)

    out = {"pasos": pasos, "datos": {"trazas": len(trz), "fim": len(fim), "segundos_trazas": t_trz},
           "inicial": {"bpb": bpb(base), "concordancia_cada_100": conc_gen(base)}}
    log(f"  D inicial: {out['inicial']}")
    docs_med = [d for d in ev_docs]
    for nombre, (fuentes, pesos) in ramas.items():
        torch.manual_seed(0)
        m = copy.deepcopy(base).train()
        opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0.0)
        it = _lotes(fuentes, pesos, 384, 8, random.Random(4))
        t = time.perf_counter()
        for s in range(pasos):
            loss, st = m.loss(next(it))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        row = {"segundos_entreno": time.perf_counter() - t, "bpb": bpb(m), "concordancia_cada_100": conc_gen(m)}
        if nombre == "texto + FIM":
            row["relleno"] = parte_c(m, lambda s: None, docs_med, modos=("voraz", "puntaje", "fim"))
        out[nombre] = row
        log(f"  D {nombre}: {row}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelos", nargs="+", default=["checkpoints/hn1-cpu-mini.pt", "checkpoints/hn1-25m-cpu.pt"])
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--largo", type=int, default=200)
    ap.add_argument("--pasos", type=int, default=300, help="pasos del ajuste fino de la parte D (0 = no se hace)")
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--salida", default="docs/resultados/lengua_datos.json")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.hilos)

    def log(s):
        print(s, flush=True)
    alm = [r["texto"] for r in corpus_local(particion="almacen")]
    ev_docs = [r["texto"] for r in corpus_local(particion="eval")]
    ng = ModeloNgramas(5).entrenar(alm)
    lexico = Lexico.desde_corpus(alm)
    lexg = LexicoGenero.desde_corpus(alm)
    cs = casos_b(a.prompts, a.largo)
    out = {"A_mutaciones": parte_a(log, ng, lexico, lexg), "modelos": []}
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for ruta in a.modelos:
        m = load_light(ruta)
        log(f"[{ruta}]")
        row = {"modelo": ruta, "B_requisitos": parte_b(m, log, cs, ng, lexico, lexg, a.largo),
               "C_relleno": parte_c(m, log, ev_docs)}
        out["modelos"].append(row)
        Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.pasos:
        log(f"[ajuste fino {a.modelos[0]}]")
        out["D_ajuste_fino"] = parte_d(a.modelos[0], log, alm, ev_docs, cs, lexg, a.pasos, a.largo)
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"listo → {a.salida}")


if __name__ == "__main__":
    main()
