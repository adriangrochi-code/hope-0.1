"""Medición de las técnicas de escritura (``hope/runtime/escritura.py``) en CPU.

Uso::

    python -m hope.bench.lenguaje --modelos checkpoints/hn1-cpu-mini.pt checkpoints/hn1-25m-cpu.pt \\
        --prompts 16 --largo 200 --salida docs/resultados/lenguaje_datos.json

Protocolo
---------
* Textos de **evaluación**: pasajes en español de ``datos_maestro/notas_calidad.jsonl`` de la
  partición ``eval`` (1 de cada 10 por hash; nunca entran al almacén kNN ni a los ejemplos de estilo).
* Cada caso: ~200 bytes de prompt (cortado en un espacio) y los ~200 bytes siguientes como referencia.
* Métricas por técnica:
  - ``rep4``: fracción de 4-gramas de palabras repetidos dentro de la salida (más bajo = mejor);
  - ``ng_bits``: bits por byte de la salida según un modelo de 5-gramas de bytes del corpus de
    almacén (fluidez «de superficie»; un bucle repetitivo puede dar valores bajos, por eso va junto a rep4);
  - ``desconocidas``: fracción de palabras (≥ 4 letras) que no aparecen en el corpus;
  - ``distinct2``: 2-gramas distintos / totales sobre todas las salidas (diversidad);
  - ``bpb``: bits por byte de la **referencia** dada el prompt (BPB condicional), cuando la técnica
    define una distribución; ``segundos``: tiempo medio por respuesta.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import zlib
from pathlib import Path

import torch

from hope.core.contrastiva import ProcesadorContrastivo
from hope.core.knn import AlmacenKNN, ProcesadorKNN, logprobs_knn
from hope.core.paso import ComputoAdaptativo, Paso, generar, logprobs_paso
from hope.runtime.escritura import (EjemplosEstilo, Escritor, Lexico, ModeloNgramas, corpus_local, distinct_n,
                                    generador_paso, mejor_de_n, palabras, pasajes, tasa_repeticion,
                                    verificadores_basicos)
from hope.train.run import load_light

LN2 = math.log(2)


def casos(n: int, largo: int, raiz: str = ".") -> list[tuple[str, str, str]]:
    """(tema, prompt, referencia) desde la partición de evaluación."""
    out = []
    for r in corpus_local(raiz, particion="eval"):
        for p in pasajes(r["texto"], max_chars=2000, min_chars=2 * largo + 40):
            b = p.encode("utf-8")
            cut = b.rfind(b" ", 0, largo) + 1
            prompt = b[:cut].decode("utf-8", errors="ignore")
            ref = b[cut:cut + largo].decode("utf-8", errors="ignore")
            out.append((r["titulo"], prompt, ref))
            break
        if len(out) >= n:
            break
    return out


def _bpb(lps: list[float]) -> float:
    return -sum(lps) / max(len(lps), 1) / LN2


def calibrar_umbral(model, textos: list[str], q: float = 0.5, senal: str = "cabeza") -> tuple[float, float]:
    """Cuantil ``q`` de la señal por byte en textos del almacén, y correlación (Spearman aprox.)
    entre la pérdida predicha por la cabeza y la pérdida real."""
    sig, pred, real = [], [], []
    for t in textos[:6]:
        x = torch.tensor([list(t.encode("utf-8")[:600])])
        with torch.no_grad():
            out = model(x[:, :-1])
        lp = torch.log_softmax(out["logits"][0].float(), -1)
        ce = -lp.gather(1, x[0, 1:, None])[:, 0]
        u = out["uncertainty"][0].float()
        ent = -(lp.exp() * lp).sum(-1)
        sig += (u if senal == "cabeza" else ent).tolist()
        pred += u.tolist()
        real += ce.tolist()
    rank = lambda v: {i: r for r, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}  # noqa: E731
    rp, rr = rank(pred), rank(real)
    n = len(pred)
    rho = 1 - 6 * sum((rp[i] - rr[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
    return sorted(sig)[int(q * (len(sig) - 1))], rho


def medir_salidas(nombre: str, salidas: list[str], segundos: list[float], ng: ModeloNgramas, lex: Lexico,
                  bpb: float | None = None, extra: dict | None = None) -> dict:
    desc = []
    for s in salidas:
        ws = [w for w in palabras(s) if len(w) >= 4]
        desc.append(sum(w not in lex.voc for w in ws) / max(len(ws), 1))
    return {"tecnica": nombre, "rep4": statistics.mean(tasa_repeticion(s) for s in salidas),
            "ng_bits": statistics.mean(ng.bits_por_byte(s) for s in salidas if s) if any(salidas) else None,
            "desconocidas": statistics.mean(desc), "distinct2": distinct_n(salidas, 2), "bpb": bpb,
            "segundos": statistics.mean(segundos), "ejemplo": salidas[0][:160], **(extra or {})}


def correr(ruta: str, aprendiz_ruta: str | None, cs, ng, lex, est: EjemplosEstilo, almacen_txt: list[str],
           largo: int, log) -> dict:
    torch.manual_seed(0)
    m = load_light(ruta)
    res: dict = {"modelo": ruta, "parametros": sum(p.numel() for p in m.parameters()), "tecnicas": [], "knn": {}}
    umbral, rho = calibrar_umbral(m, almacen_txt)
    res["umbral_cabeza"], res["spearman_cabeza_vs_perdida"] = umbral, rho
    log(f"[{ruta}] umbral (mediana) {umbral:.3f} nats · Spearman cabeza↔pérdida {rho:.3f}")

    # --- almacén kNN ---
    t0 = time.perf_counter()
    st = AlmacenKNN.construir(m, almacen_txt, max_claves=400_000, dim=64)
    res["knn"] = {"claves": st.info.claves, "dim": 64, "construccion_s": time.perf_counter() - t0,
                  "memoria_mb_exacto": st.memoria_mb(), "temp": st.temp}
    q = st.proyectar(torch.randn(64, m.cfg.d_local))
    t = time.perf_counter()
    for i in range(64):
        st.buscar(q[i:i + 1], 16)
    res["knn"]["ms_por_byte_exacto"] = (time.perf_counter() - t) / 64 * 1000
    st_ivf = AlmacenKNN(st.keys, st.vals, mean=st.mean, proj=st.proj, temp=st.temp).indexar()
    t = time.perf_counter()
    for i in range(64):
        st_ivf.buscar(q[i:i + 1], 16, nprobe=8)
    res["knn"]["ms_por_byte_ivf"] = (time.perf_counter() - t) / 64 * 1000
    res["knn"]["memoria_mb_ivf"] = st_ivf.memoria_mb()
    res["knn"]["indice"] = st_ivf.info.indice
    lams = (0.0, 0.1, 0.25, 0.4)
    acc = {lam: [] for lam in lams}
    for _tema, p, ref in cs:
        r = logprobs_knn(m, st, p, ref, lams=lams)
        for lam in lams:
            acc[lam] += r[lam]
    res["knn"]["bpb_por_lambda"] = {str(lam): _bpb(v) for lam, v in acc.items()}
    log(f"  kNN: {res['knn']}")
    lam_best = min(lams[1:], key=lambda lam: _bpb(acc[lam]))

    aprendiz = load_light(aprendiz_ruta) if aprendiz_ruta else None
    # Los núcleos ACT de CPU paran en la 1.ª iteración: el modo profundo ignora la parada (full_depth)
    ca = ComputoAdaptativo(umbral=umbral, iters_base=1, iters_profundo=2)
    ca4 = ComputoAdaptativo(umbral=umbral, iters_base=1, iters_profundo=4)
    siempre2 = ComputoAdaptativo(umbral=-math.inf, iters_base=1, iters_profundo=2)

    def con_estilo(tema, p):
        return est.anteponer(p, tema + " " + p[:200], max_bytes=600, excluir=p)

    def bpb_paso(**kw) -> float:
        lps = []
        for _tema, p, ref in cs:
            lps += logprobs_paso(m, p, ref, **kw)[0]
        return _bpb(lps)

    def bpb_estilo() -> float:
        lps = []
        for tema, p, ref in cs:
            lps += logprobs_paso(m, con_estilo(tema, p), ref)[0]
        return _bpb(lps)

    tecnicas = {}
    tecnicas["voraz"] = (lambda tema, p: generar(m, p, largo), lambda: bpb_paso())
    tecnicas["voraz 1 iteración"] = (lambda tema, p: generar(m, p, largo, n_iters=1), lambda: bpb_paso(n_iters=1))
    tecnicas["voraz 2 iteraciones"] = (lambda tema, p: generar(m, p, largo, n_iters=2), lambda: bpb_paso(n_iters=2))
    tecnicas["muestreo T=0,8 top-p 0,95"] = (lambda tema, p: generar(m, p, largo, temperature=0.8, top_p=0.95,
                                                                     seed=zlib.crc32(p.encode()) % 1000), None)
    tecnicas["siempre 2 iteraciones completas"] = (lambda tema, p: generar(m, p, largo, adaptativo=siempre2),
                                                   lambda: bpb_paso(adaptativo=siempre2))
    tecnicas["adaptativo 1→2 completas (mediana)"] = (lambda tema, p: generar(m, p, largo, adaptativo=ca),
                                                      lambda: bpb_paso(adaptativo=ca))
    tecnicas["adaptativo 1→4 completas (mediana)"] = (lambda tema, p: generar(m, p, largo, adaptativo=ca4),
                                                      lambda: bpb_paso(adaptativo=ca4))
    vs = verificadores_basicos(ngramas=ng) + [lex]
    tecnicas["MBR N=4 (chrF + verificadores)"] = (
        lambda tema, p: mejor_de_n(m, p, 4, largo, temperature=0.6, verificadores=vs).texto, None)
    tecnicas["contrastiva (aprendiz: sí mismo, 1 iteración)"] = (
        lambda tema, p: generar(m, p, largo, procesadores=[ProcesadorContrastivo(m, n_iters_aprendiz=1)]), None)
    if aprendiz is not None:
        tecnicas[f"contrastiva (aprendiz: {Path(aprendiz_ruta).stem})"] = (
            lambda tema, p: generar(m, p, largo, procesadores=[ProcesadorContrastivo(aprendiz, tau=1.0)]), None)
    tecnicas[f"kNN-LM λ={lam_best}"] = (
        lambda tema, p: generar(m, p, largo, procesadores=[ProcesadorKNN(st_ivf, lam=lam_best)]),
        lambda: res["knn"]["bpb_por_lambda"][str(lam_best)])
    tecnicas["ejemplos de estilo"] = (lambda tema, p: generar(m, con_estilo(tema, p), largo), bpb_estilo)
    esc = Escritor(m, verificadores=vs, n_candidatos=4, n_alternativas=4, temperatura=0.6, max_rondas=2)
    tecnicas["bucle de escritura (MBR + 2 rondas)"] = (lambda tema, p: esc.escribir(p, tema=tema, largo=largo).texto,
                                                     None)
    from hope.runtime.budget import SpeedTarget
    esc_t = Escritor(m, verificadores=vs, n_candidatos=4, n_alternativas=4, temperatura=0.6, max_rondas=2,
                     objetivo=SpeedTarget(tokens_per_s=25.0))
    tecnicas["bucle de escritura a 25 tok/s"] = (lambda tema, p: esc_t.escribir(p, tema=tema, largo=largo).texto,
                                                 None)
    contra = (lambda: [ProcesadorContrastivo(aprendiz, tau=1.0)]) if aprendiz is not None else \
        (lambda: [ProcesadorContrastivo(m, n_iters_aprendiz=1)])
    gen_comb = generador_paso(m, procesadores_factory=lambda: contra() + [ProcesadorKNN(st_ivf, lam=lam_best)],
                              adaptativo=ca, top_p=0.95)
    esc_c = Escritor(m, verificadores=vs, generador=gen_comb, n_candidatos=3, n_alternativas=3, temperatura=0.6,
                     max_rondas=2)
    tecnicas["combinada (estilo + contrastiva + kNN + adaptativo + bucle)"] = (
        lambda tema, p: esc_c.escribir(con_estilo(tema, p), tema=tema, largo=largo).texto, None)
    tecnicas["contrastiva + kNN (voraz)"] = (
        lambda tema, p: generar(m, p, largo, procesadores=contra() + [ProcesadorKNN(st_ivf, lam=lam_best)]), None)

    for nombre, (fn, fbpb) in tecnicas.items():
        salidas, segs = [], []
        for tema, p, _ref in cs:
            t = time.perf_counter()
            out = fn(tema, p)
            segs.append(time.perf_counter() - t)
            salidas.append(out.decode("utf-8", errors="ignore") if isinstance(out, bytes) else out)
        bpb = fbpb() if fbpb else None
        extra = {}
        if nombre.startswith("adaptativo"):
            cfg = ca if "1→2" in nombre else ca4
            fr = [generar(m, p, largo, adaptativo=cfg, informe=True).stats for _t, p, _r in cs[:4]]
            extra["frac_parches_profundos"] = sum(s.parches_profundos for s in fr) / max(sum(s.parches for s in fr), 1)
        row = medir_salidas(nombre, salidas, segs, ng, lex, bpb, extra)
        res["tecnicas"].append(row)
        log(f"  {nombre}: rep4 {row['rep4']:.3f} · ng {row['ng_bits'] or 0:.2f} · desc {row['desconocidas']:.3f}"
            f" · dist2 {row['distinct2']:.3f} · bpb {bpb if bpb is None else round(bpb, 3)} · {row['segundos']:.2f} s")
        log(f"      «{row['ejemplo'][:120]}»")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelos", nargs="+", default=["checkpoints/hn1-cpu-mini.pt", "checkpoints/hn1-25m-cpu.pt"])
    ap.add_argument("--aprendiz", default="checkpoints/hn1-cpu-mini.pt",
                    help="checkpoint aprendiz para la contrastiva de los modelos que no son él mismo")
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--largo", type=int, default=200)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--salida", default="docs/resultados/lenguaje_datos.json")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.hilos)
    t0 = time.perf_counter()
    cs = casos(a.prompts, a.largo)
    alm = [r["texto"] for r in corpus_local(particion="almacen")]
    ng = ModeloNgramas(5).entrenar(alm)
    lex = Lexico.desde_corpus(alm)
    est = EjemplosEstilo.desde_corpus(min_nota=4.0)

    def log(s):
        print(s, flush=True)
    log(f"{len(cs)} casos · corpus de almacén {sum(len(t.encode()) for t in alm)} bytes · "
        f"{len(est.pas)} pasajes de estilo")
    out = {"casos": len(cs), "largo": a.largo, "hilos": a.hilos, "modelos": []}
    for ruta in a.modelos:
        apr = a.aprendiz if Path(a.aprendiz).resolve() != Path(ruta).resolve() else None
        out["modelos"].append(correr(ruta, apr, cs, ng, lex, est, alm, a.largo, log))
        Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    out["segundos_total"] = time.perf_counter() - t0
    Path(a.salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"listo en {out['segundos_total']:.0f} s → {a.salida}")


if __name__ == "__main__":
    main()
