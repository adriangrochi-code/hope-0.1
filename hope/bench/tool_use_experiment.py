"""Experimento en CPU: ¿el currículo de herramientas mejora el uso de herramientas?
(docs/resultados/uso_herramientas.md)

Parte del mismo modelo (por defecto Hope-mini, ``checkpoints/hn1-cpu-mini.pt``) y entrena el
mismo número de pasos con dos mezclas:

* **sin**: episodios de los generadores con llamadas (``Episode.training_text(tool_calls=True)``),
  que es lo que Hope ya tenía para aprender a usar herramientas;
* **con**: lo mismo + el currículo de herramientas (``hope/data/tool_curriculum.py``), mitad y mitad.

Después mide los dos (y el modelo base, y la solución de referencia como cota superior) con el
banco sellado (``hope/eval/tool_bench.py``). La variante «con + TLM» repite el banco del modelo
«con» agregando en el contexto las herramientas sugeridas por la memoria de uso.

    python -m hope.bench.tool_use_experiment --pasos 1500 --salida runs/exp_herramientas
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def generator_texts(n_per_family: int, exclude: set[str], seed: int = 7) -> list[str]:
    from hope.data.generators import FAMILIES, generate
    from hope.tools.builtin import default_catalog
    from hope.tools.verify import verify_episode
    cat = default_catalog()
    out = []
    for fam in FAMILIES:
        for level in (1, 2):
            for ep in generate(fam, n_per_family, seed=seed, level=level):
                if ep.data["problema"] in exclude:
                    continue
                verify_episode(cat, ep)
                if ep.verification["status"] != "REFUTADO":
                    out.append(ep.training_text(tool_calls=True))
    return out


def main(argv: list[str] | None = None) -> dict:
    from hope.data.tool_curriculum import bench_problems, build_examples
    from hope.eval.tool_bench import DEFAULT_BENCH, load_bench, reference_generate, run_bench, subset_per_category
    from hope.tools.tlm import ToolUsageMemory
    from hope.train.continual import Mixture, Source, TrainConfig, Trainer
    from hope.train.run import export_light, load_light

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="checkpoints/hn1-cpu-mini.pt")
    ap.add_argument("--pasos", type=int, default=1500)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--secuencia", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-curriculo", type=int, default=6000)
    ap.add_argument("--n-generador", type=int, default=300, help="episodios por familia y nivel")
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--salida", default="runs/exp_herramientas")
    ap.add_argument("--solo", default=None, help="sin | con (correr una sola variante)")
    ap.add_argument("--items-banco", type=int, default=None, help="solo N ítems por categoría (prueba rápida)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.hilos)
    out = Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)
    bench = load_bench(DEFAULT_BENCH)
    if args.items_banco:
        bench = subset_per_category(bench, args.items_banco)
    exclude = bench_problems()
    log = lambda m: print(m, flush=True)  # noqa: E731

    t0 = time.time()
    gen = generator_texts(args.n_generador, exclude)
    tlm = ToolUsageMemory(out / "tlm_curriculo.json")
    cur = build_examples(args.n_curriculo, seed="experimento", exclude=exclude, tlm=tlm, log=log,
                         teacher_path="datos_maestro/episodios_verificados.jsonl")
    tlm.save()
    data = {"generadores_textos": len(gen), "generadores_bytes": sum(len(t.encode()) for t in gen),
            "curriculo_textos": len(cur), "curriculo_bytes": sum(len(c.text.encode()) for c in cur),
            "segundos_datos": round(time.time() - t0, 1)}
    log(json.dumps(data, ensure_ascii=False))
    mixes = {"sin": [Source.from_texts("generadores", gen, 1.0)],
             "con": [Source.from_texts("generadores", gen, 0.5),
                     Source.from_texts("herramientas", [c.text for c in cur], 0.5)]}
    results = {"datos": data, "config": vars(args)}
    base = load_light(args.base)
    results["base"] = run_bench(base.generate, bench, log=log)["resumen"]
    results["referencia"] = run_bench(reference_generate(bench), bench)["resumen"]
    log("base: " + json.dumps({k: v for k, v in results["base"].items() if k != "por_categoria"}))
    for name, sources in mixes.items():
        if args.solo and name != args.solo:
            continue
        model = load_light(args.base)
        tr = Trainer(model, TrainConfig(lr=args.lr, warmup=50, batch=args.lote, seq_bytes=args.secuencia,
                                        precision="fp32", optimizer="muon", schedule="cosine", seed=0), "cpu")
        t1 = time.time()
        rep = tr.train(Mixture(sources), args.pasos, log=log)
        export_light(tr.model, out / f"{name}.pt")
        res = run_bench(tr.model.generate, bench, log=log)
        results[name] = {"entrenamiento": {"pasos": rep.steps, "segundos": round(time.time() - t1, 1),
                                           "bpb_final": rep.train_bpb_last, "bpb_val": rep.bpb_after},
                         **res["resumen"]}
        (out / f"banco_{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"{name}: " + json.dumps({k: v for k, v in res["resumen"].items() if k != "por_categoria"}))
        if name == "con":
            import shutil
            shutil.copy(out / "tlm_curriculo.json", out / "tlm_banco.json")      # la del currículo queda intacta
            res_t = run_bench(tr.model.generate, bench, tlm=ToolUsageMemory(out / "tlm_banco.json"), log=log)
            results["con_tlm"] = res_t["resumen"]
            (out / "banco_con_tlm.json").write_text(json.dumps(res_t, indent=1, ensure_ascii=False), encoding="utf-8")
            log("con + TLM: " + json.dumps({k: v for k, v in res_t["resumen"].items() if k != "por_categoria"}))
        (out / "resultados.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
