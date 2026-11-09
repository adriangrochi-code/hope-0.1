"""Ablación de la zona visual en CPU (pedido del usuario: mejorar la calidad en imágenes sin agrandar la red).

Variantes, **todas con el mismo modelo, los mismos pasos y las mismas semillas**, medidas con el
banco sellado ``hope.eval.vision_bench``:

* ``base``        — los generadores originales (formas, texto, barras), sin mosaicos.
* ``datos``       — + los generadores nuevos (circuitos, tablas, líneas, tortas, OCR, relojes,
                    geometría, escenas) con aumentos; los documentos se reducen a la resolución
                    del modelo.
* ``mosaicos``    — ``datos`` + documentos a 2× en mosaicos (miniatura + 4 mosaicos).
* ``contrastiva`` — ``datos`` + pérdida auxiliar imagen–texto estilo SigLIP (peso 0,1).
* ``curriculo``   — ``datos`` + currículo de imágenes simples a complejas.
* ``todo``        — ``datos`` + mosaicos + contrastiva + currículo.

Cada mejora se suma a ``datos`` por separado (así se aísla su efecto); ``todo`` mide si se
combinan. Ninguna agrega parámetros.

    python -m hope.bench.vision_ablation --out runs/vision_ablation --steps 1500 --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from hope.core.hn1 import HN1Model, preset
from hope.data.vision import BASIC_KINDS, GENERATORS, ImageTextSource
from hope.eval import vision_bench as vb
from hope.train.continual import Mixture, Source, TrainConfig, Trainer

VARIANTS = {
    "base": dict(kinds=BASIC_KINDS),
    "datos": dict(),
    "mosaicos": dict(tiles=True),
    "contrastiva": dict(contrastive=0.1),
    "curriculo": dict(curriculum=True),
    "todo": dict(tiles=True, contrastive=0.1, curriculum=True),
}


def model_config(name: str = "hn1-cpu", image: int = 96, patch: int = 8, tiles: bool = False):
    return replace(preset(name), vision=True, vis_d=128, vis_layers=2, vis_heads=4, vis_image=image,
                   vis_patch=patch, vis_tiling=tiles)


def run_variant(name: str, seed: int, bench: list, *, steps: int, images: int, image: int, batch: int, seq: int,
                lr: float, preset_name: str, tools: bool, max_new: int, text: str, log=print) -> dict:
    v = VARIANTS[name]
    tiles = v.get("tiles", False)
    torch.manual_seed(seed)
    model = HN1Model(model_config(preset_name, image, tiles=tiles))
    t0 = time.time()
    src = ImageTextSource.from_procedural(images, seed=100 + seed, size=image, name=name,
                                          kinds=v.get("kinds", tuple(GENERATORS)),
                                          doc_scale=2, augment_p=0.0 if name == "base" else 0.3, tiles=tiles)
    leak = vb.contamination(bench, (it.image for it in src.items))
    if leak:
        raise RuntimeError(f"{leak} imágenes del banco aparecen en el entrenamiento")
    vision_frac = 0.9
    if v.get("curriculum"):
        src.curriculum(int(steps * vision_frac), warm=0.6)
    data_s = time.time() - t0
    tr = Trainer(model, TrainConfig(lr=lr, batch=batch, warmup=max(steps // 20, 1), precision="fp32", seed=seed,
                                    seq_bytes=seq), device="cpu")
    t0 = time.time()
    tr.train(Mixture([Source.from_texts("texto", [text], 1.0)]), steps, vision=src, vision_frac=vision_frac,
             vision_seq=seq, eval_sources=[], vision_contrastive=v.get("contrastive", 0.0),
             log=lambda m: log(f"  [{name} s{seed}] {m}"))
    train_s = time.time() - t0
    rep = vb.evaluate(model, bench, tools=tools, max_new=max_new)
    return {"variante": name, "semilla": seed, "params": sum(p.numel() for p in model.parameters()),
            "muestras": len(src.items), "segundos_datos": data_s, "segundos_entrenamiento": train_s,
            "segundos_eval": rep["segundos"], "reporte": rep}


def summarize(results: list[dict]) -> str:
    """Tabla de medias (y rango entre semillas) por variante y tarea."""
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r["variante"], []).append(r["reporte"])
    tasks = vb.TASKS
    f = lambda xs: f"{np.mean(xs):.3f}" + (f" ±{(max(xs) - min(xs)) / 2:.3f}" if len(xs) > 1 else "")  # noqa: E731
    lines = ["#### Exactitud (generación con herramientas; texto = exactitud por carácter)", "",
             "| variante | semillas | " + " | ".join(tasks) + " | promedio |",
             "|---|---|" + "---|" * (len(tasks) + 1)]
    for name, reps in by.items():
        row = [f([rp["tareas"][t]["exactitud"] for rp in reps]) for t in tasks]
        lines.append(f"| {name} | {len(reps)} | " + " | ".join(row) + f" | {f([rp['promedio_exactitud'] for rp in reps])} |")
    lines += ["", "#### Ventaja de la imagen (bpb con imagen cambiada − bpb con la correcta; más es mejor)", "",
              "| variante | " + " | ".join(tasks) + " | promedio |", "|---|" + "---|" * (len(tasks) + 1)]
    for name, reps in by.items():
        row = [f([rp["tareas"][t]["ventaja_imagen"] for rp in reps]) for t in tasks]
        lines.append(f"| {name} | " + " | ".join(row) + f" | {f([rp['promedio_ventaja_imagen'] for rp in reps])} |")
    lines += ["", "#### bpb de la respuesta de referencia con la imagen correcta (menos es mejor)", "",
              "| variante | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for name, reps in by.items():
        lines.append(f"| {name} | " + " | ".join(f([rp["tareas"][t]["bpb"] for rp in reps]) for t in tasks) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/vision_ablation")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--images", type=int, default=3000, help="imágenes procedurales de entrenamiento por corrida")
    ap.add_argument("--image", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--preset", default="hn1-cpu")
    ap.add_argument("--bench-n", type=int, default=30, help="preguntas por tarea en el banco")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--sin-herramientas", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Banco sellado: se construye una vez, se sella y cada lectura queda registrada.
    from hope.eval.sealed import SealedRegistry
    reg = SealedRegistry(out / "sellado")
    bench_path = out / f"banco_visual_{args.image}.pt"
    name = bench_path.stem
    if name not in json.loads((out / "sellado" / "manifest.json").read_text(encoding="utf-8")):
        vb.save(vb.build(args.bench_n, args.image), bench_path, {"size": args.image, "n_por_tarea": args.bench_n})
        reg.seal(name, bench_path, max_decision_reads=50, description="banco visual de la ablación")
    bench = vb.load(bench_path, registry=reg, name=name, requester="vision_ablation", decision=True)
    text = Path(__file__).resolve().parents[2].joinpath("docs", "ARQUITECTURA_HN1.md").read_text(encoding="utf-8")
    res_path = out / "results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else []
    done = {(r["variante"], r["semilla"]) for r in results}
    for seed in (int(s) for s in args.seeds.split(",")):
        for name in args.variants.split(","):
            if (name, seed) in done:
                continue
            print(f"== {name} semilla {seed}", flush=True)
            r = run_variant(name, seed, bench, steps=args.steps, images=args.images, image=args.image,
                            batch=args.batch, seq=args.seq, lr=args.lr, preset_name=args.preset,
                            tools=not args.sin_herramientas, max_new=args.max_new, text=text,
                            log=lambda m: print(m, flush=True))
            results.append(r)
            res_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(vb.table(r["reporte"]), flush=True)
    md = summarize(results)
    (out / "report.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return results


if __name__ == "__main__":
    main()
