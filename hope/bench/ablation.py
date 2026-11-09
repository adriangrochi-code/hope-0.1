"""Banco de ablación de la red experimental: ¿qué técnicas son viables para Hope?

Entrena variantes de HN-1 con **los mismos datos, la misma semilla y los mismos
pasos**, y mide bits por byte (validación), velocidad de generación y parámetros.
Las variantes salen del informe «Técnicas de generación rápida para Hope» y de la
hiperplasticidad. Los números de CPU son indicativos; la decisión se toma con la
corrida en la A770.

    python -m hope.bench.ablation --data data/f1 --preset hn1-25m --steps 2000
    python -m hope.bench.ablation --data data/f1 --variants base,ventana_gqa,convergencia
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from hope.core.hn1 import HN1Model, build_registry, preset
from hope.data.corpus import ByteData
from hope.train.continual import Mixture, Source, TrainConfig, Trainer


def _base(cfg):
    return cfg


def _ventana_gqa(cfg):
    cfg.pattern = tuple("swa" if k == "attn" else k for k in cfg.pattern[:-1]) + ("attn",)
    cfg.kv_heads = max(1, cfg.heads // 2)
    return cfg


def _local_ventana(cfg):
    cfg.local_kind, cfg.window = "swa", 128
    return cfg


def _convergencia(cfg):
    cfg.halting = "convergencia"
    return cfg


def _sesgo_compartido(cfg):
    cfg.balance, cfg.shared_expert = "sesgo", True
    return cfg


def _entropia(cfg):
    cfg.patching = "entropia"
    return cfg


VARIANTS = {
    "base": (_base, "configuración actual"),
    "ventana_gqa": (_ventana_gqa, "atención global solo en la última capa, ventana + sinks en el resto, GQA"),
    "local_ventana": (_local_ventana, "codificador/decodificador de bytes con ventana deslizante"),
    "convergencia": (_convergencia, "profundidad muestreada + parada por convergencia"),
    "sesgo_compartido": (_sesgo_compartido, "balanceo por sesgo + especialista compartido"),
    "entropia": (_entropia, "parches por entropía (estilo BLT)"),
    "rigl": (_base, "conectividad dinámica en las zonas (densidad 50 %)"),
}


def run_variant(name: str, base_cfg, train_data, val_data, *, steps: int, batch: int, seq: int,
                precision: str, seed: int = 0, gen_bytes: int = 64) -> dict:
    cfg = VARIANTS[name][0](copy.deepcopy(base_cfg))
    torch.manual_seed(seed)
    model = HN1Model(cfg)
    if cfg.patching == "entropia":
        model.fit_patcher(train_data.data)
    tr = Trainer(model, TrainConfig(lr=1e-3, warmup=max(steps // 20, 1), batch=batch, seq_bytes=seq,
                                    precision=precision, seed=seed), device="cpu")
    if name == "rigl":
        from hope.plasticity.dynamic import DynamicSparsity, HyperplasticityController
        tr.hooks.append(HyperplasticityController(model, build_registry(model),
                                                  sparsity=DynamicSparsity(model, density=0.5), rewire_every=50))
    t0 = time.time()
    tr.train(Mixture([Source("corpus", train_data, 1.0)]), steps, eval_sources=[])
    train_s = time.time() - t0
    val_bpb = tr.bpb(val_data, windows=32, seq=seq)
    model.eval()
    prompt = bytes(val_data.data[:256])
    t0 = time.time()
    out = model.generate(prompt, max_new=gen_bytes, stop=b"\x01")
    gen_s = time.time() - t0
    return {"variant": name, "description": VARIANTS[name][1], "val_bpb": val_bpb,
            "train_seconds": train_s, "generation_bytes_per_s": len(out) / max(gen_s, 1e-9),
            "params": sum(p.numel() for p in model.parameters())}


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--preset", default="hn1-25m")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-bytes", type=int, default=512)
    ap.add_argument("--precision", default="bf16-8bit")
    ap.add_argument("--out", default="runs/ablation")
    args = ap.parse_args(argv)
    d = Path(args.data)
    train, val = ByteData(d / "train.bin"), ByteData(d / "val.bin")
    results = []
    for name in args.variants.split(","):
        r = run_variant(name, preset(args.preset), train, val, steps=args.steps, batch=args.batch,
                        seq=args.seq_bytes, precision=args.precision)
        results.append(r)
        print(f"{name:18s} bpb {r['val_bpb']:.3f}  gen {r['generation_bytes_per_s']:7.1f} B/s  "
              f"train {r['train_seconds']:6.0f} s  params {r['params'] / 1e6:.2f}M", flush=True)
    base = next((r for r in results if r["variant"] == "base"), None)
    lines = ["| variante | descripción | val BPB | Δ BPB vs base | bytes/s generación | parámetros |",
             "|---|---|---|---|---|---|"]
    for r in results:
        delta = f"{r['val_bpb'] - base['val_bpb']:+.3f}" if base else "—"
        lines.append(f"| {r['variant']} | {r['description']} | {r['val_bpb']:.3f} | {delta} | "
                     f"{r['generation_bytes_per_s']:.1f} | {r['params'] / 1e6:.2f}M |")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return results


if __name__ == "__main__":
    main()
