"""Banco de ablación de la red experimental: ¿qué técnicas son viables para Hope?

Entrena variantes de HN-1 con **los mismos datos, la misma semilla y los mismos
pasos**, y mide bits por byte (validación), velocidad de generación y parámetros.
Las variantes salen del informe «Técnicas de generación rápida para Hope» y de la
hiperplasticidad. Los números de CPU son indicativos; la decisión se toma con la
corrida en la A770.

    python -m hope.bench.ablation --data data/f1 --preset hn1-25m --steps 2000
    python -m hope.bench.ablation --data data/f1 --variants base,ventana_gqa,convergencia
    python -m hope.bench.ablation --data data/pre/mix --preset hn1-cpu --steps 1500 \
        --variants base,bucle_estable,bucle_estable_ema --seeds 0,1 \
        --extra cadenas=data/cadenas:0.2 --depth-eval

``--depth-eval`` mide además la **generalización de profundidad**: exactitud en cadenas de
``hope.data.generators.chain_set`` con 1–8 pasos (se entrena con 1–4) y con 1 a 32 iteraciones
del núcleo en inferencia (más que las entrenadas).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from hope.core.hn1 import HN1Model, build_registry, preset, stable_loop
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


def _multibyte(cfg):
    cfg.multibyte = 3
    return cfg


def _bucle_estable(cfg):
    return stable_loop(cfg)


VARIANTS = {
    "base": (_base, "configuración actual"),
    "ventana_gqa": (_ventana_gqa, "atención global solo en la última capa, ventana + sinks en el resto, GQA"),
    "local_ventana": (_local_ventana, "codificador/decodificador de bytes con ventana deslizante"),
    "convergencia": (_convergencia, "profundidad muestreada + parada por convergencia"),
    "sesgo_compartido": (_sesgo_compartido, "balanceo por sesgo + especialista compartido"),
    "entropia": (_entropia, "parches por entropía (estilo BLT)"),
    "rigl": (_base, "conectividad dinámica en las zonas (densidad 50 %)"),
    "muon": (_base, "optimizador Muon en las matrices ocultas (hope/train/muon.py)"),
    "wsd": (_base, "tasa constante + enfriamiento 1−√p en el último 20 % (WSD)"),
    "multibyte": (_multibyte, "cabezas auxiliares que predicen 2–4 bytes adelante"),
    "bucle_estable": (_bucle_estable, "núcleo con reinyección convexa de la entrada, sin step_embed, "
                                      "profundidad log-normal (media 4, tope 12), parada por convergencia"),
    "bucle_estable_ema": (_bucle_estable, "bucle_estable + EMA de pesos (0,998); se evalúa con la EMA"),
}
EMA_DECAY = 0.998
DEPTH_STEPS = (1, 2, 3, 4, 5, 6, 8)
DEPTH_ITERS = (1, 2, 4, 8, 16, 32)


@torch.no_grad()
def answer_accuracy(model, problems: list[tuple[str, str]], n_iters: int | None = None, batch: int = 128) -> dict:
    """Exactitud de la respuesta de un dígito (voraz, un byte) y media de iteraciones del núcleo.
    Los enunciados se rellenan a la derecha: el modelo es causal, el relleno no cambia nada."""
    model.eval()
    dev = model.head.weight.device
    hits, iters, total = 0, 0.0, 0
    for i in range(0, len(problems), batch):
        chunk = problems[i:i + batch]
        enc = [q.encode("utf-8") for q, _ in chunk]
        width = max(len(e) for e in enc)
        x = torch.zeros(len(enc), width, dtype=torch.long, device=dev)
        for j, e in enumerate(enc):
            x[j, :len(e)] = torch.tensor(list(e), device=dev)
        out = model(x, n_iters=n_iters)
        last = torch.tensor([len(e) - 1 for e in enc], device=dev)
        pred = out["logits"][torch.arange(len(enc), device=dev), last].float().argmax(-1).tolist()
        hits += sum(bytes([p]).decode("latin-1") == a for p, (_, a) in zip(pred, chunk))
        iters += float(out["expected_iters"]) * len(enc)
        total += len(enc)
    return {"exactitud": hits / max(total, 1), "iteraciones_medias": iters / max(total, 1)}


def depth_eval(model, *, n: int = 200, seed: int = 123) -> dict:
    """Exactitud por cantidad de pasos y por iteraciones del núcleo en inferencia.

    Con ``core_step="embed"`` no se puede iterar más que ``max_iters`` (queda ``None``). En el
    bucle estable se mide con parada por convergencia («conv») y con profundidad fija («fija»)."""
    from hope.data.generators import chain_set
    sets = {k: chain_set(n, k, seed=seed) for k in DEPTH_STEPS}
    core = model.core
    can_extend = model.cfg.core_step != "embed"
    modes = {"conv": core.tol} if model.cfg.halting == "convergencia" else {"act": None}
    if model.cfg.halting == "convergencia":
        modes["fija"] = 0.0
    res = {}
    for mode, tol in modes.items():
        if tol is not None:
            old_tol, core.tol = core.tol, tol
        table = {}
        for it in DEPTH_ITERS:
            if it > model.cfg.max_iters and not can_extend:
                table[str(it)] = None
                continue
            table[str(it)] = {str(k): answer_accuracy(model, probs, n_iters=it) for k, probs in sets.items()}
        if tol is not None:
            core.tol = old_tol
        res[mode] = table
    return res


def run_variant(name: str, base_cfg, train_data, val_data, *, steps: int, batch: int, seq: int,
                precision: str, seed: int = 0, gen_bytes: int = 64, device: str = "cpu",
                extra: list[Source] | None = None, val_extra: dict | None = None, depth: bool = False) -> dict:
    cfg = VARIANTS[name][0](copy.deepcopy(base_cfg))
    torch.manual_seed(seed)
    model = HN1Model(cfg)
    if cfg.patching == "entropia":
        model.fit_patcher(train_data.data)
    tr = Trainer(model, TrainConfig(lr=1e-3, warmup=max(steps // 20, 1), batch=batch, seq_bytes=seq,
                                    precision=precision, seed=seed,
                                    optimizer="muon" if name == "muon" else "adamw",
                                    schedule="wsd" if name == "wsd" else "cosine",
                                    ema=EMA_DECAY if name.endswith("_ema") else None), device=device)
    if name == "rigl":
        from hope.plasticity.dynamic import DynamicSparsity, HyperplasticityController
        tr.hooks.append(HyperplasticityController(model, build_registry(model),
                                                  sparsity=DynamicSparsity(model, density=0.5), rewire_every=50))
    t0 = time.time()
    extra = extra or []
    mix = Mixture([Source("corpus", train_data, 1.0 - sum(e.weight for e in extra))] + list(extra))
    if name == "wsd":
        tr.train(mix, steps - steps // 5, eval_sources=[])
        tr.anneal(mix, steps // 5, eval_sources=[])
    else:
        tr.train(mix, steps, eval_sources=[])
    train_s = time.time() - t0
    raw_bpb = tr.bpb(val_data, windows=32, seq=seq) if tr.ema is not None else None
    if tr.ema is not None:                      # se evalúa (y se mide la generación) con la EMA
        tr.copy_ema_to_model()
    val_bpb = tr.bpb(val_data, windows=32, seq=seq)
    per_source = {k: tr.bpb(v, windows=32, seq=seq) for k, v in (val_extra or {}).items()}
    t1 = time.time()
    depth_res = depth_eval(model) if depth else None
    depth_s = time.time() - t1
    model.eval()
    prompt = bytes(val_data.data[:256])
    t0 = time.time()
    out = model.generate(prompt, max_new=gen_bytes, stop=b"\x01")
    gen_s = time.time() - t0
    return {"variant": name, "seed": seed, "description": VARIANTS[name][1], "val_bpb": val_bpb,
            "val_bpb_sin_ema": raw_bpb, "val_bpb_por_fuente": per_source,
            "train_seconds": train_s, "generation_bytes_per_s": len(out) / max(gen_s, 1e-9),
            "params": sum(p.numel() for p in model.parameters()), "profundidad": depth_res,
            "depth_seconds": depth_s}


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
    ap.add_argument("--device", default="cpu", help="cpu | xpu | cuda")
    ap.add_argument("--seeds", default="0", help="semillas separadas por comas")
    ap.add_argument("--extra", action="append", default=[],
                    help="fuente extra nombre=carpeta:peso (su val.bin también se evalúa)")
    ap.add_argument("--val-dirs", default="", help="carpetas extra cuyo val.bin se evalúa por separado")
    ap.add_argument("--depth-eval", action="store_true", help="generalización de profundidad (cadenas)")
    args = ap.parse_args(argv)
    d = Path(args.data)
    train, val = ByteData(d / "train.bin"), ByteData(d / "val.bin")
    extra, val_extra = [], {}
    for spec in args.extra:
        nm, rest = spec.split("=", 1)
        folder, w = rest.rsplit(":", 1)
        extra.append(Source(nm, ByteData(Path(folder) / "train.bin"), float(w)))
        val_extra[nm] = ByteData(Path(folder) / "val.bin")
    for folder in filter(None, args.val_dirs.split(",")):
        val_extra[Path(folder).name] = ByteData(Path(folder) / "val.bin")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "results.json"
    results = json.loads(done_path.read_text()) if done_path.exists() else []    # se puede retomar
    for seed in map(int, args.seeds.split(",")):
        for name in args.variants.split(","):
            if any(r["variant"] == name and r.get("seed", 0) == seed for r in results):
                continue
            r = run_variant(name, preset(args.preset), train, val, steps=args.steps, batch=args.batch,
                            seq=args.seq_bytes, precision=args.precision, device=args.device, seed=seed,
                            extra=extra, val_extra=val_extra, depth=args.depth_eval)
            results.append(r)
            done_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"{name:18s} semilla {seed}  bpb {r['val_bpb']:.3f}  gen {r['generation_bytes_per_s']:7.1f} B/s  "
                  f"train {r['train_seconds']:6.0f} s  params {r['params'] / 1e6:.2f}M", flush=True)
    lines = report_lines(results)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return results


def report_lines(results: list[dict]) -> list[str]:
    """Tabla con la media (y el rango) entre semillas, y la diferencia con la base de la misma semilla."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in results:
        by[r["variant"]].append(r)
    base = {r.get("seed", 0): r["val_bpb"] for r in by.get("base", [])}
    lines = ["| variante | semillas | val BPB (media) | rango | Δ BPB vs base (media por semilla) | "
             "bytes/s generación | parámetros |", "|---|---|---|---|---|---|---|"]
    for name, rs in by.items():
        b = [r["val_bpb"] for r in rs]
        deltas = [r["val_bpb"] - base[r.get("seed", 0)] for r in rs if r.get("seed", 0) in base]
        delta = f"{sum(deltas) / len(deltas):+.3f}" if deltas and name != "base" else "—"
        gen = sum(r["generation_bytes_per_s"] for r in rs) / len(rs)
        lines.append(f"| {name} | {len(rs)} | {sum(b) / len(b):.3f} | {min(b):.3f}–{max(b):.3f} | {delta} | "
                     f"{gen:.1f} | {rs[0]['params'] / 1e6:.2f}M |")
    if any(r.get("profundidad") for r in results):
        lines += ["", "Exactitud en cadenas (entrenadas con 1–4 pasos), media entre semillas; filas = "
                  "iteraciones del núcleo en inferencia, columnas = pasos del problema:", ""]
        for name, rs in by.items():
            for mode in (rs[0].get("profundidad") or {}):
                lines.append(f"**{name} ({mode})**")
                lines.append("")
                steps = [str(k) for k in DEPTH_STEPS]
                lines.append("| iteraciones | " + " | ".join(f"{k} pasos" for k in steps) + " | iter. medias |")
                lines.append("|---|" + "---|" * (len(steps) + 1))
                for it in map(str, DEPTH_ITERS):
                    cells = [r["profundidad"][mode].get(it) for r in rs]
                    if any(c is None for c in cells):
                        lines.append(f"| {it} | " + " | ".join("—" for _ in steps) + " | — |")
                        continue
                    acc = [sum(c[k]["exactitud"] for c in cells) / len(cells) for k in steps]
                    its = sum(c[k]["iteraciones_medias"] for c in cells for k in steps) / (len(cells) * len(steps))
                    lines.append(f"| {it} | " + " | ".join(f"{a:.2f}" for a in acc) + f" | {its:.1f} |")
                lines.append("")
    return lines


if __name__ == "__main__":
    main()
