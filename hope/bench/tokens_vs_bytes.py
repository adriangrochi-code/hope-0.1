"""¿Aprende Hope más rápido por dato con parches = tokens de Qwen y destilación? (pedido del usuario)

Compara, con **los mismos datos, el mismo tiempo de entrenamiento y Muon + WSD para todos**:

* ``espacios``  — bytes + parches por espacios y puntuación (lo actual);
* ``entropia``  — bytes + parches por entropía (n-gramas, estilo BLT);
* ``tokens``    — bytes + parches = tokens de Qwen2.5 con la regla causal de ``hope/core/tokpatch.py``;
* ``tokens_kl`` — ``tokens`` + destilación KL del próximo token de Qwen2.5-0.5B (``hope/train/distill.py``);
* ``bpe``       — línea base clásica: transformer sobre tokens BPE propios (``hope/core/baseline.py``).

Mide BPB de validación (sobre las **mismas ventanas de bytes** para todos: es la única métrica
comparable entre unidades distintas), bytes/s de entrenamiento, bytes/s de generación y la curva
BPB vs. bytes vistos. El tiempo de evaluación no cuenta en el presupuesto.

    python -m hope.bench.tokens_vs_bytes --data data/tvb/mix --logits data/tvb/maestro_qwen05 \\
        --preset hn1-cpu --minutes 15 --seeds 0,1 --out runs/tvb
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hope.core.baseline import LN2, BaselineConfig, TokenLM, load_tokenizer, token_byte_lengths, train_tokenizer
from hope.core.hn1 import HN1Model, preset
from hope.train.lowprec import prepare

VARIANTS = {
    "espacios": "bytes + parches por espacios (actual)",
    "entropia": "bytes + parches por entropía",
    "tokens": "bytes + parches = tokens de Qwen2.5 (regla causal)",
    "tokens_kl": "tokens + destilación KL del próximo token (Qwen2.5-0.5B, top-8, vocabulario reducido)",
    "bpe": "línea base: transformer sobre BPE propio (8192), mismo orden de parámetros",
}


# ---------------------------------------------------------------------------
# Ventanas de validación (las mismas para todos)
# ---------------------------------------------------------------------------
def _char_start(arr: np.ndarray, i: int) -> int:
    while i < len(arr) and arr[i] & 0xC0 == 0x80:
        i += 1
    return i


def val_windows(arr: np.ndarray, n: int, length: int) -> list[np.ndarray]:
    """``n`` ventanas de ``length + 1`` bytes repartidas en todo el archivo (español, inglés y código),
    que empiezan y terminan en un límite de carácter UTF-8."""
    starts = np.linspace(0, len(arr) - length - 8, n).astype(np.int64)
    out = []
    for s in starts:
        a = _char_start(arr, int(s))
        b = _char_start(arr, a + length + 1)
        out.append(np.asarray(arr[a:b], dtype=np.int64))
    return out


@torch.no_grad()
def hn1_bpb(model: HN1Model, windows: list[np.ndarray], batch: int = 16) -> float:
    model.eval()
    ce, nb = 0.0, 0
    for i in range(0, len(windows), batch):
        group = windows[i:i + batch]
        width = min(len(w) for w in group)
        x = torch.from_numpy(np.stack([w[:width] for w in group]))
        out = model(x[:, :-1])
        loss = F.cross_entropy(out["logits"].float().reshape(-1, 256), x[:, 1:].reshape(-1), reduction="sum")
        ce += loss.item()
        nb += x[:, 1:].numel()
    return ce / nb / LN2


def bpe_ids(tok, raw: bytes) -> list[int]:
    eod = tok.token_to_id("<eod>")
    ids: list[int] = []
    for k, doc in enumerate(raw.split(b"\x00")):
        if k:
            ids.append(eod)
        if doc:
            ids.extend(tok.encode(doc.decode("utf-8", errors="replace")).ids)
    return ids


@torch.no_grad()
def bpe_bpb(model: TokenLM, tok, tb: torch.Tensor, windows: list[np.ndarray]) -> float:
    """BPB de la línea base sobre las mismas ventanas de bytes: se tokeniza cada ventana y se cuentan
    los bytes de los tokens predichos (todos menos el primero)."""
    model.eval()
    ce, nb = 0.0, 0.0
    for w in windows:
        ids = torch.tensor([bpe_ids(tok, bytes(w.astype(np.uint8)))])
        if ids.shape[1] < 2:
            continue
        logits = model(ids[:, :-1])
        ce += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                              reduction="sum").item()
        nb += tb[ids[:, 1:]].sum().item()
    return ce / nb / LN2


# ---------------------------------------------------------------------------
# Generación de la línea base (caché de claves/valores; voraz)
# ---------------------------------------------------------------------------
@torch.no_grad()
def bpe_generate(model: TokenLM, tok, prompt: bytes, max_bytes: int, tb: torch.Tensor) -> tuple[bytes, int]:
    model.eval()
    caches = [{} for _ in model.blocks]

    def step(ids):
        h = model.embed(ids)
        for b, c in zip(model.blocks, caches):
            h = b(h, c)
        return model.head(model.norm(h))[0, -1]

    logits = step(torch.tensor([bpe_ids(tok, prompt)]))
    out, nbytes = [], 0
    while nbytes < max_bytes:
        nxt = int(logits.float().argmax())
        out.append(nxt)
        nbytes += int(tb[nxt])
        logits = step(torch.tensor([[nxt]]))
    return tok.decode(out).encode("utf-8"), nbytes


# ---------------------------------------------------------------------------
def _wsd(peak: float, step: int, warmup: int, frac: float, decay_frac: float = 0.2) -> float:
    """WSD por tiempo: calentamiento por pasos, constante, y enfriamiento 1 − √p en el último 20 %."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if frac < 1 - decay_frac:
        return peak
    p = min((frac - (1 - decay_frac)) / decay_frac, 1.0)
    return peak * max(0.0, 1 - math.sqrt(p))


def run(name: str, args, train: np.ndarray, val_arr: np.ndarray, seed: int, *, teacher=None,
        bpe=None, log=print) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    curve_w = val_windows(val_arr, args.curve_windows, args.seq_bytes)
    final_w = val_windows(val_arr, args.final_windows, args.seq_bytes)
    distiller = None
    if name == "bpe":
        tok, tok_ids, tb, avg = bpe
        cfg = BaselineConfig(vocab=tok.get_vocab_size(), d=args.bpe_d, heads=args.bpe_heads,
                             mlp_hidden=args.bpe_mlp, layers=args.bpe_layers)
        model = TokenLM(cfg)
        seq_tok = max(8, round(args.seq_bytes / avg))

        def evaluate(ws):
            return bpe_bpb(model, tok, tb, ws)
    else:
        cfg = preset(args.preset)
        if name == "entropia":
            cfg = replace(cfg, patching="entropia")
        elif name.startswith("tokens"):
            cfg = replace(cfg, patching="tokens", tokenizer=args.tokenizer)
        if name == "tokens_kl":
            from hope.train.distill import Distiller
            cfg = replace(cfg, distill_vocab=args.vocab_k)
            distiller = Distiller(teacher, weight=args.kl_weight)
        model = HN1Model(cfg)
        if name == "entropia":
            log(f"  parcheo por entropía: {model.fit_patcher(train)}")

        def evaluate(ws):
            return hn1_bpb(model, ws)
    params = sum(p.numel() for p in model.parameters())
    params_infer = params - (model.distill_head.weight.numel() if getattr(model, "distill_head", None) is not None else 0)
    opt = prepare(model, args.precision, lr=args.lr, weight_decay=0.1, seed=seed, optimizer="muon")
    budget = args.minutes * 60
    train_s, step, bytes_seen = 0.0, 0, 0.0
    next_eval = 0.0
    curve = []
    stats_last: dict = {}
    full_frac = getattr(model.cfg, "full_depth_frac", 0.0) if name != "bpe" else 0.0
    while train_s < budget and (args.max_steps is None or step < args.max_steps):
        if bytes_seen >= next_eval:
            curve.append({"bytes": bytes_seen, "segundos": train_s, "val_bpb": evaluate(curve_w)})
            log(f"  {name} s{seed}  {bytes_seen / 1e6:6.2f} MB  {train_s / 60:5.1f} min  "
                f"val {curve[-1]['val_bpb']:.3f}  {stats_last}")
            next_eval += args.eval_every_bytes
        t0 = time.time()
        lr = _wsd(args.lr, step, args.warmup, train_s / budget)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        model.train()
        if name == "bpe":
            starts = rng.integers(0, len(tok_ids) - seq_tok - 1, size=args.batch)
            x = torch.from_numpy(np.stack([tok_ids[s:s + seq_tok + 1] for s in starts]).astype(np.int64))
            loss, st = model.loss(x, tb)
            nb = float(tb[x[:, 1:]].sum())
        else:
            starts = rng.integers(0, len(train) - args.seq_bytes - 1, size=args.batch)
            x = torch.from_numpy(np.stack([train[s:s + args.seq_bytes + 1] for s in starts]).astype(np.int64))
            aux = distiller.for_batch(starts) if distiller is not None else None
            full = rng.random() < full_frac
            loss, st = model.loss(x, full_depth=full, ponder_scale=min(1.0, step / max(model.cfg.ponder_warmup, 1)),
                                  aux=aux)
            nb = float(x[:, 1:].numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        bytes_seen += nb
        train_s += time.time() - t0
        stats_last = {k: round(v, 3) for k, v in st.items() if k in ("bpb", "kl", "parches_alineados",
                                                                       "patches_per_byte")}
    final = evaluate(final_w)
    curve.append({"bytes": bytes_seen, "segundos": train_s, "val_bpb": final, "final": True})
    # Velocidad de generación (voraz, 128 bytes después de un prompt de 256 bytes de validación).
    prompt = bytes(val_arr[1000:1256].astype(np.uint8))
    t0 = time.time()
    if name == "bpe":
        _, ngen = bpe_generate(model, tok, prompt, args.gen_bytes, tb)
    else:
        model.eval()
        ngen = len(model.generate(prompt, max_new=args.gen_bytes, stop=b"\x01"))
    gen_s = time.time() - t0
    res = {"variant": name, "seed": seed, "description": VARIANTS[name], "val_bpb": final,
           "train_seconds": train_s, "steps": step, "bytes_seen": bytes_seen,
           "train_bytes_per_s": bytes_seen / max(train_s, 1e-9), "generation_bytes_per_s": ngen / max(gen_s, 1e-9),
           "params": params, "params_inference": params_infer, "curve": curve,
           "patches_per_byte": stats_last.get("patches_per_byte"), "kl_last": stats_last.get("kl"),
           "aligned_last": stats_last.get("parches_alineados")}
    log(f"{name:10s} semilla {seed}  val BPB {final:.4f}  {bytes_seen / 1e6:.2f} MB vistos  "
        f"{res['train_bytes_per_s']:,.0f} B/s entren.  {res['generation_bytes_per_s']:.0f} B/s gen.  "
        f"{params / 1e6:.2f}M parámetros")
    return res


def report(results: list[dict]) -> list[str]:
    from collections import defaultdict
    by = defaultdict(list)
    for r in results:
        by[r["variant"]].append(r)
    base = {r["seed"]: r["val_bpb"] for r in by.get("espacios", [])}
    lines = ["| variante | semillas | val BPB (media) | por semilla | Δ vs espacios | MB vistos | "
             "bytes/s entren. | bytes/s gen. | parámetros (inferencia) | bytes/parche |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in VARIANTS:
        rs = sorted(by.get(name, []), key=lambda r: r["seed"])
        if not rs:
            continue
        m = sum(r["val_bpb"] for r in rs) / len(rs)
        deltas = [r["val_bpb"] - base[r["seed"]] for r in rs if r["seed"] in base]
        d = f"{sum(deltas) / len(deltas):+.3f}" if deltas and name != "espacios" else "—"
        ppb = rs[0].get("patches_per_byte")
        per_seed = " / ".join("%.3f" % r["val_bpb"] for r in rs)
        lines.append(
            f"| {name} | {len(rs)} | {m:.3f} | {per_seed} | {d} | "
            f"{sum(r['bytes_seen'] for r in rs) / len(rs) / 1e6:.2f} | "
            f"{sum(r['train_bytes_per_s'] for r in rs) / len(rs):,.0f} | "
            f"{sum(r['generation_bytes_per_s'] for r in rs) / len(rs):.0f} | "
            f"{rs[0]['params'] / 1e6:.2f}M ({rs[0]['params_inference'] / 1e6:.2f}M) | "
            f"{(1 / ppb) if ppb else float('nan'):.2f} |")
    return lines


def curve_table(results: list[dict], step_mb: float = 1.0) -> list[str]:
    """BPB (media entre semillas) en cada múltiplo de ``step_mb`` MB vistos (interpolación lineal)."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in results:
        by[r["variant"]].append(r)
    top = min(max(c["bytes"] for c in r["curve"]) for r in results) / 1e6
    marks = [step_mb * i for i in range(1, int(top / step_mb) + 1)]
    lines = ["| MB vistos | " + " | ".join(n for n in VARIANTS if n in by) + " |",
             "|---|" + "---|" * len([n for n in VARIANTS if n in by])]
    for mk in marks:
        cells = []
        for name in VARIANTS:
            if name not in by:
                continue
            vals = []
            for r in by[name]:
                xs = [c["bytes"] / 1e6 for c in r["curve"]]
                ys = [c["val_bpb"] for c in r["curve"]]
                if mk <= xs[-1]:
                    vals.append(float(np.interp(mk, xs, ys)))
            cells.append(f"{sum(vals) / len(vals):.3f}" if len(vals) == len(by[name]) else "—")
        lines.append(f"| {mk:.0f} | " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="carpeta con train.bin y val.bin")
    ap.add_argument("--logits", default=None, help="carpeta de hope.train.distill logits (para tokens_kl)")
    ap.add_argument("--preset", default="hn1-cpu")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--minutes", type=float, default=15.0, help="tiempo de entrenamiento por corrida")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-bytes", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=75)
    ap.add_argument("--precision", default="bf16-8bit")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--vocab-k", type=int, default=4096)
    ap.add_argument("--kl-weight", type=float, default=0.25)
    ap.add_argument("--bpe-vocab", type=int, default=8192)
    ap.add_argument("--bpe-d", type=int, default=256)
    ap.add_argument("--bpe-heads", type=int, default=4)
    ap.add_argument("--bpe-mlp", type=int, default=768)
    ap.add_argument("--bpe-layers", type=int, default=6)
    ap.add_argument("--eval-every-bytes", type=float, default=500_000)
    ap.add_argument("--curve-windows", type=int, default=32)
    ap.add_argument("--final-windows", type=int, default=256)
    ap.add_argument("--gen-bytes", type=int, default=128)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="runs/tvb")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    d = Path(args.data)
    train = np.fromfile(d / "train.bin", dtype=np.uint8)
    val_arr = np.fromfile(d / "val.bin", dtype=np.uint8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    results = json.loads(path.read_text()) if path.exists() else []          # se puede retomar
    names = args.variants.split(",")
    teacher = bpe = None
    if "tokens_kl" in names:
        from hope.train.distill import TeacherLogits
        teacher = TeacherLogits(args.logits, vocab_k=args.vocab_k)
        print(f"maestro: {teacher.meta['entries']:,} entradas; cobertura del vocabulario reducido "
              f"(K = {args.vocab_k}): {teacher.vocab_coverage(args.vocab_k):.3f}")
    if "bpe" in names:
        tok_path = out / "bpe" / "bpe.json"
        tok = load_tokenizer(tok_path) if tok_path.exists() else None
        if tok is None:
            train_tokenizer(d / "train.bin", out / "bpe", args.bpe_vocab)
            tok = load_tokenizer(tok_path)
        ids_path = out / "bpe" / "train.tok"
        if not ids_path.exists():
            np.asarray(bpe_ids(tok, train.tobytes()), dtype=np.uint16).tofile(ids_path)
        tok_ids = np.fromfile(ids_path, dtype=np.uint16)
        tb = token_byte_lengths(tok)
        bpe = (tok, tok_ids, tb, len(train) / len(tok_ids))
    for seed in map(int, args.seeds.split(",")):
        for name in names:
            if any(r["variant"] == name and r["seed"] == seed for r in results):
                continue
            results.append(run(name, args, train, val_arr, seed, teacher=teacher, bpe=bpe))
            path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = report(results) + ["", "Curva: val BPB (media entre semillas) vs. MB vistos", ""] + curve_table(results)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return results


if __name__ == "__main__":
    main()
