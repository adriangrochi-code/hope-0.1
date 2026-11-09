"""Entrenamiento de HN-1 y de la línea base BPE (fase F1).

Ambos se evalúan en **bits por byte (BPB)** sobre la partición de validación,
así la comparación es justa aunque uno lea bytes y el otro tokens.

Ejemplos:
    # HN-1 de 25M
    python -m hope.train.lm --arch hn1 --preset hn1-25m --data data/wiki --out runs/hn1_25m

    # Línea base BPE de tamaño parecido, mismos datos y mismos bytes por paso
    python -m hope.train.lm --arch baseline --data data/wiki --out runs/bpe_25m

    # Comparar
    python -m hope.train.compare runs/hn1_25m runs/bpe_25m

``--max-minutes`` corta el entrenamiento por tiempo, para comparar con el mismo
cómputo real en la A770.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hope.core.baseline import (LN2, BaselineConfig, TokenLM, encode_bin, load_tokenizer,
                                token_byte_lengths, train_tokenizer)
from hope.core.hn1 import build_hn1, preset
from hope.data.corpus import ByteData
from hope.device import select_device


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


class TokenData:
    def __init__(self, path: Path):
        self.data = np.fromfile(path, dtype=np.uint16)

    def batch(self, batch: int, length: int, rng: np.random.Generator) -> torch.Tensor:
        length = min(length, len(self.data) - 1)
        starts = rng.integers(0, len(self.data) - length, size=batch)
        return torch.from_numpy(np.stack([self.data[s:s + length] for s in starts]).astype(np.int64))

    def windows(self, length: int, max_windows: int):
        n = min((len(self.data) - 1) // length, max_windows)
        for i in range(n):
            yield torch.from_numpy(self.data[i * length:(i + 1) * length].astype(np.int64)).unsqueeze(0)


def setup(args):
    data_dir = Path(args.data)
    if args.arch == "hn1":
        cfg = preset(args.preset)
        if args.hnhs:
            cfg.hnhs = True
        model, reg = build_hn1(cfg)
        train, val = ByteData(data_dir / "train.bin"), ByteData(data_dir / "val.bin")
        seq = args.seq_bytes + 1

        def step_loss(x, step=0, rng=None):
            # Contra el colapso del bucle interno (detectado con el diagnóstico de uso):
            # parte de los pasos usa todas las iteraciones y el costo de iterar sube gradualmente.
            full = rng is not None and rng.random() < cfg.full_depth_frac
            return model.loss(x, full_depth=full, ponder_scale=min(1.0, step / max(cfg.ponder_warmup, 1)))

        def eval_bpb(m, device, amp):
            tot, n = 0.0, 0
            for w in val.windows(seq, args.eval_windows):
                with torch.no_grad(), amp():
                    _, st = m.loss(w.to(device))
                tot += st["ce"] * (w.shape[1] - 1)
                n += w.shape[1] - 1
            return tot / max(n, 1) / LN2
        params = reg.report(model).as_dict()
        extra = {"config": cfg.to_dict()}
    else:
        tok_path = data_dir / "bpe.json"
        tok = load_tokenizer(tok_path) if tok_path.exists() else train_tokenizer(data_dir / "train.bin", data_dir, args.vocab)
        if not tok_path.exists():
            tok = load_tokenizer(tok_path)
        for split in ("train", "val"):
            if not (data_dir / f"{split}.tok").exists():
                encode_bin(tok, data_dir / f"{split}.bin", data_dir / f"{split}.tok")
        tb = token_byte_lengths(tok)
        train, val = TokenData(data_dir / "train.tok"), TokenData(data_dir / "val.tok")
        avg = (data_dir / "train.bin").stat().st_size / max(len(train.data), 1)
        seq = max(8, round(args.seq_bytes / avg)) + 1          # mismos bytes por ventana que HN-1
        bcfg = BaselineConfig(vocab=tok.get_vocab_size(), layers=args.baseline_layers)
        model = TokenLM(bcfg)

        def step_loss(x, step=0, rng=None):
            return model.loss(x, tb.to(x.device))

        def eval_bpb(m, device, amp):
            ce_sum, n_bytes = 0.0, 0.0
            for w in val.windows(seq, args.eval_windows):
                w = w.to(device)
                with torch.no_grad(), amp():
                    _, st = m.loss(w, tb.to(device))
                b = tb.to(device)[w[:, 1:]].sum().item()
                ce_sum += st["bpb"] * LN2 * b
                n_bytes += b
            return ce_sum / max(n_bytes, 1) / LN2
        total = sum(p.numel() for p in model.parameters())
        params = {"total": total, "active_expected": total, "trainable": total, "unregistered": 0}
        extra = {"config": bcfg.__dict__, "avg_bytes_per_token": avg, "seq_tokens": seq - 1}
    return model, train, seq, step_loss, eval_bpb, params, extra


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["hn1", "baseline"], default="hn1")
    ap.add_argument("--preset", default="hn1-25m")
    ap.add_argument("--hnhs", action="store_true", help="activar poblaciones E/I (ablación)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-bytes", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--baseline-layers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model, train, seq, step_loss, eval_bpb, params, extra = setup(args)
    model.to(device)
    use_bf16 = device.type in ("xpu", "cuda") and not args.no_bf16

    def amp():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    bytes_seen = 0.0
    history = []
    step = 0
    for step in range(args.steps):
        if args.max_minutes and time.time() - t0 > args.max_minutes * 60:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        x = train.batch(args.batch, seq, rng).to(device)
        model.train()
        with amp():
            loss, st = step_loss(x, step, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        bytes_seen += args.batch * args.seq_bytes
        row = {"step": step, "train_bpb": st["bpb"], "lr": opt.param_groups[0]["lr"],
               "elapsed_s": time.time() - t0, "bytes_seen": bytes_seen,
               **{k: v for k, v in st.items() if k not in ("bpb", "ce")}}
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            row["val_bpb"] = eval_bpb(model, device, amp)
            history.append(row)
            print(f"paso {step:6d}  train_bpb {st['bpb']:.3f}  val_bpb {row['val_bpb']:.3f}  "
                  f"{bytes_seen / (time.time() - t0):,.0f} bytes/s", flush=True)
        log.write(json.dumps(row) + "\n")
    model.eval()
    final_val = eval_bpb(model, device, amp)
    elapsed = time.time() - t0
    summary = {"arch": args.arch, "preset": args.preset if args.arch == "hn1" else "baseline",
               "hnhs": args.hnhs, "device": str(device), "bf16": use_bf16, "steps": step + 1,
               "bytes_seen": bytes_seen, "elapsed_s": elapsed, "bytes_per_s": bytes_seen / elapsed,
               "final_val_bpb": final_val, "params": params, **extra}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "summary": summary}, out / "model.pt")
    print(f"final: val_bpb {final_val:.4f}  ({elapsed / 60:.1f} min)")
    return summary


if __name__ == "__main__":
    main()
