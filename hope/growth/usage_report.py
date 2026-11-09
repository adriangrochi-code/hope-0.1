"""Reporte de uso de un checkpoint de HN-1 (diagnóstico LTH, ASG §9).

    python -m hope.growth.usage_report --run runs/hn1_25m --data data/wiki

Mide sobre la partición de validación:
  * uso de cada especialista;
  * peso de cada iteración del núcleo recurrente y mejora de la pérdida por iteración extra;
  * fracción de neuronas ociosas por población (importancia de Taylor en el decil inferior).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hope.core.hn1 import HN1Config, HN1Model
from hope.data.corpus import ByteData
from hope.growth.lth import expert_usage, iteration_usage, unit_importance


def load_run(run: Path) -> HN1Model:
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    model = HN1Model(HN1Config.from_dict(ckpt["summary"]["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model


def report(model: HN1Model, windows: list[torch.Tensor]) -> dict:
    def loss_n(m, x, n):
        o = m(x[:, :-1], n_iters=n)
        return F.cross_entropy(o["logits"].reshape(-1, 256).float(), x[:, 1:].reshape(-1)), o

    def loss(m, x):
        return m.loss(x)[0]

    model.eval()
    with torch.no_grad():
        routes = [model(w[:, :-1])["routes"][0] for w in windows]
    units = unit_importance(model, windows, loss)
    idle = {}
    for name, st in units.items():
        t = st.taylor
        thresh = 0.1 * t.median() if t.numel() else 0
        idle[name] = round(float((t <= thresh).float().mean()), 3)
    return {
        "expert_usage": [round(u, 4) for u in expert_usage(routes, model.cfg.n_experts).tolist()],
        "iterations": iteration_usage(model, windows, loss_n),
        "idle_unit_fraction (Taylor < 10% de la mediana)": idle,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq-bytes", type=int, default=512)
    args = ap.parse_args()
    model = load_run(Path(args.run))
    val = ByteData(Path(args.data) / "val.bin")
    windows = list(val.windows(args.seq_bytes + 1, args.windows))
    rep = report(model, windows)
    (Path(args.run) / "usage.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
