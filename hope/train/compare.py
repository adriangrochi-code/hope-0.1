"""Compara corridas de entrenamiento: ``python -m hope.train.compare runs/a runs/b ...``"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    rows = []
    for d in argv if argv is not None else sys.argv[1:]:
        s = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
        rows.append((d, s))
    print("| corrida | arquitectura | parámetros | activos | bytes vistos | minutos | bytes/s | val BPB |")
    print("|---|---|---|---|---|---|---|---|")
    for d, s in sorted(rows, key=lambda r: r[1]["final_val_bpb"]):
        p = s["params"]
        arch = s["preset"] + (" +HNHS" if s.get("hnhs") else "")
        print(f"| {d} | {arch} | {p['total'] / 1e6:.1f}M | {p['active_expected'] / 1e6:.1f}M | "
              f"{s['bytes_seen'] / 1e6:.1f}M | {s['elapsed_s'] / 60:.1f} | {s['bytes_per_s']:,.0f} | "
              f"{s['final_val_bpb']:.4f} |")


if __name__ == "__main__":
    main()
