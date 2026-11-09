"""Interfaz de línea de comandos del guardián, para uso humano.

    python -m guardian.cli --config configs/guardian.yaml pending
    python -m guardian.cli --config configs/guardian.yaml approve <id> --by <nombre> [--note ...]
    python -m guardian.cli --config configs/guardian.yaml deny <id> --by <nombre> [--note ...]
    python -m guardian.cli --config configs/guardian.yaml verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from guardian.core import Guardian, GuardianConfig


def load(config_path: str) -> Guardian:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return Guardian(GuardianConfig(**raw))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="guardian")
    ap.add_argument("--config", default="configs/guardian.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pending")
    for name in ("approve", "deny"):
        p = sub.add_parser(name)
        p.add_argument("request_id")
        p.add_argument("--by", required=True)
        p.add_argument("--note", default="")
    sub.add_parser("verify")
    args = ap.parse_args(argv)

    g = load(args.config)
    if args.cmd == "pending":
        print(json.dumps(g.pending(), indent=2, ensure_ascii=False))
    elif args.cmd == "approve":
        g.approve(args.request_id, args.by, args.note)
        print(f"aprobada {args.request_id}")
    elif args.cmd == "deny":
        g.deny(args.request_id, args.by, args.note)
        print(f"denegada {args.request_id}")
    elif args.cmd == "verify":
        ok, idx = g.audit.verify()
        print("registro íntegro" if ok else f"registro ALTERADO en la entrada {idx}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
