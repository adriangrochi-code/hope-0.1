"""Interfaz de línea de comandos del guardián, **para uso humano**.

    python -m guardian.cli estado                          # techo, modo y estimación de memoria
    python -m guardian.cli limite 1B --by <nombre>         # autorizar crecer hasta 1.000 millones
    python -m guardian.cli limite 500M --by <nombre> --modo USER_APPROVAL_REQUIRED
    python -m guardian.cli pending
    python -m guardian.cli approve <id> --by <nombre> [--note ...]
    python -m guardian.cli deny <id> --by <nombre> [--note ...]
    python -m guardian.cli verify
    python -m guardian.cli red                             # permisos de internet (configs/web.yaml) y uso

El techo es un **máximo autorizado, no una meta** (ASG §1): subirlo no hace
crecer a Hope. Solo le permite proponer crecimiento con evidencia, que se
prueba en copias experimentales y se consolida según el modo y el umbral del 10%.
Hope no puede cambiar el techo: la acción ``budget.change`` le está prohibida.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from guardian.core import Guardian, GuardianConfig
from hope.growth.budget import GrowthBudget, GrowthMode
from hope.train.lowprec import BYTES_PER_PARAM, max_params_for

SUFFIX = {"k": 10**3, "m": 10**6, "millones": 10**6, "b": 10**9, "g": 10**9, "mil millones": 10**9, "t": 10**12}



def load(config_path: str) -> Guardian:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return Guardian(GuardianConfig(**raw))


def parse_count(text: str) -> int:
    """'1B', '1b', '1.5B', '500M', '200 millones', '1.000.000.000', '1_000_000_000' → entero."""
    t = text.strip().lower().replace("_", "")
    m = re.fullmatch(r"([\d.,]+)\s*(k|m|b|g|t|mil millones|millones)?", t)
    if not m:
        raise ValueError(f"no entiendo la cantidad '{text}' (ejemplos: 1B, 500M, 1.000.000.000)")
    num, suffix = m.group(1), m.group(2)
    if suffix is None:
        value = int(num.replace(".", "").replace(",", ""))
    else:
        value = int(round(float(num.replace(",", ".")) * SUFFIX[suffix]))
    if value <= 0:
        raise ValueError("el límite tiene que ser positivo")
    return value


def human(n: float) -> str:
    for size, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= size:
            return f"{n / size:.2f}".rstrip("0").rstrip(".") + unit
    return str(int(n))


def training_memory_gb(params: int, precision: str = "bf16-8bit") -> float:
    """Pesos + gradientes + estados del optimizador según la precisión (sin activaciones)."""
    return params * BYTES_PER_PARAM[precision] / 1e9


def set_limit(g: Guardian, value: int, *, by: str, mode: str | None = None, note: str = "") -> list[str]:
    """Cambia el techo (y opcionalmente el modo) conservando los comentarios del archivo."""
    path = Path(g.config.budget_path)
    text = path.read_text(encoding="utf-8")
    old = g.budget()
    text, n = re.subn(r"(?m)^(max_total_params:\s*)[^#\n]*", rf"\g<1>{value} ", text)
    if n == 0:
        text = f"max_total_params: {value}\n" + text
    if mode is not None:
        GrowthMode(mode)                                          # valida
        text, n = re.subn(r"(?m)^(mode:\s*)\S+", rf"\g<1>{mode}", text)
        if n == 0:
            text += f"\nmode: {mode}\n"
    path.write_text(text, encoding="utf-8")
    new = g.budget()
    g.audit.append("budget.changed_by_human", by=by, note=note,
                   old={"max_total_params": old.max_total_params, "mode": old.mode.value},
                   new={"max_total_params": new.max_total_params, "mode": new.mode.value})
    warnings = []
    need = training_memory_gb(value, new.training_precision)
    # Las activaciones también ocupan memoria: por encima del 80% de la VRAM ya no entra cómodo.
    if new.max_vram_gb and need > 0.8 * new.max_vram_gb:
        warnings.append(
            f"entrenar {human(value)} de parámetros necesita ~{need:.0f} GB solo para pesos, gradientes y "
            f"optimizador en modo {new.training_precision}, más las activaciones; la GPU configurada tiene "
            f"{new.max_vram_gb:g} GB (entran ~{human(max_params_for(new.max_vram_gb, new.training_precision))}). "
            f"Hará falta más hardware o entrenar por partes (congelar regiones).")
    if value < old.max_total_params:
        warnings.append("el nuevo techo es más bajo que el anterior: los cambios que lo superen se rechazarán, "
                        "pero no se elimina nada de lo que ya existe")
    return warnings


def status(g: Guardian) -> dict:
    b: GrowthBudget = g.budget()
    return {
        "techo_de_parametros": f"{b.max_total_params:,}".replace(",", ".") + f" ({human(b.max_total_params)})",
        "modo": b.mode.value,
        "cambio_importante_requiere_aprobacion_si_supera": f"{b.consolidation.max_auto_param_fraction:.0%} de los parámetros",
        "precision_de_entrenamiento": b.training_precision,
        "memoria_estimada_para_entrenar_al_techo_gb": round(training_memory_gb(b.max_total_params,
                                                                                 b.training_precision), 1),
        "vram_configurada_gb": b.max_vram_gb,
        "parametros_entrenables_que_entran_en_la_vram": (human(max_params_for(b.max_vram_gb, b.training_precision))
                                                          if b.max_vram_gb else None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="guardian", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/guardian.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estado", aliases=["status"])
    lim = sub.add_parser("limite", aliases=["limit"])
    lim.add_argument("valor", help="p. ej. 1B, 500M, 1.000.000.000")
    lim.add_argument("--by", required=True, help="quién autoriza (queda en la auditoría)")
    lim.add_argument("--modo", "--mode", dest="mode", default=None, choices=[m.value for m in GrowthMode])
    lim.add_argument("--note", default="")
    sub.add_parser("pending")
    for name in ("approve", "deny"):
        p = sub.add_parser(name)
        p.add_argument("request_id")
        p.add_argument("--by", required=True)
        p.add_argument("--note", default="")
    sub.add_parser("verify")
    sub.add_parser("red", aliases=["web"])
    args = ap.parse_args(argv)

    g = load(args.config)
    if args.cmd in ("estado", "status"):
        print(json.dumps(status(g), indent=2, ensure_ascii=False))
    elif args.cmd in ("limite", "limit"):
        value = parse_count(args.valor)
        for w in set_limit(g, value, by=args.by, mode=args.mode, note=args.note):
            print(f"aviso: {w}")
        print(json.dumps(status(g), indent=2, ensure_ascii=False))
        print("El techo es un máximo autorizado, no una meta: Hope solo crecerá si demuestra que lo necesita.")
    elif args.cmd == "pending":
        print(json.dumps(g.pending(), indent=2, ensure_ascii=False))
    elif args.cmd == "approve":
        g.approve(args.request_id, args.by, args.note)
        print(f"aprobada {args.request_id}")
    elif args.cmd == "deny":
        g.deny(args.request_id, args.by, args.note)
        print(f"denegada {args.request_id}")
    elif args.cmd in ("red", "web"):
        cfg = g.web_config()
        print(json.dumps({"archivo": g.config.web_path, "modo": cfg.mode, "dominios_permitidos": cfg.allowed_domains,
                          "dominios_bloqueados": cfg.blocked_domains, **g.network_usage(),
                          "pendientes_de_red": [k for k, v in g.pending().items()
                                                if v["action"]["kind"] == "network.access"]},
                         indent=2, ensure_ascii=False))
        print("Para cambiar los permisos, editá ese archivo (Hope no puede modificarlo).")
    elif args.cmd == "verify":
        ok, idx = g.audit.verify()
        print("registro íntegro" if ok else f"registro ALTERADO en la entrada {idx}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
