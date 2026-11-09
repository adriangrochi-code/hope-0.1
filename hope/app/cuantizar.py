"""Exportar a Hope cuantizado en 4 u 8 bits (paso 7 de la aplicación; corre como subproceso en CPU).

    python -m hope.app.cuantizar --modelo runs/hope/ligero.pt --bits 4 --salida runs/exportar/hope_q4.pt

Carga el modelo ligero (``load_light``), cuantiza sus capas lineales con ``hope.core.fast.quantize``
(por grupos, igual que el QAT de ``hope.train.qat``) y guarda pesos cuantizados + configuración.
Mide la pérdida en un texto de prueba antes y después, para ver cuánto cuesta cuantizar, y deja
un resumen JSON al lado (``<salida>.json``). ``cargar(ruta)`` lo vuelve a abrir para usarlo.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

TEXTO_PRUEBA = ("La energía se conserva: en un circuito, la potencia que entrega la fuente es igual a la suma "
                "de las potencias que disipan las resistencias. The same principle applies to every closed "
                "system.\ndef area(base, altura):\n    return base * altura / 2\n")


def guardar(model, info: dict, path: Path) -> None:
    from hope.core.fast import eager
    path.parent.mkdir(parents=True, exist_ok=True)
    with eager(model):
        sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() and v.dtype == torch.float32 else v)
              for k, v in model.state_dict().items()}
    torch.save({"config": model.cfg.to_dict(), "bits": info["bits"], "group": info.get("group", 64),
                "state_dict": sd}, path)


def cargar(path: str | Path, device: str = "cpu"):
    """Abre un modelo guardado por este módulo (capas ``QuantLinear`` con sus pesos en 4/8 bits)."""
    from hope.core.fast import quantize
    from hope.core.hn1 import HN1Config, HN1Model
    ck = torch.load(path, map_location=device, weights_only=False)
    model = HN1Model(HN1Config.from_dict(ck["config"]))
    quantize(model, bits=int(ck["bits"]), group=int(ck.get("group", 64)))
    sd = {k: (v.float() if v.dtype == torch.bfloat16 else v) for k, v in ck["state_dict"].items()}
    model.load_state_dict(sd)
    return model.eval()


@torch.no_grad()
def perdida(model, texto: str = TEXTO_PRUEBA) -> float:
    x = torch.tensor(list(texto.encode("utf-8")), dtype=torch.long).unsqueeze(0)
    loss, _ = model.loss(x, full_depth=True)
    return float(loss)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--bits", type=int, choices=(4, 8), default=4)
    ap.add_argument("--grupo", type=int, default=64)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args(argv)
    from hope.core.fast import quantize
    from hope.train.run import load_light
    torch.set_num_threads(2)
    print("PROGRESO 0.05 cargando el modelo", flush=True)
    model = load_light(args.modelo, "cpu")
    params = sum(p.numel() for p in model.parameters())
    antes = perdida(model)
    print(f"PROGRESO 0.35 cuantizando a {args.bits} bits", flush=True)
    info = quantize(model, bits=args.bits, group=args.grupo)
    info["group"] = args.grupo
    despues = perdida(model)
    print("PROGRESO 0.6 guardando", flush=True)
    sal = Path(args.salida)
    guardar(model, info, sal)
    print("PROGRESO 0.75 comprobando que se vuelve a abrir", flush=True)
    re = cargar(sal)
    abierto = perdida(re)
    torch.manual_seed(0)
    muestra = re.generate("La ciudad de Buenos Aires", max_new=120, temperature=0.7)
    res = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "modelo": str(args.modelo), "archivo": str(sal),
           "bits": args.bits, "parametros": params, "capas": info["capas"],
           "mb_pesos_lineales_antes": round(info["bytes_antes"] / 1e6, 2),
           "mb_pesos_lineales_despues": round(info["bytes_despues"] / 1e6, 2),
           "mb_archivo": round(sal.stat().st_size / 1e6, 2),
           "mb_ligero": round(Path(args.modelo).stat().st_size / 1e6, 2),
           "perdida_original": round(antes, 4), "perdida_cuantizado": round(despues, 4),
           "perdida_al_reabrir": round(abierto, 4),
           "muestra": muestra.decode("utf-8", errors="replace")}
    Path(str(sal) + ".json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "muestra"}, ensure_ascii=False), flush=True)
    print("PROGRESO 1.0 listo", flush=True)


if __name__ == "__main__":
    main()
