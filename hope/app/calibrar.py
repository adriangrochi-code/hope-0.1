"""Calibración para la aplicación (corre como subproceso: la GPU queda libre al terminar).

    python -m hope.app.calibrar --salida runs/calibracion.json [--plan runs/planes/x.yaml] [--rapido]

1. Revisa PyTorch: versión, CUDA/XPU y si la instalación incluye la arquitectura de la GPU
   (``sm_61`` en una GTX 10xx). Si falta, deja el comando exacto de instalación.
2. Mide el cómputo con ``hope.bench.hardware.calibrate``.
3. Busca el lote máximo del plan (modelo de la etapa 1 con su precisión y descarga del optimizador).
4. Recomienda un plan y deja la tabla de ``recommend``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

CMD_CU126 = "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126"
CMD_XPU = "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/xpu"


def progreso(x: float, texto: str) -> None:
    print(f"PROGRESO {x:.2f} {texto}", flush=True)


def revisar_torch() -> dict:
    from hope.app.sistema import gpus_por_nombre, nvidia
    info: dict = {"problemas": [], "comandos": []}
    try:
        import torch
    except ImportError:
        info["problemas"].append("PyTorch no está instalado")
        info["comandos"].append(CMD_CU126 if nvidia() else CMD_XPU if _hay_arc(gpus_por_nombre()) else
                                "pip install torch")
        return info
    info["torch"] = torch.__version__
    info["cuda_compilado"] = torch.version.cuda
    info["cuda"] = bool(torch.cuda.is_available())
    info["xpu"] = bool(getattr(torch, "xpu", None) is not None and torch.xpu.is_available())
    nv = nvidia()
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        arch = f"sm_{props.major}{props.minor}"
        info.update(gpu=props.name, arquitectura=arch, arquitecturas_soportadas=torch.cuda.get_arch_list(),
                    vram_gb=round(props.total_memory / 1e9, 1))
        if arch not in torch.cuda.get_arch_list():
            info["problemas"].append(f"esta instalación de PyTorch no incluye {arch} (necesario para {props.name})")
            info["comandos"].append(CMD_CU126)
    elif nv:
        info["gpu"] = nv[0]["nombre"]
        info["problemas"].append(f"hay una {nv[0]['nombre']} pero PyTorch no ve CUDA "
                                 f"(versión instalada: {torch.__version__})")
        info["comandos"].append(CMD_CU126)
    elif info["xpu"]:
        info.update(gpu=torch.xpu.get_device_name(0),
                    vram_gb=round(torch.xpu.get_device_properties(0).total_memory / 1e9, 1))
    elif _hay_arc(gpus_por_nombre()):
        info["gpu"] = next(n for n in gpus_por_nombre() if "arc" in n.lower())
        info["problemas"].append("hay una Intel Arc pero PyTorch no ve XPU")
        info["comandos"].append(CMD_XPU)
    info["dispositivo"] = "cuda" if info["cuda"] and not info["problemas"] else "xpu" if info["xpu"] else "cpu"
    return info


def _hay_arc(names) -> bool:
    return any("arc" in n.lower() and "intel" in n.lower() for n in names)


def lote_maximo(plan_path: str, device: str) -> dict:
    import torch
    import yaml
    from hope.core.hn1 import HN1Model
    from hope.train.continual import Trainer, TrainConfig
    from hope.train.run import _auto_batch, _config
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    etapa = next((e for e in plan["etapas"] if "preset" in e), None)
    if etapa is None:
        return {"nota": "el plan continúa un modelo existente: el lote se busca al entrenar"}
    seq = int(etapa.get("secuencia", plan.get("secuencia", 2048)))
    tcfg = TrainConfig(batch=1, seq_bytes=seq, precision=plan.get("precision", "bf16-8bit"),
                       optimizer=plan.get("optimizador", "adamw"), checkpointing=bool(plan.get("checkpointing", True)),
                       offload_optimizer=bool(plan.get("offload_optimizador", False)))
    t0 = time.time()
    try:
        tr = Trainer(HN1Model(_config(etapa)), tcfg, device)
        b = _auto_batch(tr, seq, 1, int(etapa.get("lote_max", 32)))
    except (RuntimeError, MemoryError) as e:          # sin memoria aun con lote 1
        return {"preset": etapa["preset"], "lote": 0, "entra": False, "error": str(e)[:300]}
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
    out = {"preset": etapa["preset"], "secuencia": seq, "lote": b, "entra": b >= 1, "segundos": round(time.time() - t0)}
    if device == "cuda":
        out["memoria_pico_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


def elegir_plan(dev: str, rows: list[dict], continuacion: bool) -> tuple[str, str]:
    by = {r["preset"]: r for r in rows}
    if dev == "xpu":
        if continuacion:
            return "plan_a770_continuacion.yaml", "Intel Arc: continuar el modelo que viene de la 1060"
        return "plan_6m.yaml", "Intel Arc sin entrenamiento previo: plan de 6 meses"
    if dev == "cuda":
        r5 = by.get("hn1-500m", {})
        if r5.get("entra") or r5.get("entra_con_offload"):
            return "plan_gtx1060.yaml", "entra hn1-500m con visión (optimizador en la RAM)"
        return "plan_gtx1060_200m.yaml", "hn1-500m no entra: hn1-200m"
    return "plan_cpu.yaml", "sin GPU utilizable: solo sirve para pruebas"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salida", required=True)
    ap.add_argument("--plan", default=None, help="plan para buscar el lote máximo")
    ap.add_argument("--meses", type=float, default=2.0)
    ap.add_argument("--rapido", action="store_true", help="matmul chico y sin búsqueda de lote (pruebas)")
    ap.add_argument("--continuacion", action="store_true")
    args = ap.parse_args(argv)
    res: dict = {"fecha": time.strftime("%Y-%m-%d %H:%M")}
    progreso(0.05, "revisando PyTorch")
    res["torch"] = t = revisar_torch()
    for p in t["problemas"]:
        print("PROBLEMA: " + p, flush=True)
    for c in t["comandos"]:
        print("Instalá con: " + c, flush=True)
    dev = t.get("dispositivo", "cpu")
    if "torch" in t:
        from hope.bench.hardware import calibrate, recommend
        progreso(0.2, f"midiendo el cómputo ({dev})")
        try:
            prof = calibrate(dev, n=512 if args.rapido else 4096)
        except RuntimeError as e:
            print("PROBLEMA: " + str(e), flush=True)
            res["torch"]["problemas"].append(str(e))
            dev = "cpu"
            prof = calibrate("cpu", n=512)
        res["perfil"] = asdict(prof)
        print(f"cómputo efectivo medido: {prof.tflops_eff} TFLOPS ({prof.name})", flush=True)
        res["recomendacion"] = recommend(prof, args.meses)
        if args.plan and dev in ("cuda", "xpu") and not args.rapido:
            progreso(0.5, "buscando el lote máximo (puede tardar unos minutos)")
            res["lote"] = lote_maximo(args.plan, dev)
            print("lote máximo: " + json.dumps(res["lote"], ensure_ascii=False), flush=True)
        plan, motivo = elegir_plan(dev, res["recomendacion"], args.continuacion)
        if res.get("lote", {}).get("entra") is False and plan == "plan_gtx1060.yaml":
            plan, motivo = "plan_gtx1060_200m.yaml", "hn1-500m no entró ni con lote 1: hn1-200m"
        res["plan_recomendado"], res["motivo"] = plan, motivo
        res["dispositivo"] = dev
        print(f"plan recomendado: {plan} ({motivo})", flush=True)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.salida).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    progreso(1.0, "calibración terminada")


if __name__ == "__main__":
    main()
