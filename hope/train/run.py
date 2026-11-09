"""Programa de entrenamiento de largo plazo (plan de 6 meses; ARQUITECTURA_HN1 §11.1).

Un solo comando que sigue el plan por etapas y se puede cortar y retomar en cualquier momento::

    python -m hope.train.run --plan configs/plan_6m.yaml

* Cada etapa entrena por **horas** (no por pasos) y guarda un checkpoint cada
  ``checkpoint_min`` minutos en ``salida/<etapa>/``; al relanzar, sigue donde quedó.
* ``lote: auto`` busca el lote más grande que entra en la GPU (``hope/core/fast.py``).
* Programa WSD: tasa constante en todas las etapas y enfriamiento solo en la etapa con
  ``enfriar: true`` (con los mejores datos). Antes de enfriar se guarda la rama estable.
* Etapa de crecimiento (``crecer_a``): arma las operaciones que preservan la función y
  **pide aprobación humana al guardián** (cambio > 10 %). Si queda pendiente, el programa
  se detiene e indica el comando para aprobar; al relanzarlo continúa.
* Etapa de presupuesto (``presupuesto: 1000M``): crece o reduce hasta X con
  ``hope.growth.budget_target`` (§6.5), sin pasar el techo del guardián; cada paso de más
  del 10 % pide aprobación. Lo aplicado se guarda y al relanzar se replanifica desde ahí.
* Registro: ``salida/registro.jsonl`` con BPB por fuente, pasos y tiempo de cada tramo.
* Pausa segura: si existe el archivo ``salida/DETENER``, el programa termina **después del
  próximo checkpoint** (lo usa la aplicación ``python -m hope.app``). Al relanzarlo, sigue.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from hope.core.hn1 import HN1Model, build_registry, preset
from hope.data.corpus import ByteData
from hope.train.continual import Mixture, Source, Trainer, TrainConfig


def _config(etapa: dict):
    """Preset de la etapa con cambios opcionales (``config: {multibyte: 3, ...}``)."""
    from hope.core.hn1 import HN1Config
    base = preset(etapa["preset"])
    return HN1Config(**{**base.to_dict(), **etapa.get("config", {})}) if etapa.get("config") else base


def _sources(spec: dict, split: str = "train") -> list[Source]:
    out = []
    for name, val in spec.items():
        folder, weight = str(val).rsplit(":", 1)
        path = Path(folder) / f"{split}.bin"
        if path.exists() and path.stat().st_size > 1024:
            # ``train.mask`` (opcional, un byte por byte): pérdida solo en las partes marcadas con 1
            # (intentos corregidos del tutor, ``hope/teacher/tutor.py``).
            mpath = Path(folder) / f"{split}.mask"
            mask = np.memmap(mpath, dtype=np.uint8, mode="r") if mpath.exists() else None
            if mask is not None and len(mask) != path.stat().st_size:
                raise ValueError(f"{mpath} no tiene el mismo largo que {path}")
            out.append(Source(name, ByteData(path), float(weight), mask))
    if not out:
        raise FileNotFoundError(f"ninguna fuente con datos en {spec}")
    return out


def _sources_seguras(spec: dict):
    """(mezcla, evaluación) releídas del disco, o None si no se pudo (se sigue con las anteriores)."""
    try:
        mix = Mixture(_sources(spec))
    except (FileNotFoundError, OSError, ValueError):
        return None
    try:
        evals = _sources(spec, "val")
    except (FileNotFoundError, OSError, ValueError):
        evals = []
    return mix, evals


STOP_FILE = "DETENER"


def _log(out: Path, rec: dict) -> None:
    with (out / "registro.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def _state(out: Path) -> dict:
    p = out / "estado.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"etapas": {}}


def _save_state(out: Path, st: dict) -> None:
    (out / "estado.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")


def _auto_batch(trainer: Trainer, seq: int, start: int, maximum: int) -> int:
    from hope.core.fast import find_batch_size

    def step(b):
        x = torch.randint(0, 256, (b, seq + 1), device=trainer.device)
        trainer.model.train()
        loss, _ = trainer.model.loss(x, full_depth=True)
        loss.backward()
        trainer.model.zero_grad(set_to_none=True)
    return find_batch_size(step, start=start, maximum=maximum)


def cooldown_span(done_hours: float, minutes: float, hours: float) -> tuple[float, float]:
    """Fracción del enfriamiento que cubre un tramo: de las horas ya hechas a las que habrá al terminar."""
    total = max(hours, 1e-9)
    return min(done_hours / total, 1.0), min((done_hours + minutes / 60) / total, 1.0)


def export_light(model, path: Path) -> Path:
    """Solo pesos bf16 + configuración (sin optimizador): para compartir o usar el modelo."""
    from hope.core.fast import eager
    path.parent.mkdir(parents=True, exist_ok=True)
    with eager(model):
        sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    # Escritura atómica: quien lo lea mientras entrena (la pestaña «Chatear», las muestras) nunca ve un
    # archivo a medio escribir.
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"config": model.cfg.to_dict(), "state_dict": sd}, tmp)
    tmp.replace(path)
    return path


def load_light(path: str | Path, device: str = "cpu") -> HN1Model:
    from hope.core.hn1 import HN1Config
    ck = torch.load(path, map_location=device, weights_only=False)
    model = HN1Model(HN1Config.from_dict(ck["config"]))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    return model.eval()


def grow(trainer: Trainer, target: str, guardian, approval: str | None) -> tuple[bool, str | None]:
    """Crecimiento con permiso del guardián. Devuelve (hecho, id de solicitud pendiente)."""
    from guardian.core import Action
    from hope.growth.budget import ProposedChange
    from hope.growth.hn1_ops import plan_growth_to
    from hope.core.hn1 import HN1Config
    model = trainer.model
    if isinstance(target, dict):                     # cambios sobre la configuración actual
        target = HN1Config(**{**model.cfg.to_dict(), **target})
    ops = plan_growth_to(model, target)
    if not ops:                                      # ya tiene esa estructura (p. ej. vino de la 1060 con 500M)
        return True, None
    total = sum(p.numel() for p in model.parameters())
    delta = sum(op.plan(model).param_delta for op in ops)
    change = ProposedChange(f"crecer a {target} ({len(ops)} operaciones)", param_delta=delta,
                            region_delta=1, plasticity_level=4)
    action = Action("structural.consolidate", {"change": change.__dict__, "current_total": total},
                    requested_by="hope.train.run", justification="plan de 6 meses (§11.1)")
    d = guardian.decide(action, approval_id=approval)
    if not d.allowed:
        if d.pending_request:
            return False, d.pending_request
        raise PermissionError("el guardián rechazó el crecimiento: " + "; ".join(d.reasons)
                              + ". Si corresponde, subí el techo: python -m guardian.cli limite 500M --by <tu-nombre> "
                                "--modo USER_APPROVAL_REQUIRED")
    reg = build_registry(model)
    for op in ops:
        op.apply(model, reg)
    for z in model.regions:                          # habilitar los especialistas nuevos
        z.expert_mask.zero_()
    trainer.refresh_optimizer()
    return True, None


def budget(trainer: Trainer, value, guardian, approval: str | None, *, batches=None, profile=None,
           flops_base: float | None = None) -> tuple[bool, str | None, dict]:
    """Etapa ``presupuesto: X``: crecer o reducir hasta X (§6.5) con el guardián, paso por paso.
    Devuelve (hecho, id de solicitud pendiente, resumen del plan). Lo aplicado antes de un paso
    pendiente queda en el modelo (el llamador guarda el checkpoint) y al relanzar se replanifica."""
    from guardian.cli import parse_count
    from hope.growth.budget_target import apply_plan, plan_to_budget
    target = value if isinstance(value, int) else parse_count(str(value))
    model = trainer.model
    plan = plan_to_budget(model, target, profile, guardian=guardian, batches=batches,
                          offload="auto" if trainer.cfg.offload_optimizer else False, flops_base=flops_base)
    summary = plan.resumen()
    if not plan.pasos:
        return True, None, summary
    res = apply_plan(model, plan, guardian=guardian, approval=approval, batches=batches, unmask_new=True)
    if res.pasos_aplicados:
        trainer.refresh_optimizer()
    return res.hecho, res.pendiente, summary


def run(plan_path: str, device: str | None = None, max_hours: float | None = None) -> None:
    plan = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    out = Path(plan.get("salida", "runs/hope"))
    out.mkdir(parents=True, exist_ok=True)
    st = _state(out)
    t_start = time.time()
    trainer: Trainer | None = None
    last_ckpt: Path | None = Path(st["ultimo_checkpoint"]) if st.get("ultimo_checkpoint") else None
    if last_ckpt is None and plan.get("desde"):
        # Continuar un modelo entrenado en otra máquina (p. ej. la 1060 → la A770).
        last_ckpt = Path(plan["desde"])
    if str(plan.get("offload_optimizador", "")).lower() == "disco":
        # El estado del optimizador va a archivos mapeados en el disco elegido (RAM insuficiente).
        import os
        from hope.train.lowprec import offload_to_disk
        offload_to_disk(os.environ.get("HOPE_OFFLOAD_DIR") or out / "optimizador_en_disco")
    for etapa in plan["etapas"]:
        name = etapa["nombre"]
        es = st["etapas"].setdefault(name, {"horas": 0.0, "hecha": False})
        if es["hecha"]:
            continue
        tcfg = TrainConfig(lr=float(etapa.get("lr", plan.get("lr", 3e-4))), warmup=int(plan.get("warmup", 2000)),
                           batch=int(etapa.get("lote", 16)) if str(etapa.get("lote", 16)) != "auto" else 4,
                           seq_bytes=int(etapa.get("secuencia", plan.get("secuencia", 2048))),
                           precision=plan.get("precision", "bf16-8bit"), schedule="wsd",
                           optimizer=plan.get("optimizador", "adamw"),
                           grad_accum=int(etapa.get("acumular", plan.get("acumular", 1))),
                           checkpointing=bool(plan.get("checkpointing", True)),
                           segment_bytes=etapa.get("segmento"),
                           offload_optimizer=bool(plan.get("offload_optimizador", False)),
                           qat_bits=tuple(etapa["qat_bits"]) if etapa.get("qat_bits") else None)
        if trainer is None:
            trainer = (Trainer.resume(last_ckpt, tcfg, device) if last_ckpt and (last_ckpt / "model.pt").exists()
                       else Trainer(HN1Model(_config(etapa)), tcfg, device))
        else:
            trainer.cfg = tcfg
        if plan.get("compilar", True):
            from hope.core.fast import compile_model
            compile_model(trainer.model)
        if "tutoria" in etapa:
            # Etapa de tutoría on-policy (hope/teacher/tutor.py): un Qwen grande revisa los intentos de
            # Hope, da pistas, propone variaciones dirigidas y deja datos con máscara en ``datos``.
            from guardian.cli import load as load_guardian
            from hope.teacher.tutor import run_stage
            spec = etapa["tutoria"]
            stage_h = float(etapa.get("horas", spec.get("horas", 24)))
            left = stage_h - es["horas"]
            if max_hours is not None:
                left = min(left, max_hours - (time.time() - t_start) / 3600)
            t0 = time.time()
            res = run_stage(trainer, spec, out / name / "tutoria",
                            guardian=load_guardian(plan.get("guardian", "configs/guardian.yaml")),
                            max_hours=max(left, 0.0))
            es["horas"] += (time.time() - t0) / 3600
            ck = out / name
            trainer.save(ck)
            export_light(trainer.model, out / "ligero.pt")
            st["ultimo_checkpoint"] = str(ck)
            # Hecha salvo que se haya cortado por el límite global de horas (al relanzar, sigue).
            cut = max_hours is not None and (time.time() - t_start) / 3600 >= max_hours \
                and es["horas"] < stage_h
            es["hecha"] = not cut
            _save_state(out, st)
            _log(out, {"etapa": name, "tutoria": res, "horas": round(es["horas"], 3)})
            if cut:
                return
            continue
        if "crecer_a" in etapa:
            from guardian.cli import load as load_guardian
            done, pending = grow(trainer, etapa["crecer_a"], load_guardian(plan.get("guardian", "configs/guardian.yaml")),
                                 es.get("aprobacion"))
            if not done:
                es["solicitud"] = pending
                _save_state(out, st)
                print(f"\nEl crecimiento necesita tu aprobación: {pending}\n"
                      f"  python -m guardian.cli approve {pending} --by <tu-nombre>\n"
                      f"y después agregá \"aprobacion\": \"{pending}\" en {out / 'estado.json'} (etapa {name}) "
                      "y relanzá este programa.", flush=True)
                return
            es["hecha"] = True
            ck = out / name
            trainer.save(ck)
            st["ultimo_checkpoint"] = str(ck)
            _save_state(out, st)
            _log(out, {"etapa": name, "crecimiento": str(etapa["crecer_a"]),
                       "parametros": sum(p.numel() for p in trainer.model.parameters())})
            continue
        if "presupuesto" in etapa:
            from guardian.cli import load as load_guardian
            from hope.bench.plan import cost
            evals = None
            if etapa.get("fuentes"):
                vals = _sources(etapa["fuentes"], "val")
                rng = __import__("numpy").random.default_rng(0)
                evals = [torch.as_tensor(Mixture(vals).batch(2, 513, rng)[0]).long() for _ in range(4)]
            es.setdefault("flops_base", cost(trainer.model).flops_per_byte)
            done, pending, summary = budget(trainer, etapa["presupuesto"],
                                            load_guardian(plan.get("guardian", "configs/guardian.yaml")),
                                            es.get("aprobacion"), batches=evals, profile=plan.get("perfil"),
                                            flops_base=es["flops_base"])
            es.pop("aprobacion", None)
            ck = out / name
            trainer.save(ck)
            st["ultimo_checkpoint"] = str(ck)
            _log(out, {"etapa": name, "presupuesto": str(etapa["presupuesto"]), "plan": summary["pasos"],
                       "parametros": sum(p.numel() for p in trainer.model.parameters())})
            if not done:
                es["solicitud"] = pending
                _save_state(out, st)
                print(f"\nUn paso del presupuesto necesita tu aprobación: {pending}\n"
                      f"  python -m guardian.cli approve {pending} --by <tu-nombre>\n"
                      f"y después agregá \"aprobacion\": \"{pending}\" en {out / 'estado.json'} (etapa {name}) "
                      "y relanzá este programa. Para ver el plan antes: python -m hope.growth.budget_target "
                      f"--checkpoint {ck} --objetivo {etapa['presupuesto']}", flush=True)
                return
            es["hecha"] = True
            _save_state(out, st)
            continue
        if str(etapa.get("lote", 16)) == "auto":
            trainer.cfg.batch = _auto_batch(trainer, trainer.cfg.seq_bytes, 4, int(etapa.get("lote_max", 128)))
            _log(out, {"etapa": name, "lote_auto": trainer.cfg.batch})
        mix = Mixture(_sources(etapa["fuentes"]))
        evals = _sources(etapa["fuentes"], "val")
        vision = None
        if etapa.get("vision"):
            from hope.data.vision import ImageStore, ImageTextSource
            parts = []
            if Path(etapa["vision"]).exists():
                parts += ImageTextSource.from_store(ImageStore(etapa["vision"], trainer.model.cfg.vis_image)).items
            parts += ImageTextSource.from_procedural(int(etapa.get("procedurales", 20000)),
                                                     size=trainer.model.cfg.vis_image).items
            vision = ImageTextSource(parts)
        if etapa.get("enfriar") and not (out / "estable_antes_de_enfriar").exists():
            trainer.save(out / "estable_antes_de_enfriar")
        chunk = float(plan.get("checkpoint_min", 30))
        hours = float(etapa["horas"])
        while es["horas"] < hours:
            if max_hours is not None and (time.time() - t_start) / 3600 >= max_hours:
                return
            if (out / STOP_FILE).exists():          # pausa pedida justo al terminar la etapa anterior
                _log(out, {"etapa": name, "evento": "pausa", "paso": trainer.step})
                return
            remaining_h = hours - es["horas"]
            minutes = min(chunk, remaining_h * 60)
            # Las fuentes se vuelven a leer en cada tramo: lo que agrega el maestro mientras se entrena
            # (data/maestro, que llenan la aplicación o hope.teacher.harvest) entra en el próximo tramo.
            # Las vacías o ausentes se ignoran.
            nuevas = _sources_seguras(etapa["fuentes"])
            if nuevas is not None:
                mix, evals = nuevas
            t0 = time.time()
            if etapa.get("enfriar"):
                # En el enfriamiento la tasa baja según la fracción de horas cumplidas.
                steps_guess = max(int(minutes * 60 / max(es.get("s_por_paso", 1.0), 1e-3)), 1)
                rep = trainer.train(mix, steps_guess, eval_sources=evals, max_minutes=minutes, decay=True,
                                    decay_span=cooldown_span(es["horas"], minutes, hours))
            else:
                rep = trainer.train(mix, 10**9, eval_sources=evals, max_minutes=minutes, vision=vision,
                                    vision_frac=float(etapa.get("vision_frac", 0.0)),
                                    vision_seq=int(etapa.get("vision_secuencia", 512)))
            dt = time.time() - t0
            es["horas"] += dt / 3600
            es["s_por_paso"] = dt / max(rep.steps, 1)
            ck = out / name
            trainer.save(ck)
            export_light(trainer.model, out / "ligero.pt")
            st["ultimo_checkpoint"] = str(ck)
            _save_state(out, st)
            _log(out, {"etapa": name, "horas": round(es["horas"], 3), "paso": trainer.step, "pasos": rep.steps,
                       "bpb_val": rep.bpb_after, "bpb_entrenamiento": rep.train_bpb_last,
                       "bytes_por_s": rep.steps * trainer.cfg.batch * trainer.cfg.grad_accum * trainer.cfg.seq_bytes / dt})
            if (out / STOP_FILE).exists():
                _log(out, {"etapa": name, "evento": "pausa", "paso": trainer.step})
                return
        es["hecha"] = True
        _save_state(out, st)
        _log(out, {"etapa": name, "evento": "etapa_hecha", "horas": round(es["horas"], 3)})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/plan_6m.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-horas", type=float, default=None, help="cortar después de N horas (se retoma luego)")
    args = ap.parse_args(argv)
    run(args.plan, args.device, args.max_horas)


if __name__ == "__main__":
    main()
