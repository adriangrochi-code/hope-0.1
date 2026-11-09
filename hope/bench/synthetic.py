"""Banco sintético de cuellos de botella con causa conocida (ARQUITECTURA_HN1 §12.2).

Pregunta: ¿el ciclo diagnóstico → crecimiento dirigido funciona cuando la
respuesta correcta se conoce de antemano?

Cinco familias de tareas, cada una con su región en el modelo:

========  ==========================================  ===========================
familia   construcción                                causa correcta
========  ==========================================  ===========================
easy      y = x1                                      SIN_PROBLEMA
capacity  tabla aleatoria de 2048 claves (x1, x2)     CAPACIDAD_INSUFICIENTE
          vista muchas veces → memorizar
noisy     y = (3·x1 + x2) mod C con 40 % de           ETIQUETAS_ERRONEAS
          etiquetas al azar y entradas repetidas
tool      y = (x1·x2 + x3) mod C                      HERRAMIENTA_FALTANTE
memory    tabla de 20 000 claves (x1, x2, x3);        MEMORIA_RECUPERACION
          cada clave aparece una sola vez
========  ==========================================  ===========================

Criterios de éxito (§12.2):
  1. el diagnóstico identifica la causa correcta de cada familia;
  2. ASG crece solo donde la causa es capacidad;
  3. el crecimiento dirigido supera al uniforme con los mismos parámetros y el mismo entrenamiento.

**Límite de validez:** el banco valida la *mecánica* del ciclo en un caso
construido. No demuestra la hipótesis central sobre datos reales.

Uso:
    python -m hope.bench.synthetic                 # 3 semillas
    python -m hope.bench.synthetic --quick         # prueba rápida
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from guardian.core import Guardian, GuardianConfig
from hope.core.toy import build_routed_model
from hope.epistemic import EpistemicStatus
from hope.growth.asg import Outcome, apply_plan, decide, plan_widening, request
from hope.growth.diagnosis import Cause, Signals, diagnose
from hope.memory.pesam import ExperienceLog

FAMILIES = ["easy", "capacity", "noisy", "tool", "memory"]
EXPECTED = {
    "easy": Cause.SIN_PROBLEMA,
    "capacity": Cause.CAPACIDAD_INSUFICIENTE,
    "noisy": Cause.ETIQUETAS_ERRONEAS,
    "tool": Cause.HERRAMIENTA_FALTANTE,
    "memory": Cause.MEMORIA_RECUPERACION,
}
V_X = 64            # valores posibles de x1, x2, x3
C = 64              # clases de salida
SEQ = 4             # [token de familia, x1, x2, x3]
VOCAB = V_X + len(FAMILIES)


# ---------------------------------------------------------------------------
# Datos, herramientas y memoria externa
# ---------------------------------------------------------------------------
@dataclass
class Split:
    x: torch.Tensor          # [N, SEQ]
    y: torch.Tensor          # [N]


@dataclass
class Family:
    name: str
    index: int
    train: Split
    val: Split


def _with_task(idx: int, xs: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.full((xs.shape[0], 1), V_X + idx), xs], dim=1)


def make_families(seed: int) -> tuple[list[Family], dict]:
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.randint(0, V_X, shape, generator=g)  # noqa: E731
    fams: list[Family] = []

    # easy: y = x1
    xtr, xva = rnd(4000, 3), rnd(1000, 3)
    fams.append(Family("easy", 0, Split(_with_task(0, xtr), xtr[:, 0].clone()),
                       Split(_with_task(0, xva), xva[:, 0].clone())))

    # capacity: memorizar una tabla aleatoria de 2048 claves vistas muchas veces
    keys = torch.randperm(V_X * V_X, generator=g)[:2048]
    table = torch.randint(0, C, (V_X * V_X,), generator=g)

    def cap_split(reps: int) -> Split:
        k = keys.repeat(reps)
        xs = torch.stack([k // V_X, k % V_X, rnd(k.numel())], dim=1)
        return Split(_with_task(1, xs), table[k].clone())
    fams.append(Family("capacity", 1, cap_split(4), cap_split(1)))

    # noisy: función simple, 40 % de etiquetas al azar, entradas repetidas
    pool = rnd(1000, 3)
    idx = torch.randint(0, 1000, (4000,), generator=g)
    xtr = pool[idx]
    ytr = (3 * xtr[:, 0] + xtr[:, 1]) % C
    flip = torch.rand(4000, generator=g) < 0.4
    ytr[flip] = torch.randint(0, C, (int(flip.sum()),), generator=g)
    xva = rnd(1000, 3)
    fams.append(Family("noisy", 2, Split(_with_task(2, xtr), ytr),
                       Split(_with_task(2, xva), (3 * xva[:, 0] + xva[:, 1]) % C)))

    # tool: aritmética modular exacta
    tool_fn = lambda xs: (xs[:, 0] * xs[:, 1] + xs[:, 2]) % C  # noqa: E731
    xtr, xva = rnd(4000, 3), rnd(1000, 3)
    fams.append(Family("tool", 3, Split(_with_task(3, xtr), tool_fn(xtr)),
                       Split(_with_task(3, xva), tool_fn(xva))))

    # memory: hechos que aparecen una sola vez; las claves de validación no se vieron
    mkeys = torch.randperm(V_X ** 3, generator=g)[:20000]
    mvals = torch.randint(0, C, (20000,), generator=g)
    mx = torch.stack([mkeys // (V_X * V_X), (mkeys // V_X) % V_X, mkeys % V_X], dim=1)
    fams.append(Family("memory", 4, Split(_with_task(4, mx[:10000]), mvals[:10000]),
                       Split(_with_task(4, mx[10000:12000]), mvals[10000:12000])))

    resources = {
        # Catálogo de herramientas (HTE) disponibles: nombre → función sobre (x1, x2, x3).
        "tools": {"mul_add_mod": tool_fn, "sum_mod": lambda xs: xs.sum(dim=1) % C},
        # Memoria externa (AHMC/MDD): clave (x1, x2, x3) → valor.
        "store": {tuple(k.tolist()): int(v) for k, v in zip(mx, mvals)},
    }
    return fams, resources


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def _batch(fams: list[Family], per_family: int, g: torch.Generator):
    xs, ys, fs = [], [], []
    for f in fams:
        i = torch.randint(0, f.train.y.numel(), (per_family,), generator=g)
        xs.append(f.train.x[i]); ys.append(f.train.y[i]); fs.append(torch.full((per_family,), f.index))
    return torch.cat(xs), torch.cat(ys), torch.cat(fs)


def train(model, fams: list[Family], steps: int, seed: int, lr: float = 3e-3,
          per_family: int = 64) -> dict[str, list[float]]:
    """Devuelve el historial de pérdida por familia."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {f.name: [] for f in fams}
    model.train()
    for _ in range(steps):
        x, y, fam = _batch(fams, per_family, g)
        losses = F.cross_entropy(model(x, fam), y, reduction="none")
        opt.zero_grad(set_to_none=True)
        losses.mean().backward()
        opt.step()
        for f in fams:
            hist[f.name].append(losses[fam == f.index].mean().item())
    return hist


@torch.no_grad()
def accuracy(model, fams: list[Family], split: str) -> dict[str, float]:
    model.eval()
    out = {}
    for f in fams:
        s = getattr(f, split)
        fam = torch.full((s.y.numel(),), f.index)
        out[f.name] = (model(s.x, fam).argmax(-1) == s.y).float().mean().item()
    return out


def loss_trend(hist: list[float]) -> float:
    """Mejora relativa entre el penúltimo y el último décimo del entrenamiento."""
    n = max(len(hist) // 10, 1)
    prev, last = sum(hist[-2 * n:-n]) / n, sum(hist[-n:]) / n
    return (prev - last) / max(prev, 1e-8)


def label_conflict_rate(s: Split) -> float:
    labels: dict[tuple, set] = {}
    for x, y in zip(s.x.tolist(), s.y.tolist()):
        labels.setdefault(tuple(x), set()).add(y)
    repeated = [v for v in labels.values()]
    return sum(len(v) > 1 for v in repeated) / max(len(repeated), 1)


def tool_fit(s: Split, tools: dict) -> tuple[str, float]:
    best = max(((name, (fn(s.x[:, 1:]) == s.y).float().mean().item()) for name, fn in tools.items()),
               key=lambda t: t[1])
    return best


def store_fit(s: Split, store: dict) -> float:
    hits = [store.get(tuple(x), -1) == y for x, y in zip(s.x[:, 1:].tolist(), s.y.tolist())]
    return sum(hits) / max(len(hits), 1)


# ---------------------------------------------------------------------------
# Experimento completo para una semilla
# ---------------------------------------------------------------------------
@dataclass
class Config:
    base_steps: int = 3000
    probe_steps: int = 1000
    grow_steps: int = 2000
    probe_units: int = 80        # la sonda prueba el mismo crecimiento que se aplicaría
    growth_units: int = 80       # presupuesto total de crecimiento (unidades ocultas)


def run_seed(seed: int, cfg: Config, workdir: Path, log=print) -> dict:
    torch.manual_seed(seed)
    fams, res = make_families(seed)
    model, reg = build_routed_model(FAMILIES, vocab=VOCAB, seq_len=SEQ, n_classes=C)
    guardian = _bench_guardian(workdir, reg.report(model).total, cfg)
    pesam = ExperienceLog(workdir / "pesam.jsonl")
    region_of = {f.name: f.index + 1 for f in fams}

    # 1. Entrenamiento base
    t0 = time.time()
    hist = train(model, fams, cfg.base_steps, seed)
    tr, va = accuracy(model, fams, "train"), accuracy(model, fams, "val")
    log(f"  [semilla {seed}] base: {cfg.base_steps} pasos en {time.time() - t0:.0f}s")

    # 2. Señales y diagnóstico (reglas baratas primero)
    diagnoses = {}
    for f in fams:
        tname, tfit = tool_fit(f.train, res["tools"])
        sig = Signals(task=f.name, region=reg.find_by_path(f"regions.{region_of[f.name]}").id,
                      train_acc=tr[f.name], val_acc=va[f.name], loss_trend=loss_trend(hist[f.name]),
                      label_conflict_rate=label_conflict_rate(f.train),
                      best_tool=tname, tool_fit=tfit, store_fit=store_fit(f.train, res["store"]))
        dx = diagnose(sig)
        if dx.needs_capacity_probe:
            sig.capacity_probe_gain, sig.optimization_gain = _capacity_probe(
                model, reg, fams, f, region_of[f.name], cfg, seed, guardian)
            dx = diagnose(sig)
        diagnoses[f.name] = dx
        pesam.record("diagnostico", dx.status, seed=seed, diagnosis=dx.as_dict())

    targets = [n for n, d in diagnoses.items() if d.cause == Cause.CAPACIDAD_INSUFICIENTE]

    # 3. Tres brazos con el mismo entrenamiento adicional
    arms: dict[str, dict] = {}
    control = copy.deepcopy(model)
    train(control, fams, cfg.grow_steps, seed + 1)
    arms["fijo"] = {"val": accuracy(control, fams, "val"), "params": reg.report(control).total}

    plans = {}
    if targets:
        plans["dirigido"] = [region_of[t] for t in targets]
    plans["uniforme"] = [region_of[n] for n in FAMILIES]
    for arm, regions in plans.items():
        m, r = copy.deepcopy(model), copy.deepcopy(reg)
        plan = plan_widening(m, regions, cfg.growth_units, seed=seed)
        decision = request(guardian, m, r, plan, consolidate=False,
                           justification=f"brazo {arm}: {[d.cause.value for d in diagnoses.values()]}")
        if not decision.allowed:
            arms[arm] = {"denied": list(decision.reasons)}
            continue
        apply_plan(m, r, plan)
        train(m, fams, cfg.grow_steps, seed + 1)
        arms[arm] = {"val": accuracy(m, fams, "val"), "params": r.report(m).total}
        if arm == "dirigido":
            gains = {n: arms[arm]["val"][n] - arms["fijo"]["val"][n] for n in FAMILIES}
            outcome = decide(gains, targets)
            consolidation = request(guardian, model, reg, plan, consolidate=True)
            arms[arm]["decision"] = {
                "outcome": outcome.outcome.value, "reasons": outcome.reasons,
                "guardian": ("permitido" if consolidation.allowed else
                             "requiere aprobación humana" if consolidation.needs_human else
                             "denegado: " + "; ".join(consolidation.reasons)),
            }
            pesam.record("experimento_crecimiento",
                         EpistemicStatus.PROBADO if outcome.outcome != Outcome.REVERTIR else EpistemicStatus.REFUTADO,
                         seed=seed, regions=regions, gains=gains, outcome=outcome.outcome.value)

    return {
        "seed": seed,
        "base": {"train": tr, "val": va},
        "diagnoses": {n: {"cause": d.cause.value, "expected": EXPECTED[n].value,
                          "correct": d.cause == EXPECTED[n], "observed": d.observed}
                      for n, d in diagnoses.items()},
        "growth_targets": targets,
        "arms": arms,
    }


def _capacity_probe(model, reg, fams, fam: Family, region: int, cfg: Config, seed: int,
                    guardian: Guardian) -> tuple[float, float]:
    """Copia ensanchada vs. copia control, mismo entrenamiento corto (ASG §7, fase 3).

    Devuelve (ganancia de capacidad sobre el control, ganancia del control sobre el modelo base).
    """
    probe, preg = copy.deepcopy(model), copy.deepcopy(reg)
    plan = plan_widening(probe, [region], cfg.probe_units, seed=seed)
    if not request(guardian, probe, preg, plan, consolidate=False,
                   justification=f"sonda de capacidad para {fam.name}").allowed:
        return float("nan"), float("nan")
    apply_plan(probe, preg, plan)
    control = copy.deepcopy(model)
    before = accuracy(model, [fam], "train")[fam.name]
    train(probe, fams, cfg.probe_steps, seed + 100)
    train(control, fams, cfg.probe_steps, seed + 100)
    p, c = accuracy(probe, [fam], "train")[fam.name], accuracy(control, [fam], "train")[fam.name]
    return p - c, c - before


def _bench_guardian(workdir: Path, base_total: int, cfg: Config) -> Guardian:
    """Guardián con un techo propio del banco: el modelo base más un margen acotado."""
    budget = workdir / "budget.yaml"
    budget.write_text(
        f"max_total_params: {base_total + 20_000}\n"
        "mode: AUTO_CONSOLIDATION_WITHIN_LIMIT\n"
        "consolidation:\n  max_auto_param_fraction: 0.10\n", encoding="utf-8")
    return Guardian(GuardianConfig(workdir=str(workdir), protected_paths=[],
                                   budget_path=str(budget),
                                   audit_path=str(workdir / "audit.jsonl"),
                                   approvals_path=str(workdir / "approvals.json")))


# ---------------------------------------------------------------------------
# Resumen y reporte
# ---------------------------------------------------------------------------
def summarize(results: list[dict]) -> dict:
    diag_acc = statistics.mean(d["correct"] for r in results for d in r["diagnoses"].values())

    def arm_stats(arm: str, fam: str | None):
        vals = []
        for r in results:
            a = r["arms"].get(arm, {})
            if "val" in a:
                vals.append(a["val"][fam] if fam else statistics.mean(a["val"].values()))
        if not vals:
            return None
        return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals), "n": len(vals)}

    arms = sorted({a for r in results for a in r["arms"]})
    table = {arm: {"promedio": arm_stats(arm, None), **{f: arm_stats(arm, f) for f in FAMILIES}}
             for arm in arms}
    params = {arm: [r["arms"][arm].get("params") for r in results if arm in r["arms"]] for arm in arms}
    unnecessary = sum(1 for r in results for t in r["growth_targets"] if EXPECTED[t] != Cause.CAPACIDAD_INSUFICIENTE)
    d, u = table.get("dirigido", {}).get("promedio"), table.get("uniforme", {}).get("promedio")
    return {
        "diagnosis_accuracy": diag_acc,
        "unnecessary_growth_targets": unnecessary,
        "val_accuracy": table,
        "params": params,
        "criteria": {
            "1_diagnostico_correcto": diag_acc == 1.0,
            "2_sin_crecimiento_innecesario": unnecessary == 0,
            "3_dirigido_supera_uniforme": bool(d and u and d["mean"] > u["mean"]),
        },
    }


def to_markdown(summary: dict, results: list[dict], cfg: Config) -> str:
    lines = ["# Banco sintético de ASG — resultados", "",
             f"Semillas: {len(results)} · pasos base {cfg.base_steps} · sonda {cfg.probe_steps} · "
             f"crecimiento {cfg.grow_steps} · presupuesto {cfg.growth_units} unidades", "",
             "## Criterios de éxito", ""]
    for k, v in summary["criteria"].items():
        lines.append(f"- {'✅' if v else '❌'} {k.replace('_', ' ')}")
    lines += ["", f"Exactitud del diagnóstico: **{summary['diagnosis_accuracy']:.0%}**", "",
              "## Diagnóstico por familia (semilla por semilla)", "",
              "| familia | esperado | " + " | ".join(f"semilla {r['seed']}" for r in results) + " |",
              "|---|---|" + "---|" * len(results)]
    for f in FAMILIES:
        cells = [("✅ " if r["diagnoses"][f]["correct"] else "❌ ") + r["diagnoses"][f]["cause"] for r in results]
        lines.append(f"| {f} | {EXPECTED[f].value} | " + " | ".join(cells) + " |")
    lines += ["", "## Exactitud de validación (media ± desvío entre semillas)", "",
              "| brazo | parámetros | promedio | " + " | ".join(FAMILIES) + " |",
              "|---|---|---|" + "---|" * len(FAMILIES)]
    for arm, row in summary["val_accuracy"].items():
        fmt = lambda s: "—" if s is None else f"{s['mean']:.3f} ± {s['std']:.3f}"  # noqa: E731
        p = summary["params"][arm]
        lines.append(f"| {arm} | {p[0] if p else '—'} | {fmt(row['promedio'])} | "
                     + " | ".join(fmt(row[f]) for f in FAMILIES) + " |")
    lines += ["", "## Decisión de ASG y guardián", ""]
    for r in results:
        dec = r["arms"].get("dirigido", {}).get("decision")
        if dec:
            lines.append(f"- semilla {r['seed']}: {dec['outcome']} ({'; '.join(dec['reasons'])}) — "
                         f"guardián: {dec['guardian']}")
    lines += ["", "> Límite de validez: este banco valida la mecánica del ciclo diagnóstico → "
              "crecimiento en un caso construido con causas conocidas. No demuestra la hipótesis "
              "central sobre datos reales."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Banco sintético de ASG (§12.2)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=0,
                    help="usar semillas nuevas para validar sin las que se usaron al calibrar")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="runs/synthetic_bench")
    args = ap.parse_args(argv)
    cfg = Config(base_steps=600, probe_steps=200, grow_steps=400) if args.quick else Config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(torch.get_num_threads(), 1))

    results = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        with tempfile.TemporaryDirectory() as tmp:
            results.append(run_seed(seed, cfg, Path(tmp)))
    summary = summarize(results)
    (out / "results.json").write_text(json.dumps({"config": cfg.__dict__, "summary": summary,
                                                  "results": results}, indent=2, ensure_ascii=False,
                                                 default=str), encoding="utf-8")
    md = to_markdown(summary, results, cfg)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return summary


if __name__ == "__main__":
    main()
