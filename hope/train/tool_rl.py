"""Refuerzo del uso de herramientas (docs/HERRAMIENTAS_HN1.md §7.4).

Usa ``grpo_step_tools`` de ``hope/train/rlvr.py`` sobre problemas del currículo de herramientas
(``hope/data/tool_curriculum.py``, semilla propia y sin los problemas del banco):

    recompensa = calidad verificada − costo (bytes generados por Hope, con tope 0,5),
                 y el costo se cobra solo cuando la respuesta es correcta

Calidad (``tool_quality``):

* 1,0 si la respuesta es correcta y está **verificada**: salió de una llamada que funcionó, o el
  problema era trivial y se respondió directo;
* 0,5 si es correcta pero sin verificar (acertó «a ojo» un problema que pedía herramienta);
* 0 si es incorrecta. En código, «correcta» = la última función probada pasa las pruebas ocultas.

El costo ya castiga las llamadas innecesarias (más bytes y más tiempo) y premia delegar en la
herramienta cuando escribir la cuenta a mano sale más caro. Las fallas inyectadas (``error``)
enseñan a reintentar: el grupo que reintenta y acierta gana frente al que se rinde.

Recomendaciones de ``reports/Razonamiento por parámetro para Hope.md`` §1.4 aplicadas por
defecto: KL baja (1·10⁻³), y usar el RL **después** del entrenamiento supervisado con el currículo
(un modelo que nunca vio ``⟦…⟧`` casi no genera llamadas y los grupos no tienen señal).

Uso::

    python -m hope.train.tool_rl --prueba                     # humo en CPU con hn1-tiny (segundos)
    python -m hope.train.tool_rl --modelo runs/hope/ligero.pt --pasos 300 --salida runs/tool_rl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from hope.data.tool_curriculum import (FaultInjectingRunner, ToolExample, answer_ok, build_examples,
                                       default_notebook, extract_answer)
from hope.runtime.toolcalls import CALL_RE

RL_MIX = {"calculo": 0.16, "fraccion": 0.06, "estadistica": 0.06, "unidades": 0.14, "ecuacion": 0.08,
          "circuito": 0.06, "cadena": 0.08, "trivial": 0.16, "error": 0.10, "definicion": 0.06, "codigo": 0.04}


def tool_quality(item: ToolExample, text: str, catalog=None) -> float:
    """Calidad verificada de una respuesta generada (sin costo)."""
    calls = [(m.group(1), m.group(2)) for m in CALL_RE.finditer(text)]
    ans = extract_answer(text)
    if item.comparar == "codigo":
        from hope.tools.code_tools import run_tests
        code = [a for n, a in calls if n == "pruebas"]
        if not code:
            return 0.0
        try:
            return 1.0 if run_tests(code[-1].split("|||")[0].strip(), item.pruebas_ocultas, timeout_s=30)["ok"] else 0.0
        except Exception:
            return 0.0
    if not answer_ok(ans, item.respuesta, item.comparar):
        return 0.0
    if not item.necesita_herramienta:
        return 1.0
    # verificada: hubo al menos una llamada cuyo resultado no fue un error
    results = text.split("⟧ → ")[1:]
    worked = any(not r.startswith("error") and not r.startswith("denegado") for r in results)
    return 1.0 if worked else 0.5


def model_bytes(text: str) -> int:
    """Bytes que escribió Hope: el texto menos los resultados de herramientas (de «⟧ → » a fin de línea;
    un resultado de varias líneas, como el de pytest, se cuenta solo en su primera línea)."""
    import re
    return len(re.sub(r"⟧ → [^\n]*", "⟧", text).encode())


def rewarded(item: ToolExample, text: str, cost_per_byte: float = 0.001, max_penalty: float = 0.5) -> float:
    """Recompensa = calidad verificada − costo, cobrando el costo **solo si la respuesta es correcta**
    (Fatemi et al.; reports/Razonamiento por parámetro §1.4). Entre dos correctas gana la más corta."""
    q = tool_quality(item, text)
    return q - min(max_penalty, cost_per_byte * model_bytes(text)) * q if q > 0 else 0.0


def load_model(path: str | None, preset_name: str):
    from hope.core.hn1 import HN1Model, preset
    if path:
        from hope.train.run import load_light
        return load_light(path)
    torch.manual_seed(0)
    return HN1Model(preset(preset_name))


def train(model, problems: list[ToolExample], *, steps: int, batch: int = 4, group: int = 4, max_new: int = 200,
          lr: float = 1e-5, kl_weight: float = 1e-3, temperature: float = 0.8, seed: int = 0, log=print,
          max_minutes: float | None = None, cost_per_byte: float = 0.001) -> list[dict]:
    from hope.tools.builtin import default_catalog
    from hope.train.rlvr import GRPO, CostWeights, RLConfig, grpo_step_tools
    rl = GRPO(model, RLConfig(group=group, temperature=temperature, max_new=max_new, lr=lr, kl_weight=kl_weight))
    catalog, notebook = default_catalog(), default_notebook()
    rng = random.Random(seed)
    # los problemas con falla inyectada van en lotes aparte (el runner es el mismo para todo el lote)
    pools = {0: [p for p in problems if not p.falla_inyectada], 1: [p for p in problems if p.falla_inyectada]}
    # El costo va dentro de ``rewarded`` y solo para respuestas correctas: si se cobrara siempre, con la
    # mayoría de las respuestas mal la única diferencia en el grupo sería el largo y el modelo aprende a
    # callarse y dejar las herramientas (pasó: docs/resultados/uso_herramientas.md).
    w = CostWeights(per_model_byte=0.0, per_second=0.0, max_penalty=0.0)
    history, t0 = [], time.time()
    for step in range(steps):
        if max_minutes and time.time() - t0 > max_minutes * 60:
            break
        fails = 1 if pools[1] and rng.random() < len(pools[1]) / len(problems) else 0
        items = rng.sample(pools[fails], min(batch, len(pools[fails])))
        prompts = [it.prompt for it in items]
        st = grpo_step_tools(rl, prompts, lambda i, text: rewarded(items[i], text, cost_per_byte),
                             lambda: FaultInjectingRunner(catalog, memory=notebook, fail_next=fails), w)
        st = {"paso": step + 1, **{k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()}}
        history.append(st)
        if log and (step % 10 == 0 or step == steps - 1):
            log(json.dumps(st, ensure_ascii=False))
    return history


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", default=None, help="pesos livianos (export_light); sin esto, --preset al azar")
    ap.add_argument("--preset", default="hn1-tiny")
    ap.add_argument("--pasos", type=int, default=200)
    ap.add_argument("--problemas", type=int, default=400)
    ap.add_argument("--lote", type=int, default=4)
    ap.add_argument("--grupo", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl", type=float, default=1e-3)
    ap.add_argument("--max-minutos", type=float, default=None)
    ap.add_argument("--banco", default=None, help="banco para medir antes y después (por defecto el sellado)")
    ap.add_argument("--items-banco", type=int, default=None, help="evaluar solo los primeros N ítems por categoría")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--prueba", action="store_true", help="humo en CPU: hn1-tiny, 3 pasos, banco reducido")
    args = ap.parse_args(argv)
    if args.prueba:
        args.preset, args.pasos, args.problemas, args.lote, args.grupo, args.max_new = "hn1-tiny", 3, 24, 2, 3, 48
        args.items_banco = args.items_banco or 1
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    from hope.data.tool_curriculum import bench_problems
    from hope.eval.tool_bench import DEFAULT_BENCH, build_bench, load_bench, run_bench, subset_per_category
    model = load_model(args.modelo, args.preset)
    mix = dict(RL_MIX)
    if args.prueba:
        mix.pop("codigo")                        # pytest en sandbox es lo más lento: fuera del humo
    problems = build_examples(args.problemas, seed="rl", mix=mix, exclude=bench_problems(), hint_fraction=0.3)
    bench = load_bench(args.banco or DEFAULT_BENCH) if Path(args.banco or DEFAULT_BENCH).exists() else build_bench()
    if args.items_banco:
        bench = subset_per_category(bench, args.items_banco)
    gen_kw = {"max_new": max(args.max_new, 160)}
    before = run_bench(model.generate, bench, **gen_kw)["resumen"]
    print("antes:", json.dumps({k: v for k, v in before.items() if k != "por_categoria"}, ensure_ascii=False))
    history = train(model, problems, steps=args.pasos, batch=args.lote, group=args.grupo, max_new=args.max_new,
                    lr=args.lr, kl_weight=args.kl, max_minutes=args.max_minutos)
    after = run_bench(model.generate, bench, **gen_kw)["resumen"]
    print("después:", json.dumps({k: v for k, v in after.items() if k != "por_categoria"}, ensure_ascii=False))
    report = {"pasos": len(history), "problemas": len(problems), "banco_items": len(bench), "antes": before,
              "despues": after, "historial": history}
    if args.salida:
        out = Path(args.salida)
        out.mkdir(parents=True, exist_ok=True)
        from hope.train.run import export_light
        export_light(model, out / "ligero.pt")
        (out / "informe.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
