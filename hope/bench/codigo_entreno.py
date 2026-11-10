"""Experimento corto en CPU: ¿entrenar con trazas de ejecución (y FIM) mejora lo que Hope entiende del código?

Mide, **antes y después** de un ajuste corto del checkpoint sobre textos de trazas
(``hope/data/trazas.py``) con ``TrainConfig.fim_rate``:

* **predecir la salida** de programas generados *no vistos* (otra semilla): coincidencia exacta;
* **completar el medio** (FIM) de esos programas: se quita una línea y se pide el medio;
  coincidencia exacta de la línea;
* BPB sobre trazas no vistas.

Sin control (no hay corrida igual sin FIM ni sin trazas): mide si el formato se aprende, no
compara fuentes de datos. Uso::

    python -m hope.bench.codigo_entreno --checkpoint checkpoints/hn1-25m-cpu.pt --pasos 300 \\
        --out docs/resultados/codigo_entreno.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch


def _eval(model, rows: list[dict], n: int) -> dict:
    from hope.data.fim import complete_middle
    from hope.data.trazas import predict_prompt
    pred_ok = fim_ok = fim_n = 0
    examples = []
    rng = random.Random(0)
    for r in rows[:n]:
        out = model.generate(predict_prompt(r["codigo"]), max(len(r["salida"].encode()) + 8, 16),
                             stop_seq="⟦fin⟧".encode()).decode("utf-8", "replace")
        got = out.split("⟦fin⟧")[0].rstrip("\n")
        pred_ok += got == r["salida"].rstrip("\n")
        if len(examples) < 4:
            examples.append({"esperado": r["salida"].rstrip("\n"), "obtenido": got[:120]})
        lines = r["codigo"].split("\n")
        if len(lines) >= 3:
            i = rng.randrange(1, len(lines) - 1)
            pre, suf = "\n".join(lines[:i]) + "\n", "\n" + "\n".join(lines[i + 1:])
            mid = complete_middle(model, pre, suf, max_new=len(lines[i].encode()) + 16)
            fim_ok += mid.strip("\n") == lines[i]
            if fim_n < 3:
                examples.append({"fim_esperado": lines[i], "fim_obtenido": mid[:120]})
            fim_n += 1
    return {"prediccion_exacta": round(pred_ok / max(min(n, len(rows)), 1), 4),
            "fim_linea_exacta": round(fim_ok / max(fim_n, 1), 4), "n": min(n, len(rows)), "ejemplos": examples}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default="checkpoints/hn1-25m-cpu.pt")
    ap.add_argument("--programas", type=int, default=3000)
    ap.add_argument("--pasos", type=int, default=300)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--secuencia", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fim", type=float, default=0.5)
    ap.add_argument("--eval", type=int, default=60)
    ap.add_argument("--hilos", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.hilos:
        torch.set_num_threads(a.hilos)
    from hope.bench.codigo import bank, load_model
    from hope.data.trazas import build, exercise_programs, generated_programs, texts
    from hope.train.continual import Mixture, Source, Trainer, TrainConfig
    t0 = time.time()
    train_rows = build(generated_programs(a.programas, seed=1) + exercise_programs(bank()))
    test_rows = build(generated_programs(4 * a.eval, seed=99))
    seen = {r["codigo"] for r in train_rows}
    test_rows = [r for r in test_rows if r["codigo"] not in seen][:a.eval]
    model = load_model(a.checkpoint)
    before = _eval(model, test_rows, a.eval)
    tr = Trainer(model, TrainConfig(lr=a.lr, batch=a.lote, seq_bytes=a.secuencia, precision="fp32", warmup=20,
                                    schedule="cosine", fim_rate=a.fim), device="cpu")
    train_src = Source.from_texts("trazas", texts(train_rows), 1.0)
    test_src = Source.from_texts("trazas_nuevas", texts(test_rows), 1.0)
    rep = tr.train(Mixture([train_src]), a.pasos, eval_sources=[test_src])
    after = _eval(tr.model.eval(), test_rows, a.eval)
    out = {"checkpoint": a.checkpoint, "programas_entreno": len(train_rows), "programas_eval": len(test_rows),
           "pasos": rep.steps, "lote": a.lote, "secuencia": a.secuencia, "fim_rate": a.fim,
           "bpb_trazas_nuevas": {"antes": round(rep.bpb_before["trazas_nuevas"], 4),
                                 "despues": round(rep.bpb_after["trazas_nuevas"], 4)},
           "antes": before, "despues": after, "minutos": round((time.time() - t0) / 60, 1)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
