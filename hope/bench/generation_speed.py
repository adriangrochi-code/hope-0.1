"""Velocidad de respuesta y de razonamiento de Hope, en tokens equivalentes (pedido del usuario).

Hope genera **bytes**, no tokens. Para compararla con modelos de tokens (Qwen,
la línea base BPE) se convierte con la relación bytes/token de un tokenizador
real:

* si hay un ``bpe.json`` entrenado con el corpus, se usa su relación medida;
* si no, ~4 bytes por token, una aproximación habitual para español e inglés
  (se aclara en el reporte que es aproximada).

Mide:
  * **prefill**: procesar un prompt largo (bytes/s);
  * **primer byte**: latencia hasta el primer byte generado;
  * **generación**: bytes/s y tokens equivalentes/s, con la caché por parche;
  * **razonamiento**: tiempo para resolver un problema paso a paso hasta
    «Resultado:», con la crítica (CRE/HCRI) y los reintentos del Loop Controller.

Uso:
    python -m hope.bench.generation_speed --preset hn1-25m
    python -m hope.bench.generation_speed --run runs/hn1_25m --tokenizer data/f1/bpe.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hope.device import select_device, synchronize

DEFAULT_BYTES_PER_TOKEN = 4.0
SAMPLE_TEXT = ("La ley de Ohm establece que la tensión es igual a la corriente multiplicada por la "
               "resistencia. A voltage regulator keeps the output voltage constant when the load changes. ") * 4


def bytes_per_token(tokenizer_path: str | None, text: str = SAMPLE_TEXT) -> tuple[float, str]:
    if tokenizer_path and Path(tokenizer_path).exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)
        n = len(tok.encode(text).ids)
        return len(text.encode()) / max(n, 1), f"medido con {tokenizer_path}"
    return DEFAULT_BYTES_PER_TOKEN, "aproximado (≈4 bytes/token en español e inglés)"


def _timed(fn, device):
    synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    synchronize(device)
    return out, time.perf_counter() - t0


@torch.no_grad()
def measure(model, *, device, bpt: float, prompt_bytes: int = 1024, gen_bytes: int = 128,
            reasoning: bool = True) -> dict:
    model.eval()
    prompt = (SAMPLE_TEXT * (prompt_bytes // len(SAMPLE_TEXT) + 1)).encode()[:prompt_bytes]
    x = torch.tensor([list(prompt)], device=device)
    model(x[:, :32])                                             # calentamiento
    _, t_prefill = _timed(lambda: model(x), device)
    _, t_first = _timed(lambda: model.generate(prompt[-512:], max_new=1, stop=b"\x01"), device)
    out, t_gen = _timed(lambda: model.generate(prompt[-512:], max_new=gen_bytes, stop=b"\x01"), device)
    n = max(len(out), 1)
    rep = {
        "prefill_bytes_per_s": prompt_bytes / t_prefill,
        "prefill_tokens_equiv_per_s": prompt_bytes / t_prefill / bpt,
        "first_byte_latency_s": t_first,
        "generation_bytes_per_s": n / t_gen,
        "generation_tokens_equiv_per_s": n / t_gen / bpt,
        "generated_bytes": n,
    }
    if reasoning:
        from hope.constraints.critic import critique
        from hope.constraints.solver import analyze
        from hope.runtime.loop import solve_with_review
        from hope.teacher.cycle import extract_answer
        problem = "¿Qué tensión cae en un resistor de 220 ohm con 0.05 A?"
        an = analyze(problem)
        base = f"Tema: ley de Ohm\nProblema: {problem}\n{an.hint()}\n"
        produced = []

        def attempt(feedback):
            text = model.generate(base + feedback, max_new=300, stop_after_line=b"Resultado:").decode(
                "utf-8", errors="replace")
            produced.append(len(text.encode()))
            return text, extract_answer(text)
        res, t_reason = _timed(lambda: solve_with_review(attempt, lambda t, a: critique(problem, t, a, an),
                                                         max_iterations=2), device)
        total = sum(produced)
        rep.update({"reasoning_seconds": t_reason, "reasoning_iterations": res.iterations,
                    "reasoning_bytes": total, "reasoning_tokens_equiv": total / bpt,
                    "reasoning_stop": res.stop_reason, "reasoning_verdict": res.critique.verdict})
    return rep


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="hn1-25m")
    ap.add_argument("--run", default=None, help="carpeta con model.pt (si no, modelo sin entrenar del preset)")
    ap.add_argument("--tokenizer", default=None, help="bpe.json para medir bytes por token")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gen-bytes", type=int, default=128)
    ap.add_argument("--bf16", action="store_true", help="pesos en bf16 (como se entrena por defecto)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    device = select_device(args.device)
    if args.run:
        from hope.growth.usage_report import load_run
        model = load_run(Path(args.run))
    else:
        from hope.core.hn1 import build_hn1
        model, _ = build_hn1(args.preset)
    model.to(device)
    if args.bf16:
        model.to(torch.bfloat16)
    bpt, how = bytes_per_token(args.tokenizer)
    rep = {"device": str(device), "model": args.run or args.preset, "bytes_per_token": bpt,
           "bytes_per_token_source": how,
           "params": sum(p.numel() for p in model.parameters()),
           **measure(model, device=device, bpt=bpt, gen_bytes=args.gen_bytes)}
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    return rep


if __name__ == "__main__":
    main()
