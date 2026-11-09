"""Mide la generación en lote y la caché de conversación (§3.9.1; ``hope/core/batched.py``).

* **Lote**: bytes/s de ``generate`` (una secuencia por vez) frente a ``generate_batch`` con
  B = 1, 4 y 8, con prompts distintos y con el mismo prompt (mejor-de-N).
* **Conversación**: tiempo de un turno con ``Conversation`` (solo procesa lo nuevo) frente a
  ``generate`` con todo el historial, para historiales de 500 y 2000 bytes.

Se desactiva el byte de parada para que todas las secuencias generen la misma cantidad.

Uso:
    python -m hope.bench.batch_speed --preset hn1-cpu
    python -m hope.bench.batch_speed --preset hn1-25m --device xpu
"""

from __future__ import annotations

import argparse
import time

import torch

from hope.core.batched import BatchStats, Conversation, generate_batch
from hope.core.hn1 import build_hn1
from hope.device import synchronize

TEXT = ("La ley de Ohm establece que la tensión es igual a la corriente multiplicada por la "
        "resistencia. Un regulador mantiene constante la tensión de salida cuando cambia la carga. ")


def _timed(fn, device):
    synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    synchronize(device)
    return out, time.perf_counter() - t0


def bench_batch(model, device, new: int = 64, sizes=(1, 4, 8)) -> list[dict]:
    rows = []
    prompts = [TEXT[i * 7:i * 7 + 40 + 9 * i] for i in range(max(sizes))]       # largos distintos
    outs, t = _timed(lambda: [model.generate(p, new, stop=b"") for p in prompts[:max(sizes)]], device)
    rows.append({"modo": "generate (una por vez)", "B": max(sizes), "bytes_s": sum(map(len, outs)) / t})
    modes = [("prompts distintos", False, 0.0), ("mismo prompt", True, 0.0),
             ("mismo prompt, muestreo T=1 (mejor-de-N)", True, 1.0)]
    for name, same, temp in modes:
        for b in sizes:
            ps = [prompts[0]] * b if same else prompts[:b]
            st = BatchStats()
            outs, t = _timed(lambda: generate_batch(model, ps, new, stop=b"", temperature=temp, stats=st), device)
            rows.append({"modo": "generate_batch, " + name, "B": b, "bytes_s": sum(map(len, outs)) / t,
                         "parches_por_pasada_global": round(st.patches_per_global_call, 2)})
    return rows


def bench_conversation(model, device, history_sizes=(500, 2000), new: int = 32) -> list[dict]:
    rows = []
    turn = "\nUsuario: ¿y la potencia?\nHope:"
    for n in history_sizes:
        hist = (TEXT * (n // len(TEXT) + 1))[:n]
        r = {}
        for key, k in (("primer_byte", 1), ("turno", new)):
            conv = Conversation(model, max_context=n + 512)
            conv.say(hist, new, stop=b"")                                    # primer turno (prefill)
            full = bytes(conv.history) + turn.encode()
            _, r[key + "_cache_s"] = _timed(lambda: conv.say(turn, k, stop=b""), device)
            ref, r[key + "_completo_s"] = _timed(lambda: model.generate(full, k, stop=b"", max_context=n + 512),
                                                 device)
            assert bytes(conv.history[-len(ref):]) == ref                    # misma salida
        rows.append({"historial": len(full), **r, "bytes_procesados": len(turn.encode()) + 1})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="hn1-cpu")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--new", type=int, default=64)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    dev = torch.device(a.device)
    model, _ = build_hn1(a.preset, dev)
    model.eval()
    model.generate(TEXT[:40], 4, stop=b"")                                    # calentamiento
    print(f"preset {a.preset}, dispositivo {a.device}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parámetros")
    print("| modo | B | bytes/s | parches por pasada global |\n|---|---|---|---|")
    for r in bench_batch(model, dev, a.new):
        print(f"| {r['modo']} | {r['B']} | {r['bytes_s']:.0f} | {r.get('parches_por_pasada_global', '—')} |")
    print("\n| historial (bytes) | bytes procesados en el turno | primer byte: con caché / completo (ms) | "
          f"turno de {a.new // 2} bytes: con caché / completo (ms) |\n|---|---|---|---|")
    for r in bench_conversation(model, dev, new=a.new // 2):
        print(f"| {r['historial']} | {r['bytes_procesados']} | {1e3 * r['primer_byte_cache_s']:.0f} / "
              f"{1e3 * r['primer_byte_completo_s']:.0f} | {1e3 * r['turno_cache_s']:.0f} / "
              f"{1e3 * r['turno_completo_s']:.0f} |")

if __name__ == "__main__":
    main()
