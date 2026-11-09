"""Muestra de generación con el modelo ligero (``salida/ligero.pt``) para la aplicación.

    python -m hope.app.muestra --modelo runs/hope/ligero.pt --salida runs/hope/muestras.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROMPTS = [("es", "La ciudad de Buenos Aires"), ("en", "The history of science"),
           ("código", "def fibonacci(n):\n")]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--bytes", type=int, default=200)
    ap.add_argument("--temperatura", type=float, default=0.7)
    args = ap.parse_args(argv)
    import torch
    from hope.train.run import load_light
    torch.set_num_threads(2)
    print("PROGRESO 0.05 cargando el modelo", flush=True)
    model = load_light(args.modelo, "cpu")
    out = []
    for i, (lang, prompt) in enumerate(PROMPTS):
        torch.manual_seed(i)
        t0 = time.time()
        with torch.no_grad():
            raw = model.generate(prompt, max_new=args.bytes, temperature=args.temperatura)
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        out.append({"idioma": lang, "prompt": prompt, "texto": text, "segundos": round(time.time() - t0, 1)})
        print(f"PROGRESO {(i + 1) / len(PROMPTS):.2f} muestra {lang} lista", flush=True)
    sal = Path(args.salida)
    hist = []
    if sal.exists():
        try:
            hist = json.loads(sal.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            hist = []
    hist = (hist + [{"fecha": time.strftime("%Y-%m-%d %H:%M"), "modelo_mtime": Path(args.modelo).stat().st_mtime,
                     "muestras": out}])[-20:]
    sal.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    print("muestras guardadas en " + str(sal), flush=True)


if __name__ == "__main__":
    main()
