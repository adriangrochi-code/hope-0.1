"""Genera ``static/compat.css``: lo que los navegadores viejos necesitan para ver la pantalla igual.

* ``grid-gap`` en cada grilla con ``gap`` (Firefox 52–60 y Chrome 57–65 solo conocen ``grid-gap``).
* Márgenes en lugar de ``gap`` en los contenedores flex, bajo ``.no-flexgap`` (lo pone compat.js si el
  navegador no sabe flex-gap: Firefox < 63, Chrome < 84).
* Modo liviano (``.lite``): sin animaciones, transiciones ni sombras.

Uso: ``python -m hope.cnc.app.compat_css`` (una prueba verifica que el archivo esté al día).
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).parent / "static"
SOURCES = ("cnc.css", "tema_tisma.css")
# contenedores flex cuyo «display: flex» está en otra regla que el «gap»
FLEX_EXTRA = {".screen": "column"}
HEADER = "/* GENERADO por hope/cnc/app/compat_css.py a partir de cnc.css: no editar a mano. */\n"


def _rules(css: str):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    depth_stack = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        sel, body = m.group(1).strip(), m.group(2)
        if sel.startswith("@") or not body.strip():
            continue
        depth_stack.append(sel)
        yield sel, body


def _decls(body: str) -> dict[str, str]:
    out = {}
    for part in body.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def generate() -> str:
    merged: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for name in SOURCES:
        for sel, body in _rules((STATIC / name).read_text(encoding="utf-8")):
            for s in (x.strip() for x in sel.split(",")):
                if s not in merged:
                    merged[s] = {}
                    order.append(s)
                merged[s].update(_decls(body))
    grid, flex = [], []
    for s in order:
        d = merged[s]
        gap = d.get("gap")
        if not gap or "@" in s or s.startswith(":"):
            continue
        disp = d.get("display", "")
        if "grid" in disp:
            grid.append(f"{s} {{ grid-gap: {gap}; }}")
        elif "flex" in disp or s in FLEX_EXTRA:
            parts = gap.split()
            row_gap, col_gap = (parts[0], parts[1]) if len(parts) > 1 else (parts[0], parts[0])
            column = d.get("flex-direction", "").startswith("column") or FLEX_EXTRA.get(s) == "column"
            if column:
                flex.append(f".no-flexgap {s} > * + * {{ margin-top: {row_gap}; }}")
            else:
                flex.append(f".no-flexgap {s} > * + * {{ margin-left: {col_gap}; }}")
                if "wrap" in d.get("flex-wrap", ""):
                    flex.append(f".no-flexgap {s} > * {{ margin-bottom: {row_gap}; }}")
    lite = (".lite *, .lite *:before, .lite *:after { animation: none !important; transition: none !important; "
            "box-shadow: none !important; text-shadow: none !important; }\n"
            ".lite .led.blink, .lite .reset-led.blink { opacity: 1; }\n")
    return (HEADER + "\n/* grillas: grid-gap para Firefox 52–60 y Chrome 57–65 */\n" + "\n".join(grid) +
            "\n\n/* flexbox sin gap (Firefox < 63, Chrome < 84) */\n" + "\n".join(flex) +
            "\n\n/* modo liviano para PCs viejas */\n" + lite)


def main() -> int:
    out = STATIC / "compat.css"
    out.write_text(generate(), encoding="utf-8")
    print(f"{out}: {len(out.read_text(encoding='utf-8').splitlines())} líneas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
