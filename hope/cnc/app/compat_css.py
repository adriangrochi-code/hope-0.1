"""Generates ``static/compat.css``: what old browsers need to render the screen the same way.

What it is and what it is for
-----------------------------
TISMA CNC must look the same in Firefox 52 ESR (the last one for Windows XP/Vista) and Chrome 57 as
in a current one. ``cnc.css`` and ``tema_tisma.css`` are written with modern CSS (``gap``), and this
script reads those rules and writes the equivalents for old browsers:

* ``grid-gap`` on every grid with ``gap`` (Firefox 52–60 and Chrome 57–65 only know ``grid-gap``).
* Margins instead of ``gap`` in flex containers, under ``.no-flexgap`` (set by compat.js when the
  browser lacks flex-gap: Firefox < 63, Chrome < 84).
* Lightweight mode (``.lite``): no animations, transitions or shadows.

How it fits in
--------------
* Reads: ``static/cnc.css`` and ``static/tema_tisma.css`` (in that order; the last rule wins, as in
  the browser).
* Writes: ``static/compat.css``, which ``index.html`` loads after the other two sheets.
* Uses the classes ``static/compat.js`` puts on ``<html>``: ``no-flexgap`` (detected) and ``lite``
  (lightweight mode: automatic on 1–2 core PCs, with ``?liviano=1`` or with the checkbox on screen).
* ``tests/test_cnc_compat.py`` compares :func:`generate` with the file: if ``cnc.css`` changes and it
  is not regenerated, the test fails.

Limitations (on purpose, to keep it simple): it does not understand rules nested inside ``@media``
nor selectors starting with ``:``; a flex container whose ``display: flex`` is in a different rule
than its ``gap`` is declared by hand in ``FLEX_EXTRA``.

Usage: ``python -m hope.cnc.app.compat_css`` (a test checks that the file is up to date).
"""

from __future__ import annotations

import re
from pathlib import Path

# ==================================================================================================
# ======================================== CONFIGURATION ===========================================
# ==================================================================================================

# Page folder: the stylesheets are read from here and compat.css is written here.
STATIC = Path(__file__).parent / "static"
# Stylesheets to read, in the order index.html loads them (the theme overrides cnc.css).
SOURCES = ("cnc.css", "tema_tisma.css")
# Flex containers whose «display: flex» is in a different rule than their «gap»
# (selector → direction; "column" = children stack one below the other, so the gap is margin-top).
FLEX_EXTRA = {".screen": "column"}
# First line of the generated file: warns not to edit it by hand (edits would be lost on regeneration).
HEADER = "/* GENERADO por hope/cnc/app/compat_css.py a partir de cnc.css: no editar a mano. */\n"


# ==================================================================================================
# ======================================= READING THE CSS ==========================================
# ==================================================================================================

def _rules(css: str):
    """Walks the simple ``selector { declarations }`` rules of a stylesheet.

    css: full CSS text. Strips comments and skips rules starting with ``@`` and rules without
    declarations. Since the regular expression does not allow nested braces, from an ``@media {…}``
    only the inner rules show up, loose (they still count for computing the gaps).
    Returns (generator) ``(selector, body)`` pairs; the selector may hold several, comma-separated.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    depth_stack = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        sel, body = m.group(1).strip(), m.group(2)
        if sel.startswith("@") or not body.strip():
            continue
        depth_stack.append(sel)
        yield sel, body


def _decls(body: str) -> dict[str, str]:
    """Turns a rule body (``a: 1; b: 2``) into ``{"a": "1", "b": "2"}``.

    Splits at the first ``:`` of each declaration, so values containing ``:`` (``url(data:…)``)
    stay whole.
    """
    out = {}
    for part in body.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


# ==================================================================================================
# ========================================== GENERATION ============================================
# ==================================================================================================

def generate() -> str:
    """Builds the full text of ``compat.css`` (writes nothing; see :func:`main`).

    1. Merges all declarations of each selector from the ``SOURCES`` sheets (as the browser would:
       the last one wins), remembering the order they appear in so the output is stable.
    2. For each selector with ``gap``:
       * if it is a grid → ``grid-gap`` with the same value;
       * if it is flex → under ``.no-flexgap``, a margin on every child but the first (``> * + *``):
         ``margin-top`` for a column, ``margin-left`` for a row; and if the row wraps into several
         (``flex-wrap: wrap``), also ``margin-bottom`` for the gap between rows.
       ``gap: row column`` is honored; a single value applies to both.
    3. Appends the lightweight mode (``.lite``).
    Returns the CSS text; it must match ``static/compat.css`` byte for byte.
    """
    merged: dict[str, dict[str, str]] = {}       # selector → merged declarations
    order: list[str] = []                        # selectors in order of appearance (reproducible output)
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
        # Without gap there is nothing to translate; «selectors» with @ or starting with «:» are leftovers
        # of @media/@keyframes or of :root (variables), not containers.
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
    # Lightweight mode: on old PCs every animation or shadow costs CPU that the status refresh needs.
    # Blinking LEDs (RESET, etc.) stay steady and clearly visible.
    lite = (".lite *, .lite *:before, .lite *:after { animation: none !important; transition: none !important; "
            "box-shadow: none !important; text-shadow: none !important; }\n"
            ".lite .led.blink, .lite .reset-led.blink { opacity: 1; }\n")
    return (HEADER + "\n/* grillas: grid-gap para Firefox 52–60 y Chrome 57–65 */\n" + "\n".join(grid) +
            "\n\n/* flexbox sin gap (Firefox < 63, Chrome < 84) */\n" + "\n".join(flex) +
            "\n\n/* modo liviano para PCs viejas */\n" + lite)


def main() -> int:
    """Writes ``static/compat.css`` with :func:`generate` and reports how many lines it has.

    Returns 0 (process exit code).
    """
    out = STATIC / "compat.css"
    out.write_text(generate(), encoding="utf-8")
    print(f"{out}: {len(out.read_text(encoding='utf-8').splitlines())} líneas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
