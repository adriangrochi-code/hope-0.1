"""Genera ``static/compat.css``: lo que los navegadores viejos necesitan para ver la pantalla igual.

Qué es y para qué sirve
-----------------------
TISMA CNC tiene que verse igual en Firefox 52 ESR (el último de Windows XP/Vista) y Chrome 57 que en
uno actual. ``cnc.css`` y ``tema_tisma.css`` se escriben con CSS moderno (``gap``), y este script
lee esas reglas y escribe las equivalentes para navegadores viejos:

* ``grid-gap`` en cada grilla con ``gap`` (Firefox 52–60 y Chrome 57–65 solo conocen ``grid-gap``).
* Márgenes en lugar de ``gap`` en los contenedores flex, bajo ``.no-flexgap`` (lo pone compat.js si el
  navegador no sabe flex-gap: Firefox < 63, Chrome < 84).
* Modo liviano (``.lite``): sin animaciones, transiciones ni sombras.

Cómo encaja
-----------
* Lee: ``static/cnc.css`` y ``static/tema_tisma.css`` (en ese orden; la última regla gana, como en el
  navegador).
* Escribe: ``static/compat.css``, que ``index.html`` carga después de las otras dos hojas.
* Usa las clases que pone ``static/compat.js`` en ``<html>``: ``no-flexgap`` (detectada) y ``lite``
  (modo liviano: automático en PCs de 1–2 núcleos, con ``?liviano=1`` o con la casilla de la pantalla).
* ``tests/test_cnc_compat.py`` compara :func:`generate` con el archivo: si se cambia ``cnc.css`` y no se
  regenera, la prueba falla.

Limitaciones (a propósito, para mantenerlo simple): no entiende reglas anidadas dentro de ``@media``
ni selectores con ``:`` al principio; un contenedor flex cuyo ``display: flex`` está en otra regla que
su ``gap`` se declara a mano en ``FLEX_EXTRA``.

Uso: ``python -m hope.cnc.app.compat_css`` (una prueba verifica que el archivo esté al día).
"""

from __future__ import annotations

import re
from pathlib import Path

# ==================================================================================================
# ========================================= CONFIGURACIÓN ==========================================
# ==================================================================================================

# Carpeta de la página: de acá se leen las hojas de estilo y acá se escribe compat.css.
STATIC = Path(__file__).parent / "static"
# Hojas de estilo a leer, en el orden en que las carga index.html (el tema pisa a cnc.css).
SOURCES = ("cnc.css", "tema_tisma.css")
# contenedores flex cuyo «display: flex» está en otra regla que el «gap»
# (selector → dirección; "column" = los hijos van uno debajo del otro, así el hueco es margin-top).
FLEX_EXTRA = {".screen": "column"}
# Primera línea del archivo generado: avisa que no se edite a mano (se perdería al regenerarlo).
HEADER = "/* GENERADO por hope/cnc/app/compat_css.py a partir de cnc.css: no editar a mano. */\n"


# ==================================================================================================
# ======================================= LECTURA DEL CSS ==========================================
# ==================================================================================================

def _rules(css: str):
    """Recorre las reglas simples ``selector { declaraciones }`` de una hoja de estilo.

    css: texto CSS completo. Saca los comentarios, saltea las reglas que empiezan con ``@`` y las que
    no tienen declaraciones. Como la expresión regular no admite llaves anidadas, de un ``@media {…}``
    solo aparecen las reglas de adentro sueltas (que igual valen para calcular los huecos).
    Devuelve (generador) pares ``(selector, cuerpo)``; el selector puede tener varios separados por coma.
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
    """Convierte el cuerpo de una regla (``a: 1; b: 2``) en ``{"a": "1", "b": "2"}``.

    Parte en el primer ``:`` de cada declaración, así los valores con ``:`` adentro (``url(data:…)``)
    quedan enteros.
    """
    out = {}
    for part in body.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


# ==================================================================================================
# ========================================= GENERACIÓN =============================================
# ==================================================================================================

def generate() -> str:
    """Arma el texto completo de ``compat.css`` (no escribe nada; ver :func:`main`).

    1. Junta todas las declaraciones de cada selector de las hojas de ``SOURCES`` (como haría el
       navegador: la última gana), recordando el orden en que aparecen para que la salida sea estable.
    2. Por cada selector con ``gap``:
       * si es grilla → ``grid-gap`` con el mismo valor;
       * si es flex → bajo ``.no-flexgap``, margen en cada hijo menos el primero (``> * + *``):
         ``margin-top`` en columna, ``margin-left`` en fila; y si la fila se parte en varias
         (``flex-wrap: wrap``), además ``margin-bottom`` para el hueco entre filas.
       ``gap: fila columna`` se respeta; con un solo valor vale para los dos.
    3. Agrega el modo liviano (``.lite``).
    Devuelve el texto CSS; tiene que coincidir byte a byte con ``static/compat.css``.
    """
    merged: dict[str, dict[str, str]] = {}       # selector → declaraciones combinadas
    order: list[str] = []                        # selectores en orden de aparición (salida reproducible)
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
        # Sin gap no hay nada que traducir; los «selectores» con @ o que empiezan con «:» son restos de
        # @media/@keyframes o de :root (variables), no contenedores.
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
    # Modo liviano: en PCs viejas cada animación o sombra cuesta CPU que necesita el refresco del estado.
    # Los LED que parpadean (RESET, etc.) quedan fijos y bien visibles.
    lite = (".lite *, .lite *:before, .lite *:after { animation: none !important; transition: none !important; "
            "box-shadow: none !important; text-shadow: none !important; }\n"
            ".lite .led.blink, .lite .reset-led.blink { opacity: 1; }\n")
    return (HEADER + "\n/* grillas: grid-gap para Firefox 52–60 y Chrome 57–65 */\n" + "\n".join(grid) +
            "\n\n/* flexbox sin gap (Firefox < 63, Chrome < 84) */\n" + "\n".join(flex) +
            "\n\n/* modo liviano para PCs viejas */\n" + lite)


def main() -> int:
    """Escribe ``static/compat.css`` con :func:`generate` e informa cuántas líneas quedaron.

    Devuelve 0 (código de salida del proceso).
    """
    out = STATIC / "compat.css"
    out.write_text(generate(), encoding="utf-8")
    print(f"{out}: {len(out.read_text(encoding='utf-8').splitlines())} líneas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
