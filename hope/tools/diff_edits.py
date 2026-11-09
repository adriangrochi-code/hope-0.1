"""Ediciones como diff buscar/reemplazar (formato estilo Aider).

Reescribir un archivo entero para cambiar dos líneas es caro (bytes generados) y peligroso (un
modelo chico «olvida» partes del archivo). Con buscar/reemplazar Hope escribe solo lo que cambia:

    ruta/archivo.py
    <<<<<<< BUSCAR
    def doble(x):
        return x + x + 1
    =======
    def doble(x):
        return 2 * x
    >>>>>>> REEMPLAZAR

(también se aceptan ``SEARCH``/``REPLACE`` y la ruta dentro de un bloque ```).

Reglas al aplicar (``apply_blocks``):

* el texto buscado tiene que aparecer **exactamente una vez** en el archivo (si aparece 0 o
  más de una vez la edición se rechaza con el motivo, para que Hope agregue contexto);
* si no aparece tal cual, se intenta ignorando espacios al final de las líneas (otra vez,
  una sola coincidencia);
* ``BUSCAR`` vacío crea el archivo (si no existe) o agrega al final;
* varios bloques sobre el mismo archivo se aplican en orden; si uno falla, no se aplica nada.

El harness (``FolderTools.apply_edits_checked``) escribe las ediciones a través del guardián,
corre las pruebas y **vuelve atrás** si fallan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BLOCK_RE = re.compile(
    r"(?:^|\n)(?:```[\w-]*\n)?(?P<path>[^\n<>=`]+?)[ \t]*\n"
    r"<{5,9} ?(?:BUSCAR|SEARCH)[^\n]*\n(?P<search>.*?)(?:^|\n)={5,9}[ \t]*\n(?P<replace>.*?)(?:^|\n)>{5,9} ?(?:REEMPLAZAR|REPLACE)[^\n]*",
    re.DOTALL | re.M)


class EditError(ValueError):
    pass


@dataclass
class SearchReplace:
    path: str
    search: str
    replace: str


def parse_blocks(text: str) -> list[SearchReplace]:
    out = []
    for m in BLOCK_RE.finditer(text or ""):
        path = m.group("path").strip().strip("`*").strip()
        if path.lower().startswith(("ruta:", "archivo:", "file:")):
            path = path.split(":", 1)[1].strip()
        out.append(SearchReplace(path, m.group("search"), m.group("replace")))
    return out


def _count_lines_match(content: str, search: str) -> list[tuple[int, int]]:
    """Coincidencias ignorando espacios al final de cada línea: (inicio, fin) en ``content``."""
    want = [ln.rstrip() for ln in search.split("\n")]
    while want and not want[-1]:
        want.pop()
    if not want:
        return []
    lines = content.split("\n")
    offs, pos = [], 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    hits = []
    for i in range(len(lines) - len(want) + 1):
        if all(lines[i + j].rstrip() == want[j] for j in range(len(want))):
            end_line = i + len(want) - 1
            hits.append((offs[i], offs[end_line] + len(lines[end_line])))
    return hits


def apply_one(content: str | None, block: SearchReplace) -> str:
    if not block.search.strip():
        if content is None:
            return block.replace if block.replace.endswith("\n") else block.replace + "\n"
        return content + ("" if content.endswith("\n") or not content else "\n") + block.replace
    if content is None:
        raise EditError(f"{block.path}: el archivo no existe y el bloque BUSCAR no está vacío")
    n = content.count(block.search)
    if n == 1:
        return content.replace(block.search, block.replace, 1)
    if n > 1:
        raise EditError(f"{block.path}: el texto buscado aparece {n} veces; agregá líneas de contexto para que sea único")
    hits = _count_lines_match(content, block.search)
    if len(hits) == 1:
        a, b = hits[0]
        rep = block.replace[:-1] if block.replace.endswith("\n") and not block.search.endswith("\n") else block.replace
        return content[:a] + rep.rstrip("\n") + content[b:]
    if len(hits) > 1:
        raise EditError(f"{block.path}: el texto buscado aparece {len(hits)} veces (sin contar espacios finales)")
    first = block.search.strip().split("\n")[0][:80]
    raise EditError(f"{block.path}: el texto buscado no aparece en el archivo (primera línea: «{first}»)")


def apply_blocks(blocks: list[SearchReplace], read) -> dict[str, str]:
    """Aplica los bloques en orden. ``read(ruta) -> contenido | None``. Devuelve ruta → contenido nuevo.
    Lanza ``EditError`` (sin aplicar nada) si algún bloque falla."""
    current: dict[str, str | None] = {}
    for b in blocks:
        if b.path not in current:
            current[b.path] = read(b.path)
        current[b.path] = apply_one(current[b.path], b)
    return {k: v for k, v in current.items() if v is not None}


FORMAT_HELP = (
    "Para cambiar parte de un archivo usá un bloque buscar/reemplazar (el texto buscado tiene que aparecer "
    "una sola vez):\nruta/archivo.py\n<<<<<<< BUSCAR\n<líneas actuales>\n=======\n<líneas nuevas>\n>>>>>>> REEMPLAZAR")
