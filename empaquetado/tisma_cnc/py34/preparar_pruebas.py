"""Arma una carpeta con TISMA CNC y sus pruebas traducidos a Python 3.4, para correrlas con un Python 3.4
real (el de Windows XP):

    python empaquetado/tisma_cnc/py34/preparar_pruebas.py /tmp/x34
    cd /tmp/x34 && python3.4 -m pytest -q tests          # pytest 4.6 (el último para 3.4) y PyYAML 5.1

Quedan afuera las pruebas que revisan el empaquetado y el estilo del código fuente (no el programa).

Para qué sirve: el traductor (``a_py34.py``) solo garantiza que la SINTAXIS sea de 3.4; que la app
ande de verdad en 3.4 (con la biblioteca estándar de 3.4 y los parches de ``_tisma34.py``) lo
demuestra correr toda la suite traducida en un Python 3.4 real. Lo usa ``tests/test_cnc_py34.py``
cuando se define ``TISMA_PY34=/ruta/python3.4``.

La carpeta DESTINO se borra entera y se vuelve a crear. Necesita Python 3.9+ (``ast.unparse``).
"""
# Note: this module docstring stays in Spanish on purpose: it is printed as usage help
# (``raise SystemExit(__doc__)`` at the end of the file), so the user reads it.

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# ==================================================================================================
# ========================================= CONFIGURATION ==========================================
# ==================================================================================================

# Folder of this file (empaquetado/tisma_cnc/py34) and repository root (three levels up).
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
# To import a_py34 (from this folder) and construir (from the parent folder) without them being packages.
sys.path[:0] = [str(AQUI), str(AQUI.parent)]

from a_py34 import traducir_carpeta
from construir import copiar_app

# Tests that are neither translated nor copied: they check the packaging (they download, call
# construir.py), the style/compatibility of the modern source code or the translator itself; they make
# no sense on 3.4 (and they use things from the build Python, such as ast.unparse).
SIN_TRADUCIR = {"test_cnc_package.py", "test_cnc_compat.py", "test_cnc_py34.py"}


# ==================================================================================================
# ============================================= BUILD ==============================================
# ==================================================================================================

def preparar(destino: Path) -> int:
    """Builds in ``destino`` the app and the ``tests/test_cnc*.py`` tests translated to Python 3.4.

    Copies to a temporary folder the same things that go into the XP package (``copiar_app``), plus
    ``configs/`` (some tests read configurations from the repository) and the tests; translates
    everything into ``destino``, adds ``_tisma34.py`` (untranslated: it is already Python 3.4) and a
    ``tests/conftest.py`` of its own.

    Parameters: ``destino``, output folder (deleted if it exists).
    Returns: how many modules were translated.
    Raises: ``NoTraducible`` if some file uses something 3.4 lacks.
    """
    if destino.exists():
        shutil.rmtree(destino)
    with tempfile.TemporaryDirectory() as tmp:
        fuente = Path(tmp)
        copiar_app(fuente)
        shutil.copytree(RAIZ / "configs", fuente / "configs")
        (fuente / "tests").mkdir()
        for t in sorted((RAIZ / "tests").glob("test_cnc*.py")):
            if t.name not in SIN_TRADUCIR:
                shutil.copy2(t, fuente / "tests" / t.name)
        n = traducir_carpeta(fuente, destino)
    shutil.copy2(AQUI / "_tisma34.py", destino / "_tisma34.py")
    (destino / "tests" / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    return n


# pytest 4.6 (the last one for 3.4) gives a tmp_path from pathlib2, which 3.4's pathlib does not accept
# This conftest.py replaces the tmp_path fixture with one that returns a plain pathlib.Path built from
# tmpdir (py.path), so the tests can mix it with pathlib paths without errors.
CONFTEST = """import pathlib

import pytest


@pytest.fixture
def tmp_path(tmpdir):
    return pathlib.Path(str(tmpdir))
"""


# ==================================================================================================
# ========================================== COMMAND LINE ==========================================
# ==================================================================================================
# Usage: python preparar_pruebas.py DESTINO. With any other number of arguments it prints the module
# docstring as help and exits with an error.

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(f"{preparar(Path(sys.argv[1]))} módulos traducidos a Python 3.4 en {sys.argv[1]}")
