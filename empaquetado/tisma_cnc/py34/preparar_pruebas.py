"""Arma una carpeta con TISMA CNC y sus pruebas traducidos a Python 3.4, para correrlas con un Python 3.4
real (el de Windows XP):

    python empaquetado/tisma_cnc/py34/preparar_pruebas.py /tmp/x34
    cd /tmp/x34 && python3.4 -m pytest -q tests          # pytest 4.6 (el último para 3.4) y PyYAML 5.1

Quedan afuera las pruebas que revisan el empaquetado y el estilo del código fuente (no el programa).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
sys.path[:0] = [str(AQUI), str(AQUI.parent)]

from a_py34 import traducir_carpeta
from construir import copiar_app

SIN_TRADUCIR = {"test_cnc_package.py", "test_cnc_compat.py", "test_cnc_py34.py"}


def preparar(destino: Path) -> int:
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


# pytest 4.6 (el último para 3.4) da tmp_path de pathlib2, que el pathlib de 3.4 no acepta
CONFTEST = """import pathlib

import pytest


@pytest.fixture
def tmp_path(tmpdir):
    return pathlib.Path(str(tmpdir))
"""


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(f"{preparar(Path(sys.argv[1]))} módulos traducidos a Python 3.4 en {sys.argv[1]}")
