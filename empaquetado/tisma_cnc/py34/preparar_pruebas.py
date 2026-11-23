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

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# ==================================================================================================
# ========================================= CONFIGURACIÓN ==========================================
# ==================================================================================================

# Carpeta de este archivo (empaquetado/tisma_cnc/py34) y raíz del repositorio (tres niveles arriba).
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
# Para importar a_py34 (de esta carpeta) y construir (de la carpeta de arriba) sin que sean paquetes.
sys.path[:0] = [str(AQUI), str(AQUI.parent)]

from a_py34 import traducir_carpeta
from construir import copiar_app

# Pruebas que no se traducen ni se copian: revisan el empaquetado (descargan, llaman a construir.py),
# el estilo/compatibilidad del código fuente moderno o el propio traductor; no tienen sentido en 3.4
# (y usan cosas del Python que arma, como ast.unparse).
SIN_TRADUCIR = {"test_cnc_package.py", "test_cnc_compat.py", "test_cnc_py34.py"}


# ==================================================================================================
# ============================================ ARMADO ==============================================
# ==================================================================================================

def preparar(destino: Path) -> int:
    """Arma en ``destino`` la app y las pruebas ``tests/test_cnc*.py`` traducidas a Python 3.4.

    Copia a una carpeta temporal lo mismo que va en el paquete XP (``copiar_app``), más ``configs/``
    (algunas pruebas leen configuraciones del repositorio) y las pruebas; traduce todo a ``destino``,
    agrega ``_tisma34.py`` (sin traducir: ya es Python 3.4) y un ``tests/conftest.py`` propio.

    Parámetros: ``destino``, carpeta de salida (se borra si existe).
    Devuelve: cuántos módulos se tradujeron.
    Errores: ``NoTraducible`` si algún archivo usa algo que 3.4 no tiene.
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


# pytest 4.6 (el último para 3.4) da tmp_path de pathlib2, que el pathlib de 3.4 no acepta
# Este conftest.py reemplaza el «fixture» tmp_path por uno que devuelve un pathlib.Path común armado a
# partir de tmpdir (py.path), así las pruebas pueden mezclarlo con rutas de pathlib sin errores.
CONFTEST = """import pathlib

import pytest


@pytest.fixture
def tmp_path(tmpdir):
    return pathlib.Path(str(tmpdir))
"""


# ==================================================================================================
# ======================================= LÍNEA DE COMANDOS ========================================
# ==================================================================================================
# Uso: python preparar_pruebas.py DESTINO. Con otra cantidad de argumentos muestra el docstring del
# módulo como ayuda y sale con error.

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(f"{preparar(Path(sys.argv[1]))} módulos traducidos a Python 3.4 en {sys.argv[1]}")
