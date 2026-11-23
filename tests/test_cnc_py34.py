"""Edición para Windows XP: el traductor a Python 3.4 (empaquetado/tisma_cnc/py34/a_py34.py).

Cada caso corre el código original y el traducido y compara: tienen que dar lo mismo. Si hay un Python
3.4 a mano (variable TISMA_PY34 o «python3.4» en el PATH), además se corre toda la suite traducida en él.
"""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 9):  # noqa: UP036  (la suite también corre en 3.7)
    pytest.skip("el traductor necesita Python 3.9+ (ast.unparse); se corre al armar el paquete", allow_module_level=True)

AQUI = Path(__file__).resolve().parents[1] / "empaquetado" / "tisma_cnc" / "py34"
sys.path.insert(0, str(AQUI))
import _tisma34
from a_py34 import NoTraducible, traducir

CASOS = {
    "fstring": 'x, w, d = 3.14159, 10, 3\nr = [f"{x}", f"{x:.2f}", f"{x!r:>{w}}", f"{{lit}} {x:{w}.{d}f}|", f"{w=}"]',
    "fstring_anidada": 'a = {"k": 1}\nr = f"v={a[\'k\']} {f\'{a["k"]:03d}\'} {{}}"',
    "dict_merge": 'a, b = {"x": 1}, {"y": 2}\nr = {**a, "z": 3, **b, "x": 9}',
    "listas": 'a, b = [1, 2], (3,)\nr = ([*a, 0, *b], (*a, *b), {*a, 5}, [x for x in [*a]])',
    "llamadas": 'def f(*a, **k):\n    return a, sorted(k.items())\nl, d = [1], {"p": 2}\nr = (f(*l, 2, *l), f(1, **d, q=3), f(**d, **{"z": 0}))',
    "anotaciones": 'from __future__ import annotations\nx: int = 4\ny: list[str]\ndef g(a: int, *b: str, c: float = 1.0, **d: object) -> int:\n    z: int = a\n    return z + len(b)\nr = (x, g(1, "a"))',
    "dataclass": ('from dataclasses import dataclass, field, replace, asdict\n'
                  '@dataclass(frozen=True)\nclass P:\n    """doc"""\n    x: float\n    y: float = 2.0\n    tags: list = field(default_factory=list)\n'
                  '    def norma(self) -> float:\n        return (self.x ** 2 + self.y ** 2) ** 0.5\n'
                  'p = P(3.0, 4.0)\nr = (p.norma(), replace(p, y=0.0), asdict(p), p == P(3.0, 4.0), repr(P(1.0)))'),
    "type_checking": 'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from typing import Callable\n    F = Callable[[int], int]\nr = 1',
}


def correr(src: str) -> object:
    ns = {"__name__": "caso"}
    exec(compile(src, "<caso>", "exec"), ns)  # noqa: S102
    return repr(ns["r"])                       # repr: las clases de cada ejecución son distintas


@pytest.mark.parametrize("nombre", sorted(CASOS))
def test_traduccion_da_lo_mismo(nombre):
    src = CASOS[nombre]
    out = traducir(src, nombre)
    assert "f'" not in out and 'f"' not in out
    ast.parse(out)
    t = ast.parse(out)
    assert not any(isinstance(n, (ast.JoinedStr, ast.AnnAssign)) for n in ast.walk(t))
    assert correr(out) == correr(src)


def test_dataclass_deja_la_lista_de_campos():
    out = traducir(CASOS["dataclass"])
    assert "__dc_fields__ = ('x', 'y', 'tags')" in out
    assert "import _tisma34" in out


@pytest.mark.parametrize("src", ["if (n := 3):\n    pass", "def f(a, /, b):\n    pass", "async def f():\n    pass"])
def test_rechaza_lo_que_no_sabe_traducir(src):
    with pytest.raises(NoTraducible):
        traducir(src)


def test_typing_en_tiempo_de_ejecucion_se_rechaza():
    with pytest.raises(NoTraducible, match="typing"):
        traducir("from typing import Callable\nF = Callable[[int], int]")


def test_todo_tisma_cnc_se_traduce():
    raiz = Path(__file__).resolve().parents[1]
    archivos = sorted((raiz / "hope" / "cnc").rglob("*.py")) + [raiz / "guardian" / "core.py", raiz / "guardian" / "audit.py"]
    for f in archivos:
        traducir(f.read_text(encoding="utf-8"), str(f))   # NoTraducible si algo nuevo rompe XP


def test_shim_no_toca_nada_en_python_moderno():
    import dataclasses
    import math
    assert _tisma34._tc_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert _tisma34._tc_list([1], (2,)) == [1, 2]
    assert dataclasses.__file__ and math.inf == float("inf")          # los de verdad, no los del shim


def _py34():
    exe = os.environ.get("TISMA_PY34") or shutil.which("python3.4")
    return exe if exe and Path(exe).exists() else None


@pytest.mark.skipif(_py34() is None, reason="sin Python 3.4 (TISMA_PY34=/ruta/python3.4 para probar la edición XP)")
def test_suite_traducida_en_python_34(tmp_path):
    raiz = Path(__file__).resolve().parents[1]
    destino = tmp_path / "x34"
    subprocess.run([sys.executable, str(AQUI / "preparar_pruebas.py"), str(destino)], check=True, cwd=raiz)
    out = subprocess.run([_py34(), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=destino,
                         capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
