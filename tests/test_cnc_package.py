"""TISMA CNC packages for old PCs (empaquetado/tisma_cnc/construir.py), without downloading anything:
the file subset is enough to run the app outside the repository, the launchers are safe for the
Windows console and the installer script is complete."""

import http.client
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("construir", "empaquetado/tisma_cnc/construir.py")
construir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(construir)


def test_app_subset_runs_alone(tmp_path):
    app = tmp_path / "app"
    files = construir.copiar_app(app)
    assert "hope/cnc/app/static/cnc.js" in files and "hope/cnc/perfiles/material/mdf.json" in files
    assert not any("__pycache__" in f or f.endswith(".pyc") for f in files)
    assert not (app / "hope" / "train").exists() and not (app / "hope" / "core").exists()   # no PyTorch at all
    # outside the repository, with data in another folder, without the user's variables
    work = tmp_path / "cwd"
    work.mkdir()
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(app), "PYTHONNOUSERSITE": "1",
           "TISMA_CNC_DATOS": str(tmp_path / "datos"), "HOME": str(tmp_path)}
    proc = subprocess.Popen([sys.executable, "-m", "hope.cnc", "app", "--sin-ventana", "--puerto", "8815"],
                            cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        cfg = None
        for _ in range(60):
            time.sleep(0.2)
            try:
                c = http.client.HTTPConnection("127.0.0.1", 8815, timeout=2)
                c.request("GET", "/api/config", headers={"Host": "127.0.0.1:8815"})
                cfg = json.loads(c.getresponse().read())
                break
            except OSError:
                continue
        assert cfg and cfg["marca"] == "TISMA CNC" and cfg["nombre"] == "Router genérico"
        assert (tmp_path / "datos" / "guardian" / "guardian.json").exists()
        assert not (work / "runs").exists()                                   # nothing written in the current folder
    finally:
        proc.kill()
        proc.wait(10)


def test_windows_launchers_and_paths():
    bat = construir.BAT_PORTABLE.replace("{datos}", r"%APPDATA%\TISMA CNC")
    assert bat.isascii(), "la consola de Windows usa otra página de códigos: el .bat tiene que ser ASCII"
    assert "^(" in bat and "python\\python.exe" in bat and "TISMA_CNC_DATOS" in bat
    pth = construir.pth_embebido()
    assert pth.splitlines() == ["python37.zip", ".", "..\\app", "Lib\\site-packages", "import site"]


def test_installer_script_is_complete():
    script = construir.NSI
    for k in ("salida", "icono", "origen", "ver", "vnum"):
        assert f"@{k}@" in script
    assert "{" not in script.replace("${LANG_SPANISH}", "")            # no template leftovers
    assert "RequestExecutionLevel user" in script and "WriteUninstaller" in script
    assert "@oninit@" in script and "ucrtbase.dll" in construir.NSI_UCRT    # XP neither has nor needs UCRT


def test_xp_launcher_and_path_file():
    bat = construir.BAT_XP.replace("{datos}", r"%APPDATA%\TISMA CNC")
    assert bat.isascii() and "^(" in bat
    assert '"%~dp0python\\python.exe" -E -s -m hope.cnc app' in bat      # no variables nor another Python
    assert "ucrtbase" not in bat
    assert "Python 3.4.4" in construir.LEEME_XP and "Vista SP2, 7, 8" not in construir.LEEME_XP
    url, sha = construir.PY_XP
    assert url.endswith("/3.4.4/python-3.4.4.msi") and len(sha) == 64
    assert "5.1.2" in construir.PYYAML_XP[0] and len(construir.PYYAML_XP[1]) == 64


@pytest.mark.skipif(not Path("runs/empaquetado/python-3.4.4.msi").exists() or not shutil.which("msiextract")
                    or sys.version_info < (3, 9),  # the translation uses ast.unparse
                    reason="sin python-3.4.4.msi en caché, sin msiextract (apt install msitools) o Python < 3.9")
def test_xp_package_layout(tmp_path):
    construir.armar_xp(tmp_path, "0.0.0-prueba", tmp_path / "trabajo")
    import zipfile
    z = zipfile.ZipFile(next(tmp_path.glob("*windows-xp.zip")))
    names = z.namelist()
    for n in ("python/python.exe", "python/python34.dll", "python/msvcr100.dll", "python/Lib/os.py",
              "python/Lib/site-packages/tisma_cnc.pth", "python/Lib/site-packages/yaml/__init__.py",
              "python/Lib/site-packages/serial/__init__.py", "app/_tisma34.py", "app/hope/cnc/session.py",
              "TISMA CNC.bat", "LEEME.txt"):
        assert "TISMA CNC/" + n in names, n
    assert not any("/tkinter/" in n or "/test/" in n or n.endswith("_tkinter.pyd") for n in names)
    session = z.read("TISMA CNC/app/hope/cnc/session.py").decode("utf-8")
    import ast
    assert "import _tisma34" in session
    assert not any(isinstance(n, (ast.JoinedStr, ast.AnnAssign)) for n in ast.walk(ast.parse(session)))
    assert construir.funciones_post_xp(tmp_path / "trabajo" / "windows-xp" / "TISMA CNC" / "python") == []


def test_windows_python_runs_on_vista():
    """The Windows Python must not import functions that do not exist on Vista (3.8 imports
    GetActiveProcessorCount, from Windows 7: that is why the packages use 3.7.9)."""
    assert "3.7.9" in construir.DESCARGAS["python-x86"][0] and construir.PY_WIN == "37"


@pytest.mark.skipif(importlib.util.find_spec("pefile") is None or
                    not Path("runs/empaquetado/python-3.7.9-embed-win32.zip").exists(),
                    reason="necesita pefile y la descarga en caché")
def test_embedded_python_has_no_windows7_functions():
    for z in ("python-3.7.9-embed-win32.zip", "python-3.7.9-embed-amd64.zip"):
        p = Path("runs/empaquetado") / z
        if p.exists():
            assert construir.funciones_post_vista(p) == []
    vieja = Path("runs/empaquetado/python-3.8.10-embed-win32.zip")
    if vieja.exists():                          # the reason not to use 3.8: it does not start on Vista
        assert any("GetActiveProcessorCount" in f for f in construir.funciones_post_vista(vieja))


def test_downloads_are_pinned():
    for url, sha in construir.DESCARGAS.values():
        assert url.startswith("https://www.python.org/") and len(sha) == 64
    for ruedas in construir.RUEDAS.values():
        assert all(len(sha) == 64 for _, _, sha in ruedas)


@pytest.mark.skipif(not Path("runs/empaquetado/python-3.7.9-embed-win32.zip").exists(),
                    reason="sin la descarga en caché (python empaquetado/tisma_cnc/construir.py)")
def test_windows_zip_layout(tmp_path):
    construir.armar_windows("x86", tmp_path, "0.0.0-prueba", tmp_path / "trabajo")
    import zipfile
    names = zipfile.ZipFile(next(tmp_path.glob("*windows-x86.zip"))).namelist()
    assert "TISMA CNC/python/python.exe" in names and "TISMA CNC/python/python37._pth" in names
    assert "TISMA CNC/app/hope/cnc/session.py" in names and "TISMA CNC/TISMA CNC.bat" in names
    assert any(n.endswith("yaml/_yaml.cp37-win32.pyd") for n in names)
