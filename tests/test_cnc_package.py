"""Paquetes de TISMA CNC para PCs viejas (empaquetado/tisma_cnc/construir.py), sin descargar nada:
el subconjunto de archivos alcanza para correr la app fuera del repositorio, los lanzadores son
seguros para la consola de Windows y el script del instalador queda completo."""

import http.client
import importlib.util
import json
import os
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
    assert not (app / "hope" / "train").exists() and not (app / "hope" / "core").exists()   # nada de PyTorch
    # fuera del repositorio, con datos en otra carpeta, sin variables del usuario
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
        assert not (work / "runs").exists()                                   # nada escrito en la carpeta actual
    finally:
        proc.kill()
        proc.wait(10)


def test_windows_launchers_and_paths():
    bat = construir.BAT_PORTABLE.replace("{datos}", r"%APPDATA%\TISMA CNC")
    assert bat.isascii(), "la consola de Windows usa otra página de códigos: el .bat tiene que ser ASCII"
    assert "^(" in bat and "python\\python.exe" in bat and "TISMA_CNC_DATOS" in bat
    pth = construir.pth_embebido()
    assert pth.splitlines() == ["python38.zip", ".", "..\\app", "Lib\\site-packages", "import site"]


def test_installer_script_is_complete():
    script = construir.NSI
    for k in ("salida", "icono", "origen", "ver", "vnum"):
        assert f"@{k}@" in script
    assert "{" not in script.replace("${LANG_SPANISH}", "")            # sin restos de plantillas
    assert "RequestExecutionLevel user" in script and "WriteUninstaller" in script
    assert "ucrtbase.dll" in script


def test_downloads_are_pinned():
    for url, sha in construir.DESCARGAS.values():
        assert url.startswith("https://www.python.org/") and len(sha) == 64
    for ruedas in construir.RUEDAS.values():
        assert all(len(sha) == 64 for _, _, sha in ruedas)


@pytest.mark.skipif(not Path("runs/empaquetado/python-3.8.10-embed-win32.zip").exists(),
                    reason="sin la descarga en caché (python empaquetado/tisma_cnc/construir.py)")
def test_windows_zip_layout(tmp_path):
    construir.armar_windows("x86", tmp_path, "0.0.0-prueba", tmp_path / "trabajo")
    import zipfile
    names = zipfile.ZipFile(next(tmp_path.glob("*windows-x86.zip"))).namelist()
    assert "TISMA CNC/python/python.exe" in names and "TISMA CNC/python/python38._pth" in names
    assert "TISMA CNC/app/hope/cnc/session.py" in names and "TISMA CNC/TISMA CNC.bat" in names
    assert any(n.endswith("yaml/_yaml.cp38-win32.pyd") for n in names)
