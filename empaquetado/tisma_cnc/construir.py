"""Builds TISMA CNC ready for old PCs, with nothing to install by hand.

    python empaquetado/tisma_cnc/construir.py                 # everything that can be built here
    python empaquetado/tisma_cnc/construir.py --solo windows-x86 linux
    python empaquetado/tisma_cnc/construir.py --solo instalador   # needs makensis (NSIS)

Output in ``dist/``:

* ``TISMA-CNC-<version>-windows-x86.zip`` and ``…-windows-x64.zip``: portable editions. Unzip them into
  any folder (or USB stick) and open ``TISMA CNC.bat``. They ship **embedded Python 3.7.9**
  (the last official one that starts on **Windows Vista**: 3.8 already imports GetActiveProcessorCount,
  which is a Windows 7 function), pyserial and PyYAML. The data lives in ``datos/`` next to it.
* ``TISMA-CNC-<version>-instalador.exe``: installer (NSIS, 32-bit: works on 32- or 64-bit Windows Vista,
  7, 8, 10 and 11) that needs no administrator rights; shortcuts on the desktop and in the Start menu;
  data in ``%APPDATA%\\TISMA CNC`` (not deleted on uninstall).
* ``TISMA-CNC-<version>-windows-xp.zip`` and ``…-instalador-xp.exe``: edition for **Windows XP SP3** and
  later with **Python 3.4.4** (the last official one for XP, from the MSI installer, which is opened with
  ``msiextract`` from msitools). The code is translated automatically to Python 3.4 (``py34/a_py34.py``)
  and ``py34/_tisma34.py`` adds what that standard library is missing.
* ``TISMA-CNC-<version>-linux-x86_64.tar.gz``: portable for Linux from 2014 on (glibc 2.17), with its
  own Python 3.8 (python-build-standalone).
* ``TISMA-CNC-<version>-universal.tar.gz``: code only (≈ 1 MB) for any system with
  **Python 3.7 or newer**: 32-bit Linux (XP-era PCs), Raspberry Pi and other ARM boards,
  BSD, macOS. TISMA CNC uses only the standard library; it ships pyserial and PyYAML in pure Python.

Everything downloaded is verified against a fixed SHA-256 (``DESCARGAS``); downloads are kept in
``runs/empaquetado/`` so they are not repeated. Standard library only (plus Pillow, optional, for the icon).

How it fits with the rest
-------------------------

* **Uses**: the code of ``hope/cnc`` (and the minimum of ``hope`` and ``guardian``) copied by
  ``copiar_app``; ``py34/a_py34.py`` (translator to Python 3.4) and ``py34/_tisma34.py`` (compatibility
  layer) for the XP edition; optional external tools: ``pip`` (download wheels), ``msiextract`` (open the
  Python 3.4 MSI), ``makensis`` (installers), ``uv`` (portable Python for Linux), ``pefile`` (check that
  the .dll/.exe files do not require a newer Windows) and Pillow (icon).
* **Used by**: people building a release (by hand) and the tests ``tests/test_cnc_package.py`` (they load
  this file as a module, downloading nothing except what is already cached) and
  ``py34/preparar_pruebas.py`` (reuses ``copiar_app``).

Conventions
-----------

* Paths with ``pathlib``; ``RAIZ`` is the repository root.
* Each ``armar_*`` builds in a work folder (``--trabajo``, default ``runs/empaquetado/armado``), which it
  deletes and recreates, and leaves the compressed result in ``--destino`` (default ``dist/``).
* Texts that go to Windows (``.bat``, ``LEEME.txt``) are written with CRLF line endings: Notepad on
  XP/Vista does not understand bare LF (it shows everything on one line).
* Any problem (SHA-256 mismatch, a missing tool, a .dll that does not start on the target Windows)
  stops with ``SystemExit`` and a message in Spanish: better not to build than to build something broken.
* This script runs on the build PC (Python 3.9+ for the XP edition because of ``ast.unparse``), not on
  the old PC: it may use modern Python.

Example
-------

    $ python empaquetado/tisma_cnc/construir.py --solo windows-x86
    descargando https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-win32.zip
    dist/TISMA-CNC-0.1.0-20261123-abc1234-windows-x86.zip  (12.3 MB)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ==================================================================================================
# ========================================= CONFIGURATION ==========================================
# ==================================================================================================

# --- paths and names ------------------------------------------------------------------------------

# Repository root (this file is in empaquetado/tisma_cnc/, two levels down).
RAIZ = Path(__file__).resolve().parents[2]
# Visible program name: folder inside the ZIP, shortcuts, window title.
NOMBRE = "TISMA CNC"
# Prefix of the files generated in dist/ (no spaces, so no quoting is needed).
ARCHIVO = "TISMA-CNC"
# Download cache and default work folder. It lives in runs/ (ignored by git). Deleting it only forces
# everything to be downloaded again.
CACHE = RAIZ / "runs" / "empaquetado"

# --- embedded Python for Windows Vista and later --------------------------------------------------

# Verified downloads. The Python ZIPs match the MD5 published at
# https://www.python.org/downloads/release/python-379/ (60f77740…, 97c6558d…).
#
# Why Python 3.7.9: it is the last one with an official "embeddable" package that starts on Windows
# Vista. Python 3.8 already imports Windows 7 functions (GetActiveProcessorCount) and Vista shows "entry
# point not found" when opening it (see FUNCIONES_WIN7 and funciones_post_vista, which check it
# automatically).
# Why a fixed SHA-256: python.org publishes MD5, which no longer protects against deliberate tampering;
# the official MD5 was checked once and the SHA-256 of the same file was recorded here. If someone
# changes the file on the way (proxy, mirror, truncated download), the build stops. Changing versions
# requires updating URL and SHA-256 together (and re-running the pefile check).
#
# "Short" version of the embedded Python: gives the name of python37.zip (standard library) and
# python37._pth (search paths). It must match the version in DESCARGAS.
PY_WIN = "37"                      # python37.zip, python37._pth
DESCARGAS = {
    "python-x64": ("https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-amd64.zip",
                   "18627a097adf47829a847053febac5532376075243e233bd9ec61d6ea09dee1f"),
    "python-x86": ("https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-win32.zip",
                   "010cef379e1096cebf667e8f2c92a72dc455c927c7f7b8ca8e029975d81946bf"),
}

# --- third-party packages (wheels) ----------------------------------------------------------------

# Compiled PyYAML wheels per system: (pip platform, file, SHA-256). PyYAML 6.0.1 ships wheels for
# CPython 3.7 (cp37), the embedded Python for Windows. The Linux one is for the portable Python 3.8 (cp38)
# and "manylinux2014" = glibc 2.17, the same baseline as the Python from python_linux(). The Python
# version requested from pip is taken from the file name (see rueda()).
RUEDAS = {   # (pip platform, file, sha256)
    "x64": [("win_amd64", "PyYAML-6.0.1-cp37-cp37m-win_amd64.whl",
             "4fb147e7a67ef577a588a0e2c17b6db51dda102c71de36f8549b6816a96e1867")],
    "x86": [("win32", "PyYAML-6.0.1-cp37-cp37m-win32.whl",
             "9046c58c4395dff28dd494285c82ba00b546adfc7ef001486fbf0324bc174fba")],
    "linux": [("manylinux2014_x86_64", "PyYAML-6.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
               "7e07cbde391ba96ab58e532ff4803f79c4129397514e1413a7dc761ccd755735")],
}
# PyYAML 6.0.1 source (URL, SHA-256) for the universal package: only the pure-Python part is copied
# from it (pyyaml_puro), which runs on any processor (ARM, 32-bit…) without compiling anything.
PYYAML_FUENTE = (("https://files.pythonhosted.org/packages/cd/e5/af35f7ea75cf72f2cd079c95ee16797de7cd71f29ea7c68ae5ce7be1eda0/"
                  "PyYAML-6.0.1.tar.gz"), "bfdf460b1736c775f2ba9f6a92bca30bc2095067b8a9d77876d1fad6cc3b4a43")

# --- Windows XP edition (Python 3.4) --------------------------------------------------------------

# Windows XP: Python 3.4.4 (the last official one for XP SP3), 32-bit MSI installer extracted with
# msiextract (msitools). MD5 published at https://www.python.org/downloads/release/python-344/: e96268f7….
# Why the MSI and not a ZIP: Python 3.4 had no "embeddable" edition (it appeared in 3.5, which no longer
# starts on XP). The 32-bit one is used because it runs on both 32- and 64-bit XP.
PY_XP = ("https://www.python.org/ftp/python/3.4.4/python-3.4.4.msi",
         "46c8f9f63cf02987e8bf23934b2f471e1868b24748c5bb551efcf4863b43ca6c")
# PyYAML 5.1.2: the last one that accepts Python 3.4 (in pure Python)
# There are no compiled wheels for 3.4 on Windows: the source is used, in pure Python.
PYYAML_XP = (("https://files.pythonhosted.org/packages/e3/e8/b3212641ee2718d556df0f23f78de8303f068fe29cdaa7a91018849582fe/"
              "PyYAML-5.1.2.tar.gz"), "01adf0b6c6f61bd11af6e10ca52b7d4057dd0be0343eb9283c878cf3af56aee4")
# pyserial 3.5 (serial port to the controller): "py2.py3-none-any" wheel, i.e. pure Python that works
# in every edition (Vista, XP with 3.4, Linux, universal). Same format as the RUEDAS entries.
PYSERIAL = ("any", "pyserial-3.5-py2.py3-none-any.whl", "c4451db6ba391ca6ca299fb3ec7bae67a5c55dde170964c7a14ceefec02f2cf0")

# --- what is copied from the application ----------------------------------------------------------

# The minimum of the repository that TISMA CNC uses (the rest of Hope —PyTorch, training— is left out)
# Individual files (paths relative to RAIZ). If hope/cnc starts importing another module from hope or
# from the guardian, it must be added here: the test test_cnc_package starts the copied app and detects it.
ARCHIVOS_APP = [
    "hope/__init__.py", "hope/epistemic.py", "hope/growth/__init__.py", "hope/growth/budget.py",
    "hope/tools/__init__.py", "hope/tools/catalog.py",
    "guardian/__init__.py", "guardian/audit.py", "guardian/core.py",
]
# Folders copied whole (with the web interface: HTML, JS, CSS, profiles).
CARPETAS_APP = ["hope/cnc"]
# Folder or file names that are never copied (Python cache and linter configuration).
EXCLUIR = {"__pycache__", "ruff.toml"}

# --- standard library pruning ---------------------------------------------------------------------

# From XP's Python 3.4: what the app does not use (Tk, tests, documentation, sqlite…)
# Paths relative to python/Lib. Shrinks the ZIP by several tens of MB. If the app started using one of
# them (for example sqlite3 or unittest), it would fail only on XP with ImportError: remove it from this
# list. "site-packages" is pruned because it comes with pip/setuptools from the MSI; it is then created
# empty and filled only with our packages.
PODAR_XP_LIB = ["test", "idlelib", "tkinter", "turtledemo", "lib2to3", "ensurepip", "unittest/test", "distutils",
                "ctypes/test", "sqlite3", "msilib", "venv", "pydoc_data", "turtle.py", "site-packages"]
# Files in python/DLLs that are not copied: Tk/Tcl, sqlite, MSI, CPython test modules and icons.
PODAR_XP_DLL = ["_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "_sqlite3.pyd", "sqlite3.dll", "_msi.pyd", "_testcapi.pyd",
                "_testbuffer.pyd", "_testimportmultiple.pyd", "_ctypes_test.pyd", "py.ico", "pyc.ico"]
# Parts of the standard library the app does not use (they shrink the Linux package)
# Paths relative to python/lib/python3.8 of the portable Python. "config-3.8-…" holds what is needed to
# compile C extensions, which is not done here.
PODAR_LINUX = ["test", "idlelib", "tkinter", "turtledemo", "lib2to3/tests", "ensurepip", "unittest/test",
               "distutils/tests", "ctypes/test", "sqlite3/test", "config-3.8-x86_64-linux-gnu"]


# ==================================================================================================
# ===================================== VERSION AND DOWNLOADS ======================================
# ==================================================================================================

def version() -> str:
    """Version carried by the package names: ``<hope version>-<YYYYMMDD>-<short commit>``.

    The date is UTC (the same in any time zone). Without git or outside a repository, the commit
    becomes ``local``. Adds ``RAIZ`` to ``sys.path`` so ``hope`` can be imported.

    Returns: for example ``"0.1.0-20261123-abc1234"``.
    """
    sys.path.insert(0, str(RAIZ))
    from hope import __version__
    try:
        sha = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "local"
    return f"{__version__}-{datetime.now(timezone.utc):%Y%m%d}-{sha}"


# --- verification and cache -----------------------------------------------------------------------

def sha256(path: Path) -> str:
    """SHA-256 (lowercase hex) of the file ``path``, read 1 MB at a time so it is never fully loaded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def descargar(url: str, esperado: str, destino: Path) -> Path:
    """Downloads ``url`` to ``destino`` unless it is already cached with the ``esperado`` SHA-256.

    It is first downloaded to ``<destino>.tmp`` and renamed only once the SHA-256 matches: that way a
    truncated or tampered download never stays in the cache under the good name.

    Returns: ``destino``.
    Raises: ``SystemExit`` if the SHA-256 does not match; ``urllib`` network errors (120 s timeout).
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists() or sha256(destino) != esperado:
        print(f"descargando {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        tmp = destino.with_suffix(destino.suffix + ".tmp")
        tmp.write_bytes(data)
        if sha256(tmp) != esperado:
            tmp.unlink()
            raise SystemExit(f"SHA-256 distinto en {url}: descarga alterada o incompleta")
        tmp.replace(destino)
    return destino


def rueda(plataforma: str, nombre: str, esperado: str) -> Path:
    """Fetches the wheel ``nombre`` with ``pip download`` (cached in ``runs/empaquetado/ruedas``).

    Parameters: pip ``plataforma`` (``win32``, ``win_amd64``, ``manylinux2014_x86_64`` or ``any`` for
    pure Python), exact ``nombre`` of the ``.whl`` file and its ``esperado`` SHA-256.

    Why pip and not a fixed URL: pip knows how to find the right wheel for a platform and Python
    version other than this PC's (``--platform``/``--python-version``); the SHA-256 still guarantees the
    file is the one that was tested.

    Returns: the wheel path. Raises: ``SystemExit`` if the SHA-256 does not match;
    ``CalledProcessError`` if pip fails (no network, misspelled name).
    """
    destino = CACHE / "ruedas" / nombre
    if destino.exists() and sha256(destino) == esperado:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    # Wheel name: <package>-<version>-<Python tag>-… (PEP 427), e.g. PyYAML-6.0.1-cp37-…
    paquete, ver, tag = nombre.split("-")[:3]
    # "cp37" → "3.7"; "py2.py3" wheels (pure Python) state no version: any will do, 3.8 is requested.
    pyver = f"{tag[2]}.{tag[3:]}" if tag.startswith("cp") else "3.8"   # the version comes from the wheel name
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--python-version", pyver,
           "-d", str(destino.parent), f"{paquete}=={ver}"]
    if plataforma != "any":
        # pip only accepts --platform together with --only-binary (it cannot compile for another system).
        cmd[5:5] = ["--platform", plataforma]
    subprocess.run(cmd, check=True)
    if sha256(destino) != esperado:
        raise SystemExit(f"SHA-256 distinto en {nombre}")
    return destino


def extraer_rueda(whl: Path, site: Path) -> None:
    """Installs a wheel by unzipping it into ``site`` (a wheel is a ZIP with the package ready to use).

    Enough for wheels without scripts or data files, such as pyserial and PyYAML: pip is not needed on
    the old PC (the embedded Python for Windows does not even include it).
    """
    with zipfile.ZipFile(whl) as z:
        z.extractall(site)


# ==================================================================================================
# ====================================== APPLICATION AND ICON ======================================
# ==================================================================================================

def copiar_app(destino: Path) -> list[str]:
    """Copies the TISMA CNC code (and the minimum of Hope and the guardian) to ``destino``.

    Copies ``ARCHIVOS_APP`` and the ``CARPETAS_APP`` folders without ``EXCLUIR`` or ``.pyc``, keeping the
    repository folder structure (``destino/hope/cnc/...``).

    Returns: the copied paths, relative and with "/", in a fixed order (``sorted``) so that two builds
    of the same commit give the same package. Also used by the tests and ``py34/preparar_pruebas.py``.
    """
    copiados = []
    for rel in ARCHIVOS_APP:
        (destino / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAIZ / rel, destino / rel)
        copiados.append(rel)
    for carpeta in CARPETAS_APP:
        for f in sorted((RAIZ / carpeta).rglob("*")):
            rel = f.relative_to(RAIZ)
            if f.is_dir() or EXCLUIR & set(rel.parts) or f.suffix == ".pyc":
                continue
            (destino / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, destino / rel)
            copiados.append(rel.as_posix())
    return copiados


def icono(destino: Path) -> Path | None:
    """logo.ico (TISMA's hexagon and violet pyramid) for the shortcuts.

    Draws the logo at 256×256 and saves it as ``.ico`` with every size Windows uses (16 to 256):
    that way Windows does not have to scale it down and it looks sharp on the desktop and Start menu.

    Returns: ``destino``, or ``None`` if Pillow is missing (the package is built anyway, without an
    icon; the NSIS installer does require it).
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    s = 256
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # The drawing coordinates are in a 110-unit box (those of the original logo.svg); k converts
    # units to pixels.
    k = s / 110.0

    def pts(*xy):
        """Flat list x1, y1, x2, y2… (in logo units) → points in pixels, shifted 3 px to the right."""
        return [(x * k + 3, y * k) for x, y in zip(xy[::2], xy[1::2])]
    # Light-grey outlined hexagon and the pyramid: four faces (grey, violet, dark violet, grey) and
    # the base. The colors are those of the TISMA logo.
    d.polygon(pts(50, 4, 95, 29, 95, 81, 50, 106, 5, 81, 5, 29), outline=(206, 206, 206, 255), width=int(5 * k))
    d.polygon(pts(50, 15, 16, 78, 31, 74), fill=(206, 206, 206, 255))
    d.polygon(pts(50, 15, 31, 74, 50, 84), fill=(148, 55, 193, 255))
    d.polygon(pts(50, 15, 50, 84, 69, 74), fill=(82, 19, 134, 255))
    d.polygon(pts(50, 15, 69, 74, 84, 78), fill=(169, 169, 171, 255))
    d.polygon(pts(16, 78, 31, 74, 50, 84, 69, 74, 84, 78, 50, 97), fill=(189, 189, 191, 255))
    destino.parent.mkdir(parents=True, exist_ok=True)
    im.save(destino, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return destino


# ==================================================================================================
# ================================= TEXTS SHIPPED IN THE PACKAGES ==================================
# ==================================================================================================

# LEEME.txt (README) of every edition. Filled in with str.format: {version}, {como} (how it is opened in
# that edition) and {datos} (where the data is kept). On Windows it is converted to CRLF when written.
# LEEME_XP is derived from this text by replacing whole paragraphs: if the wording of the requirements
# changes, check LEEME_XP (test_cnc_package checks that the XP edition does not say "Vista SP2, 7, 8").
LEEME = """TISMA CNC — control CNC con la distribución de Mach3 (fresadora, router y plasma)
Versión {version}

CÓMO SE USA
  {como}

  La primera vez se abre con el simulador (sin máquina) para probar. La máquina, el material, la
  herramienta y el proceso se eligen arriba, como en un slicer de impresión 3D. Para conectar la
  máquina real: pestaña Perfiles (Alt-3) → Máquina → Controlador (grbl, ok = Marlin, linuxcnc) y Puerto.

REQUISITOS
  * Windows Vista SP2, 7, 8, 10 u 11 (32 o 64 bits), o Linux de 2014 en adelante.
  * Un navegador de 2017 en adelante: Firefox 52 ESR o más nuevo, Chrome o Edge; en Windows Vista,
    Firefox 52 ESR o Supermium (Chromium actual para Vista y XP). Internet Explorer NO sirve (la
    pantalla avisa qué instalar).
  * Windows Vista SP2, 7 SP1 y 8: hace falta la actualización KB2999226 de Microsoft (Universal C
    Runtime), que casi todas las PCs ya tienen si pasaron por Windows Update. Si al abrir aparece
    «falta api-ms-win-crt-runtime-l1-1-0.dll», instalarla (hay versión para Vista).

DATOS
  Los perfiles propios, los archivos recientes y el registro del guardián quedan en:
  {datos}

SEGURIDAD
  Mover una máquina real pide aprobación en pantalla (RESET y cada Cycle Start) y queda registrado.
  La parada de emergencia es la seta física: ningún programa la reemplaza.

Más información: docs/CNC_HOPE.md en el repositorio.
"""

# Launcher of the Windows editions with Python 3.7 (portable and installer). {datos} is replaced with
# str.replace (not format: the BAT has its own braces and %) by each edition's data folder.
# What it does, in order: goes to its own folder (%~dp0, works from a USB stick with any drive letter);
# checks that ucrtbase.dll exists (Universal C Runtime, KB2999226), without which Python 3.7 does not
# start, and if missing explains it instead of showing a cryptic DLL error; forces UTF-8 in Python
# (PYTHONUTF8) because the Windows console uses another code page; and if the app exits with an error,
# keeps the window open (pause) so it can be read. The messages have no accents: cmd.exe would garble them.
BAT_PORTABLE = r"""@echo off
rem TISMA CNC portable: doble clic. Cerrar esta ventana cierra TISMA CNC.
setlocal
cd /d "%~dp0"
title TISMA CNC
if not exist "%SystemRoot%\System32\ucrtbase.dll" if not exist "%SystemRoot%\SysWOW64\ucrtbase.dll" (
  echo Falta el "Universal C Runtime" de Windows ^(actualizacion KB2999226^).
  echo Instalala desde Windows Update y volve a abrir TISMA CNC.
  pause
  exit /b 1
)
set "PYTHONUTF8=1"
if "%TISMA_CNC_DATOS%"=="" set "TISMA_CNC_DATOS={datos}"
echo Abriendo TISMA CNC... ^(cerrar esta ventana cierra el control^)
"%~dp0python\python.exe" -m hope.cnc app %*
if errorlevel 1 pause
"""

# Launcher of the portable Linux edition. Uses only the folder's Python (not the system one);
# PYTHONNOUSERSITE keeps packages installed with "pip install --user" from mixing in. The data goes to
# ~/.local/share/tisma-cnc (XDG standard) unless TISMA_CNC_DATOS is set.
SH_LINUX = """#!/bin/sh
# TISMA CNC portable para Linux: ./tisma-cnc.sh  (cerrar la terminal o Ctrl+C cierra el control)
D="$(cd "$(dirname "$0")" && pwd)"
export TISMA_CNC_DATOS="${TISMA_CNC_DATOS:-$HOME/.local/share/tisma-cnc}"
export PYTHONPATH="$D/app"
export PYTHONNOUSERSITE=1
exec "$D/python/bin/python3" -m hope.cnc app "$@"
"""

# Optional Linux script (portable and universal) that creates the .desktop shortcut in the menu and, if
# it exists, on the desktop (English or Spanish folder name). Terminal=true: the terminal window shows
# the log and closing it closes the control, just like on Windows.
SH_ACCESO = """#!/bin/sh
# Crea el acceso directo en el menú de aplicaciones (y en el escritorio si existe).
D="$(cd "$(dirname "$0")" && pwd)"
F="$HOME/.local/share/applications/tisma-cnc.desktop"
mkdir -p "$(dirname "$F")"
cat > "$F" <<EOF
[Desktop Entry]
Type=Application
Name=TISMA CNC
Comment=Control CNC (fresadora, router y plasma)
Exec="$D/tisma-cnc.sh"
Icon=$D/app/hope/cnc/app/static/logo.svg
Terminal=true
Categories=Engineering;
EOF
chmod +x "$F"
[ -d "$HOME/Desktop" ] && cp "$F" "$HOME/Desktop/" && chmod +x "$HOME/Desktop/tisma-cnc.desktop"
[ -d "$HOME/Escritorio" ] && cp "$F" "$HOME/Escritorio/" && chmod +x "$HOME/Escritorio/tisma-cnc.desktop"
echo "Listo: TISMA CNC está en el menú de aplicaciones."
"""


# ==================================================================================================
# ============================== COMPATIBILITY CHECK WITH OLD WINDOWS ==============================
# ==================================================================================================

# Windows 7 or later functions: if the embedded Python imports any of them, it does not start on Vista
# ("entry point not found"). Python 3.8.10 imports GetActiveProcessorCount: hence 3.7.9.
# It is a list of kernel32 (and related) functions that appeared in Windows 7 and that some Python or
# library might import. It is not exhaustive: it is a safety net for when the Python version changes.
# Add here any other one discovered (extra entries do no harm).
FUNCIONES_WIN7 = {
    "GetActiveProcessorCount", "GetActiveProcessorGroupCount", "GetSystemTimePreciseAsFileTime", "WaitOnAddress",
    "WakeByAddressSingle", "PowerCreateRequest", "GetLogicalProcessorInformationEx", "CreateRemoteThreadEx",
    "GetCurrentProcessorNumberEx", "SetThreadGroupAffinity", "PathCchCanonicalizeEx", "PathCchCombineEx",
    "PathCchSkipRoot", "CreateFile2", "GetOverlappedResultEx", "TryAcquireSRWLockExclusive",
    "TryAcquireSRWLockShared", "ResolveLocaleName",
}


def funciones_post_vista(zip_python: Path) -> list[str]:
    """Windows 7+ functions imported by the .exe/.dll/.pyd files of the embedded Python (needs pefile;
    without pefile it returns [] and warns about it).

    Reads the import table of each executable inside the ZIP (without extracting it to disk) and looks
    for names from ``FUNCIONES_WIN7``. Why it is needed: Windows gives no warning until it runs; on
    Vista, a Python that imports a Windows 7 function does not even open. Better to find out at build time.

    Parameters: ``zip_python``, the "embeddable" ZIP from python.org.
    Returns: ``["file.dll: Function", …]``; empty if everything starts on Vista (or if pefile is missing).
    """
    try:
        import pefile
    except ImportError:
        print("aviso: sin pefile no se revisa que el Python arranque en Vista (pip install pefile)")
        return []
    malas = []
    with zipfile.ZipFile(zip_python) as z:
        for n in z.namelist():
            if not n.lower().endswith((".dll", ".pyd", ".exe")):
                continue
            # fast_load + parsing only the import table: much faster than analysing the whole PE.
            pe = pefile.PE(data=z.read(n), fast_load=True)
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
            # Without imports the attribute does not exist; i.name is None for imports by number (ordinal).
            for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
                malas += [f"{n}: {i.name.decode()}" for i in e.imports if i.name and i.name.decode() in FUNCIONES_WIN7]
    return malas


# Windows Vista or later functions: if Python 3.4 imported any of them, it would not start on XP
# Includes the Windows 7 ones (FUNCIONES_WIN7) plus those that appeared in Vista: SRW locks, condition
# variables, GetTickCount64, inet_pton, etc. Same criterion: a safety net, not a complete list.
FUNCIONES_VISTA = FUNCIONES_WIN7 | {
    "GetTickCount64", "InitializeSRWLock", "AcquireSRWLockExclusive", "ReleaseSRWLockExclusive", "AcquireSRWLockShared",
    "ReleaseSRWLockShared", "InitializeConditionVariable", "SleepConditionVariableCS", "SleepConditionVariableSRW",
    "WakeConditionVariable", "WakeAllConditionVariable", "InitOnceExecuteOnce", "InitOnceBeginInitialize",
    "GetFinalPathNameByHandleW", "CreateSymbolicLinkW", "CancelIoEx", "GetFileInformationByHandleEx",
    "SetFileInformationByHandle", "GetErrorMode", "inet_pton", "inet_ntop", "GetQueuedCompletionStatusEx",
    "CompareStringEx", "LCMapStringEx", "GetLocaleInfoEx", "GetUserDefaultLocaleName", "QueryFullProcessImageNameW",
    "CreateEventExW", "CreateMutexExW", "CreateSemaphoreExW", "GetThreadUILanguage", "RegGetValueW", "GetProductInfo",
    "if_nametoindex", "if_indextoname", "WSAPoll", "SHGetKnownFolderPath", "GetDynamicTimeZoneInformation",
}


def funciones_post_xp(carpeta: Path) -> list[str]:
    """Vista+ functions imported by the .exe/.dll/.pyd files in ``carpeta``, and the files whose header
    requires a Windows newer than XP (5.1). Needs pefile.

    Checks two things, because either one prevents starting on XP:

    * the minimum subsystem version in the PE header (``MajorSubsystemVersion.Minor``): if it is higher
      than 5.1 (XP), XP says "is not a valid Win32 application" even if it uses no new function;
    * the import table, against ``FUNCIONES_VISTA``.

    Parameters: ``carpeta`` with the already pruned Python (walked entirely).
    Returns: ``["file: reason", …]``; empty if everything starts on XP (or if pefile is missing).
    """
    try:
        import pefile
    except ImportError:
        print("aviso: sin pefile no se revisa que el Python arranque en XP (pip install pefile)")
        return []
    malas = []
    for f in sorted(carpeta.rglob("*")):
        if f.suffix.lower() not in (".dll", ".pyd", ".exe"):
            continue
        pe = pefile.PE(str(f), fast_load=True)
        oh = pe.OPTIONAL_HEADER
        if (oh.MajorSubsystemVersion, oh.MinorSubsystemVersion) > (5, 1):
            malas.append(f"{f.name}: pide Windows {oh.MajorSubsystemVersion}.{oh.MinorSubsystemVersion}")
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            malas += [f"{f.name}: {i.name.decode()}" for i in e.imports if i.name and i.name.decode() in FUNCIONES_VISTA]
        pe.close()   # releases the file (pefile keeps it memory-mapped)
    return malas


# ==================================================================================================
# ========================= WINDOWS EDITION (VISTA AND LATER, PYTHON 3.7) ==========================
# ==================================================================================================

def pth_embebido() -> str:
    """python37._pth: the embedded Python only sees its library, the app and its packages (nothing else).

    With a ``._pth`` file next to ``python.exe``, Python ignores PYTHONPATH and the Windows registry:
    that way another Python installed on the PC does not mix in. One path per line, relative to the
    ``python`` folder:

    * ``python37.zip``: the standard library (it comes zipped in the embedded package);
    * ``.``: the C modules (``.pyd``) next to ``python.exe``;
    * ``..\\app``: the TISMA CNC code;
    * ``Lib\\site-packages``: pyserial and PyYAML;
    * ``import site``: enables the ``site`` module (the embedded package ships with it disabled).
    """
    return f"python{PY_WIN}.zip\n.\n..\\app\nLib\\site-packages\nimport site\n"


def armar_windows(arq: str, dist: Path, ver: str, trabajo: Path) -> Path:
    """Builds the portable Windows ZIP with embedded Python 3.7.9.

    Parameters: ``arq`` = ``"x86"`` (32-bit; also works on 64-bit Windows) or ``"x64"``; ``dist``,
    output folder; ``ver``, version (see ``version()``); ``trabajo``, build folder.

    Steps: downloads and verifies Python, checks with pefile that it starts on Vista, unzips it, writes
    the ``._pth``, installs the wheels, copies the app, writes the BAT and LEEME (CRLF), generates the
    icon and zips everything into ``dist/TISMA-CNC-<ver>-windows-<arq>.zip`` (with the "TISMA CNC"
    folder inside).

    Returns: the built folder (``trabajo/windows-<arq>/TISMA CNC``), reused by ``armar_instalador``.
    Raises: ``SystemExit`` if the Python does not start on Vista or a SHA-256 check fails.
    """
    url, sha = DESCARGAS[f"python-{arq}"]
    zpy = descargar(url, sha, CACHE / Path(url).name)
    malas = funciones_post_vista(zpy)
    if malas:
        raise SystemExit("este Python no arranca en Windows Vista: " + ", ".join(malas))
    base = trabajo / f"windows-{arq}" / NOMBRE
    if base.parent.exists():
        shutil.rmtree(base.parent)   # always built from scratch: no leftovers from a previous build
    py = base / "python"
    py.mkdir(parents=True)
    with zipfile.ZipFile(zpy) as z:
        z.extractall(py)
    (py / f"python{PY_WIN}._pth").write_text(pth_embebido(), encoding="ascii")
    site = py / "Lib" / "site-packages"
    site.mkdir(parents=True)
    for plat, nombre, s in RUEDAS[arq] + [PYSERIAL]:
        extraer_rueda(rueda(plat, nombre, s), site)
    copiar_app(base / "app")
    # %~dp0datos: the data stays next to the BAT (truly portable: it travels with the folder).
    (base / "TISMA CNC.bat").write_text(BAT_PORTABLE.replace("{datos}", r"%~dp0datos").replace("\n", "\r\n"),
                                        encoding="utf-8")
    (base / "LEEME.txt").write_text(LEEME.format(version=ver, como="Doble clic en «TISMA CNC.bat».",
                                                 datos="la carpeta «datos» al lado de TISMA CNC.bat")
                                    .replace("\n", "\r\n"), encoding="utf-8")
    ico = icono(base / "logo.ico")
    salida = dist / f"{ARCHIVO}-{ver}-windows-{arq}.zip"
    with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(base.rglob("*")):
            if f.is_file():
                # Paths relative to base.parent: the ZIP holds the "TISMA CNC/" folder and unzips cleanly.
                z.write(f, f.relative_to(base.parent).as_posix())
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB){'' if ico else '  (sin ícono: falta Pillow)'}")
    return base


# ==================================================================================================
# ================================ WINDOWS XP EDITION (PYTHON 3.4) =================================
# ==================================================================================================

# Launcher of the XP edition (portable and instalador-xp). Same as BAT_PORTABLE but without the UCRT
# check (Python 3.4 ships its own msvcr100.dll) or PYTHONUTF8 (it does not exist in 3.4: _tisma34 fixes
# the console). -E ignores the system's PYTHON* variables and -s the user's site-packages, so that a
# Python 2 or 3 installed on the PC does not mix in. Written in ASCII: any accented letter would make the
# build fail (and XP's cmd.exe would garble it).
BAT_XP = r"""@echo off
rem TISMA CNC para Windows XP: doble clic. Cerrar esta ventana cierra TISMA CNC.
setlocal
cd /d "%~dp0"
title TISMA CNC
if "%TISMA_CNC_DATOS%"=="" set "TISMA_CNC_DATOS={datos}"
echo Abriendo TISMA CNC... ^(cerrar esta ventana cierra el control^)
rem -E -s: ignorar cualquier otro Python instalado; la app se encuentra por python\Lib\site-packages\tisma_cnc.pth
"%~dp0python\python.exe" -E -s -m hope.cnc app %*
if errorlevel 1 pause
"""


def msi_extraer(msi: Path, destino: Path) -> None:
    """Extracts the files of the Python 3.4 MSI installer to ``destino`` with ``msiextract`` (msitools).

    This gets XP's Python without running the installer (or having Windows). ``destino`` must not
    exist. Raises: ``SystemExit`` if msiextract is missing; ``CalledProcessError`` if it fails.
    """
    if not shutil.which("msiextract"):
        raise SystemExit("falta msiextract para abrir el instalador de Python 3.4: apt install msitools")
    destino.mkdir(parents=True)
    subprocess.run(["msiextract", "-C", str(destino), str(msi)], check=True, stdout=subprocess.DEVNULL)


def pyyaml_puro(url_sha: tuple[str, str], lib: Path) -> None:
    """Copies the pure-Python ``yaml`` package (lib3/yaml from the source) to ``lib``.

    Parameters: ``url_sha`` = (URL, SHA-256) of the PyYAML ``.tar.gz`` (``PYYAML_FUENTE`` or
    ``PYYAML_XP``); ``lib``, folder where ``lib/yaml/`` ends up.

    Why the source and not a wheel: PyYAML publishes no pure-Python wheels and the C extension
    (``_yaml``) would have to be compiled for each processor. Without it PyYAML uses its Python loader,
    slower but more than enough for the TISMA CNC profiles.

    Inside the tar the paths are ``PyYAML-x.y.z/lib/yaml/…`` (PyYAML 6) or ``…/lib3/yaml/…`` (PyYAML 5,
    which also ships ``lib/`` for Python 2).
    """
    url, sha = url_sha
    fuente = descargar(url, sha, CACHE / "ruedas" / Path(url).name)
    with tarfile.open(fuente) as t:                  # nothing compiled: any processor
        for m in t.getmembers():
            partes = m.name.split("/")
            if len(partes) > 3 and partes[1:3] in (["lib", "yaml"], ["lib3", "yaml"]) and m.isfile():
                if partes[1] == "lib" and any(x.name.split("/")[1:3] == ["lib3", "yaml"] for x in t.getmembers()):
                    continue                         # PyYAML 5: lib/ is Python 2, lib3/ is Python 3
                # Written by hand (not extractall): only what is under yaml/ gets in, with paths built here.
                destino = lib / "yaml" / "/".join(partes[3:])
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(t.extractfile(m).read())


def armar_xp(dist: Path, ver: str, trabajo: Path) -> Path:
    """Windows XP SP3 (and 2000/Vista/7 onwards, 32 or 64 bits): official Python 3.4.4 and the translated code.

    Steps: extracts the Python 3.4.4 MSI, copies only what is needed (``PODAR_XP_*``), checks with pefile
    that nothing requires a Windows newer than XP, builds a clean ``site-packages`` with pyserial and
    pure-Python PyYAML 5.1.2, copies the app to a temporary folder and translates it to Python 3.4
    (``py34/a_py34.py``), adds ``_tisma34.py`` untranslated (it is already written for 3.4), writes BAT
    (ASCII) and LEEME_XP (CRLF) and zips into ``dist/TISMA-CNC-<ver>-windows-xp.zip``.

    Why ``tisma_cnc.pth`` and not a ``._pth`` as on Vista: ``._pth`` appeared in Python 3.6.

    Returns: the built folder (``trabajo/windows-xp/TISMA CNC``).
    Raises: ``SystemExit`` if this script runs on Python < 3.9, msiextract is missing, something does not
    start on XP or a SHA-256 fails; ``NoTraducible`` (from ``a_py34``) if the code uses something 3.4 lacks.
    """
    if sys.version_info < (3, 9):  # noqa: UP036
        raise SystemExit("la edición XP se arma con Python 3.9 o más nuevo (la traducción usa ast.unparse)")
    sys.path.insert(0, str(Path(__file__).parent / "py34"))
    from a_py34 import traducir_carpeta

    url, sha = PY_XP
    msi = descargar(url, sha, CACHE / Path(url).name)
    base = trabajo / "windows-xp" / NOMBRE
    if base.parent.exists():
        shutil.rmtree(base.parent)
    crudo = trabajo / "windows-xp" / "msi"
    msi_extraer(msi, crudo)
    py = base / "python"
    py.mkdir(parents=True)
    # Only what is needed to run is taken from the MSI (no Scripts/, Tools/, Doc/, include/, libs/, tcl/).
    for n in ("python.exe", "pythonw.exe", "python34.dll", "msvcr100.dll", "LICENSE.txt"):
        shutil.copy2(crudo / n, py / n)
    shutil.copytree(crudo / "DLLs", py / "DLLs", ignore=shutil.ignore_patterns(*PODAR_XP_DLL))
    shutil.copytree(crudo / "Lib", py / "Lib", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for q in PODAR_XP_LIB:
        f = py / "Lib" / q
        if f.is_dir():
            shutil.rmtree(f)
        elif f.exists():
            f.unlink()
    malas = funciones_post_xp(py)
    if malas:
        raise SystemExit("este Python no arranca en Windows XP: " + ", ".join(malas))
    site = py / "Lib" / "site-packages"
    site.mkdir()
    # python/Lib/site-packages → ../../../ = the "TISMA CNC" folder → app/ (where the translated code is).
    (site / "tisma_cnc.pth").write_text("../../../app\n", encoding="ascii")    # relative to site-packages
    extraer_rueda(rueda(*PYSERIAL), site)
    pyyaml_puro(PYYAML_XP, site)
    for d in site.glob("*.dist-info"):
        shutil.rmtree(d)   # pip metadata: useless without pip on the PC
    fuente = trabajo / "windows-xp" / "fuente"
    copiar_app(fuente)
    n = traducir_carpeta(fuente, base / "app")                # modern Python → 3.4
    shutil.copy2(Path(__file__).parent / "py34" / "_tisma34.py", base / "app" / "_tisma34.py")
    shutil.rmtree(fuente)
    (base / "TISMA CNC.bat").write_text(BAT_XP.replace("{datos}", r"%~dp0datos").replace("\n", "\r\n"),
                                        encoding="ascii")
    (base / "LEEME.txt").write_text(LEEME_XP.format(version=ver, como="Doble clic en «TISMA CNC.bat».",
                                                    datos="la carpeta «datos» al lado de TISMA CNC.bat")
                                    .replace("\n", "\r\n"), encoding="utf-8")
    ico = icono(base / "logo.ico")
    salida = dist / f"{ARCHIVO}-{ver}-windows-xp.zip"
    with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(base.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(base.parent).as_posix())
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB, {n} módulos traducidos a Python 3.4)"
          f"{'' if ico else '  (sin ícono: falta Pillow)'}")
    return base


# LEEME of the XP edition: the general LEEME with three changes (system requirements, no KB2999226
# update and the browser recommended for XP). Careful: str.replace gives no warning if it does not find
# the text; if LEEME is edited, these fragments must still match letter for letter.
LEEME_XP = LEEME.replace("""  * Windows Vista SP2, 7, 8, 10 u 11 (32 o 64 bits), o Linux de 2014 en adelante.""",
                         """  * Windows XP SP3 (32 o 64 bits) en adelante: esta edición trae Python 3.4.4, el último para XP.
    En Vista o más nuevo conviene la edición normal (Python 3.7, más rápida).""").replace(
    """  * Windows Vista SP2, 7 SP1 y 8: hace falta la actualización KB2999226 de Microsoft (Universal C
    Runtime), que casi todas las PCs ya tienen si pasaron por Windows Update. Si al abrir aparece
    «falta api-ms-win-crt-runtime-l1-1-0.dll», instalarla (hay versión para Vista).""",
    """  * En XP no hace falta ninguna actualización de Microsoft: Python 3.4 trae su propio msvcr100.dll.""").replace(
    "en Windows Vista,\n    Firefox 52 ESR o Supermium (Chromium actual para Vista y XP)",
    "en Windows XP y Vista,\n    Firefox 52 ESR o Supermium (Chromium actual para XP y Vista)")


# ==================================================================================================
# ======================================= INSTALLERS (NSIS) ========================================
# ==================================================================================================

# NSIS script for both installers. The placeholders are @name@ (not {name}) because NSIS uses ${…}
# for its own constants. armar_instalador fills them in:
#   @salida@  path of the generated .exe      @icono@  logo.ico          @origen@  folder to package
#   @ver@     version as text                 @vnum@   version X.Y.Z.W   @oninit@  .onInit function (or nothing)
# Decisions: "Unicode true" (NSIS 3) for accented letters; RequestExecutionLevel user + installing into
# %LOCALAPPDATA%\Programs: no administrator needed (many workshop PCs lack the password); the
# uninstaller is registered in HKCU (the user's Add or Remove Programs); solid LZMA compression (the
# smallest installer); the shortcuts open the BAT minimized (SW_SHOWMINIMIZED) so the console does not
# cover the browser; uninstalling does not delete the data in %APPDATA%\TISMA CNC.
NSI = r"""; Instalador de TISMA CNC (generado por empaquetado/tisma_cnc/construir.py)
Unicode true
!include "MUI2.nsh"
Name "TISMA CNC"
OutFile "@salida@"
InstallDir "$LOCALAPPDATA\Programs\TISMA CNC"
RequestExecutionLevel user
SetCompressor /SOLID lzma
!define MUI_ICON "@icono@"
!define MUI_UNICON "@icono@"
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\TISMA CNC.bat"
!define MUI_FINISHPAGE_RUN_TEXT "Abrir TISMA CNC"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "Spanish"
VIProductVersion "@vnum@"
VIAddVersionKey /LANG=${LANG_SPANISH} "ProductName" "TISMA CNC"
VIAddVersionKey /LANG=${LANG_SPANISH} "FileDescription" "Instalador de TISMA CNC"
VIAddVersionKey /LANG=${LANG_SPANISH} "ProductVersion" "@ver@"
VIAddVersionKey /LANG=${LANG_SPANISH} "FileVersion" "@vnum@"

@oninit@
Section "TISMA CNC"
  SetOutPath "$INSTDIR"
  File /r "@origen@\*"
  CreateDirectory "$SMPROGRAMS\TISMA CNC"
  CreateShortCut "$SMPROGRAMS\TISMA CNC\TISMA CNC.lnk" "$INSTDIR\TISMA CNC.bat" "" "$INSTDIR\logo.ico" 0 SW_SHOWMINIMIZED
  CreateShortCut "$SMPROGRAMS\TISMA CNC\Desinstalar TISMA CNC.lnk" "$INSTDIR\desinstalar.exe"
  CreateShortCut "$DESKTOP\TISMA CNC.lnk" "$INSTDIR\TISMA CNC.bat" "" "$INSTDIR\logo.ico" 0 SW_SHOWMINIMIZED
  WriteUninstaller "$INSTDIR\desinstalar.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TISMA CNC" "DisplayName" "TISMA CNC"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TISMA CNC" "DisplayVersion" "@ver@"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TISMA CNC" "Publisher" "TISMA"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TISMA CNC" "DisplayIcon" "$INSTDIR\logo.ico"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TISMA CNC" "UninstallString" '"$INSTDIR\desinstalar.exe"'
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TISMA CNC" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TISMA CNC" "NoRepair" 1
SectionEnd

Section "Uninstall"
  RMDir /r "$INSTDIR"
  Delete "$DESKTOP\TISMA CNC.lnk"
  RMDir /r "$SMPROGRAMS\TISMA CNC"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\TISMA CNC"
  ; los perfiles propios y el registro del guardián quedan en %APPDATA%\TISMA CNC
SectionEnd
"""


# .onInit function of the normal installer (Python 3.7): warns at the start if the Universal C Runtime
# (KB2999226) is missing. It does not prevent installing: the update can be installed later. The XP
# installer does not include it (Python 3.4 does not use the UCRT).
NSI_UCRT = r"""Function .onInit
  IfFileExists "$WINDIR\System32\ucrtbase.dll" ucrt_ok 0
  IfFileExists "$WINDIR\SysWOW64\ucrtbase.dll" ucrt_ok 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "A este Windows le falta el «Universal C Runtime» (actualización KB2999226).$\r$\nTISMA CNC se instala igual, pero para abrirlo instalá esa actualización desde Windows Update."
  ucrt_ok:
FunctionEnd
"""


def armar_instalador(dist: Path, ver: str, trabajo: Path, xp: bool = False) -> Path:
    """NSIS installer. With ``xp``: the Python 3.4 edition (Windows XP SP3 and later, no UCRT).

    Starts from the folder already built by ``armar_windows("x86")`` (32-bit: works on 32- and 64-bit
    Windows) or ``armar_xp``; builds it if missing. Copies it, changes the BAT and LEEME so the data goes
    to ``%APPDATA%\\TISMA CNC`` (the install folder may not belong to the user forever), fills in the
    ``NSI`` script and runs ``makensis``.

    Returns: the ``.exe`` path. Raises: ``SystemExit`` if makensis or the icon (Pillow) is missing;
    ``CalledProcessError`` if makensis fails.
    """
    if not shutil.which("makensis"):
        raise SystemExit("falta makensis (NSIS): apt install nsis, o en Windows https://nsis.sourceforge.io")
    origen = "windows-xp" if xp else "windows-x86"
    base = trabajo / origen / NOMBRE
    if not base.exists():
        armar_xp(dist, ver, trabajo) if xp else armar_windows("x86", dist, ver, trabajo)
    inst = trabajo / ("instalador-xp" if xp else "instalador") / NOMBRE
    if inst.parent.exists():
        shutil.rmtree(inst.parent)
    shutil.copytree(base, inst)
    bat = BAT_XP if xp else BAT_PORTABLE
    (inst / "TISMA CNC.bat").write_text(bat.replace("{datos}", r"%APPDATA%\TISMA CNC").replace("\n", "\r\n"),
                                        encoding="ascii" if xp else "utf-8")
    (inst / "LEEME.txt").write_text((LEEME_XP if xp else LEEME).format(version=ver, como="Acceso directo «TISMA CNC» en el escritorio o el "
                                                 "menú Inicio.", datos=r"%APPDATA%\TISMA CNC")
                                    .replace("\n", "\r\n"), encoding="utf-8")
    ico = inst / "logo.ico"
    if not ico.exists():
        raise SystemExit("falta el ícono (instalar Pillow para generarlo)")
    salida = dist / f"{ARCHIVO}-{ver}-instalador{'-xp' if xp else ''}.exe"
    # VIProductVersion requires four numbers "X.Y.Z.W": "0.1.0-20261123-abc" becomes "0.1.0.0" (padded
    # with zeros and cut at four).
    nums = [int(x) for x in ver.split("-")[0].split(".")] + [0, 0, 0]
    script = NSI
    for k, v in {"salida": salida, "icono": ico, "origen": inst, "ver": ver, "oninit": "" if xp else NSI_UCRT,
                 "vnum": ".".join(str(n) for n in nums[:4])}.items():
        script = script.replace(f"@{k}@", str(v))
    nsi = inst.parent / "tisma_cnc.nsi"
    # UTF-8 with BOM: the only way makensis recognizes the script as UTF-8 (accents in LEEME and messages).
    nsi.write_text(script, encoding="utf-8-sig")
    subprocess.run(["makensis", "-V2", str(nsi)], check=True)   # -V2: warnings and errors only
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB)")
    return salida


# ==================================================================================================
# ============================== PORTABLE LINUX EDITION (PYTHON 3.8) ===============================
# ==================================================================================================

def python_linux() -> Path:
    """Portable Python 3.8 (python-build-standalone) installed by ``uv``: glibc 2.17 and later.

    Why that one: python-build-standalone is compiled against glibc 2.17 (CentOS 7, 2014), so the
    copied folder starts on almost any x86_64 Linux from that era on, without installing anything.
    If ``uv`` does not have it, it installs it.

    Returns: the Python root (the folder holding ``bin/`` and ``lib/python3.8``).
    Raises: ``SystemExit`` if what uv found is not a complete portable installation;
    ``OSError``/``CalledProcessError`` if there is no ``uv``.
    """
    try:
        exe = subprocess.run(["uv", "python", "find", "3.8"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        subprocess.run(["uv", "python", "install", "3.8"], check=True)
        exe = subprocess.run(["uv", "python", "find", "3.8"], capture_output=True, text=True, check=True).stdout.strip()
    raiz = Path(exe).resolve().parents[1]   # <root>/bin/python3.8 → <root>
    if not (raiz / "lib" / "python3.8").is_dir():
        raise SystemExit(f"no es un Python portable: {raiz}")
    return raiz


def armar_linux(dist: Path, ver: str, trabajo: Path) -> Path:
    """Builds ``dist/TISMA-CNC-<ver>-linux-x86_64.tar.gz``: portable Python 3.8, pruned, with the app.

    Copies the Python from ``python_linux()`` keeping the symbolic links (``bin/python3`` →
    ``python3.8``), prunes ``PODAR_LINUX`` and the C headers, installs PyYAML (manylinux2014 wheel) and
    pyserial, copies the app and writes the two scripts (executable) and the LEEME (with LF: it is Linux).

    Returns: the built folder (``trabajo/linux/tisma-cnc``).
    """
    base = trabajo / "linux" / "tisma-cnc"
    if base.parent.exists():
        shutil.rmtree(base.parent)
    base.mkdir(parents=True)
    shutil.copytree(python_linux(), base / "python", symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    lib = base / "python" / "lib" / "python3.8"
    for p in PODAR_LINUX:
        shutil.rmtree(lib / p, ignore_errors=True)
    shutil.rmtree(base / "python" / "include", ignore_errors=True)
    site = lib / "site-packages"
    site.mkdir(exist_ok=True)
    for plat, nombre, s in RUEDAS["linux"] + [PYSERIAL]:
        extraer_rueda(rueda(plat, nombre, s), site)
    copiar_app(base / "app")
    for nombre, texto in (("tisma-cnc.sh", SH_LINUX), ("crear-acceso-directo.sh", SH_ACCESO)):
        f = base / nombre
        f.write_text(texto, encoding="utf-8")
        # Execute permission for everyone (like chmod a+x); the tar.gz keeps it.
        f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (base / "LEEME.txt").write_text(LEEME.format(version=ver, como="./tisma-cnc.sh  (y ./crear-acceso-directo.sh "
                                                 "para tenerlo en el menú)", datos="~/.local/share/tisma-cnc"),
                                    encoding="utf-8")
    salida = dist / f"{ARCHIVO}-{ver}-linux-x86_64.tar.gz"
    with tarfile.open(salida, "w:gz", compresslevel=9) as t:
        t.add(base, arcname="tisma-cnc")
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB)")
    return base


# ==================================================================================================
# =============================== UNIVERSAL EDITION (SYSTEM PYTHON) ================================
# ==================================================================================================

# Launcher of the universal package: looks for Python 3.7 or newer among the usual names (first
# "python3", then specific versions from newest to oldest) and starts with the first one that works.
# app/ holds the code and lib/ pyserial and PyYAML in pure Python. If there is none, it explains how to
# install it.
SH_UNIVERSAL = """#!/bin/sh
# TISMA CNC con el Python del sistema (3.7 o más nuevo): Linux de 32 o 64 bits, Raspberry Pi, BSD, macOS.
D="$(cd "$(dirname "$0")" && pwd)"
for P in python3 python3.13 python3.12 python3.11 python3.10 python3.9 python3.8 python3.7; do
  if command -v "$P" >/dev/null 2>&1 && "$P" -c "import sys; sys.exit(sys.version_info < (3, 7))" 2>/dev/null; then
    export TISMA_CNC_DATOS="${TISMA_CNC_DATOS:-$HOME/.local/share/tisma-cnc}"
    export PYTHONPATH="$D/app:$D/lib"
    export PYTHONNOUSERSITE=1
    exec "$P" -m hope.cnc app "$@"
  fi
done
echo "TISMA CNC necesita Python 3.7 o más nuevo (sudo apt install python3)."
exit 1
"""


def armar_universal(dist: Path, ver: str, trabajo: Path) -> Path:
    """Builds ``dist/TISMA-CNC-<ver>-universal.tar.gz``: code only, no Python (≈ 1 MB).

    For what the other editions do not cover: 32-bit Linux, ARM (Raspberry Pi), BSD, macOS. Everything
    it ships is pure Python (the app, pyserial and the ``yaml`` from the PyYAML 6.0.1 source), so it
    runs on any processor with Python 3.7+.

    Returns: the built folder (``trabajo/universal/tisma-cnc``).
    """
    base = trabajo / "universal" / "tisma-cnc"
    if base.parent.exists():
        shutil.rmtree(base.parent)
    copiar_app(base / "app")
    lib = base / "lib"
    lib.mkdir(parents=True)
    extraer_rueda(rueda(*PYSERIAL), lib)
    pyyaml_puro(PYYAML_FUENTE, lib)
    for nombre, texto in (("tisma-cnc.sh", SH_UNIVERSAL), ("crear-acceso-directo.sh", SH_ACCESO)):
        f = base / nombre
        f.write_text(texto, encoding="utf-8")
        f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (base / "LEEME.txt").write_text(LEEME.format(version=ver, como="./tisma-cnc.sh  (usa el Python 3.7+ del sistema; "
                                                 "./crear-acceso-directo.sh para el menú)",
                                                 datos="~/.local/share/tisma-cnc"), encoding="utf-8")
    salida = dist / f"{ARCHIVO}-{ver}-universal.tar.gz"
    with tarfile.open(salida, "w:gz", compresslevel=9) as t:
        t.add(base, arcname="tisma-cnc")
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB)")
    return base


# ==================================================================================================
# ========================================== COMMAND LINE ==========================================
# ==================================================================================================

def main(argv: list[str] | None = None) -> int:
    """Entry point: reads the options and builds each requested edition.

    Without ``--solo`` it builds everything possible on this PC: always Windows x86/x64, Linux and
    universal; the XP edition only if ``msiextract`` is present; the installers only if ``makensis`` is
    present (and the XP one also needs ``msiextract``). That way ``construir.py`` without options does
    not fail because of a missing tool.

    Parameters: ``argv``, the arguments (without the program name); ``None`` = ``sys.argv[1:]``.
    Returns: 0 (errors stop earlier with ``SystemExit``).
    """
    ap = argparse.ArgumentParser(description="Arma TISMA CNC para Windows y Linux (PCs viejas)")
    ap.add_argument("--solo", nargs="*", choices=["windows-x86", "windows-x64", "instalador", "windows-xp",
                                                  "instalador-xp", "linux", "universal"],
                    help="qué armar (por defecto: todo lo posible)")
    ap.add_argument("--destino", default=str(RAIZ / "dist"))
    ap.add_argument("--trabajo", default=str(CACHE / "armado"))
    a = ap.parse_args(argv)
    dist, trabajo = Path(a.destino), Path(a.trabajo)
    dist.mkdir(parents=True, exist_ok=True)
    ver = version()
    que = a.solo or ["windows-x86", "windows-x64", "linux", "universal"] + \
        (["windows-xp"] if shutil.which("msiextract") else []) + \
        (["instalador"] if shutil.which("makensis") else []) + \
        (["instalador-xp"] if shutil.which("makensis") and shutil.which("msiextract") else [])
    # Order matters: the installers come after the editions they start from (and reuse them).
    for q in que:
        if q == "windows-xp":
            armar_xp(dist, ver, trabajo)
        elif q.startswith("windows"):
            armar_windows(q.split("-")[1], dist, ver, trabajo)
        elif q in ("instalador", "instalador-xp"):
            armar_instalador(dist, ver, trabajo, xp=q.endswith("xp"))
        elif q == "linux":
            armar_linux(dist, ver, trabajo)
        elif q == "universal":
            armar_universal(dist, ver, trabajo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
