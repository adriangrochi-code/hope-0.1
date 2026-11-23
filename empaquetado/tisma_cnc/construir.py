"""Arma TISMA CNC listo para PCs viejas, sin instalar nada a mano.

    python empaquetado/tisma_cnc/construir.py                 # todo lo que se pueda armar acá
    python empaquetado/tisma_cnc/construir.py --solo windows-x86 linux
    python empaquetado/tisma_cnc/construir.py --solo instalador   # necesita makensis (NSIS)

Resultado en ``dist/``:

* ``TISMA-CNC-<versión>-windows-x86.zip`` y ``…-windows-x64.zip``: portables. Se descomprimen en
  cualquier carpeta (o pendrive) y se abre ``TISMA CNC.bat``. Traen **Python 3.7.9 embebido**
  (el último oficial que arranca en **Windows Vista**: 3.8 ya importa GetActiveProcessorCount, que es
  de Windows 7), pyserial y PyYAML. Los datos quedan en ``datos/`` al lado.
* ``TISMA-CNC-<versión>-instalador.exe``: instalador (NSIS, 32 bits: sirve en Windows Vista, 7, 8, 10
  y 11 de 32 o 64 bits) sin permisos de administrador; accesos directos en el escritorio y el menú Inicio;
  datos en ``%APPDATA%\\TISMA CNC`` (no se borran al desinstalar).
* ``TISMA-CNC-<versión>-windows-xp.zip`` e ``…-instalador-xp.exe``: edición para **Windows XP SP3** en
  adelante con **Python 3.4.4** (el último oficial para XP, del instalador MSI, que se abre con
  ``msiextract`` de msitools). El código se traduce solo a Python 3.4 (``py34/a_py34.py``) y
  ``py34/_tisma34.py`` agrega lo que le falta a esa biblioteca estándar.
* ``TISMA-CNC-<versión>-linux-x86_64.tar.gz``: portable para Linux desde 2014 (glibc 2.17), con su
  propio Python 3.8 (python-build-standalone).
* ``TISMA-CNC-<versión>-universal.tar.gz``: solo el código (≈ 1 MB) para cualquier sistema con
  **Python 3.7 o más nuevo**: Linux de 32 bits (PCs de la época de XP), Raspberry Pi y otras ARM,
  BSD, macOS. TISMA CNC usa solo la biblioteca estándar; trae pyserial y PyYAML en Python puro.

Todo lo que se descarga se verifica con SHA-256 fijo (``DESCARGAS``); las descargas quedan en
``runs/empaquetado/`` para no repetirlas. Solo biblioteca estándar (más Pillow, opcional, para el ícono).
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

RAIZ = Path(__file__).resolve().parents[2]
NOMBRE = "TISMA CNC"
ARCHIVO = "TISMA-CNC"
CACHE = RAIZ / "runs" / "empaquetado"

# Descargas verificadas. Los ZIP de Python coinciden con el MD5 publicado en
# https://www.python.org/downloads/release/python-379/ (60f77740…, 97c6558d…).
PY_WIN = "37"                      # python37.zip, python37._pth
DESCARGAS = {
    "python-x64": ("https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-amd64.zip",
                   "18627a097adf47829a847053febac5532376075243e233bd9ec61d6ea09dee1f"),
    "python-x86": ("https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-win32.zip",
                   "010cef379e1096cebf667e8f2c92a72dc455c927c7f7b8ca8e029975d81946bf"),
}
RUEDAS = {   # (plataforma de pip, archivo, sha256)
    "x64": [("win_amd64", "PyYAML-6.0.1-cp37-cp37m-win_amd64.whl",
             "4fb147e7a67ef577a588a0e2c17b6db51dda102c71de36f8549b6816a96e1867")],
    "x86": [("win32", "PyYAML-6.0.1-cp37-cp37m-win32.whl",
             "9046c58c4395dff28dd494285c82ba00b546adfc7ef001486fbf0324bc174fba")],
    "linux": [("manylinux2014_x86_64", "PyYAML-6.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
               "7e07cbde391ba96ab58e532ff4803f79c4129397514e1413a7dc761ccd755735")],
}
PYYAML_FUENTE = (("https://files.pythonhosted.org/packages/cd/e5/af35f7ea75cf72f2cd079c95ee16797de7cd71f29ea7c68ae5ce7be1eda0/"
                  "PyYAML-6.0.1.tar.gz"), "bfdf460b1736c775f2ba9f6a92bca30bc2095067b8a9d77876d1fad6cc3b4a43")
# Windows XP: Python 3.4.4 (el último oficial para XP SP3), instalador MSI de 32 bits que se extrae con
# msiextract (msitools). MD5 publicado en https://www.python.org/downloads/release/python-344/: e96268f7….
PY_XP = ("https://www.python.org/ftp/python/3.4.4/python-3.4.4.msi",
         "46c8f9f63cf02987e8bf23934b2f471e1868b24748c5bb551efcf4863b43ca6c")
# PyYAML 5.1.2: la última que acepta Python 3.4 (en Python puro)
PYYAML_XP = (("https://files.pythonhosted.org/packages/e3/e8/b3212641ee2718d556df0f23f78de8303f068fe29cdaa7a91018849582fe/"
              "PyYAML-5.1.2.tar.gz"), "01adf0b6c6f61bd11af6e10ca52b7d4057dd0be0343eb9283c878cf3af56aee4")
PYSERIAL = ("any", "pyserial-3.5-py2.py3-none-any.whl", "c4451db6ba391ca6ca299fb3ec7bae67a5c55dde170964c7a14ceefec02f2cf0")

# Lo mínimo del repositorio que usa TISMA CNC (lo demás de Hope —PyTorch, entrenamiento— no va)
ARCHIVOS_APP = [
    "hope/__init__.py", "hope/epistemic.py", "hope/growth/__init__.py", "hope/growth/budget.py",
    "hope/tools/__init__.py", "hope/tools/catalog.py",
    "guardian/__init__.py", "guardian/audit.py", "guardian/core.py",
]
CARPETAS_APP = ["hope/cnc"]
EXCLUIR = {"__pycache__", "ruff.toml"}
# Partes de la biblioteca estándar que la app no usa (achican el paquete de Linux)
# Del Python 3.4 de XP: lo que la app no usa (Tk, pruebas, documentación, sqlite…)
PODAR_XP_LIB = ["test", "idlelib", "tkinter", "turtledemo", "lib2to3", "ensurepip", "unittest/test", "distutils",
                "ctypes/test", "sqlite3", "msilib", "venv", "pydoc_data", "turtle.py", "site-packages"]
PODAR_XP_DLL = ["_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "_sqlite3.pyd", "sqlite3.dll", "_msi.pyd", "_testcapi.pyd",
                "_testbuffer.pyd", "_testimportmultiple.pyd", "_ctypes_test.pyd", "py.ico", "pyc.ico"]
PODAR_LINUX = ["test", "idlelib", "tkinter", "turtledemo", "lib2to3/tests", "ensurepip", "unittest/test",
               "distutils/tests", "ctypes/test", "sqlite3/test", "config-3.8-x86_64-linux-gnu"]


def version() -> str:
    sys.path.insert(0, str(RAIZ))
    from hope import __version__
    try:
        sha = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "local"
    return f"{__version__}-{datetime.now(timezone.utc):%Y%m%d}-{sha}"


# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def descargar(url: str, esperado: str, destino: Path) -> Path:
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
    destino = CACHE / "ruedas" / nombre
    if destino.exists() and sha256(destino) == esperado:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    paquete, ver, tag = nombre.split("-")[:3]
    pyver = f"{tag[2]}.{tag[3:]}" if tag.startswith("cp") else "3.8"   # la versión sale del nombre de la rueda
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--python-version", pyver,
           "-d", str(destino.parent), f"{paquete}=={ver}"]
    if plataforma != "any":
        cmd[5:5] = ["--platform", plataforma]
    subprocess.run(cmd, check=True)
    if sha256(destino) != esperado:
        raise SystemExit(f"SHA-256 distinto en {nombre}")
    return destino


def extraer_rueda(whl: Path, site: Path) -> None:
    with zipfile.ZipFile(whl) as z:
        z.extractall(site)


# ---------------------------------------------------------------------------
def copiar_app(destino: Path) -> list[str]:
    """Copia el código de TISMA CNC (y lo mínimo de Hope y del guardián) a ``destino``."""
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
    """logo.ico (hexágono y pirámide violeta de TISMA) para los accesos directos."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    s = 256
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = s / 110.0

    def pts(*xy):
        return [(x * k + 3, y * k) for x, y in zip(xy[::2], xy[1::2])]
    d.polygon(pts(50, 4, 95, 29, 95, 81, 50, 106, 5, 81, 5, 29), outline=(206, 206, 206, 255), width=int(5 * k))
    d.polygon(pts(50, 15, 16, 78, 31, 74), fill=(206, 206, 206, 255))
    d.polygon(pts(50, 15, 31, 74, 50, 84), fill=(148, 55, 193, 255))
    d.polygon(pts(50, 15, 50, 84, 69, 74), fill=(82, 19, 134, 255))
    d.polygon(pts(50, 15, 69, 74, 84, 78), fill=(169, 169, 171, 255))
    d.polygon(pts(16, 78, 31, 74, 50, 84, 69, 74, 84, 78, 50, 97), fill=(189, 189, 191, 255))
    destino.parent.mkdir(parents=True, exist_ok=True)
    im.save(destino, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return destino


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

SH_LINUX = """#!/bin/sh
# TISMA CNC portable para Linux: ./tisma-cnc.sh  (cerrar la terminal o Ctrl+C cierra el control)
D="$(cd "$(dirname "$0")" && pwd)"
export TISMA_CNC_DATOS="${TISMA_CNC_DATOS:-$HOME/.local/share/tisma-cnc}"
export PYTHONPATH="$D/app"
export PYTHONNOUSERSITE=1
exec "$D/python/bin/python3" -m hope.cnc app "$@"
"""

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


# Funciones de Windows 7 o posterior: si el Python embebido importa alguna, no arranca en Vista
# («no se encontró el punto de entrada»). Python 3.8.10 importa GetActiveProcessorCount: por eso 3.7.9.
FUNCIONES_WIN7 = {
    "GetActiveProcessorCount", "GetActiveProcessorGroupCount", "GetSystemTimePreciseAsFileTime", "WaitOnAddress",
    "WakeByAddressSingle", "PowerCreateRequest", "GetLogicalProcessorInformationEx", "CreateRemoteThreadEx",
    "GetCurrentProcessorNumberEx", "SetThreadGroupAffinity", "PathCchCanonicalizeEx", "PathCchCombineEx",
    "PathCchSkipRoot", "CreateFile2", "GetOverlappedResultEx", "TryAcquireSRWLockExclusive",
    "TryAcquireSRWLockShared", "ResolveLocaleName",
}


def funciones_post_vista(zip_python: Path) -> list[str]:
    """Funciones de Windows 7+ que importan los .exe/.dll/.pyd del Python embebido (necesita pefile;
    sin pefile devuelve [] y lo avisa)."""
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
            pe = pefile.PE(data=z.read(n), fast_load=True)
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
            for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
                malas += [f"{n}: {i.name.decode()}" for i in e.imports if i.name and i.name.decode() in FUNCIONES_WIN7]
    return malas


# Funciones de Windows Vista o posterior: si Python 3.4 importara alguna, no arrancaría en XP
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
    """Funciones de Vista+ que importan los .exe/.dll/.pyd de ``carpeta``, y los que piden un Windows
    mayor que XP (5.1) en el encabezado. Necesita pefile."""
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
        pe.close()
    return malas


def pth_embebido() -> str:
    """python37._pth: el Python embebido solo ve su biblioteca, la app y sus paquetes (nada del sistema)."""
    return f"python{PY_WIN}.zip\n.\n..\\app\nLib\\site-packages\nimport site\n"


def armar_windows(arq: str, dist: Path, ver: str, trabajo: Path) -> Path:
    url, sha = DESCARGAS[f"python-{arq}"]
    zpy = descargar(url, sha, CACHE / Path(url).name)
    malas = funciones_post_vista(zpy)
    if malas:
        raise SystemExit("este Python no arranca en Windows Vista: " + ", ".join(malas))
    base = trabajo / f"windows-{arq}" / NOMBRE
    if base.parent.exists():
        shutil.rmtree(base.parent)
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
                z.write(f, f.relative_to(base.parent).as_posix())
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB){'' if ico else '  (sin ícono: falta Pillow)'}")
    return base


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
    if not shutil.which("msiextract"):
        raise SystemExit("falta msiextract para abrir el instalador de Python 3.4: apt install msitools")
    destino.mkdir(parents=True)
    subprocess.run(["msiextract", "-C", str(destino), str(msi)], check=True, stdout=subprocess.DEVNULL)


def pyyaml_puro(url_sha: tuple[str, str], lib: Path) -> None:
    """Copia el paquete ``yaml`` en Python puro (lib3/yaml de la fuente) a ``lib``."""
    url, sha = url_sha
    fuente = descargar(url, sha, CACHE / "ruedas" / Path(url).name)
    with tarfile.open(fuente) as t:                  # sin compilar: cualquier procesador
        for m in t.getmembers():
            partes = m.name.split("/")
            if len(partes) > 3 and partes[1:3] in (["lib", "yaml"], ["lib3", "yaml"]) and m.isfile():
                if partes[1] == "lib" and any(x.name.split("/")[1:3] == ["lib3", "yaml"] for x in t.getmembers()):
                    continue                         # PyYAML 5: lib/ es Python 2, lib3/ es Python 3
                destino = lib / "yaml" / "/".join(partes[3:])
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(t.extractfile(m).read())


def armar_xp(dist: Path, ver: str, trabajo: Path) -> Path:
    """Windows XP SP3 (y 2000/Vista/7 en adelante, 32 o 64 bits): Python 3.4.4 oficial y el código traducido."""
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
    (site / "tisma_cnc.pth").write_text("../../../app\n", encoding="ascii")    # relativo a site-packages
    extraer_rueda(rueda(*PYSERIAL), site)
    pyyaml_puro(PYYAML_XP, site)
    for d in site.glob("*.dist-info"):
        shutil.rmtree(d)
    fuente = trabajo / "windows-xp" / "fuente"
    copiar_app(fuente)
    n = traducir_carpeta(fuente, base / "app")                # Python moderno → 3.4
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


LEEME_XP = LEEME.replace("""  * Windows Vista SP2, 7, 8, 10 u 11 (32 o 64 bits), o Linux de 2014 en adelante.""",
                         """  * Windows XP SP3 (32 o 64 bits) en adelante: esta edición trae Python 3.4.4, el último para XP.
    En Vista o más nuevo conviene la edición normal (Python 3.7, más rápida).""").replace(
    """  * Windows Vista SP2, 7 SP1 y 8: hace falta la actualización KB2999226 de Microsoft (Universal C
    Runtime), que casi todas las PCs ya tienen si pasaron por Windows Update. Si al abrir aparece
    «falta api-ms-win-crt-runtime-l1-1-0.dll», instalarla (hay versión para Vista).""",
    """  * En XP no hace falta ninguna actualización de Microsoft: Python 3.4 trae su propio msvcr100.dll.""").replace(
    "en Windows Vista,\n    Firefox 52 ESR o Supermium (Chromium actual para Vista y XP)",
    "en Windows XP y Vista,\n    Firefox 52 ESR o Supermium (Chromium actual para XP y Vista)")


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


NSI_UCRT = r"""Function .onInit
  IfFileExists "$WINDIR\System32\ucrtbase.dll" ucrt_ok 0
  IfFileExists "$WINDIR\SysWOW64\ucrtbase.dll" ucrt_ok 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "A este Windows le falta el «Universal C Runtime» (actualización KB2999226).$\r$\nTISMA CNC se instala igual, pero para abrirlo instalá esa actualización desde Windows Update."
  ucrt_ok:
FunctionEnd
"""


def armar_instalador(dist: Path, ver: str, trabajo: Path, xp: bool = False) -> Path:
    """Instalador NSIS. Con ``xp``: la edición con Python 3.4 (Windows XP SP3 en adelante, sin UCRT)."""
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
    nums = [int(x) for x in ver.split("-")[0].split(".")] + [0, 0, 0]
    script = NSI
    for k, v in {"salida": salida, "icono": ico, "origen": inst, "ver": ver, "oninit": "" if xp else NSI_UCRT,
                 "vnum": ".".join(str(n) for n in nums[:4])}.items():
        script = script.replace(f"@{k}@", str(v))
    nsi = inst.parent / "tisma_cnc.nsi"
    nsi.write_text(script, encoding="utf-8-sig")
    subprocess.run(["makensis", "-V2", str(nsi)], check=True)
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB)")
    return salida


def python_linux() -> Path:
    """Python 3.8 portable (python-build-standalone) que instala ``uv``: glibc 2.17 en adelante."""
    try:
        exe = subprocess.run(["uv", "python", "find", "3.8"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        subprocess.run(["uv", "python", "install", "3.8"], check=True)
        exe = subprocess.run(["uv", "python", "find", "3.8"], capture_output=True, text=True, check=True).stdout.strip()
    raiz = Path(exe).resolve().parents[1]
    if not (raiz / "lib" / "python3.8").is_dir():
        raise SystemExit(f"no es un Python portable: {raiz}")
    return raiz


def armar_linux(dist: Path, ver: str, trabajo: Path) -> Path:
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
        f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (base / "LEEME.txt").write_text(LEEME.format(version=ver, como="./tisma-cnc.sh  (y ./crear-acceso-directo.sh "
                                                 "para tenerlo en el menú)", datos="~/.local/share/tisma-cnc"),
                                    encoding="utf-8")
    salida = dist / f"{ARCHIVO}-{ver}-linux-x86_64.tar.gz"
    with tarfile.open(salida, "w:gz", compresslevel=9) as t:
        t.add(base, arcname="tisma-cnc")
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB)")
    return base


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


def main(argv: list[str] | None = None) -> int:
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
