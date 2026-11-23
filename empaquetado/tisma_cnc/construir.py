"""Arma TISMA CNC listo para PCs viejas, sin instalar nada a mano.

    python empaquetado/tisma_cnc/construir.py                 # todo lo que se pueda armar acá
    python empaquetado/tisma_cnc/construir.py --solo windows-x86 linux
    python empaquetado/tisma_cnc/construir.py --solo instalador   # necesita makensis (NSIS)

Resultado en ``dist/``:

* ``TISMA-CNC-<versión>-windows-x86.zip`` y ``…-windows-x64.zip``: portables. Se descomprimen en
  cualquier carpeta (o pendrive) y se abre ``TISMA CNC.bat``. Traen **Python 3.8.10 embebido**
  (el último para Windows 7), pyserial y PyYAML. Los datos quedan en ``datos/`` al lado.
* ``TISMA-CNC-<versión>-instalador.exe``: instalador (NSIS, 32 bits: sirve en Windows 7, 8, 10 y 11
  de 32 o 64 bits) sin permisos de administrador; accesos directos en el escritorio y el menú Inicio;
  datos en ``%APPDATA%\\TISMA CNC`` (no se borran al desinstalar).
* ``TISMA-CNC-<versión>-linux-x86_64.tar.gz``: portable para Linux desde 2014 (glibc 2.17), con su
  propio Python 3.8 (python-build-standalone).

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
# https://www.python.org/downloads/release/python-3810/ (3acb1d7d…, 659adf42…).
DESCARGAS = {
    "python-x64": ("https://www.python.org/ftp/python/3.8.10/python-3.8.10-embed-amd64.zip",
                   "abbe314e9b41603dde0a823b76f5bbbe17b3de3e5ac4ef06b759da5466711271"),
    "python-x86": ("https://www.python.org/ftp/python/3.8.10/python-3.8.10-embed-win32.zip",
                   "760dc79bcb434ee80b1001a30bb6f798287881851bac6d8137867894d40ef1fc"),
}
RUEDAS = {   # (plataforma de pip, archivo, sha256)
    "x64": [("win_amd64", "PyYAML-6.0.1-cp38-cp38-win_amd64.whl",
             "1e2722cc9fbb45d9b87631ac70924c11d3a401b2d7f410cc0e3bbf249f2dca62")],
    "x86": [("win32", "PyYAML-6.0.1-cp38-cp38-win32.whl",
             "184c5108a2aca3c5b3d3bf9395d50893a7ab82a38004c8f61c258d4428e80206")],
    "linux": [("manylinux2014_x86_64", "PyYAML-6.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
               "7e07cbde391ba96ab58e532ff4803f79c4129397514e1413a7dc761ccd755735")],
}
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
    paquete, ver = nombre.split("-")[:2]
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--python-version", "3.8",
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
  * Windows 7 SP1, 8, 10 u 11 (32 o 64 bits), o Linux de 2014 en adelante.
  * Un navegador de 2017 en adelante: Firefox ESR, Chrome o Edge. Internet Explorer NO sirve
    (la pantalla avisa qué instalar).
  * Windows 7 SP1 y 8: hacen falta dos actualizaciones de Microsoft que casi todas las PCs ya tienen
    si pasaron por Windows Update: KB2533623 y KB2999226 (Universal C Runtime). Si al abrir aparece
    «falta api-ms-win-crt-runtime-l1-1-0.dll» o «no se encontró el punto de entrada», instalarlas.

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


def pth_embebido() -> str:
    """python38._pth: el Python embebido solo ve su biblioteca, la app y sus paquetes (nada del sistema)."""
    return "python38.zip\n.\n..\\app\nLib\\site-packages\nimport site\n"


def armar_windows(arq: str, dist: Path, ver: str, trabajo: Path) -> Path:
    url, sha = DESCARGAS[f"python-{arq}"]
    zpy = descargar(url, sha, CACHE / Path(url).name)
    base = trabajo / f"windows-{arq}" / NOMBRE
    if base.parent.exists():
        shutil.rmtree(base.parent)
    py = base / "python"
    py.mkdir(parents=True)
    with zipfile.ZipFile(zpy) as z:
        z.extractall(py)
    (py / "python38._pth").write_text(pth_embebido(), encoding="ascii")
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

Function .onInit
  IfFileExists "$WINDIR\System32\ucrtbase.dll" ucrt_ok 0
  IfFileExists "$WINDIR\SysWOW64\ucrtbase.dll" ucrt_ok 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "A este Windows le falta el «Universal C Runtime» (actualización KB2999226).$\r$\nTISMA CNC se instala igual, pero para abrirlo instalá esa actualización desde Windows Update."
  ucrt_ok:
FunctionEnd

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


def armar_instalador(dist: Path, ver: str, trabajo: Path) -> Path:
    if not shutil.which("makensis"):
        raise SystemExit("falta makensis (NSIS): apt install nsis, o en Windows https://nsis.sourceforge.io")
    base = trabajo / "windows-x86" / NOMBRE
    if not base.exists():
        armar_windows("x86", dist, ver, trabajo)
    inst = trabajo / "instalador" / NOMBRE
    if inst.parent.exists():
        shutil.rmtree(inst.parent)
    shutil.copytree(base, inst)
    (inst / "TISMA CNC.bat").write_text(BAT_PORTABLE.replace("{datos}", r"%APPDATA%\TISMA CNC").replace("\n", "\r\n"),
                                        encoding="utf-8")
    (inst / "LEEME.txt").write_text(LEEME.format(version=ver, como="Acceso directo «TISMA CNC» en el escritorio o el "
                                                 "menú Inicio.", datos=r"%APPDATA%\TISMA CNC")
                                    .replace("\n", "\r\n"), encoding="utf-8")
    ico = inst / "logo.ico"
    if not ico.exists():
        raise SystemExit("falta el ícono (instalar Pillow para generarlo)")
    salida = dist / f"{ARCHIVO}-{ver}-instalador.exe"
    nums = [int(x) for x in ver.split("-")[0].split(".")] + [0, 0, 0]
    script = NSI
    for k, v in {"salida": salida, "icono": ico, "origen": inst, "ver": ver,
                 "vnum": ".".join(str(n) for n in nums[:4])}.items():
        script = script.replace(f"@{k}@", str(v))
    nsi = trabajo / "instalador" / "tisma_cnc.nsi"
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Arma TISMA CNC para Windows y Linux (PCs viejas)")
    ap.add_argument("--solo", nargs="*", choices=["windows-x86", "windows-x64", "instalador", "linux"],
                    help="qué armar (por defecto: todo lo posible)")
    ap.add_argument("--destino", default=str(RAIZ / "dist"))
    ap.add_argument("--trabajo", default=str(CACHE / "armado"))
    a = ap.parse_args(argv)
    dist, trabajo = Path(a.destino), Path(a.trabajo)
    dist.mkdir(parents=True, exist_ok=True)
    ver = version()
    que = a.solo or ["windows-x86", "windows-x64", "linux"] + (["instalador"] if shutil.which("makensis") else [])
    for q in que:
        if q.startswith("windows"):
            armar_windows(q.split("-")[1], dist, ver, trabajo)
        elif q == "instalador":
            armar_instalador(dist, ver, trabajo)
        elif q == "linux":
            armar_linux(dist, ver, trabajo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
