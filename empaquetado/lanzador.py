"""Lanzador liviano de Hope (``Hope.exe``). Solo biblioteca estándar: torch NO va dentro del .exe.

Qué hace al hacer doble clic:

1. Busca la carpeta de Hope (la del .exe o una superior con ``hope/`` y ``configs/``).
2. Si falta el entorno ``.venv``, busca un Python 3.10+ instalado; si no hay, avisa y abre la página
   de descarga. Si hay, corre ``instalar_windows.bat`` (o ``instalar_linux.sh``) en una ventana visible,
   eligiendo la variante de PyTorch según la placa de video (CUDA 12.6 para NVIDIA, XPU para Intel Arc).
3. Arranca ``python -m hope.app`` sin consola. Si ya estaba abierta, solo abre el navegador.

``Hope.exe --comprobar`` revisa todo sin instalar ni abrir nada (lo usa el workflow de GitHub Actions)
y deja el resultado en ``hope_lanzador.log``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

WINDOWS = os.name == "nt"
PUERTO = 8765
URL = f"http://127.0.0.1:{PUERTO}"
TITULO = "Hope"


def aviso(texto: str, error: bool = False) -> None:
    if WINDOWS:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, texto, TITULO, 0x10 if error else 0x40)
    else:
        print(texto, file=sys.stderr if error else sys.stdout)


def log(base: Path, texto: str) -> None:
    try:
        with open(base / "hope_lanzador.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + texto + "\n")
    except OSError:
        pass


def carpeta_hope() -> Path | None:
    inicio = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    for d in (inicio, *inicio.parents[:3]):
        if (d / "hope" / "app" / "server.py").exists() and (d / "configs").is_dir():
            return d
    return None


def python_venv(base: Path, sin_consola: bool = False) -> Path:
    if WINDOWS:
        return base / ".venv" / "Scripts" / ("pythonw.exe" if sin_consola else "python.exe")
    return base / ".venv" / "bin" / "python"


def python_sistema() -> list[str] | None:
    """Un Python 3.10+ instalado (el lanzador «py» de Windows o python/python3 en el PATH)."""
    cands = [["py", "-3"]] if WINDOWS else []
    cands += [["python"], ["python3"]]
    for c in cands:
        if not shutil.which(c[0]):
            continue
        try:
            r = subprocess.run(c + ["-c", "import sys; print(sys.version_info >= (3, 10))"], capture_output=True,
                               text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW if WINDOWS else 0)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.stdout.strip() == "True":
            return c
    return None


def variante_torch() -> str:
    if shutil.which("nvidia-smi"):
        return "cu126"
    try:
        if WINDOWS:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                               capture_output=True, text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
            nombres = r.stdout
        else:
            nombres = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10).stdout
        if "intel" in nombres.lower() and "arc" in nombres.lower():
            return "a770"
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "cpu"


def ya_abierta() -> bool:
    try:
        with urllib.request.urlopen(URL + "/api/estado", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def instalar(base: Path, variante: str) -> bool:
    if WINDOWS:
        env = dict(os.environ, HOPE_SIN_PAUSA="1")
        p = subprocess.run(["cmd", "/c", str(base / "instalar_windows.bat"), variante], cwd=base, env=env,
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        p = subprocess.run([str(base / "instalar_linux.sh"), variante], cwd=base)
    return p.returncode == 0 and python_venv(base).exists()


def comprobar(base: Path | None) -> int:
    lineas = [f"carpeta de Hope: {base}", f"entorno: {python_venv(base) if base else '-'} "
              f"({'existe' if base and python_venv(base).exists() else 'falta'})",
              f"python del sistema: {python_sistema()}", f"variante de PyTorch: {variante_torch()}",
              f"app abierta: {ya_abierta()}"]
    destino = base or Path.cwd()
    for line in lineas:
        log(destino, "comprobar: " + line)
    print("\n".join(lineas))
    return 0 if base else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    base = carpeta_hope()
    if "--comprobar" in argv:
        return comprobar(base)
    if base is None:
        aviso("No encuentro la carpeta de Hope.\n\nPoné Hope.exe dentro de la carpeta de Hope (la que tiene "
              "las carpetas «hope» y «configs») y volvé a abrirlo.", error=True)
        return 1
    log(base, f"inicio en {base}")
    if ya_abierta():
        webbrowser.open(URL)
        return 0
    py_disco = [str(python_venv(base))] if python_venv(base).exists() else python_sistema()
    if py_disco is None:
        aviso("Hope necesita Python 3.10 o más nuevo.\n\nSe abre la página de descarga: instalalo marcando "
              "«Add python.exe to PATH» y volvé a abrir Hope.", error=True)
        webbrowser.open("https://www.python.org/downloads/")
        return 1
    if not (base / "hope_ubicacion.json").exists():
        # Elegir el disco (fuera del disco del sistema) en una ventana visible: pide confirmar con Enter.
        flags = subprocess.CREATE_NEW_CONSOLE if WINDOWS else 0
        r = subprocess.run(py_disco + ["-m", "hope.app.discos", "--elegir"], cwd=base, creationflags=flags)
        if r.returncode != 0 or not (base / "hope_ubicacion.json").exists():
            aviso("No se pudo elegir el disco para Hope. Volvé a abrirlo o usá Hope.bat.", error=True)
            return 1
    r = subprocess.run(py_disco + ["-m", "hope.app.discos", "--entorno"], cwd=base, capture_output=True, text=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if WINDOWS else 0)
    for line in r.stdout.splitlines():                # cachés y temporales en el disco elegido
        k, _, v = line.partition("=")
        if k and v:
            os.environ[k] = v
    if not python_venv(base).exists():
        py = python_sistema()
        if py is None:
            aviso("Hope necesita Python 3.10 o más nuevo.\n\nSe abre la página de descarga: instalalo marcando "
                  "«Add python.exe to PATH» y volvé a abrir Hope.", error=True)
            webbrowser.open("https://www.python.org/downloads/")
            return 1
        variante = variante_torch()
        aviso(f"Primera vez: se instala Hope con PyTorch ({variante}).\n\nSe abre una ventana con el progreso; "
              "puede tardar varios minutos (PyTorch pesa ~2,5 GB). Al terminar, Hope se abre solo.")
        log(base, f"instalando ({variante}) con {py}")
        if not instalar(base, variante):
            log(base, "la instalación falló")
            aviso("La instalación falló. Mirá los mensajes de la ventana de instalación o corré "
                  "instalar_windows.bat a mano.", error=True)
            return 1
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS) if WINDOWS else 0
    py = python_venv(base, sin_consola=True)
    if not py.exists():
        py = python_venv(base)
    salida = open(base / "runs" / "app_salida.log", "a", encoding="utf-8") if (base / "runs").exists() \
        else subprocess.DEVNULL
    subprocess.Popen([str(py), "-m", "hope.app"], cwd=base, creationflags=flags, stdout=salida,
                     stderr=subprocess.STDOUT, start_new_session=not WINDOWS)
    log(base, "aplicación lanzada")
    for _ in range(60):                       # la app abre el navegador sola; si tarda, avisamos
        if ya_abierta():
            return 0
        time.sleep(1)
    aviso(f"Hope tarda en arrancar. Si no se abre el navegador, entrá a {URL} o mirá runs/app_salida.log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
