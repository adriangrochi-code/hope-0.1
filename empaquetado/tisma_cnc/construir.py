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

Cómo encaja con el resto
------------------------

* **Usa**: el código de ``hope/cnc`` (y lo mínimo de ``hope`` y ``guardian``) que copia ``copiar_app``;
  ``py34/a_py34.py`` (traductor a Python 3.4) y ``py34/_tisma34.py`` (capa de compatibilidad) para la
  edición XP; herramientas externas opcionales: ``pip`` (bajar ruedas), ``msiextract`` (abrir el MSI de
  Python 3.4), ``makensis`` (instaladores), ``uv`` (Python portable de Linux), ``pefile`` (revisar que los
  .dll/.exe no pidan un Windows más nuevo) y Pillow (ícono).
* **Lo usan**: las personas que arman una versión (a mano) y las pruebas
  ``tests/test_cnc_package.py`` (cargan este archivo como módulo, sin descargar nada salvo lo que ya esté
  en caché) y ``py34/preparar_pruebas.py`` (reusa ``copiar_app``).

Convenciones
------------

* Rutas con ``pathlib``; ``RAIZ`` es la raíz del repositorio.
* Cada ``armar_*`` arma en una carpeta de trabajo (``--trabajo``, por defecto ``runs/empaquetado/armado``),
  que borra y vuelve a crear, y deja el resultado comprimido en ``--destino`` (por defecto ``dist/``).
* Los textos que van a Windows (``.bat``, ``LEEME.txt``) se escriben con fin de línea CRLF: el Bloc de
  notas de XP/Vista no entiende LF solo (muestra todo en una línea).
* Cualquier problema (SHA-256 distinto, falta una herramienta, un .dll que no arranca en el Windows
  pedido) corta con ``SystemExit`` y un mensaje en castellano: mejor no armar que armar algo roto.
* Este script corre en la PC que arma (Python 3.9+ para la edición XP por ``ast.unparse``), no en la PC
  vieja: puede usar Python moderno.

Ejemplo
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
# ========================================= CONFIGURACIÓN ==========================================
# ==================================================================================================

# --- rutas y nombres ------------------------------------------------------------------------------

# Raíz del repositorio (este archivo está en empaquetado/tisma_cnc/, dos niveles abajo).
RAIZ = Path(__file__).resolve().parents[2]
# Nombre visible del programa: carpeta dentro del ZIP, accesos directos, título de la ventana.
NOMBRE = "TISMA CNC"
# Prefijo de los archivos que se generan en dist/ (sin espacios, para que no haya que poner comillas).
ARCHIVO = "TISMA-CNC"
# Caché de descargas y carpeta de trabajo por defecto. Está en runs/ (ignorado por git). Borrarla solo
# obliga a descargar todo de nuevo.
CACHE = RAIZ / "runs" / "empaquetado"

# --- Python embebido para Windows Vista en adelante -----------------------------------------------

# Descargas verificadas. Los ZIP de Python coinciden con el MD5 publicado en
# https://www.python.org/downloads/release/python-379/ (60f77740…, 97c6558d…).
#
# Por qué Python 3.7.9: es el último con instalador «embeddable» oficial que arranca en Windows Vista.
# Python 3.8 ya importa funciones de Windows 7 (GetActiveProcessorCount) y Vista muestra «no se encontró
# el punto de entrada» al abrirlo (ver FUNCIONES_WIN7 y funciones_post_vista, que lo revisan solo).
# Por qué SHA-256 fijo: python.org publica MD5, que ya no sirve contra alteraciones a propósito; se
# comprobó una vez el MD5 oficial y se anotó acá el SHA-256 del mismo archivo. Si alguien cambia el
# archivo en el camino (proxy, espejo, descarga cortada), el armado se detiene. Cambiar de versión
# obliga a actualizar URL y SHA-256 juntos (y volver a correr la revisión con pefile).
#
# Versión «corta» del Python embebido: da el nombre de python37.zip (biblioteca estándar) y
# python37._pth (rutas de búsqueda). Tiene que coincidir con la versión de DESCARGAS.
PY_WIN = "37"                      # python37.zip, python37._pth
DESCARGAS = {
    "python-x64": ("https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-amd64.zip",
                   "18627a097adf47829a847053febac5532376075243e233bd9ec61d6ea09dee1f"),
    "python-x86": ("https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-win32.zip",
                   "010cef379e1096cebf667e8f2c92a72dc455c927c7f7b8ca8e029975d81946bf"),
}

# --- paquetes de terceros (ruedas) ----------------------------------------------------------------

# Ruedas compiladas de PyYAML por sistema: (plataforma de pip, archivo, SHA-256). PyYAML 6.0.1 trae
# ruedas para CPython 3.7 (cp37), el Python embebido de Windows. La de Linux es para el Python 3.8 portable (cp38) y
# «manylinux2014» = glibc 2.17, la misma base que el Python de python_linux(). La versión de Python que
# se le pide a pip sale del nombre del archivo (ver rueda()).
RUEDAS = {   # (plataforma de pip, archivo, sha256)
    "x64": [("win_amd64", "PyYAML-6.0.1-cp37-cp37m-win_amd64.whl",
             "4fb147e7a67ef577a588a0e2c17b6db51dda102c71de36f8549b6816a96e1867")],
    "x86": [("win32", "PyYAML-6.0.1-cp37-cp37m-win32.whl",
             "9046c58c4395dff28dd494285c82ba00b546adfc7ef001486fbf0324bc174fba")],
    "linux": [("manylinux2014_x86_64", "PyYAML-6.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
               "7e07cbde391ba96ab58e532ff4803f79c4129397514e1413a7dc761ccd755735")],
}
# Fuente de PyYAML 6.0.1 (URL, SHA-256) para el paquete universal: de ahí se copia solo la parte en
# Python puro (pyyaml_puro), que anda en cualquier procesador (ARM, 32 bits…) sin compilar nada.
PYYAML_FUENTE = (("https://files.pythonhosted.org/packages/cd/e5/af35f7ea75cf72f2cd079c95ee16797de7cd71f29ea7c68ae5ce7be1eda0/"
                  "PyYAML-6.0.1.tar.gz"), "bfdf460b1736c775f2ba9f6a92bca30bc2095067b8a9d77876d1fad6cc3b4a43")

# --- edición Windows XP (Python 3.4) --------------------------------------------------------------

# Windows XP: Python 3.4.4 (el último oficial para XP SP3), instalador MSI de 32 bits que se extrae con
# msiextract (msitools). MD5 publicado en https://www.python.org/downloads/release/python-344/: e96268f7….
# Por qué el MSI y no un ZIP: Python 3.4 no tenía edición «embeddable» (apareció en 3.5, que ya no
# arranca en XP). Se usa el de 32 bits porque corre tanto en XP de 32 como de 64 bits.
PY_XP = ("https://www.python.org/ftp/python/3.4.4/python-3.4.4.msi",
         "46c8f9f63cf02987e8bf23934b2f471e1868b24748c5bb551efcf4863b43ca6c")
# PyYAML 5.1.2: la última que acepta Python 3.4 (en Python puro)
# No hay ruedas compiladas para 3.4 en Windows: se usa la fuente, en Python puro.
PYYAML_XP = (("https://files.pythonhosted.org/packages/e3/e8/b3212641ee2718d556df0f23f78de8303f068fe29cdaa7a91018849582fe/"
              "PyYAML-5.1.2.tar.gz"), "01adf0b6c6f61bd11af6e10ca52b7d4057dd0be0343eb9283c878cf3af56aee4")
# pyserial 3.5 (puerto serie hacia el controlador): rueda «py2.py3-none-any», o sea Python puro que sirve
# en todas las ediciones (Vista, XP con 3.4, Linux, universal). Mismo formato que las entradas de RUEDAS.
PYSERIAL = ("any", "pyserial-3.5-py2.py3-none-any.whl", "c4451db6ba391ca6ca299fb3ec7bae67a5c55dde170964c7a14ceefec02f2cf0")

# --- qué se copia de la aplicación ----------------------------------------------------------------

# Lo mínimo del repositorio que usa TISMA CNC (lo demás de Hope —PyTorch, entrenamiento— no va)
# Archivos sueltos (rutas relativas a RAIZ). Si hope/cnc empieza a importar otro módulo de hope o del
# guardián, hay que agregarlo acá: la prueba test_cnc_package arranca la app copiada y lo detecta.
ARCHIVOS_APP = [
    "hope/__init__.py", "hope/epistemic.py", "hope/growth/__init__.py", "hope/growth/budget.py",
    "hope/tools/__init__.py", "hope/tools/catalog.py",
    "guardian/__init__.py", "guardian/audit.py", "guardian/core.py",
]
# Carpetas que se copian enteras (con la interfaz web: HTML, JS, CSS, perfiles).
CARPETAS_APP = ["hope/cnc"]
# Nombres de carpeta o archivo que no se copian nunca (caché de Python y configuración del linter).
EXCLUIR = {"__pycache__", "ruff.toml"}

# --- poda de la biblioteca estándar ---------------------------------------------------------------

# Del Python 3.4 de XP: lo que la app no usa (Tk, pruebas, documentación, sqlite…)
# Rutas relativas a python/Lib. Achica el ZIP varias decenas de MB. Si la app empezara a usar alguno
# (por ejemplo sqlite3 o unittest), fallaría solo en XP con ImportError: sacarlo de esta lista.
# «site-packages» se poda porque viene con pip/setuptools del MSI; después se crea vacío y se llena solo
# con lo nuestro.
PODAR_XP_LIB = ["test", "idlelib", "tkinter", "turtledemo", "lib2to3", "ensurepip", "unittest/test", "distutils",
                "ctypes/test", "sqlite3", "msilib", "venv", "pydoc_data", "turtle.py", "site-packages"]
# Archivos de python/DLLs que no se copian: Tk/Tcl, sqlite, MSI, módulos de prueba de CPython e íconos.
PODAR_XP_DLL = ["_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "_sqlite3.pyd", "sqlite3.dll", "_msi.pyd", "_testcapi.pyd",
                "_testbuffer.pyd", "_testimportmultiple.pyd", "_ctypes_test.pyd", "py.ico", "pyc.ico"]
# Partes de la biblioteca estándar que la app no usa (achican el paquete de Linux)
# Rutas relativas a python/lib/python3.8 del Python portable. «config-3.8-…» trae lo necesario para
# compilar extensiones en C, que acá no se hace.
PODAR_LINUX = ["test", "idlelib", "tkinter", "turtledemo", "lib2to3/tests", "ensurepip", "unittest/test",
               "distutils/tests", "ctypes/test", "sqlite3/test", "config-3.8-x86_64-linux-gnu"]


# ==================================================================================================
# ====================================== VERSIÓN Y DESCARGAS =======================================
# ==================================================================================================

def version() -> str:
    """Versión que llevan los nombres de los paquetes: ``<versión de hope>-<AAAAMMDD>-<commit corto>``.

    La fecha es UTC (la misma en cualquier huso horario). Sin git o fuera de un repositorio, el commit
    queda como ``local``. Agrega ``RAIZ`` a ``sys.path`` para poder importar ``hope``.

    Devuelve: por ejemplo ``"0.1.0-20261123-abc1234"``.
    """
    sys.path.insert(0, str(RAIZ))
    from hope import __version__
    try:
        sha = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "local"
    return f"{__version__}-{datetime.now(timezone.utc):%Y%m%d}-{sha}"


# --- verificación y caché -------------------------------------------------------------------------

def sha256(path: Path) -> str:
    """SHA-256 (hexadecimal, minúsculas) del archivo ``path``, leído de a 1 MB para no cargarlo entero."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def descargar(url: str, esperado: str, destino: Path) -> Path:
    """Descarga ``url`` a ``destino`` salvo que ya esté en caché con el SHA-256 ``esperado``.

    Se baja primero a ``<destino>.tmp`` y se renombra recién cuando el SHA-256 coincide: así una
    descarga cortada o alterada nunca queda en caché con el nombre bueno.

    Devuelve: ``destino``.
    Errores: ``SystemExit`` si el SHA-256 no coincide; errores de red de ``urllib`` (tiempo límite 120 s).
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
    """Consigue la rueda ``nombre`` con ``pip download`` (en caché en ``runs/empaquetado/ruedas``).

    Parámetros: ``plataforma`` de pip (``win32``, ``win_amd64``, ``manylinux2014_x86_64`` o ``any`` para
    Python puro), ``nombre`` exacto del archivo ``.whl`` y su SHA-256 ``esperado``.

    Por qué pip y no una URL fija: pip sabe encontrar la rueda justa para otra plataforma y otra versión
    de Python que la de esta PC (``--platform``/``--python-version``); el SHA-256 igual garantiza que el
    archivo es el que se probó.

    Devuelve: la ruta de la rueda. Errores: ``SystemExit`` si el SHA-256 no coincide;
    ``CalledProcessError`` si pip falla (sin red, nombre mal escrito).
    """
    destino = CACHE / "ruedas" / nombre
    if destino.exists() and sha256(destino) == esperado:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    # Nombre de rueda: <paquete>-<versión>-<etiqueta de Python>-… (PEP 427), p. ej. PyYAML-6.0.1-cp37-…
    paquete, ver, tag = nombre.split("-")[:3]
    # «cp37» → «3.7»; las ruedas «py2.py3» (Python puro) no dicen versión: cualquiera sirve, se pide 3.8.
    pyver = f"{tag[2]}.{tag[3:]}" if tag.startswith("cp") else "3.8"   # la versión sale del nombre de la rueda
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--python-version", pyver,
           "-d", str(destino.parent), f"{paquete}=={ver}"]
    if plataforma != "any":
        # pip solo acepta --platform junto con --only-binary (no puede compilar para otro sistema).
        cmd[5:5] = ["--platform", plataforma]
    subprocess.run(cmd, check=True)
    if sha256(destino) != esperado:
        raise SystemExit(f"SHA-256 distinto en {nombre}")
    return destino


def extraer_rueda(whl: Path, site: Path) -> None:
    """Instala una rueda descomprimiéndola en ``site`` (una rueda es un ZIP con el paquete listo).

    Alcanza para ruedas sin scripts ni archivos de datos, como pyserial y PyYAML: no hace falta pip en
    la PC vieja (el Python embebido de Windows ni siquiera lo trae).
    """
    with zipfile.ZipFile(whl) as z:
        z.extractall(site)


# ==================================================================================================
# ======================================= APLICACIÓN E ÍCONO =======================================
# ==================================================================================================

def copiar_app(destino: Path) -> list[str]:
    """Copia el código de TISMA CNC (y lo mínimo de Hope y del guardián) a ``destino``.

    Copia ``ARCHIVOS_APP`` y las carpetas de ``CARPETAS_APP`` sin ``EXCLUIR`` ni ``.pyc``, respetando la
    estructura de carpetas del repositorio (``destino/hope/cnc/...``).

    Devuelve: las rutas copiadas, relativas y con «/», en orden fijo (``sorted``) para que dos armados
    del mismo commit den el mismo paquete. La usan también las pruebas y ``py34/preparar_pruebas.py``.
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
    """logo.ico (hexágono y pirámide violeta de TISMA) para los accesos directos.

    Dibuja el logo a 256×256 y lo guarda como ``.ico`` con todos los tamaños que usa Windows (16 a 256):
    así Windows no tiene que achicarlo y se ve nítido en el escritorio y en el menú Inicio.

    Devuelve: ``destino``, o ``None`` si no está Pillow (el paquete se arma igual, sin ícono; el
    instalador NSIS sí lo exige).
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    s = 256
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # Las coordenadas del dibujo están en un cuadro de 110 unidades (las del logo.svg original); k pasa
    # de unidades a píxeles.
    k = s / 110.0

    def pts(*xy):
        """Lista plana x1, y1, x2, y2… (en unidades del logo) → puntos en píxeles, corridos 3 px a la derecha."""
        return [(x * k + 3, y * k) for x, y in zip(xy[::2], xy[1::2])]
    # Hexágono de contorno gris claro y la pirámide: cuatro caras (gris, violeta, violeta oscuro, gris) y
    # la base. Los colores son los del logo de TISMA.
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
# ================================= TEXTOS QUE VAN EN LOS PAQUETES =================================
# ==================================================================================================

# LEEME.txt de todas las ediciones. Se completa con str.format: {version}, {como} (cómo se abre en esa
# edición) y {datos} (dónde quedan los datos). En Windows se le pasa a CRLF al escribirlo. LEEME_XP sale
# de este texto reemplazando párrafos enteros: si se cambia la redacción de los requisitos, revisar
# LEEME_XP (test_cnc_package controla que la edición XP no diga «Vista SP2, 7, 8»).
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

# Lanzador de las ediciones Windows con Python 3.7 (portable e instalador). {datos} se reemplaza con
# str.replace (no format: el BAT tiene llaves y % propios) por la carpeta de datos de cada edición.
# Qué hace, en orden: va a su propia carpeta (%~dp0, funciona desde un pendrive con cualquier letra);
# revisa que exista ucrtbase.dll (Universal C Runtime, KB2999226), sin el cual Python 3.7 no abre, y si
# falta lo explica en vez de mostrar un error críptico de DLL; fuerza UTF-8 en Python (PYTHONUTF8) porque
# la consola de Windows usa otra página de códigos; y si la app termina con error, deja la ventana
# abierta (pause) para que se pueda leer. Los mensajes van sin tildes: cmd.exe los mostraría mal.
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

# Lanzador de la edición portable de Linux. Usa solo el Python de la carpeta (no el del sistema);
# PYTHONNOUSERSITE evita que se mezclen paquetes instalados con «pip install --user». Los datos van a
# ~/.local/share/tisma-cnc (estándar XDG) salvo que se defina TISMA_CNC_DATOS.
SH_LINUX = """#!/bin/sh
# TISMA CNC portable para Linux: ./tisma-cnc.sh  (cerrar la terminal o Ctrl+C cierra el control)
D="$(cd "$(dirname "$0")" && pwd)"
export TISMA_CNC_DATOS="${TISMA_CNC_DATOS:-$HOME/.local/share/tisma-cnc}"
export PYTHONPATH="$D/app"
export PYTHONNOUSERSITE=1
exec "$D/python/bin/python3" -m hope.cnc app "$@"
"""

# Script opcional de Linux (portable y universal) que crea el acceso directo .desktop en el menú y, si
# existe, en el escritorio (en inglés o castellano). Terminal=true: la ventana de terminal muestra el
# registro y cerrarla cierra el control, igual que en Windows.
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
# ========================= REVISIÓN DE COMPATIBILIDAD CON WINDOWS VIEJOS ==========================
# ==================================================================================================

# Funciones de Windows 7 o posterior: si el Python embebido importa alguna, no arranca en Vista
# («no se encontró el punto de entrada»). Python 3.8.10 importa GetActiveProcessorCount: por eso 3.7.9.
# Es una lista de las funciones de kernel32 y compañía que aparecieron en Windows 7 y que algún Python
# o biblioteca podría importar. No es exhaustiva: es una red de seguridad para cuando se cambie de
# versión de Python. Agregar acá cualquier otra que se descubra (no hace daño que sobren).
FUNCIONES_WIN7 = {
    "GetActiveProcessorCount", "GetActiveProcessorGroupCount", "GetSystemTimePreciseAsFileTime", "WaitOnAddress",
    "WakeByAddressSingle", "PowerCreateRequest", "GetLogicalProcessorInformationEx", "CreateRemoteThreadEx",
    "GetCurrentProcessorNumberEx", "SetThreadGroupAffinity", "PathCchCanonicalizeEx", "PathCchCombineEx",
    "PathCchSkipRoot", "CreateFile2", "GetOverlappedResultEx", "TryAcquireSRWLockExclusive",
    "TryAcquireSRWLockShared", "ResolveLocaleName",
}


def funciones_post_vista(zip_python: Path) -> list[str]:
    """Funciones de Windows 7+ que importan los .exe/.dll/.pyd del Python embebido (necesita pefile;
    sin pefile devuelve [] y lo avisa).

    Lee la tabla de importaciones de cada ejecutable dentro del ZIP (sin descomprimirlo a disco) y busca
    nombres de ``FUNCIONES_WIN7``. Por qué hace falta: Windows no avisa hasta que se ejecuta; en Vista,
    un Python que importa una función de Windows 7 ni siquiera abre. Mejor enterarse al armar.

    Parámetros: ``zip_python``, el ZIP «embeddable» de python.org.
    Devuelve: ``["archivo.dll: Función", …]``; vacío si todo arranca en Vista (o si falta pefile).
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
            # fast_load + parsear solo la tabla de importaciones: mucho más rápido que analizar todo el PE.
            pe = pefile.PE(data=z.read(n), fast_load=True)
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
            # Sin importaciones no existe el atributo; i.name es None en las importadas por número (ordinal).
            for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
                malas += [f"{n}: {i.name.decode()}" for i in e.imports if i.name and i.name.decode() in FUNCIONES_WIN7]
    return malas


# Funciones de Windows Vista o posterior: si Python 3.4 importara alguna, no arrancaría en XP
# Incluye las de Windows 7 (FUNCIONES_WIN7) más las que aparecieron en Vista: candados SRW, variables
# de condición, GetTickCount64, inet_pton, etc. Mismo criterio: red de seguridad, no lista completa.
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
    mayor que XP (5.1) en el encabezado. Necesita pefile.

    Revisa dos cosas, porque cualquiera de las dos impide arrancar en XP:

    * la versión mínima de subsistema del encabezado PE (``MajorSubsystemVersion.Minor``): si es mayor
      que 5.1 (XP), XP dice «no es una aplicación Win32 válida» aunque no use ninguna función nueva;
    * la tabla de importaciones, contra ``FUNCIONES_VISTA``.

    Parámetros: ``carpeta`` con el Python ya podado (se recorre entera).
    Devuelve: ``["archivo: motivo", …]``; vacío si todo arranca en XP (o si falta pefile).
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
        pe.close()   # suelta el archivo (pefile lo deja mapeado en memoria)
    return malas


# ==================================================================================================
# ======================== EDICIÓN WINDOWS (VISTA EN ADELANTE, PYTHON 3.7) =========================
# ==================================================================================================

def pth_embebido() -> str:
    """python37._pth: el Python embebido solo ve su biblioteca, la app y sus paquetes (nada del sistema).

    Con un archivo ``._pth`` al lado de ``python.exe``, Python ignora PYTHONPATH y el registro de
    Windows: así otro Python instalado en la PC no se mezcla. Una ruta por línea, relativas a la carpeta
    ``python``:

    * ``python37.zip``: la biblioteca estándar (viene comprimida en el ZIP embebido);
    * ``.``: los módulos en C (``.pyd``) que están junto a ``python.exe``;
    * ``..\\app``: el código de TISMA CNC;
    * ``Lib\\site-packages``: pyserial y PyYAML;
    * ``import site``: activa el módulo ``site`` (por defecto el embebido lo trae apagado).
    """
    return f"python{PY_WIN}.zip\n.\n..\\app\nLib\\site-packages\nimport site\n"


def armar_windows(arq: str, dist: Path, ver: str, trabajo: Path) -> Path:
    """Arma el ZIP portable de Windows con Python 3.7.9 embebido.

    Parámetros: ``arq`` = ``"x86"`` (32 bits; sirve también en Windows de 64) o ``"x64"``; ``dist``,
    carpeta de salida; ``ver``, versión (ver ``version()``); ``trabajo``, carpeta de armado.

    Pasos: descarga y verifica Python, revisa con pefile que arranque en Vista, lo descomprime, escribe
    el ``._pth``, instala las ruedas, copia la app, escribe el BAT y el LEEME (CRLF), genera el ícono y
    comprime todo en ``dist/TISMA-CNC-<ver>-windows-<arq>.zip`` (con la carpeta «TISMA CNC» adentro).

    Devuelve: la carpeta armada (``trabajo/windows-<arq>/TISMA CNC``), que reusa ``armar_instalador``.
    Errores: ``SystemExit`` si el Python no arranca en Vista o falla una verificación de SHA-256.
    """
    url, sha = DESCARGAS[f"python-{arq}"]
    zpy = descargar(url, sha, CACHE / Path(url).name)
    malas = funciones_post_vista(zpy)
    if malas:
        raise SystemExit("este Python no arranca en Windows Vista: " + ", ".join(malas))
    base = trabajo / f"windows-{arq}" / NOMBRE
    if base.parent.exists():
        shutil.rmtree(base.parent)   # se arma siempre de cero: nada de restos de un armado anterior
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
    # %~dp0datos: los datos quedan al lado del BAT (portable de verdad: se llevan con la carpeta).
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
                # Rutas relativas a base.parent: el ZIP trae la carpeta «TISMA CNC/» y no ensucia al descomprimir.
                z.write(f, f.relative_to(base.parent).as_posix())
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB){'' if ico else '  (sin ícono: falta Pillow)'}")
    return base


# ==================================================================================================
# ================================ EDICIÓN WINDOWS XP (PYTHON 3.4) =================================
# ==================================================================================================

# Lanzador de la edición XP (portable e instalador-xp). Igual que BAT_PORTABLE pero sin revisar el UCRT
# (Python 3.4 trae su propio msvcr100.dll) ni PYTHONUTF8 (no existe en 3.4: la consola la arregla
# _tisma34). -E ignora las variables PYTHON* del sistema y -s el site-packages del usuario, para que un
# Python 2 o 3 instalado en la PC no se mezcle. Se escribe en ASCII: cualquier tilde haría fallar el
# armado (y cmd.exe de XP la mostraría mal).
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
    """Extrae los archivos del instalador MSI de Python 3.4 a ``destino`` con ``msiextract`` (msitools).

    Así se consigue el Python de XP sin ejecutar el instalador (ni tener Windows). ``destino`` no debe
    existir. Errores: ``SystemExit`` si falta msiextract; ``CalledProcessError`` si falla.
    """
    if not shutil.which("msiextract"):
        raise SystemExit("falta msiextract para abrir el instalador de Python 3.4: apt install msitools")
    destino.mkdir(parents=True)
    subprocess.run(["msiextract", "-C", str(destino), str(msi)], check=True, stdout=subprocess.DEVNULL)


def pyyaml_puro(url_sha: tuple[str, str], lib: Path) -> None:
    """Copia el paquete ``yaml`` en Python puro (lib3/yaml de la fuente) a ``lib``.

    Parámetros: ``url_sha`` = (URL, SHA-256) del ``.tar.gz`` de PyYAML (``PYYAML_FUENTE`` o
    ``PYYAML_XP``); ``lib``, carpeta donde queda ``lib/yaml/``.

    Por qué la fuente y no una rueda: PyYAML no publica ruedas en Python puro y la extensión en C
    (``_yaml``) habría que compilarla para cada procesador. Sin ella PyYAML usa su cargador en Python,
    más lento pero de sobra para los perfiles de TISMA CNC.

    Dentro del tar las rutas son ``PyYAML-x.y.z/lib/yaml/…`` (PyYAML 6) o ``…/lib3/yaml/…`` (PyYAML 5,
    que además trae ``lib/`` para Python 2).
    """
    url, sha = url_sha
    fuente = descargar(url, sha, CACHE / "ruedas" / Path(url).name)
    with tarfile.open(fuente) as t:                  # sin compilar: cualquier procesador
        for m in t.getmembers():
            partes = m.name.split("/")
            if len(partes) > 3 and partes[1:3] in (["lib", "yaml"], ["lib3", "yaml"]) and m.isfile():
                if partes[1] == "lib" and any(x.name.split("/")[1:3] == ["lib3", "yaml"] for x in t.getmembers()):
                    continue                         # PyYAML 5: lib/ es Python 2, lib3/ es Python 3
                # Se escribe a mano (no extractall): solo entra lo que está bajo yaml/, con rutas armadas acá.
                destino = lib / "yaml" / "/".join(partes[3:])
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(t.extractfile(m).read())


def armar_xp(dist: Path, ver: str, trabajo: Path) -> Path:
    """Windows XP SP3 (y 2000/Vista/7 en adelante, 32 o 64 bits): Python 3.4.4 oficial y el código traducido.

    Pasos: extrae el MSI de Python 3.4.4, copia solo lo necesario (``PODAR_XP_*``), revisa con pefile
    que nada pida un Windows más nuevo que XP, arma un ``site-packages`` limpio con pyserial y PyYAML 5.1.2
    en Python puro, copia la app a una carpeta temporal y la traduce a Python 3.4 (``py34/a_py34.py``),
    agrega ``_tisma34.py`` sin traducir (ya está escrito para 3.4), escribe BAT (ASCII) y LEEME_XP (CRLF)
    y comprime en ``dist/TISMA-CNC-<ver>-windows-xp.zip``.

    Por qué ``tisma_cnc.pth`` y no un ``._pth`` como en Vista: el ``._pth`` apareció en Python 3.6.

    Devuelve: la carpeta armada (``trabajo/windows-xp/TISMA CNC``).
    Errores: ``SystemExit`` si este script corre en Python < 3.9, falta msiextract, algo no arranca en
    XP o falla un SHA-256; ``NoTraducible`` (de ``a_py34``) si el código usa algo que 3.4 no tiene.
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
    # Del MSI se toma solo lo que hace falta para correr (sin Scripts/, Tools/, Doc/, include/, libs/, tcl/).
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
    # python/Lib/site-packages → ../../../ = la carpeta «TISMA CNC» → app/ (ahí está el código traducido).
    (site / "tisma_cnc.pth").write_text("../../../app\n", encoding="ascii")    # relativo a site-packages
    extraer_rueda(rueda(*PYSERIAL), site)
    pyyaml_puro(PYYAML_XP, site)
    for d in site.glob("*.dist-info"):
        shutil.rmtree(d)   # metadatos de pip: sin pip en la PC no sirven para nada
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


# LEEME de la edición XP: el LEEME general con tres cambios (requisitos de sistema, sin la
# actualización KB2999226 y el navegador recomendado para XP). Ojo: str.replace no avisa si no encuentra
# el texto; si se edita LEEME, estos fragmentos tienen que seguir coincidiendo letra por letra.
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
# ====================================== INSTALADORES (NSIS) =======================================
# ==================================================================================================

# Guion de NSIS para los dos instaladores. Las marcas son @nombre@ (no {nombre}) porque NSIS usa ${…}
# para sus propias constantes. Las completa armar_instalador:
#   @salida@  ruta del .exe que genera        @icono@  logo.ico          @origen@  carpeta a empaquetar
#   @ver@     versión en texto                @vnum@   versión X.Y.Z.W   @oninit@  función .onInit (o nada)
# Decisiones: «Unicode true» (NSIS 3) para las tildes; RequestExecutionLevel user + instalar en
# %LOCALAPPDATA%\Programs: no pide administrador (muchas PCs de taller no tienen la clave); el
# desinstalador se registra en HKCU (Agregar o quitar programas del usuario); compresión LZMA sólida (el
# instalador más chico); los accesos directos abren el BAT minimizado (SW_SHOWMINIMIZED) para que la
# consola no tape el navegador; al desinstalar no se borran los datos de %APPDATA%\TISMA CNC.
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


# Función .onInit del instalador normal (Python 3.7): avisa al empezar si falta el Universal C Runtime
# (KB2999226). No impide instalar: se puede instalar la actualización después. El instalador XP no la
# lleva (Python 3.4 no usa el UCRT).
NSI_UCRT = r"""Function .onInit
  IfFileExists "$WINDIR\System32\ucrtbase.dll" ucrt_ok 0
  IfFileExists "$WINDIR\SysWOW64\ucrtbase.dll" ucrt_ok 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "A este Windows le falta el «Universal C Runtime» (actualización KB2999226).$\r$\nTISMA CNC se instala igual, pero para abrirlo instalá esa actualización desde Windows Update."
  ucrt_ok:
FunctionEnd
"""


def armar_instalador(dist: Path, ver: str, trabajo: Path, xp: bool = False) -> Path:
    """Instalador NSIS. Con ``xp``: la edición con Python 3.4 (Windows XP SP3 en adelante, sin UCRT).

    Parte de la carpeta ya armada por ``armar_windows("x86")`` (32 bits: sirve en Windows de 32 y 64) o
    ``armar_xp``; si no existe la arma. La copia, cambia el BAT y el LEEME para que los datos vayan a
    ``%APPDATA%\\TISMA CNC`` (la carpeta de instalación puede no ser del usuario para siempre), completa
    el guion ``NSI`` y corre ``makensis``.

    Devuelve: la ruta del ``.exe``. Errores: ``SystemExit`` si falta makensis o el ícono (Pillow);
    ``CalledProcessError`` si makensis falla.
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
    # VIProductVersion exige cuatro números «X.Y.Z.W»: de «0.1.0-20261123-abc» queda «0.1.0.0» (se
    # completa con ceros y se corta en cuatro).
    nums = [int(x) for x in ver.split("-")[0].split(".")] + [0, 0, 0]
    script = NSI
    for k, v in {"salida": salida, "icono": ico, "origen": inst, "ver": ver, "oninit": "" if xp else NSI_UCRT,
                 "vnum": ".".join(str(n) for n in nums[:4])}.items():
        script = script.replace(f"@{k}@", str(v))
    nsi = inst.parent / "tisma_cnc.nsi"
    # UTF-8 con BOM: es la única forma en que makensis reconoce el guion como UTF-8 (tildes del LEEME y mensajes).
    nsi.write_text(script, encoding="utf-8-sig")
    subprocess.run(["makensis", "-V2", str(nsi)], check=True)   # -V2: solo avisos y errores
    print(f"{salida}  ({salida.stat().st_size / 1e6:.1f} MB)")
    return salida


# ==================================================================================================
# ============================== EDICIÓN LINUX PORTABLE (PYTHON 3.8) ===============================
# ==================================================================================================

def python_linux() -> Path:
    """Python 3.8 portable (python-build-standalone) que instala ``uv``: glibc 2.17 en adelante.

    Por qué ese: python-build-standalone se compila contra glibc 2.17 (CentOS 7, 2014), así que la
    carpeta copiada arranca en casi cualquier Linux x86_64 de esa época en adelante, sin instalar nada.
    Si ``uv`` no lo tiene, lo instala.

    Devuelve: la raíz del Python (la carpeta que tiene ``bin/`` y ``lib/python3.8``).
    Errores: ``SystemExit`` si lo que encontró uv no es una instalación portable completa;
    ``OSError``/``CalledProcessError`` si no hay ``uv``.
    """
    try:
        exe = subprocess.run(["uv", "python", "find", "3.8"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        subprocess.run(["uv", "python", "install", "3.8"], check=True)
        exe = subprocess.run(["uv", "python", "find", "3.8"], capture_output=True, text=True, check=True).stdout.strip()
    raiz = Path(exe).resolve().parents[1]   # <raíz>/bin/python3.8 → <raíz>
    if not (raiz / "lib" / "python3.8").is_dir():
        raise SystemExit(f"no es un Python portable: {raiz}")
    return raiz


def armar_linux(dist: Path, ver: str, trabajo: Path) -> Path:
    """Arma ``dist/TISMA-CNC-<ver>-linux-x86_64.tar.gz``: Python 3.8 portable, podado, con la app.

    Copia el Python de ``python_linux()`` manteniendo los enlaces simbólicos (``bin/python3`` →
    ``python3.8``), poda ``PODAR_LINUX`` y los encabezados de C, instala PyYAML (rueda manylinux2014) y
    pyserial, copia la app y escribe los dos scripts (ejecutables) y el LEEME (con LF: es Linux).

    Devuelve: la carpeta armada (``trabajo/linux/tisma-cnc``).
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
        # Permiso de ejecución para todos (como chmod a+x); el tar.gz lo conserva.
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
# ============================= EDICIÓN UNIVERSAL (PYTHON DEL SISTEMA) =============================
# ==================================================================================================

# Lanzador del paquete universal: busca un Python 3.7 o más nuevo entre los nombres habituales (primero
# «python3», después versiones concretas de la más nueva a la más vieja) y arranca con el primero que
# sirva. app/ tiene el código y lib/ pyserial y PyYAML en Python puro. Si no hay ninguno, explica cómo
# instalarlo.
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
    """Arma ``dist/TISMA-CNC-<ver>-universal.tar.gz``: solo código, sin Python (≈ 1 MB).

    Para lo que no cubren las otras ediciones: Linux de 32 bits, ARM (Raspberry Pi), BSD, macOS. Todo
    lo que trae es Python puro (la app, pyserial y el ``yaml`` de la fuente de PyYAML 6.0.1), así que
    anda en cualquier procesador con Python 3.7+.

    Devuelve: la carpeta armada (``trabajo/universal/tisma-cnc``).
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
# ======================================= LÍNEA DE COMANDOS ========================================
# ==================================================================================================

def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: lee las opciones y arma cada edición pedida.

    Sin ``--solo`` arma todo lo que se puede en esta PC: siempre Windows x86/x64, Linux y universal;
    la edición XP solo si hay ``msiextract``; los instaladores solo si hay ``makensis`` (y el XP, además,
    ``msiextract``). Así ``construir.py`` sin opciones no falla por una herramienta que falta.

    Parámetros: ``argv``, los argumentos (sin el nombre del programa); ``None`` = ``sys.argv[1:]``.
    Devuelve: 0 (los errores cortan antes con ``SystemExit``).
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
    # El orden importa: los instaladores van después de las ediciones de las que parten (y las reusan).
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
