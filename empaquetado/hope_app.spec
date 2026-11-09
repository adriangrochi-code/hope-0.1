# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: lanzador liviano Hope.exe (solo biblioteca estándar; torch NO va adentro).
#   pip install pyinstaller
#   python -m PyInstaller --noconfirm --clean empaquetado/hope_app.spec
# Resultado: dist/Hope.exe (en Linux, dist/Hope). Va dentro de la carpeta de Hope, junto a hope/ y configs/.
import os

aqui = SPECPATH

a = Analysis(
    [os.path.join(aqui, "lanzador.py")],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    # Nada pesado: el lanzador solo busca Python, instala el entorno y arranca la app.
    excludes=["torch", "numpy", "yaml", "pyarrow", "tkinter", "unittest", "pydoc", "test"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Hope",
    debug=False,
    strip=False,
    upx=False,
    console=False,                     # sin ventana de consola
    icon=os.path.join(aqui, "hope.ico"),
)
