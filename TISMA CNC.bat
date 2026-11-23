@echo off
rem TISMA CNC: doble clic para abrir el control CNC (pantalla estilo Mach3, docs/CNC_HOPE.md).
rem No necesita PyTorch ni placa de video: alcanza con Python 3.8+ (Windows 7 en adelante, sirve en PCs viejas).
setlocal
cd /d "%~dp0"
title TISMA CNC
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
%PY% -c "import sys; sys.exit(sys.version_info < (3, 8))" >nul 2>nul
if errorlevel 1 (
  echo TISMA CNC necesita Python 3.8 o mas nuevo.
  echo En Windows 7 instala Python 3.8.10 ^(el ultimo para Windows 7^): https://www.python.org/downloads/release/python-3810/
  echo Marca "Add Python to PATH" al instalar.
  pause
  exit /b 1
)
%PY% -c "import yaml, serial" >nul 2>nul
if errorlevel 1 (
  echo Primera vez: instalando pyyaml y pyserial...
  %PY% -m pip install --user pyyaml pyserial
  if errorlevel 1 (
    echo No se pudo instalar pyyaml/pyserial: se abre igual con el simulador. Para una maquina real hacen falta.
    pause
  )
)
echo Abriendo TISMA CNC... ^(cerrar esta ventana cierra el control^)
%PY% -m hope.cnc app %*
if errorlevel 1 pause
