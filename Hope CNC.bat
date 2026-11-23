@echo off
rem Hope CNC: doble clic para abrir el control CNC (pantalla estilo Mach3, docs/CNC_HOPE.md).
rem No necesita PyTorch ni placa de video: alcanza con Python 3.10+ (sirve en PCs viejas).
setlocal
cd /d "%~dp0"
title Hope CNC
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
%PY% -c "import yaml, serial" >nul 2>nul
if errorlevel 1 (
  echo Primera vez: instalando pyyaml y pyserial...
  %PY% -m pip install --user pyyaml pyserial
  if errorlevel 1 (
    echo No se pudo instalar. Hace falta Python 3.10 o mas nuevo: https://www.python.org/downloads/
    pause
    exit /b 1
  )
)
echo Abriendo Hope CNC... ^(cerrar esta ventana cierra el control^)
%PY% -m hope.cnc app %*
if errorlevel 1 pause
