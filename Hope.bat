@echo off
rem Hope: doble clic para abrir la aplicacion de entrenamiento.
rem 1) La primera vez elige un disco que NO sea el del sistema (pide confirmar con Enter) y pone ahi todo:
rem    entorno de Python, datos, checkpoints, caches y temporales.
rem 2) Instala lo que falta (Python + PyTorch segun la placa de video).
rem 3) Abre http://127.0.0.1:8765.
setlocal
cd /d "%~dp0"
title Hope

set "PY=python"
where py >nul 2>nul && set "PY=py -3"
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"

if not exist hope_ubicacion.json (
  echo Hope necesita un disco con espacio para los datos y los checkpoints.
  %PY% -m hope.app.discos --elegir
  if errorlevel 1 (
    echo.
    echo No se pudo elegir el disco. ^(Hace falta Python 3.10 o mas nuevo: https://www.python.org/downloads/^)
    pause
    exit /b 1
  )
)
rem Caches de pip, Hugging Face y PyTorch y temporales: en el disco elegido.
for /f "usebackq tokens=1,* delims==" %%A in (`%PY% -m hope.app.discos --entorno`) do set "%%A=%%B"

if not exist .venv\Scripts\python.exe (
  echo Primera vez: instalando Hope. Puede tardar varios minutos ^(PyTorch pesa ~2,5 GB^).
  set "VARIANTE=cpu"
  where nvidia-smi >nul 2>nul && set "VARIANTE=cu126"
  powershell -NoProfile -Command "if (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Intel.*Arc' }) { exit 3 }" >nul 2>nul
  if errorlevel 3 if not errorlevel 4 set "VARIANTE=a770"
  call :instalar
  if errorlevel 1 (
    echo.
    echo La instalacion fallo. Revisa los mensajes de arriba.
    pause
    exit /b 1
  )
)
echo Abriendo Hope en el navegador... ^(cerrar esta ventana cierra la interfaz; el entrenamiento sigue^)
.venv\Scripts\python.exe -m hope.app %*
exit /b %errorlevel%

:instalar
echo Variante de PyTorch: %VARIANTE%
set "HOPE_SIN_PAUSA=1"
call instalar_windows.bat %VARIANTE%
exit /b %errorlevel%
