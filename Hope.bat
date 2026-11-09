@echo off
rem Hope: doble clic para abrir la aplicacion de entrenamiento.
rem La primera vez instala el entorno (Python + PyTorch); despues abre http://127.0.0.1:8765.
setlocal
cd /d "%~dp0"
title Hope
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
