@echo off
rem Abre la aplicacion de Hope en el navegador (http://127.0.0.1:8765).
rem Cerrar esta ventana cierra la interfaz; el entrenamiento en curso se retoma al volver a abrirla.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Falta el entorno: corre primero instalar_windows.bat
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m hope.app %*
pause
