@echo off
rem Instala Hope en Windows 10/11: entorno virtual .venv + PyTorch + dependencias.
rem   instalar_windows.bat          GTX 1060 / GTX 10xx (PyTorch con CUDA 12.6, incluye sm_61)
rem   instalar_windows.bat a770     Intel Arc A770 (PyTorch con XPU)
rem   instalar_windows.bat cpu      sin GPU (solo pruebas)
setlocal
cd /d "%~dp0"
set "VARIANTE=%~1"
if "%VARIANTE%"=="" set "VARIANTE=cu126"
set "INDEX=https://download.pytorch.org/whl/cu126"
if /i "%VARIANTE%"=="a770" set "INDEX=https://download.pytorch.org/whl/xpu"
if /i "%VARIANTE%"=="xpu" set "INDEX=https://download.pytorch.org/whl/xpu"
if /i "%VARIANTE%"=="cpu" set "INDEX=https://download.pytorch.org/whl/cpu"

set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo Hace falta Python 3.10 o mas nuevo: https://www.python.org/downloads/  ^(marca "Add python.exe to PATH"^)
  goto error
)

if not exist .venv\Scripts\python.exe (
  echo Creando el entorno virtual .venv ...
  %PY% -m venv .venv
  if errorlevel 1 goto error
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto error
echo Instalando PyTorch desde %INDEX% ...
"%VPY%" -m pip install --upgrade torch --index-url %INDEX%
if errorlevel 1 goto error
"%VPY%" -m pip install --upgrade numpy pyyaml pyarrow Pillow sympy pint z3-solver pytest
if errorlevel 1 goto error

echo.
"%VPY%" -c "import torch; x = getattr(torch, 'xpu', None); print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| XPU:', bool(x and x.is_available()))"
echo.
echo Listo. Para abrir la aplicacion: iniciar_hope.bat  (o: .venv\Scripts\python -m hope.app)
echo Si usas la A770, instala antes el driver de graficos de Intel mas reciente.
if not defined HOPE_SIN_PAUSA pause
exit /b 0

:error
echo.
echo La instalacion fallo. Revisa el mensaje de arriba.
if not defined HOPE_SIN_PAUSA pause
exit /b 1
