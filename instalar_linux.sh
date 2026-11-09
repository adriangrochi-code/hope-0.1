#!/usr/bin/env bash
# Instala Hope en Linux: entorno virtual .venv + PyTorch + dependencias.
#   ./instalar_linux.sh          GTX 1060 / GTX 10xx (PyTorch con CUDA 12.6, incluye sm_61)
#   ./instalar_linux.sh a770     Intel Arc A770 (PyTorch con XPU)
#   ./instalar_linux.sh cpu      sin GPU (solo pruebas)
set -euo pipefail
cd "$(dirname "$0")"
variante="${1:-cu126}"
case "$variante" in
  a770|xpu) index="https://download.pytorch.org/whl/xpu" ;;
  cpu)      index="https://download.pytorch.org/whl/cpu" ;;
  *)        index="https://download.pytorch.org/whl/cu126" ;;
esac
py="${PYTHON:-python3}"
"$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "Hace falta Python 3.10 o más nuevo (y el paquete python3-venv)."; exit 1; }
[ -x .venv/bin/python ] || "$py" -m venv .venv
vpy=.venv/bin/python
"$vpy" -m pip install --upgrade pip
echo "Instalando PyTorch desde $index ..."
"$vpy" -m pip install --upgrade torch --index-url "$index"
"$vpy" -m pip install --upgrade numpy pyyaml pyarrow Pillow sympy pint z3-solver pytest
"$vpy" -c "import torch; x = getattr(torch, 'xpu', None); print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| XPU:', bool(x and x.is_available()))"
echo
echo "Listo. Para abrir la aplicación: ./iniciar_hope.sh  (o: .venv/bin/python -m hope.app)"
command -v pdftotext >/dev/null || echo "Opcional: instalá poppler-utils para leer PDF (sudo apt install poppler-utils)."
