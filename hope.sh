#!/usr/bin/env bash
# Hope: abre la aplicación de entrenamiento (doble clic en el gestor de archivos o ./hope.sh).
# La primera vez instala el entorno (elige CUDA 12.6, XPU o CPU según la placa de video).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  variante=cpu
  if command -v nvidia-smi >/dev/null 2>&1; then variante=cu126
  elif command -v lspci >/dev/null 2>&1 && lspci | grep -Eiq 'VGA.*Intel.*Arc|Display.*Intel.*Arc'; then variante=a770
  fi
  echo "Primera vez: instalando Hope (PyTorch $variante). Puede tardar varios minutos."
  ./instalar_linux.sh "$variante" || { echo "La instalación falló."; read -r -p "Enter para cerrar"; exit 1; }
fi
exec .venv/bin/python -m hope.app "$@"
