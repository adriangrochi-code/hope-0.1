#!/usr/bin/env bash
# Abre la aplicación de Hope (http://127.0.0.1:8765). El entrenamiento sigue aunque se cierre la interfaz.
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Falta el entorno: corré primero ./instalar_linux.sh"; exit 1; }
exec .venv/bin/python -m hope.app "$@"
