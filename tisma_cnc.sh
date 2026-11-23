#!/usr/bin/env bash
# TISMA CNC: abre el control CNC (pantalla estilo Mach3, docs/CNC_HOPE.md). Solo necesita Python 3.8+.
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" -c "import sys; sys.exit(sys.version_info < (3, 8))" || { echo "TISMA CNC necesita Python 3.8 o más nuevo"; exit 1; }
"$PY" -c "import yaml, serial" 2>/dev/null || "$PY" -m pip install --user pyyaml pyserial
exec "$PY" -m hope.cnc app "$@"
