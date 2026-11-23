#!/usr/bin/env bash
# TISMA CNC: abre el control CNC (pantalla estilo Mach3, docs/CNC_HOPE.md). Solo necesita Python 3.10+.
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" -c "import yaml, serial" 2>/dev/null || "$PY" -m pip install --user pyyaml pyserial
exec "$PY" -m hope.cnc app "$@"
