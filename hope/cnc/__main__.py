"""Hope CNC desde la consola (docs/CNC_HOPE.md).

    python -m hope.cnc revisar  pieza.nc [--maquina configs/cnc_maquina.yaml]
    python -m hope.cnc traducir pieza.nc --dialecto grbl --salida pieza_grbl.nc
    python -m hope.cnc simular  pieza.nc
    python -m hope.cnc estado   [--maquina …]
    python -m hope.cnc enviar   pieza.nc [--aprobacion <id>]
    python -m hope.cnc app      [--maquina …] [--puerto 8766] [--sin-ventana]   (pantalla estilo Mach3)

``enviar`` en un controlador real crea una solicitud en el guardián la primera vez; se aprueba con
``python -m guardian.cli approve <id> --by <tu-nombre>`` y se vuelve a ejecutar con ``--aprobacion``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hope.cnc.backends import make_backend
from hope.cnc.check import check_text, report_text
from hope.cnc.controller import Controller
from hope.cnc.dialect import DIALECTS, emit_text
from hope.cnc.gcode import interpret
from hope.cnc.machine import load_machine

DEFAULT_MACHINE = "configs/cnc_maquina.yaml"


def _machine(path: str | None):
    if path is None and Path(DEFAULT_MACHINE).exists():
        path = DEFAULT_MACHINE
    return load_machine(path)


def _ask(msg: str) -> bool:
    try:
        return input(f"PAUSA: {msg}. Enter para seguir, «a» para abortar: ").strip().lower() != "a"
    except EOFError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hope.cnc", description=__doc__.splitlines()[0])
    ap.add_argument("comando", choices=["revisar", "traducir", "simular", "estado", "enviar", "app"])
    ap.add_argument("programa", nargs="?")
    ap.add_argument("--maquina", default=None, help=f"YAML de la máquina (por defecto {DEFAULT_MACHINE})")
    ap.add_argument("--dialecto", choices=DIALECTS, default=None)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    ap.add_argument("--aprobacion", default=None)
    ap.add_argument("--puerto", type=int, default=8766, help="puerto local de la app")
    ap.add_argument("--sin-ventana", action="store_true", help="app: no abrir la ventana")
    a = ap.parse_args(argv)
    if a.comando == "app":
        from hope.cnc.app.server import run_app
        return run_app(a.maquina, a.puerto, window=not a.sin_ventana, guardian_path=a.guardian)
    m = _machine(a.maquina)

    if a.comando not in ("estado",) and not a.programa:
        ap.error("falta el programa")
    text = Path(a.programa).read_text(encoding="utf-8", errors="replace") if a.programa else ""

    if a.comando in ("revisar", "simular"):
        r = check_text(text, m)
        print(json.dumps(r, ensure_ascii=False, indent=2) if a.json else report_text(r))
        return 0 if r["runnable"] else 1
    if a.comando == "traducir":
        prog = interpret(text, m)
        if not prog.ok:
            print("\n".join(str(d) for d in prog.errors), file=sys.stderr)
            return 1
        from hope.cnc.dialect import dialect_for
        dialect = a.dialecto or dialect_for(m)
        out = emit_text(prog, m, dialect)
        if a.salida:
            Path(a.salida).write_text(out, encoding="utf-8")
            print(f"{len(out.splitlines())} líneas → {a.salida}")
        else:
            sys.stdout.write(out)
        return 0

    backend = make_backend(m.controller, m)
    backend.connect()
    try:
        if a.comando == "estado":
            st = backend.status()
            print(f"{st.state}  máquina {st.mpos}  trabajo {st.wpos}")
            return 0
        guardian = None
        if backend.physical:
            from hope.cnc.controller import load_guardian
            guardian = load_guardian(a.guardian)
        ctl = Controller(m, backend, guardian, dialect=a.dialecto)

        def progress(sent, acked, total):
            print(f"\r{acked}/{total} líneas", end="", flush=True)

        r = ctl.run(text, approval_id=a.aprobacion, operator=_ask, progress=progress)
        print()
        if r["estado"] == "pendiente":
            print(f"Hace falta aprobación humana: python -m guardian.cli approve {r['solicitud']} --by <tu-nombre>\n"
                  f"y después: python -m hope.cnc enviar {a.programa} --aprobacion {r['solicitud']}")
            print(report_text(r["informe"]))
            return 2
        if r["estado"] != "completado":
            print(f"{r['estado']}: {r.get('motivo') or '; '.join(r.get('errores') or []) or r.get('alarma')}")
            if r["estado"] == "rechazado":
                print(report_text(r["informe"]))
            return 1
        print(f"completado: {r['confirmadas']} líneas")
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
