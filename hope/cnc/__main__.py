"""Hope CNC desde la consola (docs/CNC_HOPE.md).

    python -m hope.cnc revisar  pieza.nc [--maquina configs/cnc_maquina.yaml]
    python -m hope.cnc traducir pieza.nc --dialecto grbl --salida pieza_grbl.nc
    python -m hope.cnc simular  pieza.nc
    python -m hope.cnc estado   [--maquina …]
    python -m hope.cnc enviar   pieza.nc [--aprobacion <id>]
    python -m hope.cnc app      [--maquina …] [--puerto 8766] [--sin-ventana]   (Mach3-style screen)

On a real controller, ``enviar`` (send) creates a request in the guardian the first time; it is approved
with ``python -m guardian.cli approve <id> --by <your-name>`` and run again with ``--aprobacion``.

Commands
--------
* ``revisar`` / ``simular`` (check / simulate): check report (``check``); with ``--json``, the full report.
* ``traducir`` (translate): G-code normalized for the controller (``dialect``); to the screen or ``--salida``.
* ``estado`` (status): state and position reported by the machine's controller.
* ``enviar`` (send): runs on the controller (``controller.Controller``; guardian if it is a real machine).
* ``app``: the Mach3-style screen (``app.server``) in the browser or in its own window.
* ``medir`` (measure): performance of this PC (``medir``); does not touch the machine.

Exit codes: 0 = fine; 1 = the program has errors, or the send was rejected/aborted;
2 = human approval is missing (``enviar``). Useful for scripts.

How it fits
-----------
It is only the console entry point: it builds the machine (``machine``/``profiles``) and the backend
(``backends.make_backend``) and calls the modules above. The first line of this text is the
``--help`` description (``__doc__.splitlines()[0]``), so it stays in Spanish: do not replace it with
something long.
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

# ==================================================================================================
# ========================================= CONFIGURATION ==========================================
# ==================================================================================================

# Default machine when ``--maquina`` is not given: the repository's example YAML, relative to the
# folder it is run from. If it does not exist (installed package), the machine selected in the app is used.
DEFAULT_MACHINE = "configs/cnc_maquina.yaml"

# ==================================================================================================
# ============================================ HELPERS =============================================
# ==================================================================================================


def _machine(path: str | None, datos: str | None = None):
    """The given machine; otherwise configs/cnc_maquina.yaml (repository); otherwise the one selected in
    the app (profiles and origins from the data folder), as in the installed package.

    ``path``: machine YAML or None; ``datos``: the app's data folder (None = the usual one).
    Returns a ``Machine``. An invalid YAML raises ``load_machine``'s error; an unreadable or missing
    profile selection is not an error: ``fix_selection`` picks a valid profile."""
    if path is None and Path(DEFAULT_MACHINE).exists():
        path = DEFAULT_MACHINE
    if path is not None:
        return load_machine(path)
    # Imports in here: the app pulls in more dependencies and they are only needed in this case.
    from hope.cnc.app.server import data_dir
    from hope.cnc.profiles import ProfileStore, effective, fix_selection
    user = data_dir(datos)
    store = ProfileStore(user_dir=user / "perfiles")
    try:
        sel = json.loads((user / "perfiles" / "seleccion.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sel = {}
    return effective(store, fix_selection(store, sel))[0]


def _ask(msg: str) -> bool:
    """M0 pause on the console: True = continue (Enter), False = abort («a» or end of input,
    e.g. when run without a terminal: aborting is the safe choice)."""
    try:
        return input(f"PAUSA: {msg}. Enter para seguir, «a» para abortar: ").strip().lower() != "a"
    except EOFError:
        return False


# ==================================================================================================
# ========================================== COMMAND LINE ==========================================
# ==================================================================================================


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``python -m hope.cnc``. ``argv``: arguments (None = the console's).
    Returns the exit code (see the module header)."""
    ap = argparse.ArgumentParser(prog="hope.cnc", description=__doc__.splitlines()[0])
    ap.add_argument("comando", choices=["revisar", "traducir", "simular", "estado", "enviar", "app", "medir"])
    ap.add_argument("programa", nargs="?")
    ap.add_argument("--maquina", default=None, help=f"YAML de la máquina (por defecto {DEFAULT_MACHINE})")
    ap.add_argument("--dialecto", choices=DIALECTS, default=None)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--guardian", default=None, help="configuración del guardián (por defecto configs/guardian.yaml "
                                                    "en el repositorio o una propia en la carpeta de datos)")
    ap.add_argument("--datos", default=None, help="carpeta de datos (perfiles propios, recientes, guardián)")
    ap.add_argument("--aprobacion", default=None)
    ap.add_argument("--puerto", type=int, default=8766, help="puerto local de la app")
    ap.add_argument("--sin-ventana", action="store_true", help="app: no abrir la ventana")
    ap.add_argument("--lineas", type=int, default=20000, help="medir: tamaño del programa de prueba")
    a = ap.parse_args(argv)
    if a.comando == "medir":                       # performance of this PC (does not touch the machine)
        from hope.cnc.medir import main as medir_main
        return medir_main(a.lineas)
    if a.comando == "app":
        from hope.cnc.app.server import run_app
        return run_app(a.maquina, a.puerto, window=not a.sin_ventana, guardian_path=a.guardian, datos=a.datos)
    m = _machine(a.maquina, a.datos)

    # --- commands that do not touch the controller ------------------------------------------------
    if a.comando not in ("estado",) and not a.programa:
        ap.error("falta el programa")
    # errors="replace": a badly encoded accent in a CAM comment does not prevent reading the program
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

    # --- commands that talk to the controller -----------------------------------------------------

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
            guardian = load_guardian(a.guardian or "configs/guardian.yaml")
        ctl = Controller(m, backend, guardian, dialect=a.dialecto)

        def progress(sent, acked, total):
            """Progress on a single console line («\r» rewrites it)."""
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
        backend.close()                            # release the serial port no matter what


if __name__ == "__main__":
    raise SystemExit(main())
