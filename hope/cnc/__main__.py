"""Hope CNC desde la consola (docs/CNC_HOPE.md).

    python -m hope.cnc revisar  pieza.nc [--maquina configs/cnc_maquina.yaml]
    python -m hope.cnc traducir pieza.nc --dialecto grbl --salida pieza_grbl.nc
    python -m hope.cnc simular  pieza.nc
    python -m hope.cnc estado   [--maquina …]
    python -m hope.cnc enviar   pieza.nc [--aprobacion <id>]
    python -m hope.cnc app      [--maquina …] [--puerto 8766] [--sin-ventana]   (pantalla estilo Mach3)

``enviar`` en un controlador real crea una solicitud en el guardián la primera vez; se aprueba con
``python -m guardian.cli approve <id> --by <tu-nombre>`` y se vuelve a ejecutar con ``--aprobacion``.

Comandos
--------
* ``revisar`` / ``simular``: informe de la revisión (``check``); con ``--json``, el informe completo.
* ``traducir``: G-code normalizado para el controlador (``dialect``); a la pantalla o a ``--salida``.
* ``estado``: estado y posición que informa el controlador de la máquina.
* ``enviar``: ejecuta en el controlador (``controller.Controller``; guardián si es una máquina real).
* ``app``: la pantalla estilo Mach3 (``app.server``) en el navegador o en su ventana.
* ``medir``: rendimiento de esta PC (``medir``); no toca la máquina.

Códigos de salida: 0 = bien; 1 = el programa tiene errores, o el envío fue rechazado/abortado;
2 = falta la aprobación humana (``enviar``). Útil para scripts.

Cómo encaja
-----------
Es solo la puerta de entrada de la consola: arma la máquina (``machine``/``profiles``) y el backend
(``backends.make_backend``) y llama a los módulos de arriba. La primera línea de este texto es la
descripción de ``--help`` (``__doc__.splitlines()[0]``): no cambiarla por algo largo.
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
# ========================================= CONFIGURACIÓN ==========================================
# ==================================================================================================

# Máquina por defecto si no se pasa ``--maquina``: el YAML de ejemplo del repositorio, relativo a la
# carpeta desde donde se corre. Si no existe (paquete instalado), se usa la máquina elegida en la app.
DEFAULT_MACHINE = "configs/cnc_maquina.yaml"

# ==================================================================================================
# =========================================== AYUDANTES ============================================
# ==================================================================================================


def _machine(path: str | None, datos: str | None = None):
    """La máquina indicada; si no, configs/cnc_maquina.yaml (repositorio); si no, la que está elegida en
    la app (perfiles y orígenes de la carpeta de datos), como en el paquete instalado.

    ``path``: YAML de la máquina o None; ``datos``: carpeta de datos de la app (None = la de siempre).
    Devuelve una ``Machine``. Un YAML inválido lanza el error de ``load_machine``; una selección de
    perfil ilegible o ausente no es error: ``fix_selection`` elige un perfil válido."""
    if path is None and Path(DEFAULT_MACHINE).exists():
        path = DEFAULT_MACHINE
    if path is not None:
        return load_machine(path)
    # Importaciones acá adentro: la app trae más dependencias y solo hacen falta en este caso.
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
    """Pausa M0 en la consola: True = seguir (Enter), False = abortar («a» o fin de la entrada,
    p. ej. si se corre sin terminal: abortar es lo seguro)."""
    try:
        return input(f"PAUSA: {msg}. Enter para seguir, «a» para abortar: ").strip().lower() != "a"
    except EOFError:
        return False


# ==================================================================================================
# ======================================== LÍNEA DE ÓRDENES ========================================
# ==================================================================================================


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de ``python -m hope.cnc``. ``argv``: argumentos (None = los de la consola).
    Devuelve el código de salida (ver el encabezado del módulo)."""
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
    if a.comando == "medir":                       # rendimiento de esta PC (no toca la máquina)
        from hope.cnc.medir import main as medir_main
        return medir_main(a.lineas)
    if a.comando == "app":
        from hope.cnc.app.server import run_app
        return run_app(a.maquina, a.puerto, window=not a.sin_ventana, guardian_path=a.guardian, datos=a.datos)
    m = _machine(a.maquina, a.datos)

    # --- comandos que no tocan el controlador ------------------------------------------------------
    if a.comando not in ("estado",) and not a.programa:
        ap.error("falta el programa")
    # errors="replace": un acento mal codificado en un comentario del CAM no impide leer el programa
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

    # --- comandos que hablan con el controlador ----------------------------------------------------

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
            """Avance en una sola línea de la consola («\r» la reescribe)."""
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
        backend.close()                            # liberar el puerto serie pase lo que pase


if __name__ == "__main__":
    raise SystemExit(main())
