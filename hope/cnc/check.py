"""Checking a program against the machine, before moving anything.

Combines the interpreter's diagnostics with the machine's: travel limits (in machine coordinates,
arcs included), cutting with the spindle off, program without an end, and the planner's time.
``runnable`` is false if there is any error: the controller does not send that program.

How it fits
-----------
* **Uses:** ``gcode`` (interpreter and ``linearize`` to check arcs point by point),
  ``machine`` (travel, machine type, offsets) and ``planner`` (times and lengths).
* **Used by:** ``controller`` (sends nothing if ``runnable`` is false), ``session`` (report when
  loading a program in the app), ``tool`` (Hope's tool), ``medir`` and the console
  (``python -m hope.cnc revisar``).

What it checks
--------------
* **Errors** (prevent sending): everything the interpreter flagged as an error and any point of
  the path outside an axis' ``min``/``max`` limits (in machine coordinates).
* **Warnings** (can be sent anyway): cuts with the spindle off (or the torch, on plasma), rapids
  with the torch on, pierces without a G4 delay, program without M2/M30.

Units and conventions
---------------------
mm and seconds. The report is a ``dict`` with Spanish keys (``"runnable"`` stays in English
because several modules read it); the «envolvente» (envelope) is the [minimum, maximum] range each
axis travels, in machine coordinates and relative to G54.

Usage example
-------------
::

    from hope.cnc.check import check_text, report_text
    r = check_text(open("pieza.nc").read(), machine)
    print(report_text(r))          # «LISTO para enviar — 0 errores, 1 avisos.» …
    if r["runnable"]: ...
"""

from __future__ import annotations

import hashlib

from hope.cnc.gcode import Diag, Program, interpret, linearize
from hope.cnc.machine import Machine
from hope.cnc.machine import axis_rank as _axis_rank
from hope.cnc.planner import Plan, plan

# ==================================================================================================
# ============================================= CHECK ==============================================
# ==================================================================================================

def _fmt_time(s: float) -> str:
    """Seconds → text «1 h 02 min 05 s» (or «2 min 05 s» if it lasts less than an hour)."""
    s = round(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h} h {m:02d} min {sec:02d} s" if h else f"{m} min {sec:02d} s"


def check_program(program: Program, machine: Machine, pl: Plan | None = None) -> dict:
    """Check ``program`` against ``machine``. ``pl``: the already computed plan of that same program,
    if any (the session reuses it for per-line times instead of planning twice).

    Walks the operations once collecting limits, spindle/torch state and plasma statistics, and
    then builds the diagnostics and the report.

    Returns a ``dict`` with, among others: ``runnable`` (bool: can be sent), ``errores`` and
    ``avisos`` (texts «línea N: …» sorted by line), ``tiempo_s`` (motion + G4, in s),
    ``avance_mm``/``rapido_mm``, ``envolvente_maquina``/``envolvente_g54`` (axis → [min, max] in mm),
    ``herramientas`` and, on plasma, ``perforaciones`` and ``corte_antorcha_mm``. Raises nothing:
    every problem in the program is reported as a diagnostic."""
    axes = machine.axis_names
    diags: list[Diag] = list(program.diags)
    lo = {a: float("inf") for a in axes}
    hi = {a: float("-inf") for a in axes}
    over: dict[tuple[str, str], int] = {}  # (axis, "min"|"max") → first line that exceeds it
    spindle_on, cold_cuts, first_cold = False, 0, None
    plasma = machine.is_plasma
    pierces, torch_len, rapid_hot, first_hot = 0, 0.0, 0, None   # pierces, mm cut, G0 with torch on
    no_delay: list[int] = []           # lines of torch-ons followed by XY motion with no G4 in between
    just_lit = None                    # line of the last torch-on, until a delay or a move is seen

    # --- walk through the operations --------------------------------------------------------------

    for op in program.ops:
        # On plasma the «spindle» (M3/M5) is the torch: every torch-on is a pierce.
        if op.kind == "spindle":
            on = op.value[0] != "off"
            if on and not spindle_on:
                pierces += 1
                just_lit = op.line
            spindle_on = on
            continue
        if op.kind == "dwell":                 # G4 after torch-on: the pierce has its delay
            just_lit = None
            continue
        if op.kind not in ("rapid", "feed", "arc", "probe"):
            continue
        # Points to check against the limits: arcs point by point (an arc can leave the travel
        # range even if it starts and ends inside it); for probing (G38.2), also the end of the
        # requested travel (op.value), because if it touches nothing the machine goes all the way there.
        pts = [op.start] + (linearize(op, machine.arc_tolerance) if op.kind == "arc" else
                            [op.end, op.value] if op.kind == "probe" else [op.end])
        for p in pts:
            for a in axes:
                v = p[a]
                lo[a], hi[a] = min(lo[a], v), max(hi[a], v)
                ax = machine.axes[a]
                # 1e-6 mm tolerance: so 4-decimal rounding does not give a false error right at the limit
                if ax.min is not None and v < ax.min - 1e-6:
                    over.setdefault((a, "min"), op.line)
                if ax.max is not None and v > ax.max + 1e-6:
                    over.setdefault((a, "max"), op.line)
        # Does it move in the XY plane? Moving only in Z is not «cutting» (it pierces or retracts).
        xy = op.kind == "arc" or any(abs(op.end.get(a, 0) - op.start.get(a, 0)) > 1e-9 for a in ("X", "Y") if a in axes)
        if plasma:
            if op.kind == "rapid" and spindle_on:
                rapid_hot += 1
                first_hot = first_hot or op.line
            if op.kind in ("feed", "arc") and xy:
                if spindle_on:
                    # cut length: XY straight line or arc (|swept angle| · radius)
                    torch_len += sum(((op.end[a] - op.start[a]) ** 2 for a in ("X", "Y") if a in axes), 0.0) ** 0.5 \
                        if op.kind == "feed" else abs(op.arc.sweep) * op.arc.radius
                    if just_lit is not None:   # moved in XY without a G4 after torch-on
                        no_delay.append(just_lit)
                else:
                    cold_cuts += 1
                    first_cold = first_cold or op.line
            if xy:
                just_lit = None
        elif op.kind in ("feed", "arc") and machine.needs_spindle and not spindle_on:
            cold_cuts += 1
            first_cold = first_cold or op.line

    # --- diagnostics -----------------------------------------------------------------------------

    for (a, side), line in sorted(over.items(), key=lambda t: t[1]):
        ax = machine.axes[a]
        lim = ax.min if side == "min" else ax.max
        val = lo[a] if side == "min" else hi[a]
        diags.append(Diag("error", line, f"{a} sale del recorrido: llega a {val:.3f} y el "
                                         f"{'mínimo' if side == 'min' else 'máximo'} es {lim:g} (máquina)"))
    if cold_cuts:
        what = "con la antorcha apagada" if plasma else "con el husillo apagado (falta M3/M4)"
        diags.append(Diag("aviso", first_cold, f"{cold_cuts} movimientos de corte {what}"))
    if rapid_hot:
        diags.append(Diag("aviso", first_hot, f"{rapid_hot} movimientos rápidos con la antorcha encendida"))
    if no_delay:
        diags.append(Diag("aviso", no_delay[0], f"{len(no_delay)} perforaciones sin retardo (G4) antes de moverse: "
                                                "el arco puede no atravesar la chapa"))
    if not program.ended and program.ops:
        diags.append(Diag("aviso", program.lines, "el programa no termina con M2/M30"))

    # --- report -----------------------------------------------------------------------------------

    if pl is None:
        pl = plan(program, machine)
    g54 = machine.offset("G54")
    moved = any(op.kind in ("rapid", "feed", "arc", "probe") for op in program.ops)
    env = {a: (lo[a], hi[a]) for a in axes} if moved else {}   # with no motion lo/hi stayed at ±inf
    errors = [d for d in diags if d.level == "error"]
    return {
        "maquina": machine.name,
        "runnable": not errors and moved,  # a program that moves nothing is not sent either
        "errores": [str(d) for d in sorted(errors, key=lambda d: d.line)],
        "avisos": [str(d) for d in sorted((d for d in diags if d.level == "aviso"), key=lambda d: d.line)],
        "lineas": program.lines,
        "operaciones": len(program.ops),
        "bloques": len(pl.blocks),
        "tiempo_s": round(pl.total_time, 2),
        "tiempo_movimiento_s": round(pl.move_time, 2),
        "pausas_operador": pl.pauses,
        "avance_mm": round(pl.feed_length, 3),
        "rapido_mm": round(pl.rapid_length, 3),
        # Degrees each rotary axis turns while cutting, e.g. {"A": 720.0}. Empty on 3-axis machines.
        "giro_grados": {a: round(v, 3) for a, v in (pl.rotary_travel or {}).items()},
        "envolvente_maquina": {a: [round(v[0], 4), round(v[1], 4)] for a, v in env.items()},
        "envolvente_g54": {a: [round(v[0] - g54[a], 4), round(v[1] - g54[a], 4)] for a, v in env.items()},
        "herramientas": program.tools,
        "tipo": machine.kind,
        **({"perforaciones": pierces, "corte_antorcha_mm": round(torch_len, 3)} if plasma else {}),
    }


# ==================================================================================================
# ====================================== SHORTCUTS AND REPORT ======================================
# ==================================================================================================


def check_text(text: str, machine: Machine, start: dict[str, float] | None = None) -> dict:
    """Interpret ``text`` (G-code) and check it with ``check_program``. ``start``: the machine's
    initial position (axis → mm, machine coordinates) if it is not the interpreter's default.

    Adds the text's ``sha256`` to the report: it identifies the exact program (the guardian uses it
    so an approval is valid only for that program)."""
    r = check_program(interpret(text, machine, start), machine)
    r["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return r


def report_text(r: dict) -> str:
    """``check_program`` report as text for a person (console, app, Hope).

    The second line is always the verdict («LISTO para enviar» / «NO se puede enviar»): the session
    shows it on its own when loading a program. Up to 20 errors and 20 warnings are listed, and it
    says how many more were left out (a broken program can have thousands)."""
    out = [f"Programa para «{r['maquina']}»: {r['lineas']} líneas, {r['bloques']} tramos planificados."]
    out.append(("LISTO para enviar" if r["runnable"] else "NO se puede enviar") +
               f" — {len(r['errores'])} errores, {len(r['avisos'])} avisos.")
    if r["bloques"]:
        out.append(f"Tiempo estimado (con aceleración y esquinas): {_fmt_time(r['tiempo_s'])}"
                   + (f" + {r['pausas_operador']} pausas del operador" if r["pausas_operador"] else ""))
        out.append(f"Corte {r['avance_mm']:.1f} mm, rápidos {r['rapido_mm']:.1f} mm."
                   + ("" if not r.get("giro_grados") else " Giro: " + ", ".join(
                       f"{a} {r['giro_grados'][a]:.0f}°" for a in sorted(r["giro_grados"], key=_axis_rank)) + "."))
        eg = r["envolvente_g54"]
        env = ", ".join(f"{a} {eg[a][0]:.3f}…{eg[a][1]:.3f}" for a in sorted(eg, key=_axis_rank))
        out.append(f"Envolvente (G54): {env}")
    if r.get("perfil"):
        out.append(f"Perfil: {r['perfil']}")
    if "perforaciones" in r:
        out.append(f"Plasma: {r['perforaciones']} perforaciones, {r['corte_antorcha_mm'] / 1000:.2f} m de corte "
                   "con la antorcha encendida.")
    if r["herramientas"]:
        out.append("Herramientas: " + ", ".join(f"T{t}" for t in r["herramientas"]))
    out += ["ERROR " + e for e in r["errores"][:20]]
    out += ["aviso " + a for a in r["avisos"][:20]]
    extra = len(r["errores"]) + len(r["avisos"]) - min(20, len(r["errores"])) - min(20, len(r["avisos"]))
    if extra > 0:
        out.append(f"… y {extra} más")
    return "\n".join(out)
