"""Revisión de un programa contra la máquina, antes de mover nada.

Junta los diagnósticos del intérprete con los de la máquina: límites de recorrido (en coordenadas
de máquina, arcos incluidos), corte con el husillo apagado, programa sin final, y el tiempo del
planificador. ``runnable`` es falso si hay algún error: el controlador no envía ese programa.
"""

from __future__ import annotations

import hashlib

from hope.cnc.gcode import Diag, Program, interpret, linearize
from hope.cnc.machine import Machine
from hope.cnc.planner import plan


def _fmt_time(s: float) -> str:
    s = round(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h} h {m:02d} min {sec:02d} s" if h else f"{m} min {sec:02d} s"


def check_program(program: Program, machine: Machine) -> dict:
    axes = machine.axis_names
    diags: list[Diag] = list(program.diags)
    lo = {a: float("inf") for a in axes}
    hi = {a: float("-inf") for a in axes}
    over: dict[tuple[str, str], int] = {}
    spindle_on, cold_cuts, first_cold = False, 0, None
    for op in program.ops:
        if op.kind == "spindle":
            spindle_on = op.value[0] != "off"
            continue
        if op.kind not in ("rapid", "feed", "arc"):
            continue
        pts = [op.start] + (linearize(op, machine.arc_tolerance) if op.kind == "arc" else [op.end])
        for p in pts:
            for a in axes:
                v = p[a]
                lo[a], hi[a] = min(lo[a], v), max(hi[a], v)
                ax = machine.axes[a]
                if ax.min is not None and v < ax.min - 1e-6:
                    over.setdefault((a, "min"), op.line)
                if ax.max is not None and v > ax.max + 1e-6:
                    over.setdefault((a, "max"), op.line)
        if op.kind != "rapid" and machine.needs_spindle and not spindle_on:
            cold_cuts += 1
            first_cold = first_cold or op.line
    for (a, side), line in sorted(over.items(), key=lambda t: t[1]):
        ax = machine.axes[a]
        lim = ax.min if side == "min" else ax.max
        val = lo[a] if side == "min" else hi[a]
        diags.append(Diag("error", line, f"{a} sale del recorrido: llega a {val:.3f} y el "
                                         f"{'mínimo' if side == 'min' else 'máximo'} es {lim:g} (máquina)"))
    if cold_cuts:
        diags.append(Diag("aviso", first_cold, f"{cold_cuts} movimientos de corte con el husillo apagado (falta M3/M4)"))
    if not program.ended and program.ops:
        diags.append(Diag("aviso", program.lines, "el programa no termina con M2/M30"))
    pl = plan(program, machine)
    g54 = machine.offset("G54")
    moved = any(op.kind in ("rapid", "feed", "arc") for op in program.ops)
    env = {a: (lo[a], hi[a]) for a in axes} if moved else {}
    errors = [d for d in diags if d.level == "error"]
    return {
        "maquina": machine.name,
        "runnable": not errors and moved,
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
        "envolvente_maquina": {a: [round(v[0], 4), round(v[1], 4)] for a, v in env.items()},
        "envolvente_g54": {a: [round(v[0] - g54[a], 4), round(v[1] - g54[a], 4)] for a, v in env.items()},
        "herramientas": program.tools,
    }


def check_text(text: str, machine: Machine, start: dict[str, float] | None = None) -> dict:
    r = check_program(interpret(text, machine, start), machine)
    r["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return r


def report_text(r: dict) -> str:
    out = [f"Programa para «{r['maquina']}»: {r['lineas']} líneas, {r['bloques']} tramos planificados."]
    out.append(("LISTO para enviar" if r["runnable"] else "NO se puede enviar") +
               f" — {len(r['errores'])} errores, {len(r['avisos'])} avisos.")
    if r["bloques"]:
        out.append(f"Tiempo estimado (con aceleración y esquinas): {_fmt_time(r['tiempo_s'])}"
                   + (f" + {r['pausas_operador']} pausas del operador" if r["pausas_operador"] else ""))
        out.append(f"Corte {r['avance_mm']:.1f} mm, rápidos {r['rapido_mm']:.1f} mm.")
        env = ", ".join(f"{a} {v[0]:.3f}…{v[1]:.3f}" for a, v in r["envolvente_g54"].items())
        out.append(f"Envolvente (G54): {env}")
    if r["herramientas"]:
        out.append("Herramientas: " + ", ".join(f"T{t}" for t in r["herramientas"]))
    out += ["ERROR " + e for e in r["errores"][:20]]
    out += ["aviso " + a for a in r["avisos"][:20]]
    extra = len(r["errores"]) + len(r["avisos"]) - min(20, len(r["errores"])) - min(20, len(r["avisos"]))
    if extra > 0:
        out.append(f"… y {extra} más")
    return "\n".join(out)
