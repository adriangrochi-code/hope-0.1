"""Revisión de un programa contra la máquina, antes de mover nada.

Junta los diagnósticos del intérprete con los de la máquina: límites de recorrido (en coordenadas
de máquina, arcos incluidos), corte con el husillo apagado, programa sin final, y el tiempo del
planificador. ``runnable`` es falso si hay algún error: el controlador no envía ese programa.

Cómo encaja
-----------
* **Usa:** ``gcode`` (intérprete y ``linearize`` para revisar los arcos punto por punto),
  ``machine`` (recorridos, tipo de máquina, orígenes) y ``planner`` (tiempos y largos).
* **Lo usan:** ``controller`` (no envía nada si ``runnable`` es falso), ``session`` (informe al
  cargar un programa en la app), ``tool`` (herramienta de Hope), ``medir`` y la consola
  (``python -m hope.cnc revisar``).

Qué revisa
----------
* **Errores** (impiden enviar): todo lo que el intérprete marcó como error y cualquier punto del
  recorrido fuera de los límites ``min``/``max`` de un eje (en coordenadas de máquina).
* **Avisos** (se puede enviar igual): cortes con el husillo apagado (o la antorcha, en plasma),
  rápidos con la antorcha encendida, perforaciones sin G4 de retardo, programa sin M2/M30.

Unidades y convenciones
-----------------------
mm y segundos. El informe es un ``dict`` con claves en español (``"runnable"`` queda en inglés
porque lo leen varios módulos); la «envolvente» es el rango [mínimo, máximo] que recorre cada eje,
en coordenadas de máquina y relativa a G54.

Ejemplo de uso
--------------
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
# =========================================== REVISIÓN =============================================
# ==================================================================================================

def _fmt_time(s: float) -> str:
    """Segundos → texto «1 h 02 min 05 s» (o «2 min 05 s» si dura menos de una hora)."""
    s = round(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h} h {m:02d} min {sec:02d} s" if h else f"{m} min {sec:02d} s"


def check_program(program: Program, machine: Machine, pl: Plan | None = None) -> dict:
    """Revisa ``program`` contra ``machine``. ``pl``: el plan ya calculado de ese mismo programa, si
    lo hay (la sesión lo reutiliza para los tiempos por línea en vez de planificar dos veces).

    Recorre las operaciones una vez juntando límites, estado del husillo/antorcha y estadísticas de
    plasma, y después arma los diagnósticos y el informe.

    Devuelve un ``dict`` con, entre otras: ``runnable`` (bool: se puede enviar), ``errores`` y
    ``avisos`` (textos «línea N: …» ordenados por línea), ``tiempo_s`` (movimiento + G4, en s),
    ``avance_mm``/``rapido_mm``, ``envolvente_maquina``/``envolvente_g54`` (eje → [mín, máx] en mm),
    ``herramientas`` y, en plasma, ``perforaciones`` y ``corte_antorcha_mm``. No lanza errores: todo
    problema del programa se informa como diagnóstico."""
    axes = machine.axis_names
    diags: list[Diag] = list(program.diags)
    lo = {a: float("inf") for a in axes}
    hi = {a: float("-inf") for a in axes}
    over: dict[tuple[str, str], int] = {}  # (eje, "min"|"max") → primera línea que se pasa
    spindle_on, cold_cuts, first_cold = False, 0, None
    plasma = machine.is_plasma
    pierces, torch_len, rapid_hot, first_hot = 0, 0.0, 0, None   # perforaciones, mm cortados, G0 encendida
    no_delay: list[int] = []           # líneas de encendidos seguidos de movimiento XY sin G4 en el medio
    just_lit = None                    # línea del último encendido, hasta ver retardo o movimiento

    # --- recorrido de las operaciones -------------------------------------------------------------

    for op in program.ops:
        # En plasma el «husillo» (M3/M5) es la antorcha: cada encendido es una perforación.
        if op.kind == "spindle":
            on = op.value[0] != "off"
            if on and not spindle_on:
                pierces += 1
                just_lit = op.line
            spindle_on = on
            continue
        if op.kind == "dwell":                 # G4 tras encender: la perforación tiene su retardo
            just_lit = None
            continue
        if op.kind not in ("rapid", "feed", "arc", "probe"):
            continue
        # Puntos a revisar contra los límites: los arcos punto por punto (un arco puede salirse del
        # recorrido aunque empiece y termine dentro); en el sensado (G38.2), también el final del
        # recorrido pedido (op.value), porque si no toca nada la máquina llega hasta ahí.
        pts = [op.start] + (linearize(op, machine.arc_tolerance) if op.kind == "arc" else
                            [op.end, op.value] if op.kind == "probe" else [op.end])
        for p in pts:
            for a in axes:
                v = p[a]
                lo[a], hi[a] = min(lo[a], v), max(hi[a], v)
                ax = machine.axes[a]
                # 1e-6 mm de tolerancia: que el redondeo de 4 decimales no dé un falso error justo en el límite
                if ax.min is not None and v < ax.min - 1e-6:
                    over.setdefault((a, "min"), op.line)
                if ax.max is not None and v > ax.max + 1e-6:
                    over.setdefault((a, "max"), op.line)
        # ¿Se mueve en el plano XY? Bajar o subir en Z solo no es «cortar» (perfora o se retira).
        xy = op.kind == "arc" or any(abs(op.end.get(a, 0) - op.start.get(a, 0)) > 1e-9 for a in ("X", "Y") if a in axes)
        if plasma:
            if op.kind == "rapid" and spindle_on:
                rapid_hot += 1
                first_hot = first_hot or op.line
            if op.kind in ("feed", "arc") and xy:
                if spindle_on:
                    # largo cortado: recta XY o arco (|ángulo barrido| · radio)
                    torch_len += sum(((op.end[a] - op.start[a]) ** 2 for a in ("X", "Y") if a in axes), 0.0) ** 0.5 \
                        if op.kind == "feed" else abs(op.arc.sweep) * op.arc.radius
                    if just_lit is not None:   # se movió en XY sin G4 después de encender
                        no_delay.append(just_lit)
                else:
                    cold_cuts += 1
                    first_cold = first_cold or op.line
            if xy:
                just_lit = None
        elif op.kind in ("feed", "arc") and machine.needs_spindle and not spindle_on:
            cold_cuts += 1
            first_cold = first_cold or op.line

    # --- diagnósticos -----------------------------------------------------------------------------

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

    # --- informe ----------------------------------------------------------------------------------

    if pl is None:
        pl = plan(program, machine)
    g54 = machine.offset("G54")
    moved = any(op.kind in ("rapid", "feed", "arc", "probe") for op in program.ops)
    env = {a: (lo[a], hi[a]) for a in axes} if moved else {}   # sin movimiento lo/hi quedaron en ±inf
    errors = [d for d in diags if d.level == "error"]
    return {
        "maquina": machine.name,
        "runnable": not errors and moved,  # un programa que no mueve nada tampoco se envía
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
        "tipo": machine.kind,
        **({"perforaciones": pierces, "corte_antorcha_mm": round(torch_len, 3)} if plasma else {}),
    }


# ==================================================================================================
# ======================================== ATAJOS E INFORME ========================================
# ==================================================================================================


def check_text(text: str, machine: Machine, start: dict[str, float] | None = None) -> dict:
    """Interpreta ``text`` (G-code) y lo revisa con ``check_program``. ``start``: posición inicial de
    la máquina (eje → mm, coordenadas de máquina) si no es la del intérprete por defecto.

    Agrega ``sha256`` del texto al informe: identifica el programa exacto (el guardián lo usa para
    que una aprobación valga solo para ese programa)."""
    r = check_program(interpret(text, machine, start), machine)
    r["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return r


def report_text(r: dict) -> str:
    """Informe de ``check_program`` en texto para una persona (consola, app, Hope).

    La segunda línea es siempre el veredicto («LISTO para enviar» / «NO se puede enviar»): la sesión
    la muestra sola al cargar un programa. Se listan hasta 20 errores y 20 avisos, y se dice cuántos
    más quedaron afuera (un programa roto puede tener miles)."""
    out = [f"Programa para «{r['maquina']}»: {r['lineas']} líneas, {r['bloques']} tramos planificados."]
    out.append(("LISTO para enviar" if r["runnable"] else "NO se puede enviar") +
               f" — {len(r['errores'])} errores, {len(r['avisos'])} avisos.")
    if r["bloques"]:
        out.append(f"Tiempo estimado (con aceleración y esquinas): {_fmt_time(r['tiempo_s'])}"
                   + (f" + {r['pausas_operador']} pausas del operador" if r["pausas_operador"] else ""))
        out.append(f"Corte {r['avance_mm']:.1f} mm, rápidos {r['rapido_mm']:.1f} mm.")
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
