"""Traducción del programa interpretado al dialecto de cada controlador.

El intérprete de Hope es la autoridad: el controlador recibe un G-code **simple y normalizado**
(G21 G90 G17 G94, coordenadas absolutas en G54), así lo que se revisó es exactamente lo que se envía.

* Los orígenes (G54–G59, G92, G10) y la compensación de largo (G43) quedan aplicados en las
  coordenadas; el controlador solo necesita su G54 (que se lee de la máquina antes de enviar).
* Ciclos fijos ya expandidos (grbl no los tiene).
* Arcos: G2/G3 con I/J/K incrementales; los de varias vueltas se dividen en cuerdas.
* Cambio de herramienta: ``T M6`` en LinuxCNC; en grbl, ``M5`` + ``M0`` (cambio manual) con mensaje.
"""

from __future__ import annotations

import math

from hope.cnc.gcode import Op, Program, linearize
from hope.cnc.machine import Machine

DIALECTS = ("grbl", "linuxcnc", "marlin")
HEADER = "G21 G90 G17 G94 G54 G91.1"
HEADERS = {"marlin": "G21 G90"}             # Marlin/Smoothie/RRF: sin G94/G54/G91.1 (respondería «Unknown command»)


def dialect_for(machine: Machine) -> str:
    """Dialecto según el controlador de la máquina (perfil): grbl, linuxcnc o marlin (controladores «ok»)."""
    kind = str((machine.controller or {}).get("tipo", "sim")).lower()
    if kind == "linuxcnc":
        return "linuxcnc"
    if kind in ("ok", "marlin", "smoothie", "smoothieware", "reprap", "rrf"):
        return "marlin"
    return "grbl"


def _n(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def emit(program: Program, machine: Machine, dialect: str = "grbl") -> list[tuple[str, int]]:
    """Líneas del dialecto con la línea del programa original de la que salen."""
    if dialect not in DIALECTS:
        raise ValueError(f"dialecto «{dialect}» desconocido (use {', '.join(DIALECTS)})")
    axes = machine.axis_names
    g54 = machine.offset("G54")
    out: list[tuple[str, int]] = [(HEADERS.get(dialect, HEADER), 0)]
    marlin = dialect == "marlin"
    feed = None
    rpm = None

    def coords(p: dict, prev: dict | None, frame: dict) -> str:
        return " ".join(f"{a}{_n(p[a] - frame[a])}" for a in axes if prev is None or abs(p[a] - prev[a]) > 1e-9)

    def lin(kind: str, op: Op, start: dict, end: dict, f: float):
        nonlocal feed
        c = coords(end, start, op.frame or g54)
        if not c:
            return
        if kind == "rapid":
            out.append((f"G0 {c}", op.line))
        else:
            fs = "" if feed == f else f" F{_n(f)}"
            feed = f
            out.append((f"G1 {c}{fs}", op.line))

    for op in program.ops:
        frame = op.frame or g54
        if op.kind in ("rapid", "feed"):
            lin(op.kind, op, op.start, op.end, op.feed)
        elif op.kind == "probe":
            # se envía el recorrido completo: el controlador para al tocar
            c = coords(op.value, op.start, frame)
            out.append((f"G38.2 {c} F{_n(op.feed)}", op.line))
            feed = None
        elif op.kind == "wcs":
            # G10 L20 con el valor que da el mismo origen en el controlador (sin G92 ni G43: ya aplicados)
            vals = " ".join(f"{a}{_n(v)}" for a, v in op.end.items() if a != "L")
            # en Marlin G10 es «retraer filamento»: el mismo efecto es G92 (la posición actual vale esto)
            out.append(((f"G92 {vals}") if marlin else f"G10 L20 P1 {vals}", op.line))
        elif op.kind == "arc":
            a, b, _ = op.arc.plane
            if abs(op.arc.sweep) > 2 * math.pi + 1e-9:
                prev = op.start
                for p in linearize(op, machine.arc_tolerance):
                    lin("feed", op, prev, p, op.feed)
                    prev = p
                continue
            letters = {"X": "I", "Y": "J", "Z": "K"}
            g = "G2" if op.arc.sweep < 0 else "G3"
            plane = {("X", "Y"): "G17", ("Z", "X"): "G18", ("Y", "Z"): "G19"}[(a, b)]
            c = " ".join(f"{ax}{_n(op.end[ax] - frame[ax])}" for ax in axes)
            ij = f"{letters[a]}{_n(op.arc.center[0] - op.start[a])} {letters[b]}{_n(op.arc.center[1] - op.start[b])}"
            fs = "" if feed == op.feed else f" F{_n(op.feed)}"
            feed = op.feed
            pre = f"{plane} " if plane != "G17" else ""
            out.append((f"{pre}{g} {c} {ij}{fs}", op.line))
            if plane != "G17":
                out.append(("G17", op.line))
        elif op.kind == "dwell":
            # Marlin: G4 P son milisegundos y G4 S segundos; grbl y LinuxCNC: P en segundos
            out.append(((f"G4 S{_n(float(op.value))}") if marlin else f"G4 P{_n(float(op.value))}", op.line))
        elif op.kind == "spindle":
            d, s = op.value
            if d == "off":
                out.append(("M5", op.line))
            else:
                ss = "" if rpm == s or (machine.is_plasma and not s) else f" S{_n(s)}"
                rpm = s
                out.append((("M3" if d == "cw" else "M4") + ss, op.line))
        elif op.kind == "coolant":
            v = op.value
            if not v:
                out.append(("M9", op.line))
            else:
                out += [(("M7" if c == "niebla" else "M8"), op.line) for c in sorted(v)]
        elif op.kind == "tool":
            if dialect == "linuxcnc":
                out.append((f"T{op.value} M6", op.line))
            else:
                msg = f"cambiar a la herramienta T{op.value}"
                out += [("M5", op.line), ((f"M0 {msg}") if marlin else f"M0 (MSG, {msg})", op.line)]
                rpm = None
        elif op.kind == "pause":
            text = str(op.value)[:60]
            out.append(((f"M0 {text}") if marlin else f"M0 (MSG, {text})", op.line))
        elif op.kind == "end":
            out.append(({"linuxcnc": "M30", "marlin": "M5"}.get(dialect, "M2"), op.line))
    return out


def emit_text(program: Program, machine: Machine, dialect: str = "grbl") -> str:
    return "\n".join(t for t, _ in emit(program, machine, dialect)) + "\n"
