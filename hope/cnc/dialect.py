"""Translation of the interpreted program into each controller's dialect.

Hope's interpreter is the authority: the controller receives **simple, normalized** G-code
(G21 G90 G17 G94, absolute coordinates in G54), so what was checked is exactly what is sent.

* Offsets (G54–G59, G92, G10) and length compensation (G43) are already applied to the
  coordinates; the controller only needs its G54 (read from the machine before sending).
* Canned cycles already expanded (grbl does not have them).
* Arcs: G2/G3 with incremental I/J/K; multi-turn arcs are split into chords.
* Tool change: ``T M6`` on LinuxCNC; on grbl, ``M5`` + ``M0`` (manual change) with a message.

Differences between dialects (what changes in the emitted text)
---------------------------------------------------------------
=====================  ======================  ======================  ==================================
What                   grbl / grblHAL          LinuxCNC                Marlin / Smoothie / RRF
=====================  ======================  ======================  ==================================
Header                 full ``HEADER``         full ``HEADER``         ``G21 G90`` (nothing else exists)
G4 dwell               ``G4 P<s>`` (seconds)   ``G4 P<s>`` (seconds)   ``G4 S<s>`` (P would be ms)
Set origin             ``G10 L20 P1 …``        ``G10 L20 P1 …``        ``G92 …`` (G10 = retract filament)
Pause message          ``M0 (MSG, text)``      ``M0 (MSG, text)``      ``M0 text``
Tool change            ``M5`` + manual ``M0``  ``M5`` + ``T<n> M6``    ``M5`` + manual ``M0``
End of program         ``M2``                  ``M30``                 ``M5`` (M30 = delete from SD)
=====================  ======================  ======================  ==================================

Each operation produces the **same number of lines** in all three dialects (see ``emit``).

How it fits
-----------
* **Uses:** ``gcode`` (the interpreted operations and ``linearize``) and ``machine`` (axes, G54,
  arc tolerance, whether it is plasma, controller type).
* **Used by:** ``controller`` (what gets sent), ``session`` (the app; «Correr desde aquí» (Run From
  Here)), ``tool``, ``medir`` and the console (``python -m hope.cnc traducir``).

Units
-----
Always mm (G21), feed F in mm/min, numbers with up to 4 decimals (0.1 µm, more than enough for any
machine and keeps lines short for the serial link).

Usage example
-------------
::

    from hope.cnc.dialect import dialect_for, emit_text
    texto = emit_text(interpret(gcode, m), m, dialect_for(m))   # «G21 G90 G17 G94 G54 G91.1\nG0 X10…»
"""

from __future__ import annotations

import math

from hope.cnc.gcode import Op, Program, linearize
from hope.cnc.machine import Machine
from hope.cnc.machine import axis_rank as _axis_rank

# ==================================================================================================
# ========================================= CONFIGURATION ==========================================
# ==================================================================================================

# Dialects ``emit`` can produce (the value of ``--dialecto`` in the console). Adding one here
# without adding it to ``emit`` is useless: ``emit`` decides with ``marlin``/``dialect == "linuxcnc"``.
DIALECTS = ("grbl", "linuxcnc", "marlin")

# First line sent: puts the controller into a known state, regardless of what a previous program
# may have left behind.
#   G21   units in mm                  G90    absolute coordinates
#   G17   XY plane for arcs            G94    F in mm per minute (not inverse time G93)
#   G54   first work system            G91.1  incremental I/J/K arc center (relative to the start)
HEADER = "G21 G90 G17 G94 G54 G91.1"
# Per-dialect headers (if a dialect is not listed, HEADER is used).
HEADERS = {"marlin": "G21 G90"}             # Marlin/Smoothie/RRF: no G94/G54/G91.1 (it would answer «Unknown command»)


# ==================================================================================================
# ========================================== TRANSLATION ===========================================
# ==================================================================================================


def dialect_for(machine: Machine) -> str:
    """Dialect according to the machine's controller (profile): grbl, linuxcnc or marlin («ok» controllers).

    Reads ``controlador.tipo`` from the profile. «ok» controllers are those that answer ``ok`` to every
    line but do not speak grbl (Marlin, Smoothieware, RepRapFirmware). Any other value, including the
    simulator ("sim"), yields grbl: it is the most basic G-code and the one almost everyone understands."""
    kind = str((machine.controller or {}).get("tipo", "sim")).lower()
    if kind == "linuxcnc":
        return "linuxcnc"
    if kind in ("ok", "marlin", "smoothie", "smoothieware", "reprap", "rrf"):
        return "marlin"
    return "grbl"


def _n(v: float) -> str:
    """Number for the G-code: up to 4 decimals, without trailing zeros or «-0» (``1.5000`` → ``1.5``,
    ``-0.00001`` → ``0``). Shorter lines = fewer bytes over the serial link."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def emit(program: Program, machine: Machine, dialect: str = "grbl",
         origins: list[int] | None = None) -> list[tuple[str, int]]:
    """Dialect lines, each with the line of the original program it comes from.

    With ``origins`` (an empty list), for each emitted line the index of the ``program.ops`` operation
    that produced it is recorded (-1 for the header). All dialects emit the SAME number of lines per
    operation: that way line N in grbl corresponds to line N in Marlin or LinuxCNC
    («Correr desde aquí» (Run From Here) re-interprets the grbl version and sends the dialect's one).

    Parameters:
        program: interpreted program (already checked: it is translated as is, without re-validating).
        machine: axes, G54 and arc tolerance.
        dialect: one of ``DIALECTS``.
        origins: empty list to fill in, or None.

    Returns a list of ``(line text, original program line)``; the header carries line 0. Raises
    ``ValueError`` if the dialect does not exist.

    The emitted G-code is **modal** like the original: F and S are sent only when they change, and
    moves only carry the axes that move (fewer bytes over the serial link)."""
    if dialect not in DIALECTS:
        raise ValueError(f"dialecto «{dialect}» desconocido (use {', '.join(DIALECTS)})")
    axes = machine.axis_names
    g54 = machine.offset("G54")
    out: list[tuple[str, int]] = [(HEADERS.get(dialect, HEADER), 0)]
    marlin = dialect == "marlin"
    feed = None                                    # last F sent (mm/min); None = must send it
    rpm = None                                     # last S sent (rpm)

    # --- helpers ----------------------------------------------------------------------------------

    def coords(p: dict, prev: dict | None, frame: dict) -> str:
        """«X… Y… Z…» of ``p`` (machine) relative to ``frame`` (the current G54), only with the axes
        that change with respect to ``prev`` (all of them if ``prev`` is None). Empty if nothing moves."""
        return " ".join(f"{a}{_n(p[a] - frame[a])}" for a in axes if prev is None or abs(p[a] - prev[a]) > 1e-9)

    def lin(kind: str, op: Op, start: dict, end: dict, f: float):
        """Append a G0 (``kind`` "rapid") or G1 from ``start`` to ``end``; ``f`` in mm/min. A
        zero-length segment emits nothing."""
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

    # --- one operation after another --------------------------------------------------------------

    for k, op in enumerate(program.ops):
        if origins is not None:                    # fill in the origin of what the previous op emitted
            origins.extend([k - 1] * (len(out) - len(origins)))
        frame = op.frame or g54                    # G54 in effect for this operation (G10 may change it mid-program)
        if op.kind in ("rapid", "feed"):
            lin(op.kind, op, op.start, op.end, op.feed)
        elif op.kind == "probe":
            # the full travel is sent: the controller stops on contact
            c = coords(op.value, op.start, frame)
            out.append((f"G38.2 {c} F{_n(op.feed)}", op.line))
            feed = None                            # G38.2 changed the modal F: the next G1 sends its own
        elif op.kind == "wcs":
            # G10 L20 with the value that gives the same origin in the controller (no G92 or G43: already applied)
            vals = " ".join(f"{a}{_n(op.end[a])}" for a in sorted(op.end, key=_axis_rank) if a != "L")
            # in Marlin G10 means «retract filament»: the same effect is G92 (the current position is this)
            out.append(((f"G92 {vals}") if marlin else f"G10 L20 P1 {vals}", op.line))
        elif op.kind == "arc":
            a, b, _ = op.arc.plane
            # More than one turn (helices): grbl and Marlin lack LinuxCNC's P word (number of turns),
            # so chords are sent; the same applies to all dialects.
            if abs(op.arc.sweep) > 2 * math.pi + 1e-9:
                prev = op.start
                for p in linearize(op, machine.arc_tolerance):
                    lin("feed", op, prev, p, op.feed)
                    prev = p
                continue
            letters = {"X": "I", "Y": "J", "Z": "K"}  # center letter for each plane axis
            g = "G2" if op.arc.sweep < 0 else "G3"     # negative sweep = clockwise (G2)
            plane = {("X", "Y"): "G17", ("Z", "X"): "G18", ("Y", "Z"): "G19"}[(a, b)]
            c = " ".join(f"{ax}{_n(op.end[ax] - frame[ax])}" for ax in axes)
            # incremental center from the start point (G91.1 in the header)
            ij = f"{letters[a]}{_n(op.arc.center[0] - op.start[a])} {letters[b]}{_n(op.arc.center[1] - op.start[b])}"
            fs = "" if feed == op.feed else f" F{_n(op.feed)}"
            feed = op.feed
            pre = f"{plane} " if plane != "G17" else ""
            out.append((f"{pre}{g} {c} {ij}{fs}", op.line))
            if plane != "G17":                     # G18/G19: return to G17 so the rest does not depend on the plane
                out.append(("G17", op.line))
        elif op.kind == "dwell":
            # Marlin: G4 P is milliseconds and G4 S seconds; grbl and LinuxCNC: P in seconds
            out.append(((f"G4 S{_n(float(op.value))}") if marlin else f"G4 P{_n(float(op.value))}", op.line))
        elif op.kind == "spindle":
            d, s = op.value
            if d == "off":
                out.append(("M5", op.line))
            else:
                # S only if it changed; on plasma without S, «S0» is not sent (the torch has no rpm)
                ss = "" if rpm == s or (machine.is_plasma and not s) else f" S{_n(s)}"
                rpm = s
                out.append((("M3" if d == "cw" else "M4") + ss, op.line))
        elif op.kind == "coolant":
            v = op.value
            if not v:
                out.append(("M9", op.line))
            else:
                # M7 mist, M8 flood; sorted(): fixed order even though it is a set (Python 3.4)
                out += [(("M7" if c == "niebla" else "M8"), op.line) for c in sorted(v)]
        elif op.kind == "tool":
            if dialect == "linuxcnc":                # M5 + T M6: two lines, same as the other dialects
                out += [("M5", op.line), (f"T{op.value} M6", op.line)]
            else:
                # grbl and Marlin do not change tools: the spindle is stopped and it pauses for a manual change
                msg = f"cambiar a la herramienta T{op.value}"
                out += [("M5", op.line), ((f"M0 {msg}") if marlin else f"M0 (MSG, {msg})", op.line)]
                rpm = None                         # after the change, resend S with the next M3/M4
        elif op.kind == "pause":
            text = str(op.value)[:60]              # short: grbl accepts lines of ~80 characters
            out.append(((f"M0 {text}") if marlin else f"M0 (MSG, {text})", op.line))
        elif op.kind == "end":
            # Marlin has no M2 and its M30 deletes a file from the SD card: just stop the spindle
            out.append(({"linuxcnc": "M30", "marlin": "M5"}.get(dialect, "M2"), op.line))
    if origins is not None:
        origins.extend([len(program.ops) - 1] * (len(out) - len(origins)))
        origins[0] = -1                            # the header does not come from any operation
    return out


def emit_text(program: Program, machine: Machine, dialect: str = "grbl") -> str:
    """Like ``emit`` but only the text, one line per row and with a final line break (ready to save
    to a file). Raises ``ValueError`` if the dialect does not exist."""
    return "\n".join(t for t, _ in emit(program, machine, dialect)) + "\n"
