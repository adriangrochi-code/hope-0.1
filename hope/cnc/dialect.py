"""Traducción del programa interpretado al dialecto de cada controlador.

El intérprete de Hope es la autoridad: el controlador recibe un G-code **simple y normalizado**
(G21 G90 G17 G94, coordenadas absolutas en G54), así lo que se revisó es exactamente lo que se envía.

* Los orígenes (G54–G59, G92, G10) y la compensación de largo (G43) quedan aplicados en las
  coordenadas; el controlador solo necesita su G54 (que se lee de la máquina antes de enviar).
* Ciclos fijos ya expandidos (grbl no los tiene).
* Arcos: G2/G3 con I/J/K incrementales; los de varias vueltas se dividen en cuerdas.
* Cambio de herramienta: ``T M6`` en LinuxCNC; en grbl, ``M5`` + ``M0`` (cambio manual) con mensaje.

Diferencias entre dialectos (lo que cambia en el texto emitido)
---------------------------------------------------------------
=====================  ======================  ======================  ==================================
Qué                    grbl / grblHAL          LinuxCNC                Marlin / Smoothie / RRF
=====================  ======================  ======================  ==================================
Encabezado             ``HEADER`` completo     ``HEADER`` completo     ``G21 G90`` (lo demás no existe)
Pausa G4               ``G4 P<s>`` (segundos)  ``G4 P<s>`` (segundos)  ``G4 S<s>`` (P sería en ms)
Fijar origen           ``G10 L20 P1 …``        ``G10 L20 P1 …``        ``G92 …`` (G10 = retraer filamento)
Mensaje de pausa       ``M0 (MSG, texto)``     ``M0 (MSG, texto)``     ``M0 texto``
Cambio de herramienta  ``M5`` + ``M0`` manual  ``M5`` + ``T<n> M6``    ``M5`` + ``M0`` manual
Fin de programa        ``M2``                  ``M30``                 ``M5`` (M30 = borrar de la SD)
=====================  ======================  ======================  ==================================

Cada operación produce la **misma cantidad de líneas** en los tres dialectos (ver ``emit``).

Cómo encaja
-----------
* **Usa:** ``gcode`` (las operaciones interpretadas y ``linearize``) y ``machine`` (ejes, G54,
  tolerancia de arcos, si es plasma, tipo de controlador).
* **Lo usan:** ``controller`` (lo que se envía), ``session`` (la app; «Correr desde aquí»),
  ``tool``, ``medir`` y la consola (``python -m hope.cnc traducir``).

Unidades
--------
Siempre mm (G21), avance F en mm/min, números con hasta 4 decimales (0.1 µm, sobra para cualquier
máquina y mantiene las líneas cortas para la conexión serie).

Ejemplo de uso
--------------
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
# ========================================= CONFIGURACIÓN ==========================================
# ==================================================================================================

# Dialectos que sabe emitir ``emit`` (el valor de ``--dialecto`` en la consola). Agregar uno acá
# sin agregarlo en ``emit`` no sirve: ``emit`` decide con ``marlin``/``dialect == "linuxcnc"``.
DIALECTS = ("grbl", "linuxcnc", "marlin")

# Primera línea que se envía: deja el controlador en un estado conocido, sin depender de lo que
# haya quedado de un programa anterior.
#   G21   unidades en mm               G90    coordenadas absolutas
#   G17   plano XY para los arcos      G94    F en mm por minuto (no tiempo inverso G93)
#   G54   primer sistema de trabajo    G91.1  centro de arco I/J/K incremental (relativo al inicio)
HEADER = "G21 G90 G17 G94 G54 G91.1"
# Encabezados distintos por dialecto (si no figura, se usa HEADER).
HEADERS = {"marlin": "G21 G90"}             # Marlin/Smoothie/RRF: sin G94/G54/G91.1 (respondería «Unknown command»)


# ==================================================================================================
# ========================================== TRADUCCIÓN ============================================
# ==================================================================================================


def dialect_for(machine: Machine) -> str:
    """Dialecto según el controlador de la máquina (perfil): grbl, linuxcnc o marlin (controladores «ok»).

    Lee ``controlador.tipo`` del perfil. «ok» son los controladores que responden ``ok`` a cada línea
    pero no hablan grbl (Marlin, Smoothieware, RepRapFirmware). Cualquier otro valor, incluido el
    simulador ("sim"), da grbl: es el G-code más básico y el que entienden casi todos."""
    kind = str((machine.controller or {}).get("tipo", "sim")).lower()
    if kind == "linuxcnc":
        return "linuxcnc"
    if kind in ("ok", "marlin", "smoothie", "smoothieware", "reprap", "rrf"):
        return "marlin"
    return "grbl"


def _n(v: float) -> str:
    """Número para el G-code: hasta 4 decimales, sin ceros de más ni «-0» (``1.5000`` → ``1.5``,
    ``-0.00001`` → ``0``). Líneas más cortas = menos bytes por la conexión serie."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def emit(program: Program, machine: Machine, dialect: str = "grbl",
         origins: list[int] | None = None) -> list[tuple[str, int]]:
    """Líneas del dialecto con la línea del programa original de la que salen.

    Con ``origins`` (una lista vacía) se anota, para cada línea emitida, el índice de la operación de
    ``program.ops`` que la produjo (-1 para el encabezado). Todos los dialectos emiten la MISMA cantidad
    de líneas por operación: así una línea N de grbl corresponde a la línea N de Marlin o LinuxCNC
    («Correr desde aquí» reinterpreta la versión grbl y envía la del dialecto).

    Parámetros:
        program: programa interpretado (ya revisado: se traduce tal cual, sin volver a validar).
        machine: ejes, G54 y tolerancia de arcos.
        dialect: uno de ``DIALECTS``.
        origins: lista vacía a completar, o None.

    Devuelve una lista de ``(texto de la línea, línea del programa original)``; el encabezado lleva
    la línea 0. Lanza ``ValueError`` si el dialecto no existe.

    El G-code emitido es **modal** como el original: F y S se mandan solo cuando cambian, y en los
    movimientos solo los ejes que se mueven (menos bytes por la conexión serie)."""
    if dialect not in DIALECTS:
        raise ValueError(f"dialecto «{dialect}» desconocido (use {', '.join(DIALECTS)})")
    axes = machine.axis_names
    g54 = machine.offset("G54")
    out: list[tuple[str, int]] = [(HEADERS.get(dialect, HEADER), 0)]
    marlin = dialect == "marlin"
    feed = None                                    # último F enviado (mm/min); None = mandarlo sí o sí
    rpm = None                                     # último S enviado (rpm)

    # --- ayudantes --------------------------------------------------------------------------------

    def coords(p: dict, prev: dict | None, frame: dict) -> str:
        """«X… Y… Z…» de ``p`` (máquina) relativo a ``frame`` (el G54 vigente), solo con los ejes que
        cambian respecto de ``prev`` (todos si ``prev`` es None). Vacío si no se mueve nada."""
        return " ".join(f"{a}{_n(p[a] - frame[a])}" for a in axes if prev is None or abs(p[a] - prev[a]) > 1e-9)

    def lin(kind: str, op: Op, start: dict, end: dict, f: float):
        """Agrega un G0 (``kind`` "rapid") o G1 de ``start`` a ``end``; ``f`` en mm/min. Un tramo
        nulo no emite nada."""
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

    # --- una operación tras otra ------------------------------------------------------------------

    for k, op in enumerate(program.ops):
        if origins is not None:                    # completar el origen de lo emitido por la op anterior
            origins.extend([k - 1] * (len(out) - len(origins)))
        frame = op.frame or g54                    # G54 vigente en esta operación (G10 puede cambiarlo en marcha)
        if op.kind in ("rapid", "feed"):
            lin(op.kind, op, op.start, op.end, op.feed)
        elif op.kind == "probe":
            # se envía el recorrido completo: el controlador para al tocar
            c = coords(op.value, op.start, frame)
            out.append((f"G38.2 {c} F{_n(op.feed)}", op.line))
            feed = None                            # G38.2 cambió el F modal: el próximo G1 manda el suyo
        elif op.kind == "wcs":
            # G10 L20 con el valor que da el mismo origen en el controlador (sin G92 ni G43: ya aplicados)
            vals = " ".join(f"{a}{_n(op.end[a])}" for a in sorted(op.end, key=_axis_rank) if a != "L")
            # en Marlin G10 es «retraer filamento»: el mismo efecto es G92 (la posición actual vale esto)
            out.append(((f"G92 {vals}") if marlin else f"G10 L20 P1 {vals}", op.line))
        elif op.kind == "arc":
            a, b, _ = op.arc.plane
            # Más de una vuelta (hélices): grbl y Marlin no tienen la palabra P (cantidad de vueltas)
            # de LinuxCNC, así que se envían cuerdas; vale igual para todos los dialectos.
            if abs(op.arc.sweep) > 2 * math.pi + 1e-9:
                prev = op.start
                for p in linearize(op, machine.arc_tolerance):
                    lin("feed", op, prev, p, op.feed)
                    prev = p
                continue
            letters = {"X": "I", "Y": "J", "Z": "K"}  # letra del centro según el eje del plano
            g = "G2" if op.arc.sweep < 0 else "G3"     # barrido negativo = horario (G2)
            plane = {("X", "Y"): "G17", ("Z", "X"): "G18", ("Y", "Z"): "G19"}[(a, b)]
            c = " ".join(f"{ax}{_n(op.end[ax] - frame[ax])}" for ax in axes)
            # centro incremental desde el punto inicial (G91.1 del encabezado)
            ij = f"{letters[a]}{_n(op.arc.center[0] - op.start[a])} {letters[b]}{_n(op.arc.center[1] - op.start[b])}"
            fs = "" if feed == op.feed else f" F{_n(op.feed)}"
            feed = op.feed
            pre = f"{plane} " if plane != "G17" else ""
            out.append((f"{pre}{g} {c} {ij}{fs}", op.line))
            if plane != "G17":                     # G18/G19: volver a G17 para que lo demás no dependa del plano
                out.append(("G17", op.line))
        elif op.kind == "dwell":
            # Marlin: G4 P son milisegundos y G4 S segundos; grbl y LinuxCNC: P en segundos
            out.append(((f"G4 S{_n(float(op.value))}") if marlin else f"G4 P{_n(float(op.value))}", op.line))
        elif op.kind == "spindle":
            d, s = op.value
            if d == "off":
                out.append(("M5", op.line))
            else:
                # S solo si cambió; en plasma sin S no se manda «S0» (la antorcha no tiene rpm)
                ss = "" if rpm == s or (machine.is_plasma and not s) else f" S{_n(s)}"
                rpm = s
                out.append((("M3" if d == "cw" else "M4") + ss, op.line))
        elif op.kind == "coolant":
            v = op.value
            if not v:
                out.append(("M9", op.line))
            else:
                # M7 niebla, M8 inundación; sorted(): orden fijo aunque sea un set (Python 3.4)
                out += [(("M7" if c == "niebla" else "M8"), op.line) for c in sorted(v)]
        elif op.kind == "tool":
            if dialect == "linuxcnc":                # M5 + T M6: dos líneas, igual que los otros dialectos
                out += [("M5", op.line), (f"T{op.value} M6", op.line)]
            else:
                # grbl y Marlin no cambian herramienta: se apaga el husillo y se pausa para el cambio manual
                msg = f"cambiar a la herramienta T{op.value}"
                out += [("M5", op.line), ((f"M0 {msg}") if marlin else f"M0 (MSG, {msg})", op.line)]
                rpm = None                         # tras el cambio, reenviar S con el próximo M3/M4
        elif op.kind == "pause":
            text = str(op.value)[:60]              # corto: grbl acepta líneas de ~80 caracteres
            out.append(((f"M0 {text}") if marlin else f"M0 (MSG, {text})", op.line))
        elif op.kind == "end":
            # Marlin no tiene M2 y su M30 borra un archivo de la SD: solo se apaga el husillo
            out.append(({"linuxcnc": "M30", "marlin": "M5"}.get(dialect, "M2"), op.line))
    if origins is not None:
        origins.extend([len(program.ops) - 1] * (len(out) - len(origins)))
        origins[0] = -1                            # el encabezado no sale de ninguna operación
    return out


def emit_text(program: Program, machine: Machine, dialect: str = "grbl") -> str:
    """Como ``emit`` pero solo el texto, una línea por renglón y con salto de línea final (listo para
    guardar en un archivo). Lanza ``ValueError`` si el dialecto no existe."""
    return "\n".join(t for t, _ in emit(program, machine, dialect)) + "\n"
