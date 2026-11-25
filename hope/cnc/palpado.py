"""Probing routines («palpado»): find edges, corners, centres, the surface, tool lengths and a height map.

What it is
----------
The probing screen every modern controller has (Mach3's probing plug-ins, UCCNC, LinuxCNC's
probe screens): the operator jogs a touch probe (or the tool with a touch plate) close to the
workpiece, picks a routine and TISMA moves, touches and sets the work zero by itself.

Routines (``ROUTINES``):
    borde             one edge (X+, X−, Y+ or Y−): the edge becomes the chosen coordinate (0).
    esquina           an outside corner: touches X and Y and zeroes both at the corner.
    centro_agujero    centre of a bore or a pocket: touches −X, +X, −Y, +Y (twice on X) and zeroes
                      X and Y at the centre; also reports the diameter.
    centro_saliente   centre of a boss (a round or square piece sticking up): goes around it from
                      above, touches both sides of X and Y and zeroes at the centre; reports the size.
    superficie        top surface: Z0 where it touches (minus the touch plate thickness).
    herramienta       tool length with a fixed tool setter: the first tool is the reference; after
                      each tool change the G54 Z is corrected by the difference.
    mapa              height map: probes a grid of points on the surface (warped boards, PCBs) so
                      the loaded program can follow it (``apply_height_map``).

How it fits
-----------
``session.py`` (``run_probe``) runs a :class:`Prober` in a background thread, with these
callables: ``send`` = the session's MDI (each line is checked against the travel limits before
it moves the machine), ``wait`` = wait until the controller is «Idle», and the ``link`` (where the
contact point comes from: :meth:`hope.cnc.links.Link.last_probe`). The probing moves are plain
``G38.2`` lines, so every controller with probing works (grbl, LinuxCNC, Marlin with G38).
``apply_height_map`` uses the interpreter (``gcode.interpret``) to rebuild the program with the
map applied. The app's «Palpado» dialog builds its form from ``ROUTINES`` and ``PROBE_FIELDS``.

Units and conventions
---------------------
mm and mm/min. Inside the :class:`Prober` everything is in **machine coordinates** (what the
controller reports); the lines it sends are in **work coordinates** (``G90``: machine − WCO, the
active offset the controller reports), because ``G53`` cannot be combined with ``G38.2``. Zeros
are set with ``G10 L20 P1`` («the current position is worth v»), the same as Mach3's «Zero X»
button, so they work whatever G92 / tool length is active. The probe tip is a ball of
``diametro_punta``: an X/Y edge is one radius beyond the centre of the ball when it touches.
Every probe touches twice: fast to find the surface, back off ``retroceso`` and slowly again for
the precise value (the slow touch is the one used).

Safety
------
* Every line goes through the session's MDI check: a probe that would go beyond the travel is
  rejected before moving.
* ``G38.2`` raises an alarm on the controller if it does not touch within the distance: the
  routine stops right there with a Spanish message («no tocó nada»).
* Moves next to the part (going down beside a boss, approaching a corner) use ``G38.3`` (probe
  without alarm): if the probe touches something it did not expect, it stops there instead of
  pushing the probe into the part, and the routine is aborted.
* The operator's Stop cancels the routine before its next line (``Prober.cancelled``).

Example
-------
::

    p = Prober(send=session.mdi, wait=session._wait_idle, link=session.link,
               settings=ProbeSettings(tip_diameter=2))
    r = p.run("centro_agujero", {"distancia": 30})
    print(r["mensajes"])            # ["Centro: X 100.012 Y 50.003 (máquina)", "Diámetro X 24.98 / Y 25.01 mm", …]
"""

from __future__ import annotations

import math
import re
import threading

# ==================================================================================================
# ======================================= FORMS (for the app) ======================================
# ==================================================================================================

# --- common probe settings ------------------------------------------------------------------------
# Shown under every routine in the dialog: (key, label, unit, default). Reasonable ranges:
#   avance          fast touch feed, mm/min. Typical 100–500. Too fast = the probe's spring travel
#                   (usually 3–5 mm) may not be enough to stop before it breaks.
#   avance_fino     slow (precise) touch feed, mm/min. Typical 20–60. Repeatability improves as it
#                   goes down; 0 = do a single touch (fast) only.
#   retroceso       back-off between the fast and the slow touch, and after each touch, mm. 1–3.
#   diametro_punta  probe ball diameter (or tool diameter when touching with the tool), mm. For a
#                   touch plate under the tool on Z only, it does not matter.
PROBE_FIELDS = [
    ("avance", "Avance de búsqueda", "mm/min", 300),
    ("avance_fino", "Avance fino (0 = un solo toque)", "mm/min", 40),
    ("retroceso", "Retroceso entre toques", "mm", 2),
    ("diametro_punta", "Diámetro de la punta", "mm", 2),
]

# --- routines -------------------------------------------------------------------------------------
# kind → (title, fields). Each field = (key, label, unit, default) or, for a choice,
# (key, label, "", default, [(value, label), ...]). Lists, not dicts, for the order on Python 3.4.
ROUTINES = {
    "borde": ("Borde (X o Y)", [
        ("direccion", "Dirección de búsqueda", "", "X+", [
            ("X+", "hacia +X (el borde está a la derecha)"), ("X-", "hacia −X (el borde está a la izquierda)"),
            ("Y+", "hacia +Y (el borde está atrás)"), ("Y-", "hacia −Y (el borde está adelante)")]),
        ("distancia", "Distancia máxima de búsqueda", "mm", 20),
        ("valor", "El borde queda en", "mm", 0),
    ]),
    "esquina": ("Esquina exterior", [
        ("esquina", "Esquina de la pieza", "", "inf_izq", [
            ("inf_izq", "adelante a la izquierda"), ("inf_der", "adelante a la derecha"),
            ("sup_izq", "atrás a la izquierda"), ("sup_der", "atrás a la derecha")]),
        ("entrada", "Cuánto entra sobre cada lado", "mm", 10),
        ("distancia", "Distancia máxima de búsqueda", "mm", 20),
    ]),
    "centro_agujero": ("Centro de un agujero", [
        ("distancia", "Recorrido máximo desde el centro", "mm", 30),
    ]),
    "centro_saliente": ("Centro de un saliente (pieza redonda o cuadrada)", [
        ("ancho_x", "Ancho aproximado en X", "mm", 50),
        ("ancho_y", "Ancho aproximado en Y", "mm", 50),
        ("bajar", "Cuánto bajar al costado", "mm", 5),
        ("holgura", "Holgura al rodearla", "mm", 5),
    ]),
    "superficie": ("Superficie (Z0)", [
        ("distancia", "Distancia máxima de búsqueda", "mm", 30),
        ("espesor_placa", "Espesor de la placa de contacto (0 = sin placa)", "mm", 0),
    ]),
    "herramienta": ("Largo de herramienta (sensor fijo)", [
        ("modo", "Qué hacer", "", "nueva", [
            ("referencia", "medir la herramienta de referencia (con el Z0 ya hecho)"),
            ("nueva", "medir la herramienta nueva y corregir Z0")]),
        ("x", "X del sensor (máquina)", "mm", 0),
        ("y", "Y del sensor (máquina)", "mm", 0),
        ("distancia", "Distancia máxima de bajada", "mm", 80),
    ]),
    "mapa": ("Mapa de superficie (chapas torcidas, plaquetas)", [
        ("x", "X inicial (pieza)", "mm", 0),
        ("y", "Y inicial (pieza)", "mm", 0),
        ("ancho", "Ancho", "mm", 100),
        ("alto", "Alto", "mm", 80),
        ("columnas", "Columnas", "", 5),
        ("filas", "Filas", "", 4),
        ("altura", "Altura de traslado (Z pieza)", "mm", 3),
        ("distancia", "Distancia máxima de búsqueda", "mm", 6),
    ]),
}

# Corner (as seen from the front of the machine) → direction towards the part on X and on Y.
# «inf_izq» (front-left corner of the part): the probe starts outside, to the left and in front,
# so the part is towards +X and +Y.
CORNER_SIDES = {"inf_izq": (1, 1), "inf_der": (-1, 1), "sup_izq": (1, -1), "sup_der": (-1, -1)}


# Invalid routine data or a failed touch. Message in Spanish, shown to the operator.
class ProbeError(ValueError):
    pass


class ProbeSettings:
    """Probe settings for a run (see ``PROBE_FIELDS`` for meanings and ranges).

    Attributes (mm, mm/min): feed, slow_feed (0 = single touch), retract, tip_diameter.
    """

    def __init__(self, feed=300.0, slow_feed=40.0, retract=2.0, tip_diameter=2.0):
        self.feed = float(feed)
        self.slow_feed = float(slow_feed)
        self.retract = float(retract)
        self.tip_diameter = float(tip_diameter)
        if self.feed <= 0 or self.slow_feed < 0 or self.retract <= 0 or self.tip_diameter < 0:
            raise ProbeError("datos de la sonda: avance y retroceso mayores que 0, diámetro y avance fino no negativos")

    @property
    def radius(self) -> float:
        """Probe ball radius (mm): what separates the touched surface from the reported centre."""
        return self.tip_diameter / 2.0

    @classmethod
    def from_params(cls, params: dict) -> ProbeSettings:
        """Builds the settings from the dialog's values (keys of ``PROBE_FIELDS``, defaults if absent)."""
        d = {k: v for k, _, _, v in PROBE_FIELDS}
        get = lambda k: _num(params.get(k, d[k]), k)
        return cls(get("avance"), get("avance_fino"), get("retroceso"), get("diametro_punta"))


def _num(v, key: str) -> float:
    """``v`` as a float (accepts «2,5»); raises ProbeError naming ``key`` if it is not a number."""
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        raise ProbeError(f"«{key}» tiene que ser un número") from None


def _fmt(v: float) -> str:
    """A coordinate for a G-code line: 4 decimals (0.1 µm, below any machine's resolution) and no
    trailing zeros, so the lines stay short for the controller's buffer."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ==================================================================================================
# ========================================== HEIGHT MAP ============================================
# ==================================================================================================

class HeightMap:
    """A grid of measured surface heights (work Z, mm) and its bilinear interpolation.

    Attributes:
        x0, y0     work coordinates of the first point (column 0, row 0), mm.
        dx, dy     spacing between columns / rows, mm (0 when there is only one column / row).
        z          rows of heights: ``z[row][col]`` (row 0 = y0).

    Outside the grid the nearest edge value is used (clamped), never extrapolated: extrapolating a
    tilt beyond what was measured could send the tool into the part.
    """

    def __init__(self, x0: float, y0: float, dx: float, dy: float, z: list):
        self.x0, self.y0, self.dx, self.dy = float(x0), float(y0), float(dx), float(dy)
        self.z = [[float(v) for v in row] for row in z]

    @property
    def cols(self) -> int:
        return len(self.z[0])

    @property
    def rows(self) -> int:
        return len(self.z)

    def at(self, x: float, y: float) -> float:
        """Interpolated surface height (work Z, mm) at work ``(x, y)``.

        Bilinear: inside the cell, first along X on the two rows, then along Y between them; the
        result is exact at the measured points and continuous across cells (no steps in the cut).
        """
        def cell(v, v0, step, n):
            """(index of the cell's first point, fraction 0..1 inside it), clamped to the grid."""
            if n == 1 or step <= 0:
                return 0, 0.0
            t = min(max((v - v0) / step, 0.0), n - 1.0)
            i = min(int(t), n - 2)
            return i, t - i
        i, fx = cell(x, self.x0, self.dx, self.cols)
        j, fy = cell(y, self.y0, self.dy, self.rows)
        row = lambda r: self.z[r][i] if self.cols == 1 else self.z[r][i] * (1 - fx) + self.z[r][i + 1] * fx
        return row(j) if self.rows == 1 else row(j) * (1 - fy) + row(j + 1) * fy

    def to_dict(self) -> dict:
        """For the app (JSON): position, spacing, heights and the range (min/max), mm."""
        flat = [v for row in self.z for v in row]
        return {"x0": self.x0, "y0": self.y0, "dx": self.dx, "dy": self.dy, "z": self.z,
                "min": min(flat), "max": max(flat)}


# G codes that make a program unsuitable for the height map (see apply_height_map): incremental
# moves, inches, other origins / offsets, tilted planes, inverse time and probing. Rotary axis
# words (A/B/C) are refused too: a map measured on a flat table means nothing on a rotated part.
_MAP_REFUSE_G = {91.0, 20.0, 92.0, 52.0, 43.0, 43.1, 10.0, 55.0, 56.0, 57.0, 58.0, 59.0, 68.2, 93.0,
                 38.2, 38.3, 31.0}
# Words a rebuilt motion line no longer needs: axes, arc / cycle words and F (each straight piece
# carries its own coordinates and feed), plus the motion G codes (G0–G3, G80–G89, G98/G99).
_MOTION_LETTERS = set("XYZIJKRQLPF")
_MOTION_G = {0.0, 1.0, 2.0, 3.0, 98.0, 99.0} | {80.0 + i for i in range(10)}
# One word of a stripped line (upper case, no spaces): letter + number.
_WORD = re.compile(r"([A-Z])([-+]?[0-9]*\.?[0-9]*)")


def _words(code: str) -> list:
    """``[(letter, text of the number), ...]`` of a line already stripped by ``strip_line``."""
    return [(m.group(1), m.group(2)) for m in _WORD.finditer(code)]


def _gnum(text: str) -> float:
    """The number of a G word (``"01"`` → 1.0, ``"38.2"`` → 38.2); -1 if it is not a number."""
    try:
        return round(float(text), 1)
    except ValueError:
        return -1.0


def apply_height_map(text: str, hmap: HeightMap, machine, segment: float = 5.0) -> str:
    """Rebuilds a program so that its Z follows the measured surface.

    Every move is replaced by straight lines (arcs too, as chords within the machine's arc
    tolerance) no longer than ``segment`` mm in XY, and each end point gets ``hmap.at(x, y)`` added to
    its Z: the depth below the real surface stays the programmed one even if the board is warped.
    Lines that do not move (M3, M6, G4, comments…) are copied unchanged.

    Args:
        text: the program (G90, mm, G54; see Raises).
        hmap: the measured :class:`HeightMap` (work coordinates of the same G54).
        machine: the machine (for the interpreter).
        segment: maximum XY length of each straight piece, mm. Smaller = follows the surface better,
            longer program. 2–10 is typical (the map's own spacing is usually 10–30 mm).

    Returns:
        The new program text (Spanish header comment saying the map was applied).

    Raises:
        ProbeError: the program uses something the rebuild cannot keep exactly (G91, G20, G92,
            G52, G43, G10, G55–G59, G68.2, G93, probing, rotary axes) or has errors.
    """
    from hope.cnc.gcode import interpret, linearize, strip_line

    lines = text.splitlines()
    for n, ln in enumerate(lines, 1):
        for letter, num in _words(strip_line(ln)[0]):
            if letter in "ABC" or (letter == "G" and _gnum(num) in _MAP_REFUSE_G):
                raise ProbeError(f"el mapa de superficie no se puede aplicar: la línea {n} usa {letter}{num} "
                                 "(hace falta un programa en G90, mm, G54, sin G92/G43 ni ejes rotativos)")
    prog = interpret(text, machine)
    errors = [d for d in prog.diags if d.level == "error"]
    if errors:
        raise ProbeError(f"el programa tiene errores: {errors[0]}")
    by_line: dict = {}
    for op in prog.ops:
        by_line.setdefault(op.line, []).append(op)

    lo, hi = min(min(r) for r in hmap.z), max(max(r) for r in hmap.z)
    head = (f"(mapa de superficie aplicado: {hmap.cols}x{hmap.rows} puntos, Z {lo:+.3f} a {hi:+.3f} mm, "
            f"tramos de {segment:g} mm)")
    out = [head]
    feed = None
    for n, ln in enumerate(lines, 1):
        moves = [op for op in by_line.get(n, []) if op.kind in ("rapid", "feed", "arc")]
        if not moves:
            out.append(ln)
            continue
        # Keep whatever else the line does (M3 S…, M8, T…) on its own line before the moves.
        rest = " ".join(letter + num for letter, num in _words(strip_line(ln)[0])
                        if letter not in _MOTION_LETTERS and not (letter == "G" and _gnum(num) in _MOTION_G))
        if rest:
            out.append(rest)
        for op in moves:
            g = op.frame or machine.offset("G54")
            if op.kind == "arc":
                pts = [op.start] + linearize(op, machine.arc_tolerance)
            else:
                pts = [op.start, op.end]
            for a, b in zip(pts, pts[1:]):
                # Rapids are not split (they travel above the part); feeds are, to follow the map.
                length = math.hypot(b["X"] - a["X"], b["Y"] - a["Y"])
                steps = 1 if op.kind == "rapid" else max(1, math.ceil(length / segment))
                for k in range(1, steps + 1):
                    t = k / steps
                    x = a["X"] + (b["X"] - a["X"]) * t - g["X"]
                    y = a["Y"] + (b["Y"] - a["Y"]) * t - g["Y"]
                    z = a["Z"] + (b["Z"] - a["Z"]) * t - g["Z"] + hmap.at(x, y)
                    words = f"X{_fmt(x)} Y{_fmt(y)} Z{_fmt(z)}"
                    if op.kind == "rapid":
                        out.append("G0 " + words)
                    else:
                        out.append("G1 " + words + (f" F{op.feed:g}" if op.feed != feed else ""))
                        feed = op.feed
    return "\n".join(out) + "\n"


# ==================================================================================================
# ============================================= PROBER =============================================
# ==================================================================================================

class Prober:
    """Runs one probing routine on a live connection (see the module header for the routines).

    Args:
        send: function that runs one MDI line (the session's ``mdi``: checked against the limits).
        wait: function that blocks until the controller is «Idle» again.
        link: the connection (``snapshot()`` for position/offsets/state, ``last_probe()`` for the
            exact contact point when the controller reports it).
        settings: :class:`ProbeSettings`.
        memory: dict kept by the session between runs (the tool-setter reference lives here).

    The session also sets ``z_top`` (machine Z maximum, mm), needed by «herramienta».

    Attributes:
        cancelled: ``threading.Event``; set it (the operator pressed Stop) and the routine raises
            ProbeError before sending its next line.
    """

    def __init__(self, send, wait, link, settings: ProbeSettings, memory: dict | None = None):
        self.send, self.wait, self.link, self.s = send, wait, link, settings
        self.memory = memory if memory is not None else {}
        self.z_top = None                   # machine Z maximum (mm), set by the session; for «herramienta»
        self.cancelled = threading.Event()
        self.messages: list = []

    # --- basic moves --------------------------------------------------------------------------------
    def _snap(self) -> dict:
        return self.link.snapshot()

    def _line(self, line: str) -> None:
        """Sends one line and waits until the machine is at rest. Stops the routine if cancelled or
        if the controller went into alarm (a G38.2 without contact, a limit switch…)."""
        if self.cancelled.is_set():
            raise ProbeError("palpado cancelado")
        self.send(line)
        self.wait()
        st = self._snap()
        if st.get("estado") == "Alarm":
            raise ProbeError(f"el controlador quedó en alarma: {st.get('alarma') or 'sin detalle'} "
                             "(desbloquear y revisar la sonda)")

    def _work(self, axis: str, machine_value: float) -> float:
        """Work coordinate of a machine value on ``axis``: machine − WCO (the active offset)."""
        return machine_value - float(self._snap().get("wco", {}).get(axis, 0.0))

    def pos(self, axis: str) -> float:
        """Current machine position of ``axis`` (mm)."""
        return float(self._snap()["mpos"][axis])

    def move(self, axis_values: dict, feed: float | None = None) -> None:
        """Straight move to machine coordinates ``{axis: mm}`` (G1 at ``feed``, or G0 if ``None``).

        Sent in work coordinates (G90) because the session's MDI checks those lines like any other.
        Axes are written in the fixed order X Y Z (dict order is not reliable on Python 3.4).
        """
        words = " ".join(f"{a}{_fmt(self._work(a, axis_values[a]))}" for a in ("X", "Y", "Z") if a in axis_values)
        self._line(f"G90 G1 {words} F{feed:g}" if feed else f"G90 G0 {words}")

    def _touch(self, axis: str, direction: int, distance: float, feed: float, code: str = "38.2") -> dict | None:
        """One probing move: ``G38.x`` from here up to ``distance`` mm towards ``direction`` (+1/−1).

        Returns the contact point (machine coordinates, all axes) or ``None`` if a ``G38.3`` did
        not touch. Where the point comes from:
          * ``link.last_probe()`` when the controller reports it (grbl's ``[PRB:…]``): the exact
            position latched when the probe input switched, not where the machine stopped after
            decelerating (that can be 0.01–0.1 mm further);
          * otherwise (Marlin), the position the machine stopped at, which with the slow touch
            is within a few µm.
        """
        before = self.link.last_probe()
        n0 = before["n"] if before else 0
        target = self.pos(axis) + direction * distance
        self._line(f"G90 G38.{code[-1]} {axis}{_fmt(self._work(axis, target))} F{feed:g}")
        r = self.link.last_probe()
        if r is not None and r["n"] != n0:
            return dict(r["pos"]) if r["ok"] else None
        if code == "38.3":
            # No report: touched if it stopped before the end of the travel.
            here = self._snap()["mpos"]
            return dict(here) if abs(here[axis] - target) > 1e-3 else None
        return dict(self._snap()["mpos"])

    def probe(self, axis: str, direction: int, distance: float) -> float:
        """Finds a surface along ``axis``: fast touch, back off, slow touch, back off again.

        Returns the machine coordinate of the probe centre at the (slow) contact. The machine ends
        ``retroceso`` mm away from the surface, so it can move sideways safely.

        Raises:
            ProbeError: it did not touch within ``distance`` mm (the controller alarm says so).
        """
        if distance <= 0:
            raise ProbeError("la distancia de búsqueda tiene que ser mayor que 0")
        c = self._touch(axis, direction, distance, self.s.feed)
        back = c[axis] - direction * self.s.retract
        self.move({axis: back}, self.s.feed)
        if self.s.slow_feed > 0:
            c = self._touch(axis, direction, 2 * self.s.retract, self.s.slow_feed)
            self.move({axis: back}, self.s.feed)
        return c[axis]

    def guarded_move(self, axis: str, target: float) -> None:
        """Moves ``axis`` to the machine coordinate ``target`` with ``G38.3`` (stops if the probe
        touches anything). Used to approach the part from the side: if it touches, the routine
        stops instead of pushing the probe into the part."""
        here = self.pos(axis)
        if abs(target - here) < 1e-6:
            return
        direction = 1 if target > here else -1
        hit = self._touch(axis, direction, abs(target - here), self.s.feed, code="38.3")
        if hit is not None:
            raise ProbeError(f"la sonda tocó algo al moverse en {axis} (en {hit[axis]:.3f}): "
                             "revisar la posición inicial y las medidas")

    def set_work(self, values: dict) -> None:
        """Sets the G54 zero so that, at the CURRENT position, each axis is worth ``values[axis]``
        (``G10 L20 P1``, like Mach3's «Zero X» button). Axes in fixed X Y Z order."""
        words = " ".join(f"{a}{_fmt(values[a])}" for a in ("X", "Y", "Z") if a in values)
        self._line(f"G10 L20 P1 {words}")

    def say(self, msg: str) -> None:
        self.messages.append(msg)

    # --- routines -------------------------------------------------------------------------------------
    def run(self, kind: str, params: dict) -> dict:
        """Runs routine ``kind`` with the dialog's ``params``.

        Returns ``{"ok": True, "tipo", "mensajes": [Spanish lines], "valores": {…}}``; ``valores``
        holds the measured numbers (machine coordinates, sizes, the height map).

        Raises:
            ProbeError: unknown routine, invalid data, no contact, unexpected contact, cancelled.
        """
        fn = getattr(self, "_r_" + kind, None)
        if kind not in ROUTINES or fn is None:
            raise ProbeError(f"rutina de palpado desconocida: {kind}")
        defaults = {f[0]: f[3] for f in ROUTINES[kind][1]}
        p = dict(defaults)
        p.update({k: v for k, v in params.items() if k in defaults})
        for f in ROUTINES[kind][1]:
            if len(f) == 4:
                p[f[0]] = _num(p[f[0]], f[0])
        self.messages = []
        values = fn(p)
        return {"ok": True, "tipo": kind, "mensajes": list(self.messages), "valores": values}

    def _r_borde(self, p: dict) -> dict:
        """One edge: probe along the chosen direction; the edge is one tip radius beyond the
        contact (the ball's surface touched it, not its centre)."""
        d = str(p["direccion"]).upper()
        if d not in ("X+", "X-", "Y+", "Y-"):
            raise ProbeError("dirección: X+, X-, Y+ o Y-")
        axis, sign = d[0], (1 if d[1] == "+" else -1)
        edge = self.probe(axis, sign, p["distancia"]) + sign * self.s.radius
        self.set_work({axis: self.pos(axis) - edge + p["valor"]})
        self.say(f"Borde {d}: {axis} {edge:.3f} (máquina) → {axis}{p['valor']:g} de la pieza")
        return {axis: edge}

    def _r_esquina(self, p: dict) -> dict:
        """Outside corner. The probe starts outside the part, diagonally from the corner and below
        its top. For X it first slides ``entrada`` mm along Y (so it faces the side of the part),
        probes X, returns; then the same for Y. Both slides use G38.3 (stop on touch)."""
        if p["esquina"] not in CORNER_SIDES:
            raise ProbeError("esquina: inf_izq, inf_der, sup_izq o sup_der")
        sx, sy = CORNER_SIDES[p["esquina"]]
        x0, y0 = self.pos("X"), self.pos("Y")
        self.guarded_move("Y", y0 + sy * p["entrada"])
        ex = self.probe("X", sx, p["distancia"]) + sx * self.s.radius
        self.move({"X": x0}, self.s.feed)
        self.move({"Y": y0}, self.s.feed)
        self.guarded_move("X", x0 + sx * p["entrada"])
        ey = self.probe("Y", sy, p["distancia"]) + sy * self.s.radius
        self.move({"Y": y0}, self.s.feed)
        self.move({"X": x0}, self.s.feed)
        self.set_work({"X": self.pos("X") - ex, "Y": self.pos("Y") - ey})
        self.say(f"Esquina: X {ex:.3f} Y {ey:.3f} (máquina) → X0 Y0 de la pieza")
        return {"X": ex, "Y": ey}

    def _bore_axis(self, axis: str, reach: float) -> tuple:
        """Touches both walls of a bore on ``axis`` from the current point and goes to the middle.
        Returns (centre, inside width between the walls), machine mm."""
        a = self.probe(axis, -1, reach)
        b = self.probe(axis, 1, reach + abs(self.pos(axis) - a))
        centre = (a + b) / 2.0
        self.move({axis: centre}, self.s.feed)
        return centre, b - a + 2 * self.s.radius

    def _r_centro_agujero(self, p: dict) -> dict:
        """Bore centre: X (−, +) → middle; Y (−, +) → middle; X again (now that Y is centred the X
        chord is the real diameter, the first one may have been a shorter chord) and zero X0 Y0."""
        cx, _ = self._bore_axis("X", p["distancia"])
        cy, dy = self._bore_axis("Y", p["distancia"])
        cx, dx = self._bore_axis("X", p["distancia"])
        self.set_work({"X": 0.0, "Y": 0.0})
        self.say(f"Centro: X {cx:.3f} Y {cy:.3f} (máquina) → X0 Y0 de la pieza")
        self.say(f"Diámetro: X {dx:.3f} / Y {dy:.3f} mm")
        return {"X": cx, "Y": cy, "diametro_x": dx, "diametro_y": dy}

    def _boss_axis(self, axis: str, width: float, down: float, clearance: float) -> tuple:
        """Touches both outer sides of a boss on ``axis``. Starts above its (approximate) centre;
        for each side: out to half the width + clearance, down ``down`` mm (G38.3), probe towards
        the centre, back up, and finally to the middle. Returns (centre, outside width), mm."""
        c0, z0 = self.pos(axis), self.pos("Z")
        out = width / 2.0 + clearance + self.s.radius
        sides = []
        for sign in (-1, 1):
            self.move({axis: c0 + sign * out}, self.s.feed)
            self.guarded_move("Z", z0 - down)
            sides.append(self.probe(axis, -sign, clearance + self.s.radius + width / 2.0))
            self.move({"Z": z0}, self.s.feed)
        a, b = sides
        centre = (a + b) / 2.0
        self.move({axis: centre}, self.s.feed)
        return centre, b - a - 2 * self.s.radius

    def _r_centro_saliente(self, p: dict) -> dict:
        """Boss centre: around it on X and then on Y (from above its centre) and zero X0 Y0."""
        if min(p["ancho_x"], p["ancho_y"], p["bajar"], p["holgura"]) <= 0:
            raise ProbeError("anchos, bajada y holgura tienen que ser mayores que 0")
        cx, wx = self._boss_axis("X", p["ancho_x"], p["bajar"], p["holgura"])
        cy, wy = self._boss_axis("Y", p["ancho_y"], p["bajar"], p["holgura"])
        self.set_work({"X": 0.0, "Y": 0.0})
        self.say(f"Centro: X {cx:.3f} Y {cy:.3f} (máquina) → X0 Y0 de la pieza")
        self.say(f"Medida: X {wx:.3f} / Y {wy:.3f} mm")
        return {"X": cx, "Y": cy, "ancho_x": wx, "ancho_y": wy}

    def _r_superficie(self, p: dict) -> dict:
        """Top surface: probe down; with a touch plate the surface is its thickness lower."""
        z = self.probe("Z", -1, p["distancia"]) - p["espesor_placa"]
        self.set_work({"Z": self.pos("Z") - z})
        self.say(f"Superficie: Z {z:.3f} (máquina) → Z0 de la pieza")
        return {"Z": z}

    def _r_herramienta(self, p: dict) -> dict:
        """Tool length with a fixed tool setter.

        Goes up to the top of Z (``G53 G0 Z<max>``), over the setter (``G53 G0 X Y``), probes down
        and comes back up. «referencia» stores where the setter's top is in WORK Z with the tool
        whose Z0 is already right. «nueva» probes the new tool and sets Z so the setter's top has
        that same work Z again (``G10 L20``): a tool 3 mm longer touches 3 mm higher and Z0 moves
        3 mm up with it. Done with work coordinates (not by reading the G54 offset) so it works
        whatever G92 or tool length offset is active and with controllers that do not report offsets.
        """
        if self.z_top is None:
            raise ProbeError("hace falta el límite máximo de Z en la máquina para subir antes de ir al sensor")
        if p["modo"] not in ("referencia", "nueva"):
            raise ProbeError("modo: referencia o nueva")
        ref = self.memory.get("sensor_z_pieza")
        if p["modo"] == "nueva" and ref is None:         # checked before moving anything
            raise ProbeError("primero medir la herramienta de referencia (con el Z0 de la pieza ya hecho)")
        self._line(f"G53 G0 Z{_fmt(self.z_top)}")
        self._line(f"G53 G0 X{_fmt(p['x'])} Y{_fmt(p['y'])}")
        c = self.probe("Z", -1, p["distancia"])
        if p["modo"] == "referencia":
            self.memory["sensor_z_pieza"] = self._work("Z", c)
            self._line(f"G53 G0 Z{_fmt(self.z_top)}")
            self.say(f"Herramienta de referencia: el sensor está en Z {self.memory['sensor_z_pieza']:.3f} de la pieza. "
                     "Ya se puede cambiar la herramienta.")
            return {"Z": c}
        change = self._work("Z", c) - ref          # > 0: the new tool is longer (touched higher in work Z)
        self.set_work({"Z": self.pos("Z") - c + ref})
        self._line(f"G53 G0 Z{_fmt(self.z_top)}")
        self.say(f"Herramienta nueva: {change:+.3f} mm respecto de la de referencia → Z0 corregido")
        return {"Z": c, "diferencia": change}

    def _r_mapa(self, p: dict) -> dict:
        """Height map: a columns × rows grid. Travels at ``altura`` (work Z) between points in a
        zig-zag (shorter than going back to the first column every row), probes down at each one
        and stores the work Z of the contact. Ends at the travel height."""
        cols, rows = int(p["columnas"]), int(p["filas"])
        if cols < 1 or rows < 1 or cols * rows > 400:
            raise ProbeError("columnas y filas: de 1 en adelante, hasta 400 puntos en total")
        if p["ancho"] < 0 or p["alto"] < 0 or (cols > 1 and p["ancho"] == 0) or (rows > 1 and p["alto"] == 0):
            raise ProbeError("ancho y alto tienen que ser mayores que 0 (o una sola columna / fila)")
        dx = p["ancho"] / (cols - 1) if cols > 1 else 0.0
        dy = p["alto"] / (rows - 1) if rows > 1 else 0.0
        z = [[0.0] * cols for _ in range(rows)]
        self._line(f"G90 G0 Z{_fmt(p['altura'])}")
        for j in range(rows):
            order = range(cols) if j % 2 == 0 else range(cols - 1, -1, -1)
            for i in order:
                self._line(f"G90 G0 X{_fmt(p['x'] + i * dx)} Y{_fmt(p['y'] + j * dy)}")
                z[j][i] = self._work("Z", self.probe("Z", -1, p["altura"] + p["distancia"]))
                self._line(f"G90 G0 Z{_fmt(p['altura'])}")
        hmap = HeightMap(p["x"], p["y"], dx, dy, z)
        d = hmap.to_dict()
        self.say(f"Mapa: {cols}×{rows} puntos, Z de {d['min']:+.3f} a {d['max']:+.3f} mm "
                 f"(diferencia {d['max'] - d['min']:.3f} mm)")
        return {"mapa": d}
