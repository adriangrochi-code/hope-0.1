"""Machining wizards («asistentes», like Mach3's Wizards): small G-code programs from a few numbers.

What it is
----------
Generators for the everyday jobs that do not deserve a CAM session: face a surface, mill a
rectangular or circular pocket, drill a bolt circle or a grid of holes, cut out a rectangle or a
circle. Each one takes its dimensions (mm, work coordinates) plus the tool and the speeds from the
selected profiles (``Herramienta`` (tool) = end mill diameter; feed, plunge and RPM computed by the
feeds-and-speeds calculator), and returns a commented G-code program the app loads like any other
file — so it is checked, simulated and approved the same way.

Available wizards (``WIZARDS``):
    planeado          facing: zig-zag passes over a rectangle, in depth steps.
    cajera_rect       rectangular pocket: rings from the centre outwards, then a finishing pass.
    cajera_circ       circular pocket: helical entry and circles outwards, then a finishing pass.
    agujeros_circulo  bolt circle: N holes on a circle (G81, or G83 pecking with «picado»).
    agujeros_grilla   grid of holes: columns × rows (same drilling cycle).
    recorte_rect      cut out a rectangle (outside the line) in depth steps, with holding tabs.
    recorte_circ      cut out a disc (outside the circle) in depth steps, with holding tabs.

How it fits
-----------
Pure functions, standard library only. Used by ``session.py`` (``run_wizard``) and the app's
«Asistentes» dialog, which reads ``WIZARDS`` to build its form. The output uses only codes the
interpreter supports (G0/G1/G2/G3, G81/G83, M3/M5…).

Units and conventions
---------------------
mm, mm/min, rpm. Z0 is the top of the stock; depths are positive numbers going down. The tool
radius compensation is done here (the controllers TISMA talks to do not have G41/G42), so the
walls end exactly on the requested size. All texts the user sees (labels, errors, comments in
the program) are in Spanish.

Example
-------
::

    gcode = generate("cajera_rect", {"x": 50, "y": 30, "ancho": 40, "alto": 20, "profundidad": 6},
                     Tooling(diameter=6, feed=1200, plunge=400, rpm=18000))
"""

from __future__ import annotations

import math


class WizardError(ValueError):
    """Invalid wizard data (message in Spanish, shown to the user)."""


class Tooling:
    """Tool and speeds the wizard uses, from the effective profiles.

    diameter (mm)       end mill diameter (the ``Herramienta`` profile).
    feed (mm/min)       cutting feed; plunge (mm/min) for Z; rpm for the spindle.
    stepover_pct (%)    radial step as % of the diameter for pockets and facing (material's
                        recommended stepover; 40 % if unknown). More = faster, rougher, more load.
    stepdown (mm)       depth per pass (material's recommendation; one diameter if unknown).
    safe_z (mm)         rapid height above the stock between moves. Typical 3–10.
    """

    def __init__(self, diameter=6.0, feed=1000.0, plunge=300.0, rpm=18000.0, stepover_pct=40.0, stepdown=None,
                 safe_z=5.0):
        self.diameter, self.feed, self.plunge, self.rpm = float(diameter), float(feed), float(plunge), float(rpm)
        self.stepover_pct = float(stepover_pct)
        self.stepdown = float(stepdown) if stepdown else float(diameter)
        self.safe_z = float(safe_z)


# Form definition for the app: (key, label, unit, default). Labels in Spanish (user-facing).
WIZARDS = {
    "planeado": ("Planeado de superficie", [
        ("x", "Esquina X", "mm", 0), ("y", "Esquina Y", "mm", 0), ("ancho", "Ancho (X)", "mm", 100),
        ("alto", "Largo (Y)", "mm", 60), ("profundidad", "Profundidad total", "mm", 1)]),
    "cajera_rect": ("Cajera rectangular", [
        ("x", "Centro X", "mm", 0), ("y", "Centro Y", "mm", 0), ("ancho", "Ancho (X)", "mm", 40),
        ("alto", "Largo (Y)", "mm", 20), ("profundidad", "Profundidad", "mm", 5)]),
    "cajera_circ": ("Cajera circular", [
        ("x", "Centro X", "mm", 0), ("y", "Centro Y", "mm", 0), ("diametro", "Diámetro", "mm", 30),
        ("profundidad", "Profundidad", "mm", 5)]),
    "agujeros_circulo": ("Agujeros en círculo (círculo de pernos)", [
        ("x", "Centro X", "mm", 0), ("y", "Centro Y", "mm", 0), ("diametro", "Diámetro del círculo", "mm", 60),
        ("cantidad", "Cantidad de agujeros", "", 6), ("angulo", "Ángulo del primero", "°", 0),
        ("profundidad", "Profundidad", "mm", 10), ("picado", "Picado (0 = sin picar)", "mm", 0)]),
    "agujeros_grilla": ("Agujeros en grilla", [
        ("x", "Primer agujero X", "mm", 0), ("y", "Primer agujero Y", "mm", 0), ("columnas", "Columnas", "", 3),
        ("filas", "Filas", "", 2), ("paso_x", "Separación en X", "mm", 20), ("paso_y", "Separación en Y", "mm", 20),
        ("profundidad", "Profundidad", "mm", 10), ("picado", "Picado (0 = sin picar)", "mm", 0)]),
    "recorte_rect": ("Recortar un rectángulo", [
        ("x", "Esquina X", "mm", 0), ("y", "Esquina Y", "mm", 0), ("ancho", "Ancho (X)", "mm", 80),
        ("alto", "Largo (Y)", "mm", 50), ("profundidad", "Espesor a cortar", "mm", 6),
        ("puentes", "Puentes por lado (0 = sin puentes)", "", 1), ("alto_puente", "Alto de los puentes", "mm", 1.5)]),
    "recorte_circ": ("Recortar un círculo", [
        ("x", "Centro X", "mm", 0), ("y", "Centro Y", "mm", 0), ("diametro", "Diámetro", "mm", 60),
        ("profundidad", "Espesor a cortar", "mm", 6), ("puentes", "Puentes (0 = sin puentes)", "", 4),
        ("alto_puente", "Alto de los puentes", "mm", 1.5)]),
}


# ==================================================================================================
# ===================================== HELPERS ====================================================
# ==================================================================================================

def _n(v: float) -> str:
    """Number for G-code: 3 decimals, no trailing zeros (a µm is below any router's resolution)."""
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _levels(depth: float, step: float) -> list:
    """Z levels (negative) from the first pass down to −depth, equal steps no bigger than ``step``:
    e.g. depth 5, step 2 → [−1.667, −3.333, −5] (three equal passes, not 2 + 2 + 1)."""
    n = max(1, math.ceil(depth / step - 1e-9))
    return [-depth * k / n for k in range(1, n + 1)]


def _num(params: dict, key: str, default=None, positive=False) -> float:
    v = params.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise WizardError(f"«{key}» tiene que ser un número") from None
    if positive and v <= 0:
        raise WizardError(f"«{key}» tiene que ser mayor que 0")
    return v


def _header(title: str, t: Tooling) -> list:
    return [f"(TISMA CNC - asistente: {title})",
            f"(fresa Ø{_n(t.diameter)} mm, avance {t.feed:.0f} mm/min, bajada {t.plunge:.0f} mm/min, {t.rpm:.0f} rpm)",
            "G21 G90 G17 G94 G54", f"S{t.rpm:.0f} M3", f"G0 Z{_n(t.safe_z)}"]


def _footer(t: Tooling) -> list:
    return [f"G0 Z{_n(t.safe_z)}", "M5", "M30"]


# ==================================================================================================
# ===================================== WIZARDS ====================================================
# ==================================================================================================

def _facing(p: dict, t: Tooling) -> list:
    """Zig-zag passes along X covering the rectangle, the tool overhanging the edges by its radius
    so the edge is faced too; stepover from the material."""
    x, y = _num(p, "x", 0), _num(p, "y", 0)
    w, h, depth = _num(p, "ancho", positive=True), _num(p, "alto", positive=True), _num(p, "profundidad", positive=True)
    r = t.diameter / 2
    step = t.diameter * t.stepover_pct / 100
    out = _header("planeado", t)
    rows = max(1, math.ceil((h + 1e-9) / step))
    ys = [y + h * k / rows for k in range(rows + 1)]
    for z in _levels(depth, t.stepdown):
        out += [f"G0 X{_n(x - r - 1)} Y{_n(ys[0])}", f"G1 Z{_n(z)} F{_n(t.plunge)}"]
        for k, yy in enumerate(ys):
            xe = x + w + r + 1 if k % 2 == 0 else x - r - 1
            out += [f"G1 Y{_n(yy)} F{_n(t.feed)}", f"G1 X{_n(xe)}"]
        out.append(f"G0 Z{_n(t.safe_z)}")
    return out + _footer(t)


def _rect_pocket(p: dict, t: Tooling) -> list:
    """Rectangular pocket: plunge at the centre, rings growing by the stepover until the wall
    (tool centre at half size − radius), then a finishing ring. The corners keep the tool radius,
    as any milled pocket."""
    cx, cy = _num(p, "x", 0), _num(p, "y", 0)
    w, h, depth = _num(p, "ancho", positive=True), _num(p, "alto", positive=True), _num(p, "profundidad", positive=True)
    r = t.diameter / 2
    hx, hy = w / 2 - r, h / 2 - r
    if hx < 0 or hy < 0:
        raise WizardError("la fresa es más grande que la cajera")
    step = t.diameter * t.stepover_pct / 100
    out = _header("cajera rectangular", t)
    rings = max(1, math.ceil(max(hx, hy) / step))
    for z in _levels(depth, t.stepdown):
        out += [f"G0 X{_n(cx)} Y{_n(cy)}", f"G1 Z{_n(z)} F{_n(t.plunge)}"]
        for k in range(1, rings + 1):
            ax, ay = hx * k / rings, hy * k / rings
            out += [f"G1 X{_n(cx - ax)} Y{_n(cy - ay)} F{_n(t.feed)}", f"G1 X{_n(cx + ax)}", f"G1 Y{_n(cy + ay)}",
                    f"G1 X{_n(cx - ax)}", f"G1 Y{_n(cy - ay)}"]
        out.append(f"G0 Z{_n(t.safe_z)}")
    return out + _footer(t)


def _circ_pocket(p: dict, t: Tooling) -> list:
    """Circular pocket: helical entry around the centre (no straight plunge for the end mill), then
    full circles growing by the stepover until the wall, at each depth."""
    cx, cy = _num(p, "x", 0), _num(p, "y", 0)
    d, depth = _num(p, "diametro", positive=True), _num(p, "profundidad", positive=True)
    r = t.diameter / 2
    R = d / 2 - r
    if R <= 0:
        raise WizardError("la fresa es más grande que la cajera")
    step = t.diameter * t.stepover_pct / 100
    rings = max(1, math.ceil(R / step))
    out = _header("cajera circular", t)
    z_prev = 0.0
    helix = min(R, r) / 2 or R                               # small helix radius for the entry
    for z in _levels(depth, t.stepdown):
        out += [f"G0 X{_n(cx + helix)} Y{_n(cy)}", f"G1 Z{_n(z_prev)} F{_n(t.plunge)}",
                f"G3 X{_n(cx + helix)} Y{_n(cy)} I{_n(-helix)} J0 Z{_n(z)} F{_n(t.plunge)}"]
        for k in range(1, rings + 1):
            rr = R * k / rings
            out += [f"G1 X{_n(cx + rr)} Y{_n(cy)} F{_n(t.feed)}", f"G3 X{_n(cx + rr)} Y{_n(cy)} I{_n(-rr)} J0"]
        out.append(f"G0 Z{_n(t.safe_z)}")
        z_prev = z
    return out + _footer(t)


def _drill(points: list, p: dict, t: Tooling, title: str) -> list:
    """Drilling cycle on the points: G81 (straight), or G83 pecking with «picado» (mm per peck,
    clears the chips; for holes deeper than ~3 diameters). R = 2 mm above the stock."""
    depth = _num(p, "profundidad", positive=True)
    peck = _num(p, "picado", 0)
    out = _header(title, t)
    cyc = f"G83 Z{_n(-depth)} R2 Q{_n(peck)}" if peck > 0 else f"G81 Z{_n(-depth)} R2"
    first = True
    for x, y in points:
        if first:
            out.append(f"G98 {cyc} X{_n(x)} Y{_n(y)} F{_n(t.plunge)}")
            first = False
        else:
            out.append(f"X{_n(x)} Y{_n(y)}")
    out.append("G80")
    return out + _footer(t)


def _bolt_circle(p: dict, t: Tooling) -> list:
    cx, cy = _num(p, "x", 0), _num(p, "y", 0)
    r = _num(p, "diametro", positive=True) / 2
    n = int(_num(p, "cantidad", positive=True))
    a0 = math.radians(_num(p, "angulo", 0))
    pts = [(cx + r * math.cos(a0 + 2 * math.pi * k / n), cy + r * math.sin(a0 + 2 * math.pi * k / n))
           for k in range(n)]
    return _drill(pts, p, t, f"{n} agujeros en círculo de Ø{_n(2 * r)}")


def _grid(p: dict, t: Tooling) -> list:
    x0, y0 = _num(p, "x", 0), _num(p, "y", 0)
    cols, rows = int(_num(p, "columnas", positive=True)), int(_num(p, "filas", positive=True))
    dx, dy = _num(p, "paso_x", 0), _num(p, "paso_y", 0)
    pts = []
    for j in range(rows):                          # zig-zag rows: shortest travel between holes
        xs = range(cols) if j % 2 == 0 else range(cols - 1, -1, -1)
        pts += [(x0 + i * dx, y0 + j * dy) for i in xs]
    return _drill(pts, p, t, f"grilla de {cols} × {rows} agujeros")


def _tabs_z(z: float, depth: float, tab_h: float) -> float:
    """Z of the tabs on a pass at ``z``: the tabs stand ``tab_h`` mm above the bottom, so passes
    deeper than that go up over them; shallower passes are not affected."""
    top = -depth + tab_h
    return max(z, top)


def _rect_cut(p: dict, t: Tooling) -> list:
    """Cut out a rectangle: the tool runs OUTSIDE the line (offset by its radius, round outer
    corners) clockwise — climb-free conventional milling on the part side for a router — in depth
    steps; on the last passes it rises over ``puentes`` tabs per side (width = 2 × diameter) so the
    part does not come loose and fly."""
    x, y = _num(p, "x", 0), _num(p, "y", 0)
    w, h, depth = _num(p, "ancho", positive=True), _num(p, "alto", positive=True), _num(p, "profundidad", positive=True)
    tabs, tab_h = int(_num(p, "puentes", 0)), _num(p, "alto_puente", 0)
    r = t.diameter / 2
    out = _header("recorte rectangular", t)
    # side list for a clockwise run around the part: start bottom-left going up the left side
    sides = [((x - r, y), (x - r, y + h)), ((x, y + h + r), (x + w, y + h + r)),
             ((x + w + r, y + h), (x + w + r, y)), ((x + w, y - r), (x, y - r))]
    corners = [(x, y + h, "G2"), (x + w, y + h, "G2"), (x + w, y, "G2"), (x, y, "G2")]
    out += [f"G0 X{_n(sides[0][0][0])} Y{_n(sides[0][0][1])}"]
    for z in _levels(depth, t.stepdown):
        out.append(f"G1 Z{_n(z)} F{_n(t.plunge)}")
        zt = _tabs_z(z, depth, tab_h) if tabs > 0 and tab_h > 0 else z
        for (a, b), (ccx, ccy, g) in zip(sides, corners):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            if zt > z and tabs > 0:
                for k in range(tabs):              # tabs centred along the side, 2 × diameter wide
                    mid = (k + 0.5) / tabs
                    half = t.diameter / L
                    for u, zz in ((mid - half, z), (mid - half, zt), (mid + half, zt), (mid + half, z)):
                        out.append(f"G1 X{_n(a[0] + (b[0] - a[0]) * u)} Y{_n(a[1] + (b[1] - a[1]) * u)} Z{_n(zz)}"
                                   f" F{_n(t.feed)}")
            out.append(f"G1 X{_n(b[0])} Y{_n(b[1])} Z{_n(z)} F{_n(t.feed)}")
            nxt = sides[(sides.index((a, b)) + 1) % 4][0]
            out.append(f"{g} X{_n(nxt[0])} Y{_n(nxt[1])} I{_n(ccx - b[0])} J{_n(ccy - b[1])}")
    out.append(f"G0 Z{_n(t.safe_z)}")
    return out + _footer(t)


def _circ_cut(p: dict, t: Tooling) -> list:
    """Cut out a disc: the tool runs outside the circle (radius + tool radius), clockwise, in depth
    steps; on the last passes it rises over ``puentes`` tabs spread around (2 × diameter wide)."""
    cx, cy = _num(p, "x", 0), _num(p, "y", 0)
    d, depth = _num(p, "diametro", positive=True), _num(p, "profundidad", positive=True)
    tabs, tab_h = int(_num(p, "puentes", 0)), _num(p, "alto_puente", 0)
    R = d / 2 + t.diameter / 2
    out = _header("recorte circular", t)
    out.append(f"G0 X{_n(cx - R)} Y{_n(cy)}")
    for z in _levels(depth, t.stepdown):
        out.append(f"G1 Z{_n(z)} F{_n(t.plunge)}")
        zt = _tabs_z(z, depth, tab_h) if tabs > 0 and tab_h > 0 else z
        if zt <= z or tabs <= 0:
            out.append(f"G2 X{_n(cx - R)} Y{_n(cy)} I{_n(R)} J0 F{_n(t.feed)}")
            continue
        half = t.diameter / R                      # tab half-width as an angle (rad)
        ang = math.pi                              # start at the left point, going clockwise (decreasing)
        for k in range(tabs):
            centre = math.pi - 2 * math.pi * (k + 0.5) / tabs
            for a, zz in ((centre + half, z), (centre + half, zt), (centre - half, zt), (centre - half, z)):
                px, py = cx + R * math.cos(a), cy + R * math.sin(a)
                if zz == z and a > centre:          # arc to the start of the tab at cutting depth
                    out.append(f"G2 X{_n(px)} Y{_n(py)} I{_n(cx - (cx + R * math.cos(ang)))} "
                               f"J{_n(cy - (cy + R * math.sin(ang)))} F{_n(t.feed)}")
                elif zz == zt and a > centre:
                    out.append(f"G1 Z{_n(zt)}")
                elif zz == zt:
                    out.append(f"G2 X{_n(px)} Y{_n(py)} I{_n(cx - (cx + R * math.cos(centre + half)))} "
                               f"J{_n(cy - (cy + R * math.sin(centre + half)))}")
                else:
                    out.append(f"G1 Z{_n(z)}")
                ang = a
        out.append(f"G2 X{_n(cx - R)} Y{_n(cy)} I{_n(cx - (cx + R * math.cos(ang)))} "
                   f"J{_n(cy - (cy + R * math.sin(ang)))} F{_n(t.feed)}")
    out.append(f"G0 Z{_n(t.safe_z)}")
    return out + _footer(t)


GENERATORS = {"planeado": _facing, "cajera_rect": _rect_pocket, "cajera_circ": _circ_pocket,
              "agujeros_circulo": _bolt_circle, "agujeros_grilla": _grid, "recorte_rect": _rect_cut,
              "recorte_circ": _circ_cut}


def generate(kind: str, params: dict, tooling: Tooling) -> str:
    """G-code text of wizard ``kind`` with ``params`` (form values; missing ones take the defaults of
    ``WIZARDS``) and the ``tooling`` from the profiles.

    Raises:
        WizardError: unknown wizard or invalid data (Spanish message for the user).
    """
    if kind not in GENERATORS:
        raise WizardError(f"asistente desconocido: {kind}")
    full = {key: default for key, _label, _unit, default in WIZARDS[kind][1]}
    full.update({k: v for k, v in (params or {}).items() if v not in (None, "")})
    return "\n".join(GENERATORS[kind](full, tooling)) + "\n"
