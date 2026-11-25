"""From a DXF drawing to plasma G-code: «open a DXF and cut it», without a separate CAM program.

What it is
----------
Takes the 2D geometry read by ``dxf.py`` and produces a ready-to-cut G-code program using the
plasma profiles (kerf and speed from the material's cut table, heights and delays). The rules are
the usual ones of plasma CAM programs (SheetCam, Hypertherm's guidelines):

1. **Contours**: loose lines and arcs are chained into closed contours (tolerance
   ``CHAIN_TOLERANCE``); open chains are reported and not cut.
2. **Parts and holes**: a contour inside an odd number of others is a hole; otherwise it is the
   outside of a part (a part inside a hole is a part again).
3. **Direction**: outsides clockwise, holes counter-clockwise. With the standard clockwise swirl of
   the plasma gas, the good (square) side of the cut is then on the right of the torch: on the
   part, for both.
4. **Kerf**: the torch path is offset by half the kerf away from the part (outward on outsides,
   into the hole on holes) so the part comes out at the drawing's size. Circles stay true arcs
   (G2/G3); other contours are offset as fine polylines.
5. **Lead-in / overcut**: each cut starts on the scrap side with a short lead-in perpendicular to
   the contour (the pierce crater stays on the scrap), and ends with a small overcut past the
   start so no nub is left. Small holes start from their centre.
6. **Order**: inside each part, its holes first, then its outside (once the outside is cut the part
   can shift); parts and holes in nearest-neighbour order to keep rapids short.
7. **Small holes**: holes smaller than ``agujero_chico_factor`` × thickness are cut slower
   (``agujero_chico_pct`` of the speed): a full-speed small hole comes out tapered and oval.

How it fits
-----------
Uses ``dxf.py`` (geometry). Used by ``session.py`` (``load_dxf``: the app loads the generated
program like any other, so it is checked, simulated and sent the same way). The resulting
program's pierce sequence comes from the machine/process profiles: with automatic piercing on, the
program only carries M3/M5 and the interpreter adds the heights; otherwise the heights are written
in the program itself.

Units and conventions
---------------------
mm and mm/min; work coordinates (the drawing's origin is the G54 zero). Points are ``(x, y)``.

Example
-------
::

    geo = dxf.read_dxf(text)
    prog = plasma_program(geo, PlasmaParams(kerf=1.5, feed=3800, thickness=3), name="brida.dxf")
    print(prog.text)            # G-code;  prog.notes: warnings in Spanish
"""

from __future__ import annotations

import math

from hope.cnc import dxf

# --- configuration ---------------------------------------------------------------------------------
# Two segment ends closer than this (mm) are the same point when chaining contours. CAD drawings
# often leave tiny gaps; 0.05 mm is invisible in a plasma cut (kerf ≈ 1–2 mm). Too big would join
# features that are really separate. Typical 0.01–0.1.
CHAIN_TOLERANCE = 0.05
# Chord tolerance (mm) when an arc of a non-circular contour is turned into short lines for the
# kerf offset. 0.02 mm is far below what plasma can show. Smaller = longer programs.
ARC_TOLERANCE = 0.02


class PlasmaParams:
    """Cutting parameters, normally filled from the effective profiles (see ``session.load_dxf``).

    kerf (mm)               width of the cut; the path is offset by half of it. 0 = no offset.
    feed (mm/min)           cutting speed (from the material's cut table).
    thickness (mm)          sheet thickness, for the small-hole rule.
    lead_in (mm)            lead-in length on the scrap side. Typical 3–6; too short and the
                            pierce crater touches the part.
    overcut (mm)            how far past the start the cut continues. Typical 0.5–2.
    small_factor            holes with diameter < factor × thickness are «small». 2.5 is the usual
                            rule of thumb (Hypertherm «True Hole» starts from 1×).
    small_pct (%)           speed for small holes, as % of ``feed``. Typical 50–70.
    auto_pierce             the interpreter adds the pierce sequence on M3 (process profile); if
                            False, the heights below are written in the program.
    pierce_height, cut_height, travel_height (mm) and pierce_delay (s): from the plasma profile.
    copies                  how many copies of the drawing to cut (grid nesting, see ``nest_grid``).
    spacing (mm)            gap between copies; it must leave room for the lead-ins and the kerf.
    sheet_w, sheet_h (mm)   usable sheet size from the G54 zero (normally the machine travel).
    """

    def __init__(self, kerf=1.5, feed=2000.0, thickness=3.0, lead_in=4.0, overcut=1.0, small_factor=2.5,
                 small_pct=60.0, auto_pierce=True, pierce_height=3.8, cut_height=1.5, travel_height=10.0,
                 pierce_delay=0.5, plunge_feed=300.0, copies=1, spacing=10.0, sheet_w=0.0, sheet_h=0.0):
        self.kerf, self.feed, self.thickness = float(kerf), float(feed), float(thickness)
        self.lead_in, self.overcut = float(lead_in), float(overcut)
        self.small_factor, self.small_pct = float(small_factor), float(small_pct)
        self.auto_pierce = bool(auto_pierce)
        self.pierce_height, self.cut_height = float(pierce_height), float(cut_height)
        self.travel_height, self.pierce_delay = float(travel_height), float(pierce_delay)
        self.plunge_feed = float(plunge_feed)
        self.copies, self.spacing = max(1, int(copies)), float(spacing)
        self.sheet_w, self.sheet_h = float(sheet_w), float(sheet_h)


class PlasmaProgram:
    """Result: the G-code ``text``, the counts and the ``notes`` (Spanish, shown to the user)."""

    def __init__(self, text: str, parts: int, holes: int, small: int, notes: list):
        self.text, self.parts, self.holes, self.small, self.notes = text, parts, holes, small, notes


# ==================================================================================================
# ===================================== CONTOURS ===================================================
# ==================================================================================================

def _close(a, b, tol=CHAIN_TOLERANCE) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def chain(segments: list) -> tuple:
    """Joins loose segments into closed contours (each a list of ``dxf.Segment`` in travel order).

    Greedy: start from any unused segment and keep adding the unused one whose end touches the
    current end (reversing it if needed) until the chain closes on its start. A grid of buckets
    (cell = tolerance) finds neighbours fast even with thousands of segments.

    Returns ``(closed contours, open chains)``; open chains are lists of segments too."""
    cell = CHAIN_TOLERANCE * 2

    def key(p):
        return (math.floor(p[0] / cell), math.floor(p[1] / cell))
    buckets: dict = {}
    for i, s in enumerate(segments):
        for end in (s.p0, s.p1):
            buckets.setdefault(key(end), []).append(i)
    used = [False] * len(segments)

    def find(p):
        kx, ky = key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in buckets.get((kx + dx, ky + dy), ()):
                    if not used[i]:
                        s = segments[i]
                        if _close(s.p0, p):
                            return i, s
                        if _close(s.p1, p):
                            return i, s.reversed()
        return None
    closed, open_ = [], []
    for i, s in enumerate(segments):
        if used[i]:
            continue
        used[i] = True
        path = [s]
        while not _close(path[-1].p1, path[0].p0):
            nxt = find(path[-1].p1)
            if nxt is None:
                break
            used[nxt[0]] = True
            path.append(nxt[1])
        (closed if _close(path[-1].p1, path[0].p0) else open_).append(path)
    return closed, open_


def polygon(contour: list) -> list:
    """Points of a closed contour (arcs as short lines, ``ARC_TOLERANCE``), without repeating the
    first point at the end."""
    pts = []
    for s in contour:
        pts.append(s.p0)
        if s.center is not None:
            pts += _arc_inner_points(s)
    return pts


def _arc_inner_points(s) -> list:
    """Intermediate points of an arc segment (excluding both ends)."""
    cx, cy = s.center
    r = math.hypot(s.p0[0] - cx, s.p0[1] - cy)
    a0 = math.atan2(s.p0[1] - cy, s.p0[0] - cx)
    a1 = math.atan2(s.p1[1] - cy, s.p1[0] - cx)
    sweep = a1 - a0
    if s.ccw and sweep <= 0:
        sweep += 2 * math.pi
    elif not s.ccw and sweep >= 0:
        sweep -= 2 * math.pi
    step = 2 * math.acos(max(-1.0, 1 - ARC_TOLERANCE / r)) if r > ARC_TOLERANCE else abs(sweep)
    n = max(1, math.ceil(abs(sweep) / step))
    return [(cx + r * math.cos(a0 + sweep * k / n), cy + r * math.sin(a0 + sweep * k / n)) for k in range(1, n)]


def area(pts: list) -> float:
    """Signed area (shoelace formula): positive = counter-clockwise."""
    return 0.5 * sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts)))


def contains(pts: list, p) -> bool:
    """Point-in-polygon by ray casting (count edge crossings of a horizontal ray to +X)."""
    x, y = p
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def offset_left(pts: list, d: float) -> list:
    """Closed polyline offset by ``d`` mm to the LEFT of the travel direction.

    Each edge is shifted along its left normal. Where the path turns left the two shifted edges
    overlap: they are trimmed at their intersection. Where it turns right they leave a gap: it is
    filled with a round join (an arc of radius d around the original corner), the shape a torch of
    that kerf really makes there. Nearly collinear edges just share their shifted point."""
    if d <= 0:
        return list(pts)
    n = len(pts)
    edges = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1e-9:
            continue
        nx, ny = -(b[1] - a[1]) / L, (b[0] - a[0]) / L
        edges.append(((a[0] + nx * d, a[1] + ny * d), (b[0] + nx * d, b[1] + ny * d), (nx, ny), b))
    out = []
    m = len(edges)
    for i in range(m):
        a0, a1, na, corner = edges[i]
        b0, b1, nb, _ = edges[(i + 1) % m]
        da = (a1[0] - a0[0], a1[1] - a0[1])
        db = (b1[0] - b0[0], b1[1] - b0[1])
        cross = da[0] * db[1] - da[1] * db[0]
        la, lb = math.hypot(*da), math.hypot(*db)
        sin_t = cross / (la * lb) if la and lb else 0.0
        if abs(sin_t) < 1e-6:                      # collinear: shifted ends coincide
            out.append(a1)
        elif sin_t > 0:                            # left turn: trim at the intersection
            t = ((b0[0] - a0[0]) * db[1] - (b0[1] - a0[1]) * db[0]) / cross
            out.append((a0[0] + da[0] * t, a0[1] + da[1] * t))
        else:                                      # right turn: round join around the corner
            ang0 = math.atan2(na[1], na[0])
            ang1 = math.atan2(nb[1], nb[0])
            sweep = ang1 - ang0
            if sweep > 0:
                sweep -= 2 * math.pi               # turning right = clockwise
            steps = max(1, math.ceil(abs(sweep) / (2 * math.acos(max(-1.0, 1 - ARC_TOLERANCE / d)))))
            for k in range(steps + 1):
                ang = ang0 + sweep * k / steps
                out.append((corner[0] + d * math.cos(ang), corner[1] + d * math.sin(ang)))
    return out


# ==================================================================================================
# ===================================== CUT PLAN ===================================================
# ==================================================================================================

class _Cut:
    """One contour to cut: a polyline (``pts``, closed) or a true circle (``circle`` = (c, r)),
    already offset and in cutting direction; ``hole`` and ``small`` flags; ``part`` = index of the
    part (outside contour) it belongs to."""

    def __init__(self, pts=None, circle=None, hole=False, small=False, part=-1, size=0.0):
        self.pts, self.circle, self.hole, self.small, self.part, self.size = pts, circle, hole, small, part, size

    def start(self):
        return self.lead[0]


def _lead(cut: _Cut, p: PlasmaParams) -> None:
    """Chooses where the cut starts and its lead-in (``cut.lead`` = [lead start, contour start]).

    Polylines start in the middle of their longest edge (a straight, clean place for the pierce
    mark); circles at 0°. The lead-in comes from the left of the travel direction, which is the
    scrap side for both outsides (clockwise) and holes (counter-clockwise). Small holes and
    circles smaller than the lead start from the centre."""
    if cut.circle is not None:
        (cx, cy), r = cut.circle
        start = (cx + r, cy)
        if cut.hole:
            L = min(p.lead_in, r * 0.8)
            cut.lead = [(cx + r - L, cy), start]
        else:
            cut.lead = [(cx + r + p.lead_in, cy), start]
        return
    pts = cut.pts
    best, bi = -1.0, 0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L > best:
            best, bi = L, i
    a, b = pts[bi], pts[(bi + 1) % len(pts)]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    cut.pts = [mid] + pts[bi + 1:] + pts[:bi + 1]          # rotate so the contour starts at mid
    nx, ny = -(b[1] - a[1]) / best, (b[0] - a[0]) / best    # left normal = scrap side
    L = p.lead_in
    if cut.hole:
        L = min(L, 0.4 * cut.size)                          # stay inside small holes
    cut.lead = [(mid[0] + nx * L, mid[1] + ny * L), mid]


def plan_cuts(geo, p: PlasmaParams) -> tuple:
    """Builds the ordered list of cuts and the notes (warnings, Spanish) from the geometry."""
    notes = []
    contours, open_chains = chain(geo.segments)
    if open_chains:
        q = open_chains[0][0].p0
        notes.append(f"{len(open_chains)} contornos abiertos no se cortan (el primero cerca de "
                     f"X{q[0]:.1f} Y{q[1]:.1f}): cerrarlos en el CAD")
    shapes = []                                     # (polygon points, circle or None)
    for c in contours:
        pts = polygon(c)
        if len(pts) >= 3 and abs(area(pts)) > 1e-6:
            shapes.append((pts, None))
    for (cx, cy), r in geo.circles:
        n = 64
        shapes.append(([(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
                        for k in range(n)], ((cx, cy), r)))
    # nesting depth: how many other shapes contain a point of this one
    depth = []
    for i, (pts, _) in enumerate(shapes):
        probe = pts[0]
        depth.append(sum(1 for j, (q, _) in enumerate(shapes) if j != i and abs(area(q)) > abs(area(pts))
                         and contains(q, probe)))
    d = p.kerf / 2.0
    cuts = []
    for i, (pts, circ) in enumerate(shapes):
        hole = depth[i] % 2 == 1
        size = 2 * math.sqrt(abs(area(pts)) / math.pi)      # equivalent diameter (mm)
        small = hole and p.thickness > 0 and size < p.small_factor * p.thickness
        if circ is not None:
            c, r = circ
            r2 = r - d if hole else r + d
            if r2 <= 0:
                notes.append(f"agujero de Ø{2 * r:.1f} mm más chico que el kerf: no se corta")
                continue
            cut = _Cut(circle=(c, r2), hole=hole, small=small, size=size)
        else:
            ccw = area(pts) > 0
            if ccw != hole:                          # outsides clockwise, holes counter-clockwise
                pts = pts[::-1]
            cut = _Cut(pts=offset_left(pts, d), hole=hole, small=small, size=size)
        # the part a hole belongs to: the smallest outside contour that contains it
        if hole:
            owners = [j for j, (q, _) in enumerate(shapes) if depth[j] == depth[i] - 1 and contains(q, pts[0])]
            cut.part = min(owners, key=lambda j: abs(area(shapes[j][0]))) if owners else -1
        else:
            cut.part = i
        _lead(cut, p)
        cuts.append(cut)
    # order: parts by nearest neighbour; inside each, its holes (nearest neighbour), then the outside
    ordered, pos = [], (0.0, 0.0)
    outsides = [c for c in cuts if not c.hole]
    holes_by_part: dict = {}
    for c in cuts:
        if c.hole:
            holes_by_part.setdefault(c.part, []).append(c)

    def nearest(items, at):
        return min(items, key=lambda c: (c.start()[0] - at[0]) ** 2 + (c.start()[1] - at[1]) ** 2)
    remaining = list(outsides)
    while remaining:
        part = nearest(remaining, pos)
        remaining.remove(part)
        hs = list(holes_by_part.pop(part.part, []))
        while hs:
            h = nearest(hs, pos)
            hs.remove(h)
            ordered.append(h)
            pos = h.start()
        ordered.append(part)
        pos = part.start()
    for hs in holes_by_part.values():               # holes without an outside (drawing of holes only)
        ordered += hs
    return ordered, notes


# ==================================================================================================
# ===================================== G-CODE =====================================================
# ==================================================================================================

def _n(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ==================================================================================================
# ===================================== NESTING ====================================================
# ==================================================================================================

def bounds(geo) -> tuple:
    """Bounding box (xmin, ymin, xmax, ymax) of the drawing, arcs and circles included (a circle by
    its full extent; an arc segment by its end points and its centre ± radius, a safe overestimate)."""
    xs, ys = [], []
    for s in geo.segments:
        xs += [s.p0[0], s.p1[0]]
        ys += [s.p0[1], s.p1[1]]
        if s.center is not None:
            r = math.hypot(s.p0[0] - s.center[0], s.p0[1] - s.center[1])
            xs += [s.center[0] - r, s.center[0] + r]
            ys += [s.center[1] - r, s.center[1] + r]
    for (cx, cy), r in geo.circles:
        xs += [cx - r, cx + r]
        ys += [cy - r, cy + r]
    return (min(xs), min(ys), max(xs), max(ys))


def nest_grid(geo, p: PlasmaParams) -> tuple:
    """Grid nesting: ``p.copies`` copies of the whole drawing in rows from the G54 zero, each moved so
    its bounding box starts at the cell corner, cells of (width + spacing) × (height + spacing), as
    many columns as fit in ``p.sheet_w`` (one column if the sheet size is unknown). Copies that do
    not fit in ``p.sheet_h`` are dropped with a note. Returns (new geometry, notes, copies placed).

    Simple on purpose (no rotation, no true-shape nesting): it is what most plasma jobs of identical
    parts need, and it is predictable for the operator."""
    notes = []
    x0, y0, x1, y1 = bounds(geo)
    w, h = x1 - x0, y1 - y0
    cw, ch = w + p.spacing, h + p.spacing
    cols = max(1, int((p.sheet_w + p.spacing) // cw)) if p.sheet_w > 0 else 1
    rows_fit = int((p.sheet_h + p.spacing) // ch) if p.sheet_h > 0 else None
    placed = p.copies
    if rows_fit is not None and rows_fit * cols < p.copies:
        placed = max(0, rows_fit * cols)
        notes.append(f"en la chapa entran {placed} copias de {p.copies} ({cols} × {rows_fit})")
    out = dxf.Geometry()
    for k in range(placed):
        dx = (k % cols) * cw - x0
        dy = (k // cols) * ch - y0

        def mv(q, dx=dx, dy=dy):
            return (q[0] + dx, q[1] + dy)
        out.segments += [dxf.Segment(mv(s.p0), mv(s.p1), mv(s.center) if s.center else None, s.ccw)
                         for s in geo.segments]
        out.circles += [(mv(c), r) for c, r in geo.circles]
    out.ignored = dict(geo.ignored)
    return out, notes, placed


def plasma_program(geo, p: PlasmaParams, name: str = "pieza.dxf") -> PlasmaProgram:
    """The complete G-code program for the drawing (see the module header for the rules)."""
    nest_notes = []
    if p.copies > 1:
        geo, nest_notes, _ = nest_grid(geo, p)
    cuts, notes = plan_cuts(geo, p)
    notes = nest_notes + notes
    small_feed = p.feed * p.small_pct / 100.0
    holes = sum(1 for c in cuts if c.hole)
    small = sum(1 for c in cuts if c.small)
    parts = len(cuts) - holes
    out = [f"(TISMA CNC - generado desde {name}: {parts} piezas, {holes} agujeros, kerf {p.kerf:g} mm)",
           f"(velocidad {p.feed:.0f} mm/min; {small} agujeros chicos a {small_feed:.0f} mm/min)",
           "G21 G90 G17 G94 G54"]
    # Start at the travel height in both modes: with automatic piercing and height sensing, the
    # probe starts from the current Z and has a limited travel (e.g. 20 mm): from the machine's top
    # it would not reach the sheet.
    out.append(f"G0 Z{_n(p.travel_height)}")
    for c in cuts:
        feed = small_feed if c.small else p.feed
        out.append(f"({'agujero' if c.hole else 'contorno'}{' chico' if c.small else ''})")
        lx, ly = c.lead[0]
        out.append(f"G0 X{_n(lx)} Y{_n(ly)}")
        if p.auto_pierce:
            out.append("M3")
        else:
            out += [f"G0 Z{_n(p.pierce_height)}", "M3", f"G4 P{p.pierce_delay:g}",
                    f"G1 Z{_n(p.cut_height)} F{_n(p.plunge_feed)}"]
        sx, sy = c.lead[1]
        out.append(f"G1 X{_n(sx)} Y{_n(sy)} F{_n(feed)}")
        if c.circle is not None:
            (cx, cy), r = c.circle
            g = "G3" if c.hole else "G2"             # holes CCW, outsides CW
            out.append(f"{g} X{_n(sx)} Y{_n(sy)} I{_n(cx - sx)} J{_n(cy - sy)}")
            if p.overcut > 0:                       # continue a little along the circle
                ang = p.overcut / r * (1 if c.hole else -1)
                ex, ey = cx + r * math.cos(ang), cy + r * math.sin(ang)
                out.append(f"{g} X{_n(ex)} Y{_n(ey)} I{_n(cx - sx)} J{_n(cy - sy)}")
        else:
            for x, y in c.pts[1:] + c.pts[:1]:
                out.append(f"G1 X{_n(x)} Y{_n(y)}")
            left = p.overcut
            k = 0
            pts = c.pts
            while left > 1e-9 and k < len(pts):
                a, b = pts[k], pts[(k + 1) % len(pts)]
                L = math.hypot(b[0] - a[0], b[1] - a[1])
                if L >= left:
                    t = left / L
                    out.append(f"G1 X{_n(a[0] + (b[0] - a[0]) * t)} Y{_n(a[1] + (b[1] - a[1]) * t)}")
                    break
                out.append(f"G1 X{_n(b[0])} Y{_n(b[1])}")
                left -= L
                k += 1
        out.append("M5")
        if not p.auto_pierce:
            out.append(f"G0 Z{_n(p.travel_height)}")
    if geo.ignored:
        notes.append("se ignoraron entidades del DXF: " + ", ".join(f"{k} ×{v}" for k, v in sorted(geo.ignored.items())))
    if not cuts:
        notes.append("no quedó nada para cortar")
    out.append("M30")
    return PlasmaProgram("\n".join(out) + "\n", parts, holes, small, notes)


def program_from_dxf(text: str, p: PlasmaParams, name: str = "pieza.dxf") -> PlasmaProgram:
    """Reads the DXF text and builds the program. Raises ``dxf.DxfError`` for unreadable files."""
    return plasma_program(dxf.read_dxf(text), p, name)
