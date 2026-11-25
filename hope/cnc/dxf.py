"""Minimal DXF reader for plasma (and router) cutting: the 2D geometry of a part drawing.

What it is
----------
Reads an ASCII DXF file (the format every CAD exports: AutoCAD, LibreCAD, QCAD, Fusion, SolidWorks,
Inkscape…) and returns its 2D geometry as straight and circular segments in millimetres. It is
the first step of «open a DXF and cut it» (see ``plasma_cam.py``), so it reads only what a cutting
drawing needs and ignores the rest (text, dimensions, hatches, 3D).

Supported entities: LINE, ARC, CIRCLE, LWPOLYLINE and POLYLINE (with bulges = arc segments),
ELLIPSE and SPLINE (approximated with short lines), and INSERT of BLOCKS (translation, scale and
rotation; nested blocks too). Binary DXF is not supported (rare in practice).

How it fits
-----------
Pure standard library (it must run on Windows XP with Python 3.4 after translation). Used by
``plasma_cam.py``. Nothing else in TISMA CNC depends on it.

Units and conventions
---------------------
* The header's ``$INSUNITS`` decides the unit (1 inch, 2 feet, 4 mm, 5 cm, 6 m); everything is
  converted to mm. Without it, millimetres are assumed (what most cutting drawings use).
* Arcs are counter-clockwise (DXF convention) from ``start`` to ``end``.
* Z is ignored: drawings are flat (an entity with a non-standard extrusion direction, mirrored in
  CAD, is mirrored back so it lands where it was drawn).

Example
-------
::

    geo = read_dxf(open("brida.dxf", encoding="latin-1").read())
    print(len(geo.segments), "segmentos,", len(geo.circles), "círculos")
"""

from __future__ import annotations

import math

# --- configuration ---------------------------------------------------------------------------------
# Chord tolerance (mm) used to turn ellipses and splines into short lines: the largest distance
# between the true curve and the lines. 0.01 mm is far below any plasma kerf (≈1–2 mm); smaller
# values only make longer programs. Typical 0.005–0.05.
CURVE_TOLERANCE = 0.01

# $INSUNITS code → millimetres per drawing unit (0 = unitless, taken as mm).
UNIT_MM = {0: 1.0, 1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0, 8: 0.0000254, 9: 0.0254, 10: 914.4}

# Safety limit: INSERT inside INSERT inside … (a block that includes itself would loop forever).
MAX_BLOCK_DEPTH = 8


class DxfError(ValueError):
    """The file is not a DXF this reader understands (message in Spanish, shown to the user)."""


class Segment:
    """A straight line (``center`` None) or a circular arc from ``p0`` to ``p1`` (mm).

    For arcs, ``center`` is the centre and ``ccw`` the direction (True = counter-clockwise, like
    G3). Points are ``(x, y)`` tuples."""

    __slots__ = ("ccw", "center", "p0", "p1")

    def __init__(self, p0, p1, center=None, ccw=True):
        self.p0, self.p1, self.center, self.ccw = p0, p1, center, ccw

    def reversed(self) -> Segment:
        """The same segment travelled the other way."""
        return Segment(self.p1, self.p0, self.center, not self.ccw)

    def __repr__(self) -> str:
        kind = "arc" if self.center else "line"
        return f"Segment({kind}, {self.p0} → {self.p1})"


class Geometry:
    """Result of ``read_dxf``: open segments (to be chained into contours) and full circles."""

    def __init__(self):
        self.segments: list[Segment] = []
        self.circles: list[tuple] = []     # (centre (x, y), radius) in mm
        self.ignored: dict = {}            # entity type → count (shown as a warning)


# ==================================================================================================
# ======================================== PARSER ==================================================
# ==================================================================================================

def _pairs(text: str) -> list:
    """The DXF as a list of (group code, value) pairs. A DXF is two lines per item: an integer code
    and its value. Raises DxfError if it does not look like ASCII DXF."""
    lines = text.splitlines()
    if len(lines) < 4:
        raise DxfError("el archivo está vacío o no es un DXF")
    out = []
    for i in range(0, len(lines) - 1, 2):
        try:
            code = int(lines[i].strip())
        except ValueError:
            raise DxfError("no es un DXF de texto (¿DXF binario?): guardarlo como DXF ASCII") from None
        out.append((code, lines[i + 1].strip()))
    return out


def _entities(pairs: list, start: int) -> tuple:
    """Reads entities from ``start`` until ENDSEC/ENDBLK. Each entity is (type, [(code, value), …]).
    Returns (entities, next index)."""
    ents = []
    i = start
    cur = None
    while i < len(pairs):
        code, val = pairs[i]
        if code == 0:
            if cur is not None:
                ents.append(cur)
            if val in ("ENDSEC", "ENDBLK"):
                return ents, i
            cur = (val, [])
        elif cur is not None:
            cur[1].append((code, val))
        i += 1
    if cur is not None:
        ents.append(cur)
    return ents, i


def _f(props: list, code: int, default: float = 0.0) -> float:
    """First value of group ``code`` as float."""
    for c, v in props:
        if c == code:
            return float(v)
    return default


def _all(props: list, code: int) -> list:
    """All values of group ``code`` as floats, in order (vertices of a polyline, spline points)."""
    return [float(v) for c, v in props if c == code]


def read_dxf(text: str) -> Geometry:
    """Parses an ASCII DXF and returns its 2D geometry in mm (see the module header).

    Raises:
        DxfError: not a DXF, or no cuttable geometry at all.
    """
    pairs = _pairs(text)
    scale = 1.0
    blocks: dict = {}
    entities: list = []
    i = 0
    while i < len(pairs):
        code, val = pairs[i]
        if code == 0 and val == "SECTION" and i + 1 < len(pairs):
            name = pairs[i + 1][1]
            if name == "HEADER":
                j = i + 2
                while j < len(pairs) and pairs[j] != (0, "ENDSEC"):
                    if pairs[j] == (9, "$INSUNITS") and j + 1 < len(pairs):
                        scale = UNIT_MM.get(int(float(pairs[j + 1][1])), 1.0)
                    j += 1
                i = j
            elif name == "BLOCKS":
                j = i + 2
                while j < len(pairs) and pairs[j] != (0, "ENDSEC"):
                    if pairs[j] == (0, "BLOCK"):
                        k = j + 1
                        props = []
                        while k < len(pairs) and pairs[k][0] != 0:
                            props.append(pairs[k])
                            k += 1
                        bname = next((v for c, v in props if c == 2), "")
                        base = (_f(props, 10), _f(props, 20))
                        ents, k = _entities(pairs, k)
                        blocks[bname] = (base, ents)
                        j = k
                    j += 1
                i = j
            elif name == "ENTITIES":
                entities, i = _entities(pairs, i + 2)
        i += 1
    geo = Geometry()
    _emit(entities, blocks, geo, _Transform(), 0)
    if scale != 1.0:
        _scale(geo, scale)
    if not geo.segments and not geo.circles:
        raise DxfError("el DXF no tiene líneas, arcos ni círculos para cortar")
    return geo


# ==================================================================================================
# ================================== ENTITIES → SEGMENTS ===========================================
# ==================================================================================================

class _Transform:
    """Placement of a block (INSERT): scale (sx, sy), rotation (degrees) and offset, applied in the
    DXF order: scale, rotate, then translate. Identity by default. A negative ``sx·sy`` mirrors,
    which also flips arc directions."""

    def __init__(self, ox=0.0, oy=0.0, sx=1.0, sy=1.0, rot=0.0):
        self.ox, self.oy, self.sx, self.sy = ox, oy, sx, sy
        self.c, self.s = math.cos(math.radians(rot)), math.sin(math.radians(rot))

    def point(self, x: float, y: float) -> tuple:
        x, y = x * self.sx, y * self.sy
        return (self.ox + x * self.c - y * self.s, self.oy + x * self.s + y * self.c)

    @property
    def mirrored(self) -> bool:
        return self.sx * self.sy < 0

    def then(self, inner: _Transform) -> _Transform:
        """Composition: ``inner`` applied first, then self (for nested INSERTs). Only uniform
        rotation/scale chains are exact; non-uniform scale of a rotated block is approximated."""
        t = _Transform()
        o = self.point(inner.ox, inner.oy)
        t.ox, t.oy = o
        t.sx, t.sy = self.sx * inner.sx, self.sy * inner.sy
        t.c = self.c * inner.c - self.s * inner.s
        t.s = self.s * inner.c + self.c * inner.s
        return t


def _arc_points(center, r, a0, a1, tol):
    """Points (excluding the first) along a CCW arc from angle a0 to a1 (radians) with chord error ≤ tol."""
    sweep = a1 - a0
    if r <= tol:
        n = 1
    else:
        step = 2 * math.acos(max(-1.0, 1 - tol / r))
        n = max(1, math.ceil(abs(sweep) / step))
    return [(center[0] + r * math.cos(a0 + sweep * k / n), center[1] + r * math.sin(a0 + sweep * k / n))
            for k in range(1, n + 1)]


def _add_arc(geo, t, center, r, a0_deg, a1_deg, flip):
    """Adds a DXF ARC (CCW from a0 to a1 degrees) placed by ``t``. Non-uniform block scale turns a
    circle into an ellipse: then it is approximated with lines."""
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    if a1 <= a0:
        a1 += 2 * math.pi
    if abs(abs(t.sx) - abs(t.sy)) > 1e-9:
        pts = [(center[0] + r * math.cos(a0), center[1] + r * math.sin(a0))] + _arc_points(center, r, a0, a1,
                                                                                           CURVE_TOLERANCE)
        _add_points(geo, t, pts, flip)
        return
    p0 = (center[0] + r * math.cos(a0), center[1] + r * math.sin(a0))
    p1 = (center[0] + r * math.cos(a1), center[1] + r * math.sin(a1))
    ccw = not (t.mirrored ^ flip)                  # a mirroring placement turns CCW arcs into CW
    geo.segments.append(Segment(t.point(*p0), t.point(*p1), t.point(*center), ccw))


def _add_points(geo, t, pts, flip=False):
    """Adds a chain of straight lines through ``pts`` (already in entity coordinates)."""
    for a, b in zip(pts, pts[1:]):
        pa, pb = t.point(*(_flip(a) if flip else a)), t.point(*(_flip(b) if flip else b))
        if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-9:
            geo.segments.append(Segment(pa, pb))


def _flip(p):
    """Mirror in X (an entity whose extrusion direction is (0, 0, −1): mirrored in CAD)."""
    return (-p[0], p[1])


def _bulge_segment(geo, t, a, b, bulge, flip):
    """A polyline segment from a to b: straight (bulge 0) or an arc whose bulge = tan(sweep / 4),
    positive = counter-clockwise (DXF definition)."""
    if flip:
        a, b, bulge = _flip(a), _flip(b), -bulge
    if abs(bulge) < 1e-12:
        pa, pb = t.point(*a), t.point(*b)
        if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) > 1e-9:
            geo.segments.append(Segment(pa, pb))
        return
    chord = math.hypot(b[0] - a[0], b[1] - a[1])
    if chord < 1e-12:
        return
    sweep = 4 * math.atan(bulge)
    r = chord / (2 * math.sin(abs(sweep) / 2))
    # centre: from the chord midpoint, perpendicular, at distance r·cos(sweep/2) to the left (CCW)
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    d = r * math.cos(sweep / 2) * (1 if bulge > 0 else -1)
    ux, uy = (b[0] - a[0]) / chord, (b[1] - a[1]) / chord
    center = (mx - uy * d, my + ux * d)
    if abs(abs(t.sx) - abs(t.sy)) > 1e-9:
        a0 = math.atan2(a[1] - center[1], a[0] - center[0])
        pts = [a] + _arc_points(center, r, a0, a0 + sweep, CURVE_TOLERANCE)
        _add_points(geo, t, pts)
        return
    geo.segments.append(Segment(t.point(*a), t.point(*b), t.point(*center), (bulge > 0) ^ t.mirrored))


def _spline_points(props):
    """Points along a SPLINE: evaluated from its control points and knots (de Boor), or its fit
    points if it has no control points. Dense enough for CURVE_TOLERANCE on typical parts."""
    deg = int(_f(props, 71, 3))
    knots = _all(props, 40)
    cx, cy = _all(props, 10), _all(props, 20)
    ctrl = list(zip(cx, cy))
    if len(ctrl) < 2 or len(knots) != len(ctrl) + deg + 1:
        fx, fy = _all(props, 11), _all(props, 21)
        return list(zip(fx, fy)) if len(fx) >= 2 else ctrl
    weights = _all(props, 41) or [1.0] * len(ctrl)
    if len(weights) != len(ctrl):
        weights = [1.0] * len(ctrl)
    lo, hi = knots[deg], knots[len(ctrl)]
    n = max(16, 24 * len(ctrl))
    pts = []
    for k in range(n + 1):
        u = lo + (hi - lo) * k / n
        pts.append(_de_boor(u, deg, knots, ctrl, weights))
    return pts


def _de_boor(u, p, t, c, w):
    """Point of a (rational) B-spline of degree p at parameter u (de Boor's algorithm)."""
    k = p
    while k < len(c) - 1 and u >= t[k + 1]:
        k += 1
    d = [(c[j + k - p][0] * w[j + k - p], c[j + k - p][1] * w[j + k - p], w[j + k - p]) for j in range(p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            den = t[j + 1 + k - r] - t[j + k - p]
            a = 0.0 if den == 0 else (u - t[j + k - p]) / den
            d[j] = tuple((1 - a) * d[j - 1][q] + a * d[j][q] for q in range(3))
    x, y, ww = d[p]
    return (x / ww, y / ww)


def _ellipse_points(props):
    """Points along an ELLIPSE: centre, major-axis end point (relative), minor/major ratio and the
    start/end parameters (radians)."""
    cx, cy = _f(props, 10), _f(props, 20)
    mx, my = _f(props, 11), _f(props, 21)
    ratio = _f(props, 40, 1.0)
    t0, t1 = _f(props, 41, 0.0), _f(props, 42, 2 * math.pi)
    if t1 <= t0:
        t1 += 2 * math.pi
    a = math.hypot(mx, my)
    ux, uy = (mx / a, my / a) if a else (1.0, 0.0)
    n = max(16, math.ceil((t1 - t0) / (2 * math.acos(max(-1.0, 1 - CURVE_TOLERANCE / max(a, 1e-9))))))
    out = []
    for k in range(n + 1):
        t = t0 + (t1 - t0) * k / n
        x, y = a * math.cos(t), a * ratio * math.sin(t)
        out.append((cx + x * ux - y * uy, cy + x * uy + y * ux))
    return out


def _emit(entities, blocks, geo, t, depth):
    """Turns DXF entities into segments/circles of ``geo``, placed by transform ``t``."""
    for kind, props in entities:
        flip = _f(props, 230, 1.0) < 0             # extrusion (0, 0, −1): mirrored in CAD
        if kind == "LINE":
            _add_points(geo, t, [(_f(props, 10), _f(props, 20)), (_f(props, 11), _f(props, 21))], flip)
        elif kind == "CIRCLE":
            c = (_f(props, 10), _f(props, 20))
            r = _f(props, 40)
            if abs(abs(t.sx) - abs(t.sy)) > 1e-9:
                _add_points(geo, t, [(c[0] + r, c[1])] + _arc_points(c, r, 0.0, 2 * math.pi, CURVE_TOLERANCE), flip)
            elif r > 0:
                geo.circles.append((t.point(*(_flip(c) if flip else c)), r * abs(t.sx)))
        elif kind == "ARC":
            c = (_f(props, 10), _f(props, 20))
            a0, a1 = _f(props, 50), _f(props, 51)
            if flip:                               # mirrored in X: angles become 180 − a and swap
                c, a0, a1 = _flip(c), 180.0 - a1, 180.0 - a0
            _add_arc(geo, t, c, _f(props, 40), a0, a1, False)
        elif kind == "LWPOLYLINE":
            xs, ys = _all(props, 10), _all(props, 20)
            bulges = _lw_bulges(props, len(xs))
            closed = int(_f(props, 70, 0)) & 1
            pts = list(zip(xs, ys))
            _polyline(geo, t, pts, bulges, closed, flip)
        elif kind == "SPLINE":
            _add_points(geo, t, _spline_points(props), flip)
        elif kind == "ELLIPSE":
            _add_points(geo, t, _ellipse_points(props), flip)
        elif kind == "INSERT":
            name = next((v for c, v in props if c == 2), "")
            if name in blocks and depth < MAX_BLOCK_DEPTH:
                (bx, by), ents = blocks[name]
                inner = _Transform(-bx, -by)
                place = _Transform(_f(props, 10), _f(props, 20), _f(props, 41, 1.0), _f(props, 42, 1.0),
                                   _f(props, 50, 0.0))
                _emit(ents, blocks, geo, t.then(place.then(inner)), depth + 1)
        elif kind in ("POLYLINE", "VERTEX", "SEQEND"):
            pass                                   # old-style polylines: handled by _old_polylines below
        else:
            geo.ignored[kind] = geo.ignored.get(kind, 0) + 1
    _old_polylines(entities, geo, t)


def _lw_bulges(props, n):
    """Bulge of each LWPOLYLINE vertex (group 42 follows the vertex it belongs to; 0 if absent)."""
    out, idx = [0.0] * n, -1
    for c, v in props:
        if c == 10:
            idx += 1
        elif c == 42 and 0 <= idx < n:
            out[idx] = float(v)
    return out


def _polyline(geo, t, pts, bulges, closed, flip):
    """Adds a polyline: each vertex to the next with its bulge (and back to the first if closed)."""
    n = len(pts)
    for k in range(n - 1 + (1 if closed and n > 2 else 0)):
        _bulge_segment(geo, t, pts[k], pts[(k + 1) % n], bulges[k], flip)


def _old_polylines(entities, geo, t):
    """Old-style POLYLINE … VERTEX … SEQEND (R12 files): vertices come as separate entities."""
    k = 0
    while k < len(entities):
        kind, props = entities[k]
        if kind == "POLYLINE":
            closed = int(_f(props, 70, 0)) & 1
            flip = _f(props, 230, 1.0) < 0
            pts, bul = [], []
            k += 1
            while k < len(entities) and entities[k][0] == "VERTEX":
                vp = entities[k][1]
                pts.append((_f(vp, 10), _f(vp, 20)))
                bul.append(_f(vp, 42, 0.0))
                k += 1
            _polyline(geo, t, pts, bul, closed, flip)
        else:
            k += 1


def _scale(geo, f):
    """Converts everything from drawing units to mm."""
    def sc(p):
        return (p[0] * f, p[1] * f)
    geo.segments = [Segment(sc(s.p0), sc(s.p1), sc(s.center) if s.center else None, s.ccw) for s in geo.segments]
    geo.circles = [(sc(c), r * f) for c, r in geo.circles]
