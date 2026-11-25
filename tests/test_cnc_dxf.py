"""Plasma from DXF: reading drawings (dxf.py) and turning them into a cutting program (plasma_cam.py).

The example ``hope/cnc/ejemplos/placa_plasma.dxf`` is a 200 × 120 mm plate with rounded corners, four
Ø10 bolt holes, a Ø60 centre hole, a Ø5 small hole and a 40 × 10 slot drawn with lines and arcs
(left with a 0.02 mm gap on purpose). The tests check the rules plasma CAM programs follow: parts
vs holes, directions, kerf, lead-in on the scrap side, holes before their part, small holes slower.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hope.cnc import dxf, plasma_cam
from hope.cnc.gcode import interpret, linearize
from hope.cnc.machine import load_machine

PLACA = Path("hope/cnc/ejemplos/placa_plasma.dxf")


def _dxf(entities: list, units: int = 4, blocks: list | None = None) -> str:
    """A minimal ASCII DXF with the given entity group lists (and optional BLOCKS)."""
    out = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", str(units), "0", "ENDSEC"]
    if blocks:
        out += ["0", "SECTION", "2", "BLOCKS"] + blocks + ["0", "ENDSEC"]
    out += ["0", "SECTION", "2", "ENTITIES"] + entities + ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(out) + "\n"


def _line(a, b):
    return ["0", "LINE", "8", "0", "10", str(a[0]), "20", str(a[1]), "11", str(b[0]), "21", str(b[1])]


def _circle(c, r):
    return ["0", "CIRCLE", "8", "0", "10", str(c[0]), "20", str(c[1]), "40", str(r)]


def _square(x, y, s):
    return _line((x, y), (x + s, y)) + _line((x + s, y), (x + s, y + s)) + _line((x + s, y + s), (x, y + s)) + \
        _line((x, y + s), (x, y))


def _cut_points(text):
    """Work XY points the torch passes with the cut on (M3…M5), per contour."""
    m = load_machine("configs/cnc_plasma.yaml")
    m.plasma.probe = False                        # no sensing: keep the program simple
    p = interpret(text, m)
    assert p.ok, [str(d) for d in p.diags]
    g54 = m.offset("G54")
    cuts, cur, on = [], None, False
    for o in p.ops:
        if o.kind == "spindle":
            on = o.value[0] != "off"
            if on:
                cur = []
                cuts.append(cur)
            continue
        if on and o.kind in ("feed", "arc"):
            seq = linearize(o, 0.01) if o.kind == "arc" else [o.end]
            cur += [(q["X"] - g54["X"], q["Y"] - g54["Y"]) for q in seq]
    return cuts


def test_reads_the_example_plate():
    geo = dxf.read_dxf(PLACA.read_text(encoding="latin-1"))
    assert len(geo.circles) == 6 and len(geo.segments) == 12 and not geo.ignored
    closed, open_ = plasma_cam.chain(geo.segments)
    assert len(closed) == 2 and not open_                     # the 0.02 mm gap of the slot is closed


def test_plate_program_follows_plasma_rules():
    prog = plasma_cam.program_from_dxf(PLACA.read_text(encoding="latin-1"),
                                      plasma_cam.PlasmaParams(kerf=1.5, feed=3800, thickness=3), "placa.dxf")
    assert (prog.parts, prog.holes, prog.small) == (1, 7, 1) and not prog.notes
    cuts = _cut_points(prog.text)
    assert len(cuts) == 8
    outside = cuts[-1]                                         # the part's outside is cut last
    xs = [x for x, _ in outside[1:]]
    assert min(xs) == pytest.approx(-0.75, abs=1e-3) and max(xs) == pytest.approx(200.75, abs=1e-3)   # kerf/2 out
    centre = [c for c in cuts if any(abs(math.hypot(x - 100, y - 60) - 29.25) < 1e-3 for x, y in c)]
    assert centre, "the Ø60 hole is cut at radius 30 − 0.75"
    assert "F2280" in prog.text                                # Ø5 < 2.5 × 3 mm: 60 % of 3800


def test_directions_lead_in_and_small_hole_from_the_centre():
    text = _dxf(_square(0, 0, 100) + _circle((50, 50), 10) + _circle((20, 20), 2))
    prog = plasma_cam.program_from_dxf(text, plasma_cam.PlasmaParams(kerf=1, feed=3000, thickness=6, lead_in=5))
    lines = prog.text.splitlines()
    assert any(t.startswith("G3 ") for t in lines) and not any(t.startswith("G2 ") for t in lines)  # holes CCW
    cuts = _cut_points(prog.text)
    sq = cuts[-1]                                                                  # outside last
    a = sum(sq[i - 1][0] * sq[i][1] - sq[i][0] * sq[i - 1][1] for i in range(1, len(sq))) / 2
    assert a < 0                                                                   # outside clockwise
    lead = [t for t in lines if t.startswith("G0 X")][-1]                          # lead of the square
    x, y = (float(v[1:]) for v in lead.split()[1:3])
    assert x < -0.5 or y < -0.5 or x > 100.5 or y > 100.5                          # starts on the scrap side
    small_lead = next(t for t in lines if t.startswith("G0 X"))                # the Ø4 hole goes first
    x, y = (float(v[1:]) for v in small_lead.split()[1:3])
    assert math.hypot(x - 20, y - 20) < 1.5                                        # lead-in from near its centre


def test_units_blocks_mirror_and_bad_files():
    inch = dxf.read_dxf(_dxf(_line((0, 0), (1, 0)), units=1))
    assert inch.segments[0].p1 == pytest.approx((25.4, 0.0))
    blk = ["0", "BLOCK", "8", "0", "2", "AGUJERO", "10", "0", "20", "0"] + _circle((0, 0), 3) + ["0", "ENDBLK"]
    ins = ["0", "INSERT", "8", "0", "2", "AGUJERO", "10", "40", "20", "10", "41", "2", "42", "2"]
    geo = dxf.read_dxf(_dxf(ins, blocks=blk))
    assert geo.circles == [((40.0, 10.0), 6.0)]                                   # placed and scaled ×2
    arc = ["0", "ARC", "8", "0", "10", "0", "20", "0", "40", "5", "50", "0", "51", "90", "230", "-1"]
    s = dxf.read_dxf(_dxf(arc)).segments[0]
    assert s.p0 == pytest.approx((0.0, 5.0)) and s.p1 == pytest.approx((-5.0, 0.0))   # mirrored back in X
    with pytest.raises(dxf.DxfError):
        dxf.read_dxf("hola\nesto no es un dxf\n")
    with pytest.raises(dxf.DxfError):
        dxf.read_dxf(_dxf(["0", "TEXT", "8", "0", "1", "solo texto"]))


def test_spline_and_ellipse_are_approximated():
    ell = ["0", "ELLIPSE", "8", "0", "10", "0", "20", "0", "11", "20", "21", "0", "40", "0.5", "41", "0", "42",
           str(2 * math.pi)]
    geo = dxf.read_dxf(_dxf(ell))
    pts = [s.p0 for s in geo.segments]
    assert max(x for x, _ in pts) == pytest.approx(20, abs=0.01) and max(y for _, y in pts) == pytest.approx(10, abs=0.02)
    closed, _ = plasma_cam.chain(geo.segments)
    assert len(closed) == 1


def test_session_opens_dxf_and_regenerates_with_the_profile():
    from hope.cnc.links import SimLink
    from hope.cnc.profiles import ProfileStore, default_selection, effective
    from hope.cnc.session import CncSession, SessionError
    store = ProfileStore(None)
    sel = default_selection(store, "Mesa de plasma 1500×3000")
    m, _ = effective(store, sel)
    s = CncSession(m, SimLink(m, autostart=False), profiles=store, selection=sel,
                   link_factory=lambda mm: SimLink(mm, autostart=False))
    r = s.load("placa.dxf", PLACA.read_text(encoding="latin-1"))
    assert r["ok"] and s.program["fuente"].startswith("0\nSECTION") and "G3" in s.program["texto"]
    first = s.program["texto"]
    thick = [e for e in s.profile_summary.get("espesores", []) if e != s.profile_summary.get("espesor")][-1]
    s.choose_profile(None, None, thick)                         # another thickness → other speed/kerf
    assert s.program["nombre"] == "placa.dxf" and s.program["texto"] != first   # regenerated from the DXF
    assert s.recent[0]["nombre"] == "placa.dxf"
    router, _ = effective(store, default_selection(store))
    r2 = CncSession(router, SimLink(router, autostart=False))
    with pytest.raises(SessionError, match="plasma"):
        r2.load("placa.dxf", PLACA.read_text(encoding="latin-1"))
