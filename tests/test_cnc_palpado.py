"""Probing routines (palpado.py): each routine finds the right edge / centre / surface on a simulated part.

The simulator (SimLink) gets a ``probe_surface`` describing a real shape (a block, a bore, a boss,
a warped sheet, a tool setter), and the routines run through a real session, so every line goes
through the MDI check and the controller's modal state. The simulated clock is stepped by hand
(``_fast_session``), so the tests take milliseconds instead of the real probing time.
"""

import math
import time

import pytest

from hope.cnc.gcode import interpret
from hope.cnc.links import GrblLink, SimLink
from hope.cnc.palpado import ROUTINES, HeightMap, ProbeError, apply_height_map
from hope.cnc.profiles import ProfileStore, default_selection, effective
from hope.cnc.session import CncSession, SessionError

R = 1.0                                       # probe tip radius used in every test (Ø2, the default)


# ==================================================================================================
# helpers: a session whose simulated time is stepped by hand, and simulated shapes
# ==================================================================================================

def _fast_session(surface, kind="Router genérico"):
    """Enabled session on SimLink (no thread) whose waits advance the simulation until «Idle»."""
    store = ProfileStore(None)
    sel = default_selection(store, kind) if kind != "Router genérico" else default_selection(store)
    m, _ = effective(store, sel)
    link = SimLink(m, autostart=False)
    link.probe_surface = surface
    s = CncSession(m, link, profiles=store, selection=sel, link_factory=lambda mm: SimLink(mm, autostart=False))

    def wait(timeout=0):
        for _ in range(100000):
            if link.state != "Run":
                return
            link.step(0.5)
    s._wait_idle = wait
    s.reset()
    return s, link


def _go(s, x, y, z):
    """Machine move (G53) to the start point of a routine."""
    s.mdi(f"G53 G0 Z{z}")
    s._wait_idle()
    s.mdi(f"G53 G0 X{x} Y{y}")
    s._wait_idle()


def _probe(s, kind, **params):
    """Runs a routine and waits for its thread; returns ``state()["palpado"]``."""
    s.run_probe(kind, params)
    for _ in range(2000):
        if s.prober is None:
            break
        time.sleep(0.005)
    return s.state()["palpado"]


def block(x0, x1, y0, y1, top):
    """Probe-contact function of a rectangular block (machine mm): the ball (radius R) touches the
    sides one radius away, the top with its tip. Straight moves only (what the routines do)."""
    def hit(a, b):
        lo = {"X": x0 - R, "Y": y0 - R, "Z": -1e9}
        hi = {"X": x1 + R, "Y": y1 + R, "Z": top}
        t_in, t_out = 0.0, 1.0
        for ax in "XYZ":
            d = b[ax] - a[ax]
            if abs(d) < 1e-12:
                if not lo[ax] <= a[ax] <= hi[ax]:
                    return None
                continue
            t1, t2 = (lo[ax] - a[ax]) / d, (hi[ax] - a[ax]) / d
            t_in, t_out = max(t_in, min(t1, t2)), min(t_out, max(t1, t2))
        if t_in > t_out or t_in <= 0:
            return None
        return {ax: a[ax] + (b[ax] - a[ax]) * t_in for ax in "XYZ"}
    return hit


def bore(cx, cy, diameter):
    """Contact inside a round hole (the probe starts inside it): first wall the ball reaches."""
    rr = diameter / 2 - R

    def hit(a, b):
        dx, dy = b["X"] - a["X"], b["Y"] - a["Y"]
        if dx == 0 and dy == 0:
            return None
        ax, ay = a["X"] - cx, a["Y"] - cy
        qa, qb, qc = dx * dx + dy * dy, 2 * (ax * dx + ay * dy), ax * ax + ay * ay - rr * rr
        t = (-qb + math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
        return None if t > 1 else {"X": a["X"] + t * dx, "Y": a["Y"] + t * dy, "Z": a["Z"]}
    return hit


# ==================================================================================================
# routines
# ==================================================================================================

@pytest.mark.parametrize("direction,start,edge", [("X+", (100, 150), 120.0), ("X-", (300, 150), 280.0),
                                                   ("Y+", (200, 80), 100.0), ("Y-", (200, 220), 200.0)])
def test_edge_in_every_direction(direction, start, edge):
    s, link = _fast_session(block(120, 280, 100, 200, -50))
    _go(s, start[0], start[1], -60)
    r = _probe(s, "borde", direccion=direction, distancia=30, valor=0)
    assert r["estado"] == "listo", r
    axis = direction[0]
    assert r["valores"][axis] == pytest.approx(edge)
    assert link.offsets()["G54"][axis] == pytest.approx(edge)          # the edge is the work 0


def test_edge_can_get_another_value():
    s, link = _fast_session(block(120, 280, 100, 200, -50))
    _go(s, 100, 150, -60)
    _probe(s, "borde", direccion="X+", distancia=30, valor=5)
    assert link.offsets()["G54"]["X"] == pytest.approx(115)            # edge = X5 → X0 is 5 mm before


@pytest.mark.parametrize("corner,start,expect", [("inf_izq", (110, 90), (120, 100)), ("sup_der", (290, 210), (280, 200))])
def test_outside_corner(corner, start, expect):
    s, link = _fast_session(block(120, 280, 100, 200, -50))
    _go(s, start[0], start[1], -60)
    r = _probe(s, "esquina", esquina=corner, entrada=10, distancia=30)
    assert r["estado"] == "listo", r
    g54 = link.offsets()["G54"]
    assert (g54["X"], g54["Y"]) == (pytest.approx(expect[0]), pytest.approx(expect[1]))
    assert (link.pos["X"], link.pos["Y"]) == (pytest.approx(start[0]), pytest.approx(start[1]))   # back home


def test_bore_centre_and_diameter_even_from_off_centre():
    s, link = _fast_session(bore(150, 120, 25))
    _go(s, 145, 116, -40)                                               # 5 mm and 4 mm off the centre
    r = _probe(s, "centro_agujero", distancia=30)
    assert r["estado"] == "listo", r
    v = r["valores"]
    assert (v["X"], v["Y"]) == (pytest.approx(150), pytest.approx(120))
    assert v["diametro_x"] == pytest.approx(25) and v["diametro_y"] == pytest.approx(25)
    g54 = link.offsets()["G54"]
    assert (g54["X"], g54["Y"]) == (pytest.approx(150), pytest.approx(120))


def test_boss_centre_and_size():
    s, link = _fast_session(block(170, 230, 90, 130, -50))              # 60 × 40 boss, centre (200, 110)
    _go(s, 196, 112, -45)                                               # above it, roughly centred
    r = _probe(s, "centro_saliente", ancho_x=60, ancho_y=40, bajar=8, holgura=5)
    assert r["estado"] == "listo", r
    v = r["valores"]
    assert (v["X"], v["Y"]) == (pytest.approx(200), pytest.approx(110))
    assert (v["ancho_x"], v["ancho_y"]) == (pytest.approx(60), pytest.approx(40))
    assert link.pos["Z"] == pytest.approx(-45)                          # ends above it


def test_boss_too_small_estimate_stops_on_the_unexpected_touch():
    s, _ = _fast_session(block(170, 230, 90, 130, -50))
    _go(s, 200, 110, -45)
    r = _probe(s, "centro_saliente", ancho_x=30, ancho_y=40, bajar=8, holgura=5)   # real width is 60
    assert r["estado"] == "error" and "tocó algo" in r["mensajes"][0]


def test_surface_with_touch_plate():
    s, link = _fast_session(block(0, 600, 0, 400, -52.5))
    _go(s, 200, 200, -40)
    r = _probe(s, "superficie", distancia=30, espesor_placa=0)
    assert r["valores"]["Z"] == pytest.approx(-52.5) and link.offsets()["G54"]["Z"] == pytest.approx(-52.5)
    s2, link2 = _fast_session(block(0, 600, 0, 400, -52.5 + 10))       # a 10 mm plate on the surface
    _go(s2, 200, 200, -30)
    _probe(s2, "superficie", distancia=30, espesor_placa=10)
    assert link2.offsets()["G54"]["Z"] == pytest.approx(-52.5)


def test_no_contact_is_an_alarm_and_a_clear_error():
    s, link = _fast_session(block(500, 600, 0, 400, -50))
    _go(s, 100, 150, -60)
    r = _probe(s, "borde", direccion="X+", distancia=20, valor=0)
    assert r["estado"] == "error" and "no tocó" in r["mensajes"][0]
    assert link.state == "Alarm"


def test_probe_beyond_the_travel_is_refused_before_moving():
    s, link = _fast_session(block(120, 280, 100, 200, -50))
    _go(s, 590, 150, -60)
    r = _probe(s, "borde", direccion="X+", distancia=30, valor=0)       # 590 + 30 > 600
    assert r["estado"] == "error" and link.pos["X"] == pytest.approx(590)


def test_tool_length_reference_and_new_tool():
    setter = {"top": -70.0}

    def hit(a, b):                                                       # tool setter at X10 Y10
        if abs(a["X"] - 10) < 1e-6 and abs(a["Y"] - 10) < 1e-6 and b["Z"] < a["Z"]:
            return dict(a, Z=setter["top"]) if b["Z"] <= setter["top"] <= a["Z"] else None
        return None
    s, link = _fast_session(hit)
    s.mdi("G10 L2 P1 Z-60")                                              # Z0 already set for tool 1
    r = _probe(s, "herramienta", modo="nueva", x=10, y=10, distancia=80)
    assert r["estado"] == "error" and "referencia" in r["mensajes"][0]
    r = _probe(s, "herramienta", modo="referencia", x=10, y=10, distancia=80)
    assert r["estado"] == "listo", r
    setter["top"] = -67.0                                                # the new tool is 3 mm longer
    r = _probe(s, "herramienta", modo="nueva", x=10, y=10, distancia=80)
    assert r["valores"]["diferencia"] == pytest.approx(3)
    assert link.offsets()["G54"]["Z"] == pytest.approx(-57)              # Z0 went 3 mm up with the tip
    assert link.pos["Z"] == pytest.approx(0)                             # ends at the top


def test_stop_cancels_the_routine():
    s, _ = _fast_session(bore(150, 120, 25))
    _go(s, 150, 120, -40)
    s.prober = None
    s.run_probe("centro_agujero", {"distancia": 30})
    s.stop()
    for _ in range(2000):
        if s.prober is None:
            break
        time.sleep(0.005)
    assert s.state()["palpado"]["estado"] in ("error", "listo")         # finished either way, never hangs


def test_session_refuses_plasma_and_a_second_routine():
    s, _ = _fast_session(None, "Mesa de plasma 1500×3000")
    with pytest.raises(SessionError, match="Sensar altura"):
        s.run_probe("superficie", {})
    s, _ = _fast_session(None)
    s.prober = object()
    with pytest.raises(SessionError, match="en marcha"):
        s.run_probe("superficie", {})
    forms = s.probe_forms()
    assert [r["tipo"] for r in forms["rutinas"]] == sorted(ROUTINES)
    assert any("opciones" in c for r in forms["rutinas"] for c in r["campos"])


# ==================================================================================================
# height map
# ==================================================================================================

def test_height_map_interpolation():
    h = HeightMap(0, 0, 10, 10, [[0, 1], [2, 3]])
    assert h.at(0, 0) == 0 and h.at(10, 10) == 3 and h.at(5, 5) == pytest.approx(1.5)
    assert h.at(-50, -50) == 0 and h.at(99, 0) == 1                     # clamped, never extrapolated
    assert HeightMap(0, 0, 0, 0, [[0.7]]).at(30, 40) == pytest.approx(0.7)


def test_map_probing_and_applying_it_to_a_program():
    tilt = lambda x, y: -40.0 + 0.01 * x - 0.02 * y                     # warped sheet, machine Z of the top

    def hit(a, b):
        if b["Z"] < a["Z"] and a["X"] == b["X"] and a["Y"] == b["Y"]:
            top = tilt(a["X"], a["Y"])
            return dict(a, Z=top) if b["Z"] <= top <= a["Z"] else None
        return None
    s, _ = _fast_session(hit)                                            # G54 = X20 Y20 Z−40
    _go(s, 20, 20, -30)
    r = _probe(s, "mapa", x=0, y=0, ancho=100, alto=60, columnas=3, filas=4, altura=3, distancia=6)
    assert r["estado"] == "listo", r
    m = r["valores"]["mapa"]
    assert m["z"][0][0] == pytest.approx(tilt(20, 20) + 40) and m["z"][3][2] == pytest.approx(tilt(120, 80) + 40)
    s.load("ranura.nc", "G21 G90 G54\nM3 S10000\nG0 Z3\nG0 X0 Y30\nG1 Z-1 F300\nG1 X100 F800\nG2 X100 Y50 I0 J10\nG0 Z3\nM30\n")
    r = s.apply_height_map(5)
    assert r["ok"] and s.program["nombre"] == "ranura.mapa.nc"
    assert "mapa de superficie aplicado" in s.program["texto"] and "M3 S10000" in s.program["texto"]
    prog = interpret(s.program["texto"], s.m)
    g = s.m.offset("G54")
    cuts = [o for o in prog.ops if o.kind == "feed" and o.end["Z"] - g["Z"] < 0]
    assert len(cuts) > 20                                                # split in ≤ 5 mm pieces
    for o in cuts:                                                       # always 1 mm under the real top
        x, y = o.end["X"], o.end["Y"]
        assert o.end["Z"] - tilt(x, y) == pytest.approx(-1, abs=1e-3)


def test_map_refuses_programs_it_cannot_keep_exact():
    h = HeightMap(0, 0, 10, 10, [[0, 0], [0, 0]])
    store = ProfileStore(None)
    m, _ = effective(store, default_selection(store))
    for bad in ("G91 G1 X10 F100", "G20\nG1 X1 F10", "G1 X1 A90 F100", "G92 X0\nG1 X1 F100", "G38.2 Z-5 F100"):
        with pytest.raises(ProbeError, match="no se puede aplicar"):
            apply_height_map(bad, h, m)
    out = apply_height_map("G90 G1 X10 Y0 Z-1 F100 (corte)\n(fin)\nM5\n", h, m)
    assert out.splitlines()[-2:] == ["(fin)", "M5"]


# ==================================================================================================
# grbl reports the exact contact
# ==================================================================================================

def test_grbl_reports_the_latched_probe_position():
    store = ProfileStore(None)
    m, _ = effective(store, default_selection(store))
    link = GrblLink(object(), m, autostart=False)
    assert link.last_probe() is None
    link.handle("[PRB:120.500,80.000,-41.250:1]")
    p = link.last_probe()
    assert p["ok"] and p["n"] == 1 and p["pos"] == {"X": 120.5, "Y": 80.0, "Z": -41.25}
    link.handle("[PRB:0.000,0.000,0.000:0]")
    assert link.last_probe()["n"] == 2 and not link.last_probe()["ok"]
