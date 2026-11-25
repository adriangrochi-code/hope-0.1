"""Rotary axes, stage 2: rollover, shortest rotation on G0, table/head mounting and centre of rotation
(the basis for 5-axis «3+2» work).

* Shortest rotation on G0 behaves like Mach3's «Shortest Rot on G0» and keeps the angle continuous.
* Rollover only wraps what the DRO shows (0–360°), never the position the controller uses.
* The toolpath view turns the workpiece around the real centre of each table axis: the «3+2» cube
  example lands its five engravings 0.5 mm deep on the five faces of the cube.
* Head-mounted axes do not turn the workpiece in the view.
"""

from pathlib import Path

import pytest

from hope.cnc.gcode import interpret
from hope.cnc.links import SimLink
from hope.cnc.machine import machine_from_dict
from hope.cnc.profiles import ProfileStore, default_selection, effective
from hope.cnc.session import CncSession, table_rotations

EJEMPLOS = Path("hope/cnc/ejemplos")


def _profile_machine(name):
    store = ProfileStore(None)
    return effective(store, default_selection(store, name))[0]


def test_shortest_rotation_on_g0_like_mach3():
    m = machine_from_dict({"ejes": {"X": {}, "Y": {}, "Z": {}, "A": {"giro_continuo": True, "vuelta_corta": True}}})
    p = interpret("G0 A350\nG0 A90\nG1 A0 F100\nG91 G0 A-200\nG90 G0 A10", m)
    ends = [round(o.end["A"], 6) for o in p.ops]
    # 0→350 turns −10; −10→90 turns +100; G1 is never shortened; G91 is literal; −200→10 turns −150
    assert ends == [-10.0, 90.0, 0.0, -200.0, -350.0]
    plain = machine_from_dict({"ejes": {"X": {}, "Y": {}, "Z": {}, "A": {}}})
    assert [o.end["A"] for o in interpret("G0 A350\nG0 A90", plain).ops] == [350.0, 90.0]   # off by default


def test_rollover_only_wraps_the_dro():
    m = machine_from_dict({"ejes": {"X": {}, "Y": {}, "Z": {}, "A": {"giro_continuo": True}}})
    link = SimLink(m, autostart=False)
    s = CncSession(m, link)
    link.interp.pos = {**link.interp.pos, "A": 725.0}
    link._pos = dict(link.interp.pos)
    st = s.state()
    assert st["trabajo_pos"]["A"] == pytest.approx(5.0) and st["estado"]["mpos"]["A"] == pytest.approx(5.0)
    assert link.snapshot()["mpos"]["A"] == pytest.approx(725.0)          # the link keeps the real angle


def test_profile_keys_for_kinematics():
    m = machine_from_dict({"ejes": {"X": {}, "Y": {}, "Z": {},
                                    "A": {"montaje": "mesa", "centro_y": 75, "centro_z": -170},
                                    "B": {"montaje": "cabezal"},
                                    "C": {"centro_x": 200}}})                 # incomplete centre → None
    assert m.axes["A"].pivot == (75.0, -170.0) and m.axes["A"].mount == "mesa"
    assert m.axes["B"].mount == "cabezal" and m.axes["C"].pivot is None
    g54 = {"X": 200.0, "Y": 75.0, "Z": -150.0}
    # head-mounted B does not turn the part; A's centre is given in G54 coordinates
    assert table_rotations(m, g54) == [["A", 0.0, -20.0], ["C", 0.0, 0.0]]


def test_3_plus_2_cube_lands_on_the_five_faces():
    m = _profile_machine("Fresadora 5 ejes (mesa A+C)")
    texto = (EJEMPLOS / "cubo_3mas2.nc").read_text(encoding="utf-8")
    assert interpret(texto, m).ok
    s = CncSession(m, SimLink(m, autostart=False))
    s.load("cubo.nc", texto)
    assert s.program["informe"]["runnable"], s.program["informe"]["errores"]
    cut = [p for p in s.toolpath()["puntos"] if p[3] == 1]
    faces = {}
    for x, y, z, _, _line in cut:
        dist = {"+X": x, "-X": -x, "+Y": y, "-Y": -y, "arriba": z + 20}    # cube 40 mm, centre 20 mm down
        face = max(dist, key=dist.get)
        faces.setdefault(face, []).append(dist[face])
    assert sorted(faces) == ["+X", "+Y", "-X", "-Y", "arriba"]
    assert all(d == pytest.approx(19.5, abs=1e-3) for v in faces.values() for d in v)   # 0.5 mm deep


def test_head_mounted_axis_does_not_turn_the_view():
    head = machine_from_dict({"ejes": {"X": {}, "Y": {}, "Z": {}, "B": {"montaje": "cabezal"}}})
    s = CncSession(head, SimLink(head, autostart=False))
    s.load("b.nc", "G0 X10 Y0 Z5 B45\nM3 S1000\nG1 Z0 F100\nG1 X20\nM30")
    pts = s.toolpath()["puntos"]
    assert [p[:3] for p in pts if p[3] == 1][-1] == [20.0, 0.0, 0.0]          # straight, not rotated


# --- inverse time feed (G93) -------------------------------------------------------------------------

G93_PROGRAM = "G1 X1 F100\nG93 G1 X11 A90 F2\nG2 X21 Y0 I5 J0 A180 F4\nG1 A200 F0.5\nG94 G1 X0 F500\nM30"


def _axis4(**extra):
    axes = {"X": {"vel_max": 20000, "acel": 50000}, "Y": {"vel_max": 20000, "acel": 50000}, "Z": {},
            "A": {"vel_max": 200000, "acel": 50000}}
    return machine_from_dict({"ejes": axes, **extra})


def test_g93_move_lasts_one_over_f_minutes():
    from hope.cnc.planner import plan
    m = _axis4()
    p = interpret(G93_PROGRAM, m)
    assert p.ok, [str(d) for d in p.diags]
    times = plan(p, m).time_by_op
    assert times[1] == pytest.approx(30.0, rel=1e-3)          # F2 → 1/2 min
    assert times[2] == pytest.approx(15.0, rel=1e-3)          # arc with F4 → 1/4 min
    assert times[3] == pytest.approx(120.0, rel=1e-3)         # F0.5 → 2 min


def test_g93_rules_like_rs274ngc():
    m = _axis4()
    assert "necesita su F" in str(interpret("G93 G1 X10 F2\nG1 X20", m).diags[0])      # F on every line
    assert "sin F" in str(interpret("G1 X1 F100\nG93 G1 X2 F2\nG94 G1 X3", m).diags[0])  # G94 needs F again
    assert "G93" in str(interpret("G93\nG81 X1 Y1 Z-1 R1 F2", m).diags[0])            # no canned cycles
    assert interpret("G93 G0 X10 A90\nG94", m).ok                                    # rapids need no F


def test_g93_output_per_controller():
    from hope.cnc.dialect import emit
    m = _axis4()
    p = interpret(G93_PROGRAM, m)
    grbl = [t for t, _ in emit(p, m, "grbl")]
    assert grbl[2] == "G93 G1 X11 A90 F2" and grbl[3].endswith(" F4") and grbl[4] == "G1 A200 F0.5"
    assert grbl[5] == "G94 G1 X0 F500"
    marlin = [t for t, _ in emit(p, m, "marlin")]                     # no G93 in Marlin: equivalent feed
    assert not any("G93" in t or "G94" in t for t in marlin)
    assert marlin[2] == "G1 X11 A90 F181.1077"                         # √(10² + 90²) × 2
    assert len(marlin) == len(grbl) == len(emit(p, m, "linuxcnc"))    # same lines in every dialect


def test_run_from_here_inside_g93():
    m = _axis4()
    s = CncSession(m, SimLink(m, autostart=False))
    s.load("g93.nc", G93_PROGRAM)
    s.next_line = 4                                                    # «G1 A200 F0.5», in G93
    lines, _ = s._lines_to_run()
    # the controller must be in G93 before «F0.5», or it would take it as 0.5 units/min
    assert "G93" in lines[0] and "G94" not in lines[0] and lines[1] == "G1 A200 F0.5"
    s.from_here = True
    lines, _ = s._lines_to_run()
    k = lines.index("G1 A200 F0.5")
    assert lines[k - 1] == "G93" and not any(t.startswith("F") for t in lines[:k])
    s.next_line, s.from_here = 5, False                # the G94 line itself: it switches back on its own
    lines, _ = s._lines_to_run()
    assert lines[1] == "G94 G1 X0 F500"
