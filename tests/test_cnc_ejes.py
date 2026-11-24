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
