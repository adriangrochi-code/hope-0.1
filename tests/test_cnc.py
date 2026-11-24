"""Hope CNC (hope/cnc, docs/CNC_HOPE.md): interpreter, planner, review, translation,
controllers (simulated, fake grbl) and the guardian's permission to move a real machine."""

import math
import sys

import pytest

from guardian.core import Guardian, GuardianConfig
from hope.cnc.backends.base import BackendError
from hope.cnc.backends.grbl import GrblBackend, describe, parse_status
from hope.cnc.backends.sim import SimBackend
from hope.cnc.check import check_text, report_text
from hope.cnc.controller import Controller
from hope.cnc.dialect import emit_text
from hope.cnc.gcode import interpret
from hope.cnc.machine import Axis, Machine, Tool, load_machine, machine_from_dict
from hope.cnc.planner import plan


def fast_machine(**kw):
    axes = {"X": Axis("X", max_rate=60000, accel=100, min=-1000, max=1000),
            "Y": Axis("Y", max_rate=60000, accel=100, min=-1000, max=1000),
            "Z": Axis("Z", max_rate=60000, accel=100, min=-1000, max=1000)}
    return Machine(axes=axes, **kw)


def moves(prog):
    return [(o.kind, {k: round(v, 4) for k, v in o.end.items()}) for o in prog.ops if o.kind in ("rapid", "feed", "arc")]


# --- planner --------------------------------------------------------------------------
def test_trapezoid_time_is_exact():
    m = fast_machine()
    # 200 mm at 100 mm/s with 100 mm/s²: accelerates 1 s (50 mm), cruises 1 s (100 mm), decelerates 1 s
    assert plan(interpret("G1 X200 F6000", m), m).total_time == pytest.approx(3.0)
    # 100 mm: triangle that just reaches 100 mm/s → 2 s
    assert plan(interpret("G1 X100 F6000", m), m).total_time == pytest.approx(2.0)


def test_collinear_segments_do_not_slow_down():
    m = fast_machine()
    split = "\n".join(f"G1 X{10 * i} F6000" for i in range(1, 21))
    assert plan(interpret(split, m), m).total_time == pytest.approx(3.0)


def test_corner_speed_follows_junction_deviation():
    m = fast_machine(junction_deviation=0.01)
    pl = plan(interpret("G1 X100 F6000\nY100", m), m)
    s = math.sqrt(0.5)
    assert pl.blocks[0].v_exit == pytest.approx(math.sqrt(100 * 0.01 * s / (1 - s)))
    assert pl.blocks[1].v_exit == 0.0
    rev = interpret("G1 X100 F6000\nX0", m)
    assert plan(rev, m).blocks[0].v_exit == 0.0                          # reversal: stops


def test_per_axis_limits_cap_speed():
    m = Machine(axes={"X": Axis("X", max_rate=600, accel=1000), "Y": Axis("Y", max_rate=6000, accel=1000)})
    b = plan(interpret("G1 X10 Y10 F100000", m), m).blocks[0]
    assert b.v_nominal == pytest.approx(10 / math.sqrt(0.5))               # X limits: 10 mm/s / cos 45°


def test_pauses_force_stops_and_dwell_adds_time():
    m = fast_machine()
    a = plan(interpret("G1 X100 F6000\nX200", m), m).total_time
    b = plan(interpret("G1 X100 F6000\nG4 P1.5\nX200", m), m)
    assert b.dwell_time == 1.5 and b.total_time > a + 1.5


# --- interpreter ----------------------------------------------------------------------
def test_units_distance_modes_and_offsets():
    m = fast_machine(offsets={"G55": {"X": 100, "Y": 50}})
    p = interpret("G20 G90 G0 X1 Y2\nG21 G91 G1 X10 F100\nG90 G55 G0 X0 Y0\nG92 X5\nG0 X0\nG92.1\n"
                  "G10 L2 P1 X7\nG54 G0 X0", m)
    assert p.ok, p.diags
    assert moves(p) == [("rapid", {"X": 25.4, "Y": 50.8, "Z": 0}), ("feed", {"X": 35.4, "Y": 50.8, "Z": 0}),
                        ("rapid", {"X": 100, "Y": 50, "Z": 0}), ("rapid", {"X": 95, "Y": 50, "Z": 0}),
                        ("rapid", {"X": 7, "Y": 50, "Z": 0})]


def test_g10_l20_and_tool_length():
    m = fast_machine(tools={2: Tool(2, length=5.0)})
    p = interpret("G0 X12 Z-3\nG10 L20 P1 X0 Z0\nG0 X1\nT2 M6\nG43 H2\nG0 Z0\nG49 G0 Z0", m)
    assert p.ok, p.diags
    assert moves(p)[1:] == [("rapid", {"X": 13, "Y": 0, "Z": -3}), ("rapid", {"X": 13, "Y": 0, "Z": 2}),
                            ("rapid", {"X": 13, "Y": 0, "Z": -3})]
    assert p.tools == [2]


def test_arcs_ij_r_full_circle_and_errors():
    m = fast_machine()
    p = interpret("G0 X10 Y0\nG2 X-10 Y0 I-10 J0 F600\nG2 X10 Y0 R10\nG3 X10 Y0 I-10\nG3 X-10 Y0 R-10", m)
    assert p.ok, p.diags
    arcs = [o.arc for o in p.ops if o.kind == "arc"]
    assert [round(a.sweep, 6) for a in arcs] == [round(-math.pi, 6), round(-math.pi, 6), round(2 * math.pi, 6),
                                                 round(math.pi, 6)]
    assert all(a.radius == pytest.approx(10) for a in arcs)
    assert arcs[0].center == pytest.approx((0, 0))
    bad = interpret("G2 X10 Y0 I3 J0 F100", m)
    assert "radio inicial" in bad.errors[0].msg
    assert "demasiado chico" in interpret("G2 X50 R10 F100", m).errors[0].msg


def test_helical_arc_planned_length():
    m = fast_machine(arc_tolerance=0.0001)
    pl = plan(interpret("G0 X10\nG3 X10 Y0 Z-5 I-10 F600", m), m)
    feed = pl.feed_length
    assert feed == pytest.approx(math.hypot(2 * math.pi * 10, 5), rel=1e-4)


def test_mach3_habits_and_unsupported():
    m = fast_machine()
    p = interpret("G0 X1 (comentario)\n/G0 X99\nG51 X2\nM101\nG1 X2\n#1=5\nG1 E2 X3 F100\nM98 P100\nG0 X4 M30\nG0 X5", m)
    msgs = [(d.level, d.line, d.msg) for d in p.diags]
    assert ("error", 3) == msgs[0][:2] and "Mach3" in msgs[0][2]
    assert any(lv == "aviso" and ln == 4 and "macro" in t for lv, ln, t in msgs)
    assert any(ln == 5 and "sin F" in t for _, ln, t in msgs)
    assert any(ln == 6 and "parámetros" in t for _, ln, t in msgs)
    assert any(ln == 7 and "impresora" in t for _, ln, t in msgs)
    assert any(ln == 8 and "M98" in t for _, ln, t in msgs)
    assert p.ended and moves(p)[-1][1]["X"] == 4                        # nothing after M30
    assert all(o.end.get("X") != 99 for o in p.ops)                      # block delete


def test_modal_group_conflict_and_repeated_letter():
    m = fast_machine()
    assert "mismo grupo" in interpret("G0 G1 X1", m).errors[0].msg
    assert "dos veces" in interpret("G0 X1 X2", m).errors[0].msg
    assert "no configurado" in interpret("G0 A10", m).errors[0].msg


def test_canned_cycles_expand_with_retract_modes():
    m = fast_machine()
    p = interpret("G0 Z10\nG98 G81 X5 Y5 Z-3 R2 F100\nX15\nG99 X25\nG80\nG0 X0", m)
    assert p.ok, p.diags
    zs = [(o.kind, round(o.end["X"], 3), round(o.end["Z"], 3)) for o in p.ops if o.source == "G81"]
    assert zs[:4] == [("rapid", 5, 10), ("rapid", 5, 2), ("feed", 5, -3), ("rapid", 5, 10)]
    assert zs[-1] == ("rapid", 25, 2)                                    # G99: stays at R
    p83 = interpret("G0 Z5\nG83 X0 Y0 Z-5 R1 Q2 F100\nG80", m)
    feeds = [round(o.end["Z"], 3) for o in p83.ops if o.kind == "feed"]
    assert feeds == [-1, -3, -5]
    assert interpret("G82 X0 Y0 Z-1 R1 P0.5 F50", m).ops[-2].kind == "dwell"
    assert "Q" in interpret("G83 X0 Y0 Z-5 R1 F100", m).errors[0].msg


# --- review ---------------------------------------------------------------------------
def test_check_soft_limits_spindle_and_report():
    m = Machine()                                     # X 0…300, Y 0…200, Z -60…0
    r = check_text("G0 X10 Y10\nG1 Z-1 F100\nG2 X10 Y10 I0 J-15\nM30", m)
    assert not r["runnable"]
    assert any("Y sale del recorrido" in e and "línea 3" in e for e in r["errores"])
    assert any("husillo apagado" in a for a in r["avisos"])
    ok = check_text("G0 X10 Y10\nM3 S10000\nG1 Z-1 F100\nG1 X50\nM5\nM30", m)
    assert ok["runnable"] and not ok["errores"] and not ok["avisos"]
    assert "LISTO" in report_text(ok) and "NO se puede" in report_text(r)
    assert ok["envolvente_g54"]["X"] == [0.0, 50.0]


def test_machine_yaml_roundtrip(tmp_path):
    m = load_machine("configs/cnc_maquina.yaml")
    assert m.axes["Z"].min == -90 and m.tools[2].length == 5.2 and m.controller["tipo"] == "sim"
    with pytest.raises(ValueError):
        machine_from_dict({"ejes": {"Q": {}}})
    with pytest.raises(ValueError):
        machine_from_dict({"tipo": "impresora"})


# --- translation ----------------------------------------------------------------------
PART = """%
(pieza de prueba)
G21 G90 G17
G55
T1 M6
S12000 M3
M8
G0 X5 Y5 Z5
G1 Z-1 F300
G2 X25 Y5 R10 F600
G18 G2 X35 Z-1 I5 K0
G17
G98 G83 X40 Y20 Z-6 R1 Q2 F150
G80
G0 Z5
M9
M30
%"""


def test_emit_roundtrip_matches_original():
    m = fast_machine(offsets={"G55": {"X": 10, "Y": 20, "Z": -2}})
    p = interpret(PART, m)
    assert p.ok, p.diags
    text = emit_text(p, m, "grbl")
    assert "G83" not in text and "M6" not in text and "M0" in text
    back = interpret(text, m)
    assert back.ok, back.diags
    assert moves(back) == moves(p)
    assert plan(back, m).total_time == pytest.approx(plan(p, m).total_time, rel=1e-6)
    assert "T1 M6" in emit_text(p, m, "linuxcnc")


# --- fake grbl ------------------------------------------------------------------------
class FakeGrbl:
    """Replies like grbl: ``ok`` per processed line, status on «?», never more than RX bytes waiting."""

    def __init__(self, rx=128, fail_on=None, alarm_on=None):
        self.rx, self.fail_on, self.alarm_on = rx, fail_on, alarm_on
        self.queue, self.out, self.lines, self.realtime = [], [b"Grbl 1.1h ['$' for help]\r\n"], [], []
        self.max_used = 0
        self.state = "Idle"

    def write(self, data: bytes):
        for ch in (b"?", b"!", b"~", b"\x18"):
            if data == ch:
                self.realtime.append(ch)
                if ch == b"?":
                    self.out.append(f"<{self.state}|MPos:1.000,2.000,3.000|FS:500,12000|WCO:1.000,0.000,0.000>\r\n"
                                    .encode())
                if ch == b"~":
                    self.state = "Idle"
                return
        for line in data.decode().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line == "$#":
                self.out += [b"[G54:10.000,20.000,-2.000]\r\n", b"[G55:0.000,0.000,0.000]\r\n", b"ok\r\n"]
                continue
            self.queue.append(line)
            self.max_used = max(self.max_used, sum(len(q) + 1 for q in self.queue))
            assert self.max_used <= self.rx

    def readline(self) -> bytes:
        if self.out:
            return self.out.pop(0)
        if self.queue:
            line = self.queue.pop(0)
            self.lines.append(line)
            if line.startswith("M0"):
                self.state = "Hold"
            if self.alarm_on and self.alarm_on in line:
                return b"ALARM:2\r\n"
            return b"error:20\r\n" if self.fail_on and self.fail_on in line else b"ok\r\n"
        return b""

    def close(self):
        pass


def test_grbl_character_counting_and_status():
    port = FakeGrbl(rx=64)
    g = GrblBackend(port, rx_buffer=64)
    g.connect()
    assert g.banner.startswith("Grbl")
    lines = [f"G1 X{i}.123 Y{i}.456 F1000" for i in range(200)]
    r = g.stream(lines)
    assert r.ok and r.acknowledged == 200 and len(port.lines) == 200
    assert 40 < port.max_used <= 64                    # buffer full but not overflowing
    st = g.status()
    assert st.state == "Idle" and st.mpos == {"X": 1, "Y": 2, "Z": 3} and st.wpos["X"] == 0 and st.spindle == 12000
    assert g.work_offsets()["G54"] == {"X": 10, "Y": 20, "Z": -2}


def test_grbl_error_and_alarm_abort():
    port = FakeGrbl(fail_on="X3")
    r = GrblBackend(port).stream([f"G1 X{i} F100" for i in range(10)])
    assert r.aborted and "error:20" in r.errors[0] and b"\x18" in port.realtime
    r2 = GrblBackend(FakeGrbl(alarm_on="X2")).stream([f"G0 X{i}" for i in range(5)])
    assert r2.aborted and "soft limit" in r2.alarm
    assert "alarma" in describe("error:9")


def test_parse_status_variants():
    s = parse_status("<Hold:0|WPos:1.5,2,3|F:250>")
    assert s.state == "Hold" and s.wpos["X"] == 1.5 and s.feed == 250
    with pytest.raises(BackendError):
        parse_status("basura")


# --- controller and guardian ------------------------------------------------------------
@pytest.fixture
def guardian(tmp_path):
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "budget.yaml").write_text("max_total_params: 1000\n")
    return Guardian(GuardianConfig(str(tmp_path), [str(gdir)], str(gdir / "budget.yaml"), str(gdir / "audit.jsonl"),
                                   str(gdir / "approvals.json")))


GOOD = "G0 X10 Y10\nM3 S10000\nG1 Z-1 F200\nG1 X40\nM0 (MSG, revisar)\nG1 X60\nM5\nM30"


def test_sim_runs_without_approval():
    m = Machine()
    sim = SimBackend(m)
    r = Controller(m, sim).run(GOOD)
    assert r["estado"] == "completado" and sim.pos["X"] == 60 and sim.clock > 0


def test_real_machine_needs_human_approval(guardian):
    m = Machine()
    port = FakeGrbl()
    ctl = Controller(m, GrblBackend(port), guardian)
    assert Controller(m, GrblBackend(FakeGrbl())).run(GOOD)["estado"] == "rechazado"     # without guardian
    r = ctl.run(GOOD)
    assert r["estado"] == "pendiente" and r["solicitud"] and not port.lines
    req = r["solicitud"]
    guardian.approve(req, "operador")
    asked = []
    done = ctl.run(GOOD, approval_id=req, operator=lambda msg: asked.append(msg) or True)
    assert done["estado"] == "completado", done
    assert asked == ["revisar"] and b"~" in port.realtime
    assert port.lines[0].startswith("G21G90") and "M2" in port.lines
    # the real G54 (10, 20, -2) is used: the first X sent is the program's, in G54
    assert port.lines[1] == "G0X10Y10"
    # the approval is consumed and is not valid for another program
    assert ctl.run(GOOD, approval_id=req)["estado"] == "pendiente"
    assert ctl.run(GOOD.replace("X60", "X61"), approval_id=req)["estado"] == "pendiente"


def test_operator_can_abort_at_pause(guardian):
    m = Machine()
    port = FakeGrbl()
    ctl = Controller(m, GrblBackend(port), guardian)
    req = ctl.run(GOOD)["solicitud"]
    guardian.approve(req, "operador")
    r = ctl.run(GOOD, approval_id=req, operator=lambda msg: False)
    assert r["estado"] == "abortado" and b"\x18" in port.realtime and "G1X60" not in port.lines


def test_programs_with_errors_never_reach_the_machine(guardian):
    port = FakeGrbl()
    r = Controller(Machine(), GrblBackend(port), guardian).run("G0 X500\nM30")
    assert r["estado"] == "rechazado" and not port.lines and not guardian.pending()


def test_linuxcnc_backend_reports_missing_module():
    from hope.cnc.backends.linuxcnc import LinuxCNCBackend
    with pytest.raises(BackendError, match="LinuxCNC"):
        LinuxCNCBackend()


# --- internal tool and console ------------------------------------------------------
@pytest.mark.skipif(sys.version_info < (3, 10), reason="el catálogo de herramientas de Hope (no TISMA CNC) es 3.10+")
def test_internal_tool_reviews_without_moving(tmp_path):
    from hope.runtime.toolcalls import HELP, ToolRunner
    from hope.tools.builtin import default_catalog
    work = tmp_path / "w"
    work.mkdir()
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "budget.yaml").write_text("max_total_params: 1000\n")
    g = Guardian(GuardianConfig(str(work), [str(gdir)], str(gdir / "budget.yaml"), str(gdir / "audit.jsonl"),
                                str(gdir / "approvals.json")))
    (work / "pieza.nc").write_text(GOOD)
    (work / "maquina.yaml").write_text("ejes:\n  X: {min: 0, max: 50}\n  Y: {min: 0, max: 50}\n  Z: {min: -10, max: 0}\n")
    cat = default_catalog(g)
    assert "cnc" in HELP and cat.spec("hte_cnc_v1").permissions
    runner = ToolRunner(cat)
    out = runner.run("cnc", "pieza.nc | dialecto=grbl")
    assert "LISTO" in out and "grbl" in out
    out2 = runner.run("cnc", "pieza.nc | maquina=maquina.yaml")
    assert "NO se puede" in out2 and "X sale del recorrido" in out2


def test_cli_review_and_translate(tmp_path, capsys):
    from hope.cnc.__main__ import main
    f = tmp_path / "p.nc"
    f.write_text(GOOD)
    assert main(["revisar", str(f)]) == 0
    assert "LISTO" in capsys.readouterr().out
    out = tmp_path / "g.nc"
    assert main(["traducir", str(f), "--dialecto", "grbl", "--salida", str(out)]) == 0
    assert out.read_text().startswith("G21 G90")
    assert main(["simular", str(f), "--json"]) == 0


# --- Mach3-specific codes (manual «Using Mach3Mill» §10.7) --------------------------------------------
def test_mach3_g12_g13_circular_pocket():
    m = fast_machine()
    p = interpret("G0 X50 Y50\nG1 Z-1 F200\nG12 I10\nG13 I5", m)
    assert p.ok, p.diags
    arcs = [o for o in p.ops if o.kind == "arc"]
    assert [round(a.arc.sweep, 4) for a in arcs] == [round(-2 * math.pi, 4), round(2 * math.pi, 4)]
    assert arcs[0].arc.center == (50, 50) and arcs[0].arc.radius == 10
    assert moves(p)[-1][1] == {"X": 50, "Y": 50, "Z": -1}                 # returns to the center
    back = interpret(emit_text(p, m), m)
    assert back.ok and moves(back) == moves(p)


def test_mach3_polar_coordinates_g16():
    m = fast_machine()
    p = interpret("G0 X10 Y5\nG16\nG0 X50 Y0\nY90\nG15\nG0 X0 Y0", m)       # example from the manual
    assert p.ok, p.diags
    assert moves(p)[1][1]["X"] == 60 and moves(p)[1][1]["Y"] == 5
    assert moves(p)[2][1]["X"] == pytest.approx(10) and moves(p)[2][1]["Y"] == 55
    assert moves(p)[3][1] == {"X": 0, "Y": 0, "Z": 0}                     # G15: Cartesian again
    assert "polares" in interpret("G16\nG2 X1 Y1 R5 F100", m).errors[0].msg


def test_mach3_g52_temporary_offset():
    m = fast_machine()
    p = interpret("G0 X4\nG52 X7\nG0 X0\nG52 X0\nG0 X0", m)                # example from the manual: X4 → -3
    assert p.ok
    assert [mv[1]["X"] for mv in moves(p)] == [4, 7, 0]
    assert "G52" in interpret("G52", m).errors[0].msg


def test_mach3_subprograms_m98_m99():
    m = fast_machine()
    text = ("G21 G90\nG52 X10 Y10\nM98 P1000\nG52 X40 Y10\nM98 P1000 L2\nG52 X0 Y0\nM30\n"
            "O1000 (cuadrado)\nG0 X0 Y0\nG1 X5 F500\nM99\n")
    p = interpret(text, m)
    assert p.ok, p.diags
    xs = [mv[1]["X"] for mv in moves(p)]
    assert xs == [10, 15, 40, 45, 40, 45]                                   # L2 repeats the subprogram
    assert p.ended
    assert "O9" in interpret("M98 P9\nM30", m).errors[0].msg
    looped = interpret("O1\nG0 X1\nM98 P1\n", m)
    assert any("profundo" in d.msg for d in looped.errors)
