"""Plasma en TISMA CNC (hope/cnc): perforación automática, sensado de altura (G38.2 y G31 de Mach3),
orígenes que cambian en marcha, revisiones propias de plasma y el panel de antorcha de la sesión."""

import math
import time
from pathlib import Path

import pytest

from hope.cnc.check import check_program, report_text
from hope.cnc.dialect import emit_text
from hope.cnc.gcode import interpret
from hope.cnc.links import SimLink
from hope.cnc.machine import (
    Axis,
    Machine,
    Plasma,
    as_controller,
    load_machine,
    machine_from_dict,
)
from hope.cnc.session import CncSession, SessionError


def plasma_machine(**pl):
    axes = {"X": Axis("X", max_rate=15000, accel=800, min=0, max=1500),
            "Y": Axis("Y", max_rate=15000, accel=800, min=0, max=3000),
            "Z": Axis("Z", max_rate=3000, accel=500, min=-80, max=0)}
    return Machine(kind="plasma", axes=axes, offsets={"G54": {"X": 10, "Y": 10, "Z": -60}}, plasma=Plasma(**pl))


def work_z(op, m):
    return round(op.end["Z"] - m.offset("G54")["Z"], 3)


def moves(prog):
    return [(o.kind, {k: round(v, 3) for k, v in o.end.items()}) for o in prog.ops
            if o.kind in ("rapid", "feed", "arc", "probe")]


def test_auto_pierce_sequence_with_probe():
    m = plasma_machine(auto_pierce=True, probe=True, pierce_height=3.8, cut_height=1.5, pierce_delay=0.6,
                       travel_height=10, float_offset=1.5, probe_depth=20)
    p = interpret("G0 X50 Y50 Z10\nM3\nG1 X100 F2000\nM5\nM30", m)
    assert p.ok, p.diags
    seq = [(o.kind, o.source) for o in p.ops if o.source == "perforación"]
    assert [k for k, _ in seq] == ["probe", "wcs", "rapid", "spindle", "dwell", "feed", "rapid"]
    probe = next(o for o in p.ops if o.kind == "probe")
    assert work_z(probe, m) == -1.5                              # contacto supuesto: chapa en Z0 + holgura
    assert round(probe.value["Z"] - m.offset("G54")["Z"], 3) == -10   # recorrido completo para revisar límites
    zs = [work_z(o, m) for o in p.ops if o.kind in ("rapid", "feed") and o.source == "perforación"]
    assert zs == [3.8, 1.5, 10]
    assert next(o for o in p.ops if o.kind == "dwell").value == 0.6
    cut = next(o for o in p.ops if o.kind == "feed" and o.source == "")
    assert work_z(cut, m) == 1.5                                 # corta a la altura de corte


def test_auto_pierce_without_probe_and_cam_mode():
    m = plasma_machine(auto_pierce=True)
    p = interpret("G0 X50 Y50 Z10\nM3\nG1 X100 F2000\nM5", m)
    assert [o.kind for o in p.ops if o.source == "perforación"] == ["rapid", "spindle", "dwell", "feed", "rapid"]
    cam = interpret("G0 X50 Y50 Z3.8\nM3\nG4 P0.5\nG1 Z1.5 F1000\nG1 X100 F2000\nM5", plasma_machine())
    assert not any(o.source == "perforación" for o in cam.ops)    # el CAM ya trae la secuencia


def test_translation_keeps_probe_and_runtime_zero():
    m = plasma_machine(auto_pierce=True, probe=True)
    p = interpret("G0 X50 Y50 Z10\nM3\nG1 X100 F2000\nM5\nG0 X200\nM3\nG2 X200 Y100 R25\nM5\nM30", m)
    text = emit_text(p, m, "grbl")
    lines = text.splitlines()
    assert lines.count("G10 L20 P1 Z-1.5") == 2 and sum(ln.startswith("G38.2 Z") for ln in lines) == 2
    assert "M3" in lines and not any(ln.startswith("M3 S") for ln in lines)     # plasma: sin S
    i = lines.index("G10 L20 P1 Z-1.5")
    assert lines[i + 1] == "G0 Z3.8"         # alturas relativas a la chapa medida, no coordenadas fijas
    back = interpret(text, as_controller(m))
    assert back.ok and moves(back) == moves(p)   # al volver a leerlo no se duplica la perforación


def test_mach3_g31_probe_and_limits():
    m = plasma_machine()
    p = interpret("G0 Z5\nG31 Z-10 F300\nG10 L20 P1 Z-1.5\nG0 Z3.8\nM30", m)
    assert p.ok, p.diags
    probe = next(o for o in p.ops if o.kind == "probe")
    assert probe.source == "G31" and work_z(probe, m) == -1.5
    deep = check_program(interpret("G0 Z5\nG38.2 Z-30 F300\nM30", m), m)
    assert any("Z sale del recorrido" in e for e in deep["errores"])    # el peor caso pasa el límite de Z
    short = interpret("G0 Z5\nG38.2 Z2 F300", m)
    assert any("puede no tocar" in d.msg for d in short.diags)
    assert "F" in interpret("G38.2 Z-5", m).errors[0].msg
    assert "G38.4" in interpret("G38.4 Z-5 F100", m).errors[0].msg


def test_plasma_checks_and_report():
    m = plasma_machine()
    text = ("G0 X10 Y10 Z3.8\nM3\nG1 X50 F2000\nG0 X60\nG2 X60 Y10 I-10 J0 F1500\nM5\n"
            "G1 X80 F2000\nM30")
    r = check_program(interpret(text, m), m)
    assert r["perforaciones"] == 1 and r["tipo"] == "plasma"
    assert r["corte_antorcha_mm"] == pytest.approx(40 + 2 * math.pi * 10, rel=1e-3)
    avisos = " ".join(r["avisos"])
    assert "rápidos con la antorcha encendida" in avisos
    assert "sin retardo" in avisos
    assert "antorcha apagada" in avisos
    assert "Plasma: 1 perforaciones" in report_text(r)
    ok = check_program(interpret("G0 X10 Y10 Z3.8\nM3\nG4 P0.5\nG1 Z1.5 F1000\nG1 X50 F2000\nM5\nM30", m), m)
    assert ok["runnable"] and not ok["avisos"]


def test_plasma_config_yaml_and_validation():
    m = load_machine("configs/cnc_plasma.yaml")
    assert m.is_plasma and m.plasma.auto_pierce and m.plasma.probe and m.plasma.float_offset == 1.5
    assert Machine(kind="plasma").plasma is not None
    with pytest.raises(ValueError, match="altura"):
        machine_from_dict({"tipo": "plasma", "plasma": {"altura_corte": 5, "altura_perforacion": 3}})
    r = check_program(interpret(Path("hope/cnc/ejemplos/plasma_brida.nc").read_text(), m), m)
    assert r["runnable"] and r["perforaciones"] == 8
    assert r["corte_antorcha_mm"] == pytest.approx(6 * (2 * math.pi * 8 + 4) + 2 * math.pi * 40 + 6
                                                   + 2 * math.pi * 100 + 6, rel=1e-3)


# --- sesión (panel de antorcha) ------------------------------------------------------------------
def run_sim(link, seconds, dt=0.01):
    for _ in range(int(seconds / dt)):
        link.step(dt)


def plasma_session():
    m = load_machine("configs/cnc_plasma.yaml")
    link = SimLink(m, autostart=False)
    s = CncSession(m, link)
    s.reset()
    return s, link


def test_session_counts_pierces_and_torch_state():
    s, link = plasma_session()
    s.load("b.nc", "G0 X50 Y50 Z10\nM3\nG1 X100 F3000\nM5\nG0 X150\nM3\nG1 Y100 F3000\nM5\nM30")
    st = s.state()
    assert st["plasma"] == {"antorcha": False, "perforaciones_total": 2, "perforaciones_hechas": 0}
    s.cycle_start()
    seen_on = False
    for _ in range(3000):
        link.step(0.01)
        st = s.state()
        seen_on |= st["plasma"]["antorcha"]
        if st["estado"]["trabajo"]["terminado"]:
            break
    assert seen_on and st["plasma"]["perforaciones_hechas"] == 2 and not st["plasma"]["antorcha"]
    tp = s.toolpath()
    assert len(tp["perforaciones"]) == 2 and tp["perforaciones"][0][:2] == [50, 50]
    assert {p[3] for p in tp["puntos"]} >= {0, 1, 3}             # rápido, corte y sensado (sin cortar)
    assert s.config()["plasma"]["sensado"] and s.config()["marca"] == "TISMA CNC"


def test_session_probe_and_test_fire_in_simulator():
    s, link = plasma_session()
    s.mdi("G0 X100 Y100 Z5")                                     # 5 mm sobre la chapa
    run_sim(link, 5)
    s.plasma_probe()
    for _ in range(400):
        run_sim(link, 0.05)
        time.sleep(0.002)
        if s.state()["trabajo_pos"]["Z"] == 10 and s.state()["estado"]["estado"] == "Idle":
            break
    assert s.state()["trabajo_pos"]["Z"] == pytest.approx(10)    # terminó en altura de traslado
    assert link.offsets()["G54"]["Z"] == pytest.approx(-60)      # chapa donde se esperaba (simulador)
    s.test_fire()
    fired = False
    for _ in range(400):
        run_sim(link, 0.02)
        time.sleep(0.002)
        fired |= s.state()["plasma"]["antorcha"]
        if fired and not s.state()["plasma"]["antorcha"]:
            break
    assert fired and not s.state()["plasma"]["antorcha"]
    router = CncSession(Machine(), SimLink(Machine(), autostart=False))
    router.reset()
    with pytest.raises(SessionError, match="plasma"):
        router.test_fire()


def test_mdi_keeps_modal_state_for_checks():
    m = Machine()
    link = SimLink(m, autostart=False)
    s = CncSession(m, link)
    s.reset()
    s.mdi("G91")
    for _ in range(3):
        s.mdi("G0 X100")
        run_sim(link, 5)
    assert link.pos["X"] == pytest.approx(300)                   # incremental, como en el controlador
    with pytest.raises(SessionError, match="recorrido"):
        s.mdi("G0 X10")                                          # 310 > 300: se revisa en incremental
    s.mdi("G90")
    s.mdi("G0 X10")
    run_sim(link, 10)
    assert link.pos["X"] == pytest.approx(10)
