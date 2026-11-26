"""Parallel port: the pulse engine (tisma_lpt.c) and its connection (paralelo.py).

The engine is built here in its simulator form (same C code, simulated port, virtual time), so
every pulse can be counted: exact step counts, never two steps of an axis in one tick, the speed
ramps of the acceleration, feed hold, probe, home and limit switches, emergency stop, pin
polarities and the charge pump. Then LptLink drives the simulator «live» (CFG INTERACTIVE) to
check the whole chain: interpreter → planner → engine → position back on screen.
The real Windows builds (32/64-bit) are only compiled when mingw is installed.
"""

import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from hope.cnc.backends.base import BackendError
from hope.cnc.paralelo import LptLink, cap_rates, engine_config
from hope.cnc.profiles import ProfileStore, default_selection, effective

SRC = Path(__file__).resolve().parent.parent / "hope" / "cnc" / "lpt"
pytestmark = pytest.mark.skipif(shutil.which("gcc") is None or sys.platform.startswith("win"),
                                reason="hace falta gcc (Linux/macOS) para compilar el simulador del motor")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """The simulator build of the engine (compiled once for this file)."""
    out = tmp_path_factory.mktemp("lpt")
    subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Werror", "-DTISMA_SIM", "-o", str(out / "sim"),
                    str(SRC / "tisma_lpt.c"), "-lm"], check=True)
    return out / "sim"


AXES_CFG = ["CFG AXIS 0 STEP 2 DIR 3 STEPLOW 0 DIRINV 0", "CFG AXIS 1 STEP 4 DIR 5 STEPLOW 0 DIRINV 0",
            "CFG AXIS 2 STEP 6 DIR 7 STEPLOW 0 DIRINV 0"]


def run(engine, *lines, trace=1):
    """Runs the simulator with the axes above plus ``lines``; returns (pulses, other lines)."""
    cfg = [ln for ln in lines if ln.startswith("CFG")]               # configuration goes before START
    cmds = [ln for ln in lines if not ln.startswith("CFG")]
    text = "\n".join(["CFG FREQ 25000", "CFG REPORT 0", f"CFG TRACE {trace}", *AXES_CFG, *cfg, "START", *cmds]) + "\n"
    out = subprocess.run([str(engine)], input=text.encode(), stdout=subprocess.PIPE, check=True).stdout.decode()
    pulses = [tuple(int(x) for x in ln.split()[1:]) for ln in out.splitlines() if ln.startswith("PULSE")]
    rest = [ln for ln in out.splitlines() if not ln.startswith(("PULSE", "PORT"))]
    return pulses, rest, out


def counts(pulses):
    c = Counter()
    for _, mask, _ in pulses:
        for i in range(6):
            if mask >> i & 1:
                c[i] += 1
    return c


def final(rest):
    """Last status line → (state, done id, steps per axis)."""
    p = [ln for ln in rest if ln.startswith("S ")][-1].split()
    return p[1], int(p[2]), [int(x) for x in p[3:9]]


# ==================================================================================================
# the engine
# ==================================================================================================

def test_exact_steps_in_a_straight_line(engine):
    pulses, rest, _ = run(engine, "M 1 12.5 10 0 200 0 8000 -6000 1234 0 0 0")
    assert counts(pulses) == {0: 8000, 1: 6000, 2: 1234}
    assert final(rest) == ("I", 1, [8000, -6000, 1234, 0, 0, 0])
    assert [ln for ln in rest if ln.startswith("S ")][-1].split()[10] == "0.0000"      # no speed at rest
    assert all(d & 2 for _, m, d in pulses if m & 2)                  # Y always with its direction negative
    # DDA: halfway through X's pulses, Y and Z are halfway too (straight line, not a dog leg)
    k = [i for i, (_, m, _) in enumerate(pulses) if m & 1][3999]
    half = counts(pulses[:k + 1])
    assert abs(half[1] - 3000) <= 1 and abs(half[2] - 617) <= 1


def test_never_two_steps_per_tick_and_the_ramp_follows_the_acceleration(engine):
    # 20 mm at 20 mm/s with 100 mm/s²: 0.2 s ramp up, 0.8 s cruise, 0.2 s down (800 steps/mm).
    pulses, _, _ = run(engine, "M 1 20 20 0 100 0 16000 0 0 0 0 0")
    ticks = [t for t, m, _ in pulses]
    assert len(set(ticks)) == len(ticks) == 16000
    total = ticks[-1] - ticks[0]
    assert total / 25000 == pytest.approx(1.2, abs=0.01)             # trapezoid time
    first_tenth = sum(1 for t in ticks if t < 2500)                  # motion starts at tick 0: s = ½·a·t² in 0.1 s
    assert first_tenth == pytest.approx(0.5 * 100 * 0.1 ** 2 * 800, rel=0.05)


def test_junction_speed_is_kept_between_blocks(engine):
    # Two collinear blocks with exit speed = cruise: no stop in between (same time as one block).
    p2, _, _ = run(engine, "M 1 10 20 20 100 0 8000 0 0 0 0 0", "M 2 10 20 0 100 0 8000 0 0 0 0 0")
    p1, _, _ = run(engine, "M 1 20 20 0 100 0 16000 0 0 0 0 0")
    assert abs((p2[-1][0] - p2[0][0]) - (p1[-1][0] - p1[0][0])) < 30


def test_feed_hold_stops_inside_the_block_and_resume_finishes_it(engine):
    pulses, rest, _ = run(engine, "M 1 20 20 0 100 0 16000 0 0 0 0 0", "@0.5 HOLD", "@1.5 RESUME")
    assert counts(pulses)[0] == 16000 and final(rest)[0] == "I"
    ticks = [t for t, m, _ in pulses]
    gap = max(b - a for a, b in zip(ticks, ticks[1:]))   # noqa: RUF007  (pairwise is 3.10+)
    assert gap > 0.5 * 25000                                          # it really stood still


def test_cancel_brakes_and_drops_the_queue(engine):
    pulses, rest, _ = run(engine, "M 1 20 20 0 100 0 16000 0 0 0 0 0", "M 2 5 20 0 100 0 0 4000 0 0 0 0",
                          "@0.5 CANCEL")
    n = counts(pulses)
    assert 5000 < n[0] < 16000 and n[1] == 0
    assert final(rest)[0] == "I" and "ALARM" not in " ".join(rest)


def test_probe_latches_the_trigger_and_alarms_without_contact(engine):
    _, rest, _ = run(engine, "CFG IN PROBE 15 0", "SIMIN PROBE 2 < -4000",
                     "M 1 10 5 0 100 0 0 0 -8000 0 0 0 P2")
    prb = [ln for ln in rest if ln.startswith("PRB")]
    assert prb == ["PRB 1 0 0 -4001 0 0 0"]                           # exactly where it switched
    assert final(rest)[2][2] < -4001                                  # braked a little further
    _, rest, _ = run(engine, "CFG IN PROBE 15 0", "M 1 2 5 0 100 0 0 0 -1600 0 0 0 P2")
    assert any(ln.startswith("ALARM sensado sin contacto") for ln in rest) and final(rest)[0] == "A"
    _, rest, _ = run(engine, "CFG IN PROBE 15 0", "M 1 2 5 0 100 0 0 0 -1600 0 0 0 P3")
    assert "PRB 0 0 0 -1600 0 0 0" in rest and final(rest)[0] == "I"


def test_home_switch_stops_and_limits_raise_an_alarm(engine):
    _, rest, _ = run(engine, "CFG IN HOME0 10 1", "SIMIN HOME0 0 < -2400", "M 7 100 5 0 100 0 -80000 0 0 0 0 0 H0")
    assert "HOMED 7" in rest and final(rest)[2][0] == -2401
    _, rest, _ = run(engine, "CFG IN LIMIT1 11 0", "SIMIN LIMIT1 1 > 800", "M 1 10 5 0 100 0 0 8000 0 0 0 0")
    assert any("fin de carrera del eje Y" in ln for ln in rest) and final(rest)[2][1] == 801
    # Backing off a switch that is still pressed: the move ignores that limit.
    _, rest, _ = run(engine, "CFG IN LIMIT0 10 0", "IN LIMIT0 1", "M 1 2 5 0 100 0 1600 0 0 0 0 0 I1")
    assert final(rest) == ("I", 1, [1600, 0, 0, 0, 0, 0])


def test_emergency_stop_cuts_everything_until_released(engine):
    _, rest, out = run(engine, "CFG IN ESTOP 15 1", "CFG OUT SPINDLE_CW 1 0", "O 1 SPINDLE_CW 1",
                       "M 2 20 20 0 100 0 16000 0 0 0 0 0", "@0.3 IN ESTOP 1", "@0.4 RESET", "@0.5 IN ESTOP 0",
                       "@0.6 RESET", "@0.7 M 3 1 5 0 100 0 800 0 0 0 0 0", trace=2)
    assert "ALARM parada de emergencia" in rest
    state, done, steps = final(rest)
    assert state == "I" and done == 3 and 800 < steps[0] < 16000      # moved again only after release + RESET
    # Spindle relay on pin 1 (control bit 0, inverted by the port hardware): on → bit 0 = 0; e-stop → off.
    ctrl = [int(ln.split()[3]) for ln in out.splitlines() if ln.startswith("PORT") and ln.split()[2] == "2"]
    assert ctrl[0] & 1 == 0 and ctrl[-1] & 1 == 1


def test_pin_polarity_and_charge_pump(engine):
    # X step active low on pin 2: idle level high (bit 0 = 1), pulses pull it to 0.
    _, _, out = run(engine, "CFG OUT CHARGEPUMP 16 0", "CFG AXIS 0 STEP 2 DIR 3 STEPLOW 1 DIRINV 1",
                    "M 1 0.01 5 0 100 0 8 0 0 0 0 0", trace=2)
    data = [int(ln.split()[3]) for ln in out.splitlines() if ln.startswith("PORT") and ln.split()[2] == "0"]
    assert data[0] & 1 == 1 and sum(1 for v in data if not v & 1) == 8
    assert all(v & 2 for v in data)                                   # +X with DIRINV: dir pin high
    ctrl = [int(ln.split()[3]) & 4 for ln in out.splitlines() if ln.startswith("PORT") and ln.split()[2] == "2"]
    assert len(ctrl) > 10 and ctrl[:4] in ([4, 0, 4, 0], [0, 4, 0, 4])  # 12.5 kHz: toggles every tick


def test_bad_configuration_is_reported(engine):
    _, rest, _ = run(engine, "CFG IN PROBE 2 0", "CFG OUT SPINDLE_CW 10 0")
    assert sum(ln.startswith("ERR") for ln in rest) == 2


def test_windows_builds_are_xp_compatible(tmp_path):
    if shutil.which("i686-w64-mingw32-gcc") is None:
        pytest.skip("sin mingw-w64 no se compilan los .exe")
    subprocess.run(["sh", str(SRC / "compilar.sh"), str(tmp_path)], check=True)
    pefile = pytest.importorskip("pefile")
    sys.path.insert(0, str(SRC.parent.parent.parent / "empaquetado" / "tisma_cnc"))
    import construir
    xp = tmp_path / "xp"
    xp.mkdir()
    shutil.copy(tmp_path / "tisma_lpt32.exe", xp)
    assert construir.funciones_post_xp(xp) == []
    for bits, machine in (("32", 0x14C), ("64", 0x8664)):
        pe = pefile.PE(str(tmp_path / f"tisma_lpt{bits}.exe"), fast_load=True)
        assert pe.FILE_HEADER.Machine == machine
        pe.close()


# ==================================================================================================
# configuration
# ==================================================================================================

def _machine():
    store = ProfileStore(None)
    m, _ = effective(store, default_selection(store))
    return m


def test_pin_table_to_engine_lines():
    m = _machine()
    lines = engine_config({"puerto": "0x278", "pines": [
        {"senal": "X paso", "pin": 2, "bajo": 1}, {"senal": "X dir", "pin": 3, "bajo": 0},
        {"senal": "Y paso", "pin": 4}, {"senal": "Y dir", "pin": 5}, {"senal": "Z paso", "pin": 6},
        {"senal": "Z dir", "pin": 7, "bajo": 1}, {"senal": "Husillo", "pin": 1, "bajo": 0},
        {"senal": "Bomba de carga", "pin": 17}, {"senal": "Parada de emergencia", "pin": 15, "bajo": 1},
        {"senal": "Límite X", "pin": 10}, {"senal": "Referencia X", "pin": 10}, {"senal": "Sonda", "pin": 13}]}, m)
    assert "CFG PORT 632" in lines and "CFG AXIS 0 STEP 2 DIR 3 STEPLOW 1 DIRINV 0" in lines
    assert "CFG AXIS 2 STEP 6 DIR 7 STEPLOW 0 DIRINV 1" in lines
    assert {"CFG OUT SPINDLE_CW 1 0", "CFG OUT CHARGEPUMP 17 0", "CFG IN ESTOP 15 1", "CFG IN LIMIT0 10 0",
            "CFG IN HOME0 10 0", "CFG IN PROBE 13 0"} <= set(lines)
    with pytest.raises(BackendError, match="pin 2"):
        engine_config({"pines": [{"senal": "X paso", "pin": 2}, {"senal": "Husillo", "pin": 2}]}, m)
    with pytest.raises(BackendError, match="eje Y"):
        engine_config({"pines": [{"senal": "X paso", "pin": 2}, {"senal": "X dir", "pin": 3}]}, m)
    with pytest.raises(BackendError, match="desconocida"):
        engine_config({"pines": [{"senal": "Turbo", "pin": 9}]}, m)
    with pytest.raises(BackendError, match="dirección"):
        engine_config({"puerto": "LPT1"}, m)


def test_speeds_are_capped_to_the_step_rate():
    m = _machine()                                        # 800 steps/mm, 5000 mm/min
    capped, notes = cap_rates(m, 25000)
    assert capped.axes["X"].max_rate == pytest.approx(25000 / 2 / 800 * 60) and "X limitado" in notes[0]
    assert m.axes["X"].max_rate == 5000                   # the profile is not touched


def test_not_on_linux_without_an_engine():
    if sys.platform.startswith("win"):
        pytest.skip("solo fuera de Windows")
    with pytest.raises(BackendError, match="LinuxCNC"):
        LptLink(_machine(), {"tipo": "paralelo"})


# ==================================================================================================
# the live link (engine simulator in interactive mode)
# ==================================================================================================

def _link(engine, *extra, cfg=None):
    c = {"tipo": "paralelo", "motor": str(engine),
         "motor_extra": ["CFG INTERACTIVE 40", "CFG REPORT 20", *extra]}
    c.update(cfg or {})
    return LptLink(_machine(), c)


def _until(fn, timeout=20.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.01)
    return False


def test_live_program_with_spindle_and_coolant(engine):
    link = _link(engine)
    try:
        link.mdi("G53 G0 X100 Y80 Z-10")
        assert _until(lambda: link.snapshot()["estado"] == "Idle")
        assert link.snapshot()["mpos"] == {"X": 100, "Y": 80, "Z": -10}
        link.start(["G90 G54", "M3 S12000", "M8", "G1 X10 Y10 F600", "G2 X30 Y10 I10 J0", "M9", "M5", "M30"])
        assert _until(lambda: link.snapshot()["trabajo"]["terminado"])
        s = link.snapshot()
        assert s["mpos"]["X"] == pytest.approx(50) and s["mpos"]["Y"] == pytest.approx(30)   # G54 = X20 Y20
        assert s["trabajo"]["confirmadas"] == 8 and s["husillo_dir"] == "off" and not s["refrigerante"]
    finally:
        link.close()


def test_live_stop_keeps_the_position_and_accepts_new_moves(engine):
    link = _link(engine)
    try:
        link.start(["G90 G53 G1 X300 F900"])
        assert _until(lambda: link.snapshot()["mpos"]["X"] > 5)
        link.stop()
        s = link.snapshot()
        assert s["estado"] == "Idle" and 5 < s["mpos"]["X"] < 300 and s["trabajo"]["error"]
        link.mdi("G53 G0 X0")
        assert _until(lambda: link.snapshot()["estado"] == "Idle" and link.snapshot()["mpos"]["X"] == 0)
    finally:
        link.close()


def test_live_probe_routine_and_homing(engine):
    from hope.cnc.palpado import Prober, ProbeSettings
    # Probe touches when Z goes below −52.5 mm (machine), i.e. −42000 steps; X home at X < 0.
    link = _link(engine, "SIMIN PROBE 2 < -42000", "SIMIN HOME0 0 < 0",
                 cfg={"pines": [*_default_pins(), {"senal": "Sonda", "pin": 13},
                                {"senal": "Referencia X", "pin": 10}],
                      "referencia": {"X": {"velocidad": 600, "lenta": 60, "retiro": 2}}})
    try:
        link.mdi("G53 G0 X50 Y50 Z-40")
        assert _until(lambda: link.snapshot()["estado"] == "Idle")

        def wait():
            time.sleep(0.05)
            _until(lambda: link.snapshot()["estado"] in ("Idle", "Alarm"))
        p = Prober(send=link.mdi, wait=wait, link=link, settings=ProbeSettings(tip_diameter=0))
        r = p.run("superficie", {"distancia": 20})
        assert r["valores"]["Z"] == pytest.approx(-52.5, abs=0.003)
        link.home()                                                   # X has a home switch: homes to X min (0)
        s = link.snapshot()
        assert s["referenciado"] and s["mpos"]["X"] == pytest.approx(2) and s["mpos"]["Y"] == pytest.approx(50)
    finally:
        link.close()


def _default_pins():
    from hope.cnc.paralelo import DEFAULT_PINS
    return list(DEFAULT_PINS)


def test_live_limit_alarm_and_unlock(engine):
    link = _link(engine, "SIMIN LIMIT1 1 > 8000",
                 cfg={"pines": [*_default_pins(), {"senal": "Límite Y", "pin": 11}]})
    try:
        link.start(["G90 G53 G1 Y100 F900"])
        assert _until(lambda: link.snapshot()["estado"] == "Alarm")
        s = link.snapshot()
        assert "fin de carrera del eje Y" in s["alarma"] and s["trabajo"]["error"]
        with pytest.raises(BackendError):
            link.mdi("G0 X1")
        link.unlock()
        assert _until(lambda: link.snapshot()["estado"] == "Idle")
    finally:
        link.close()


def test_choosing_the_parallel_profile_where_it_cannot_run_keeps_the_old_connection():
    if sys.platform.startswith("win"):
        pytest.skip("solo fuera de Windows")
    from hope.cnc.links import SimLink, make_link
    from hope.cnc.session import CncSession, SessionError
    store = ProfileStore(None)
    sel = default_selection(store)
    m, _ = effective(store, sel)
    link = SimLink(m, autostart=False)
    s = CncSession(m, link, profiles=store, selection=sel, link_factory=make_link)
    with pytest.raises(SessionError, match="LinuxCNC"):
        s.choose_profile("maquina", "Router con puerto paralelo (placa tipo Mach3)")
    assert s.link is link and link._run and s.selection == sel and s.m.name == "Router genérico"


def test_live_latency_test(engine):
    link = _link(engine)
    try:
        r = link.latency_test(1)
        assert r["max_us"] == 0 and r["frecuencia_sugerida"] == 25000 and "sirve" in r["texto"]
    finally:
        link.close()
