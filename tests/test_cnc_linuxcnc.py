"""LinuxCNC live connection (linuxcnc_vivo.py) against a fake ``linuxcnc`` module.

The fake keeps the same fields and constants as LinuxCNC's Python API (stat / command /
error_channel) and records every command, so the tests check the translation both ways: what
TISMA asks LinuxCNC to do (modes, MDI, program, jog, home, overrides) and how LinuxCNC's state is
shown (states, positions, work offset, progress, probe, offsets from the .var file).
"""

import pytest

from hope.cnc.backends.base import BackendError
from hope.cnc.links import make_link
from hope.cnc.linuxcnc_vivo import LinuxCncLink
from hope.cnc.profiles import ProfileStore, default_selection, effective


class FakeLinuxCNC:
    """The bits of LinuxCNC's ``linuxcnc`` module that the link uses."""

    RCS_DONE, RCS_EXEC, RCS_ERROR = 1, 2, 3
    STATE_ESTOP, STATE_ESTOP_RESET, STATE_OFF, STATE_ON = 1, 2, 3, 4
    MODE_MANUAL, MODE_AUTO, MODE_MDI = 1, 2, 3
    INTERP_IDLE, INTERP_READING, INTERP_PAUSED, INTERP_WAITING = 1, 2, 3, 4
    AUTO_RUN, AUTO_PAUSE, AUTO_RESUME, AUTO_STEP = 0, 1, 2, 3
    JOG_STOP, JOG_CONTINUOUS, JOG_INCREMENT = 0, 1, 2
    FLOOD_OFF, FLOOD_ON = 0, 1
    NML_ERROR, NML_TEXT, NML_DISPLAY, OPERATOR_ERROR, OPERATOR_TEXT, OPERATOR_DISPLAY = 1, 2, 3, 11, 12, 13

    def __init__(self):
        self.calls = []
        self.errors = []
        mod = self

        class Stat:
            def __init__(s):
                s.estop, s.task_state, s.task_mode = 1, mod.STATE_ESTOP, mod.MODE_MANUAL
                s.state, s.interp_state, s.paused = mod.RCS_DONE, mod.INTERP_IDLE, False
                s.actual_position = [0.0] * 9
                s.g5x_offset, s.g92_offset, s.tool_offset = [0.0] * 9, [0.0] * 9, [0.0] * 9
                s.g5x_index, s.current_vel, s.motion_line = 1, 0.0, 0
                s.spindle = [{"speed": 0.0, "direction": 0}]
                s.flood, s.mist, s.homed, s.joints = 0, 0, [0, 0, 0], 3
                s.probed_position, s.probe_tripped = [0.0] * 9, False
                s.gcodes = (10, 170, 210, 900, 940, 540, -1)
                s.tool_in_spindle, s.ini_filename = 0, ""

            def poll(s):
                pass

        class Command:
            def __getattr__(s, name):
                def call(*args):
                    mod.calls.append((name, *args))
                    st = mod.st
                    if name == "state" and args[0] == mod.STATE_ESTOP_RESET:
                        st.estop, st.task_state = 0, mod.STATE_OFF
                    elif name == "state" and args[0] == mod.STATE_ON and not st.estop:
                        st.task_state = mod.STATE_ON
                    elif name == "mode":
                        st.task_mode = args[0]
                    elif name == "auto" and args[0] in (mod.AUTO_RUN, mod.AUTO_STEP):
                        st.interp_state = mod.INTERP_READING
                    elif name == "auto" and args[0] == mod.AUTO_PAUSE:
                        st.paused = True
                    elif name == "home":
                        st.homed = [1, 1, 1]
                    elif name == "flood":
                        st.flood = args[0]
                return call

        class Errors:
            def poll(s):
                return mod.errors.pop(0) if mod.errors else None

        self.st = Stat()
        self.stat = lambda: self.st
        self.command = Command
        self.error_channel = Errors


def _link(tmp_path=None):
    store = ProfileStore(None)
    m, _ = effective(store, default_selection(store))
    fake = FakeLinuxCNC()
    return LinuxCncLink(m, {"tipo": "linuxcnc"}, module=fake), fake


def test_without_linuxcnc_installed_the_message_says_so():
    store = ProfileStore(None)
    m, _ = effective(store, default_selection(store))
    with pytest.raises(BackendError, match="LinuxCNC instalado"):
        make_link(m, {"tipo": "linuxcnc"})


def test_estop_unlock_and_state_translation():
    link, fake = _link()
    s = link.snapshot()
    assert s["estado"] == "Alarm" and "parada de emergencia" in s["alarma"]
    with pytest.raises(BackendError, match="Desbloquear"):
        link.mdi("G0 X1")
    link.unlock()
    assert link.snapshot()["estado"] == "Idle"
    fake.st.actual_position[:3] = [120.0, 80.0, -30.0]
    fake.st.g5x_offset[:3] = [20.0, 20.0, -40.0]
    fake.st.g92_offset[0] = 1.0
    fake.st.tool_offset[2] = 5.0
    fake.st.current_vel = 25.0
    fake.st.spindle = [{"speed": 12000.0, "direction": 1}]
    s = link.snapshot()
    assert s["mpos"] == {"X": 120.0, "Y": 80.0, "Z": -30.0}
    assert s["wco"] == {"X": 21.0, "Y": 20.0, "Z": -35.0} and s["avance"] == 1500.0
    assert (s["husillo"], s["husillo_dir"]) == (12000.0, "cw") and s["wcs"] == "G54"
    assert link.modal() == "G1 G17 G21 G90 G94 G54"


def test_program_run_hold_resume_and_progress():
    link, fake = _link()
    link.unlock()
    link.start(["G21 G90", "G1 X10 F500", "G1 X20", "M2"])
    assert ("mode", fake.MODE_AUTO) in fake.calls and ("auto", fake.AUTO_RUN, 0) in fake.calls
    path = next(c for c in fake.calls if c[0] == "program_open")[1]
    with open(path) as f:
        assert f.read().splitlines() == ["G21 G90", "G1 X10 F500", "G1 X20", "M2"]
    fake.st.motion_line = 3
    s = link.snapshot()
    assert s["estado"] == "Run" and s["trabajo"]["activo"] and s["trabajo"]["confirmadas"] == 2
    link.hold()
    assert link.snapshot()["estado"] == "Hold"
    fake.st.paused = False
    link.resume()
    assert fake.calls[-1] == ("auto", fake.AUTO_RESUME)
    fake.st.interp_state = fake.INTERP_IDLE                   # LinuxCNC finished the program
    s = link.snapshot()
    assert s["estado"] == "Idle" and s["trabajo"]["terminado"] and s["trabajo"]["confirmadas"] == 4


def test_single_block_steps_and_errors_abort_the_job():
    link, fake = _link()
    link.unlock()
    link.set_single(True)
    link.start(["G1 X1 F100", "G1 X2"])
    assert fake.calls[-1] == ("auto", fake.AUTO_STEP, 0)
    fake.errors.append((fake.NML_ERROR, "joint 0 on limit switch error"))
    s = link.snapshot()
    assert s["trabajo"]["error"] == "joint 0 on limit switch error" and not s["trabajo"]["activo"]
    link.stop()
    assert ("abort",) in fake.calls


def test_mdi_jog_home_overrides_and_coolant():
    link, fake = _link()
    link.unlock()
    link.mdi("G0 X10")
    assert ("mode", fake.MODE_MDI) in fake.calls and ("mdi", "G0 X10") in fake.calls
    link.jog({"X": 5.0}, 600)
    assert ("jog", fake.JOG_INCREMENT, False, 0, 10.0, 5.0) in fake.calls
    with pytest.raises(BackendError, match="recorrido"):
        link.jog({"X": -5.0}, 600)                            # X min is 0
    link.home()
    assert ("home", -1) in fake.calls and link.snapshot()["referenciado"]
    link.override("avance", "+10")
    link.override("rapido", "25")
    link.override("husillo", "-10")
    assert ("feedrate", 1.1) in fake.calls and ("rapidrate", 0.25) in fake.calls and ("spindleoverride", 0.9) in fake.calls
    link.coolant_toggle()
    assert fake.calls[-1] == ("flood", fake.FLOOD_ON) and link.snapshot()["refrigerante"]


def test_probe_result_and_offsets_from_the_var_file(tmp_path):
    link, fake = _link()
    assert link.last_probe() is None or link.last_probe()["n"] == 1
    n0 = (link.last_probe() or {"n": 0})["n"]
    fake.st.probed_position[:3] = [50.0, 60.0, -52.5]
    fake.st.probe_tripped = True
    p = link.last_probe()
    assert p["n"] == n0 + 1 and p["ok"] and p["pos"] == {"X": 50.0, "Y": 60.0, "Z": -52.5}
    assert link.last_probe()["n"] == n0 + 1                   # same result: not new
    (tmp_path / "maquina.var").write_text("5221 20.0\n5222 20.0\n5223 -40.0\n5241 100.0\n5211 1.5\n")
    (tmp_path / "maquina.ini").write_text("[EMC]\nMACHINE = x\n[RS274NGC]\nPARAMETER_FILE = maquina.var\n")
    fake.st.ini_filename = str(tmp_path / "maquina.ini")
    off = link.offsets()
    assert off["G54"] == {"X": 20.0, "Y": 20.0, "Z": -40.0} and off["G55"]["X"] == 100.0 and off["G92"]["X"] == 1.5
