"""App de PC de Hope CNC (hope/cnc/app, session, links): simulador en tiempo real, grbl en vivo
con un controlador falso, flujo de RESET / Cycle Start con el guardián y el servidor local."""

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from guardian.core import Guardian, GuardianConfig
from hope.cnc.links import GrblLink, SimLink
from hope.cnc.machine import Machine
from hope.cnc.session import CncSession, SessionError

PROG = "G0 X10 Y10\nM3 S10000\nG1 Z-1 F600\nG1 X40\nM0 (MSG, mirar la pieza)\nG1 X60\nM5\nM30"


def run_sim(link: SimLink, seconds: float, dt: float = 0.01):
    for _ in range(int(seconds / dt)):
        link.step(dt)


def sim_session(**kw):
    m = Machine()
    link = SimLink(m, autostart=False)
    return CncSession(m, link, **kw), link


@pytest.fixture
def guardian(tmp_path):
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "budget.yaml").write_text("max_total_params: 1000\n")
    return Guardian(GuardianConfig(str(tmp_path), [str(gdir)], str(gdir / "budget.yaml"), str(gdir / "audit.jsonl"),
                                   str(gdir / "approvals.json")))


# --- simulador y sesión --------------------------------------------------------------------
def test_reset_gates_manual_commands():
    s, link = sim_session()
    with pytest.raises(SessionError, match="RESET"):
        s.jog("X", 1, 1.0, False)
    with pytest.raises(SessionError, match="RESET"):
        s.cycle_start()
    assert s.reset()["habilitada"]
    s.jog("X", 1, 5.0, False)
    run_sim(link, 5)
    assert link.pos["X"] == pytest.approx(5.0)
    assert not s.reset()["habilitada"]


def test_program_runs_pauses_and_tracks_progress():
    s, link = sim_session()
    s.reset()
    r = s.load("p.nc", PROG)
    assert r["informe"]["runnable"]
    s.cycle_start()
    run_sim(link, 1.5)
    st = s.state()
    assert st["estado"]["estado"] == "Run" and 1 < st["programa"]["linea_actual"] <= 4
    assert st["programa"]["tiempo_restante"] < st["programa"]["tiempo_total"]
    run_sim(link, 30)
    st = s.state()
    assert st["estado"]["estado"] == "Hold" and st["estado"]["mensaje"] == "mirar la pieza"
    assert st["programa"]["linea_actual"] == 5 and link.pos["X"] == pytest.approx(40)
    s.cycle_start()                                     # Cycle Start reanuda, como en Mach3
    run_sim(link, 30)
    st = s.state()
    assert st["estado"]["estado"] == "Idle" and st["estado"]["trabajo"]["terminado"]
    assert link.pos["X"] == pytest.approx(60) and st["estado"]["husillo_dir"] == "off"
    s.rewind()
    assert s.state()["programa"]["linea_actual"] == 0


def test_feed_override_speeds_up_the_simulation():
    def duration(ov_clicks):
        s, link = sim_session()
        s.reset()
        s.load("p.nc", "G1 X200 F600\nM30")
        for _ in range(ov_clicks):
            s.override("avance", "+10")
        s.cycle_start()
        t = 0.0
        while s.state()["estado"]["estado"] != "Idle":
            link.step(0.05)
            t += 0.05
        return t
    assert duration(10) == pytest.approx(duration(0) / 2, rel=0.05)    # 200 % → la mitad del tiempo


def test_hold_stop_and_errors_block_start():
    s, link = sim_session()
    s.reset()
    s.load("p.nc", "G1 X500 F600")                       # fuera del recorrido (X máx 300)
    with pytest.raises(SessionError, match="errores"):
        s.cycle_start()
    s.load("p.nc", "G1 X200 F600\nM30")
    s.cycle_start()
    run_sim(link, 1)
    s.feed_hold()
    x = link.pos["X"]
    run_sim(link, 1)
    assert link.pos["X"] == x and s.state()["estado"]["estado"] == "Hold"
    s.stop()
    st = s.state()["estado"]
    assert st["estado"] == "Idle" and st["trabajo"]["error"] == "detenido por el operador"


def test_zero_mdi_goto_and_home():
    s, link = sim_session()
    s.reset()
    s.mdi("G0 X50 Y40 Z-10")
    run_sim(link, 10)
    s.zero("todo")
    assert s.state()["trabajo_pos"] == {"X": 0, "Y": 0, "Z": 0}
    assert link.offsets()["G54"] == {"X": 50, "Y": 40, "Z": -10}
    s.mdi("G0 X10")
    run_sim(link, 5)
    with pytest.raises(SessionError, match="recorrido"):
        s.mdi("G0 X400")
    s.home()
    time.sleep(0.05)
    run_sim(link, 2)
    st = s.state()["estado"]
    assert st["referenciado"] and st["mpos"] == {"X": 0, "Y": 0, "Z": 0}


def test_continuous_jog_stops_at_limit_and_on_release():
    s, link = sim_session()
    s.reset()
    s.jog("Y", 1, None, True)                            # continuo y rápido hacia +Y
    run_sim(link, 0.5)
    s.jog_stop()
    y = link.pos["Y"]
    assert 0 < y < 200 and s.state()["estado"]["estado"] == "Idle"
    s.jog("Y", 1, None, True)
    run_sim(link, 30)
    assert link.pos["Y"] == pytest.approx(200)            # se frena en el límite
    with pytest.raises(SessionError, match="límite"):
        s.jog("Y", 1, None, False)


def test_toolpath_points_in_work_coordinates():
    m = Machine(offsets={"G54": {"X": 10, "Y": 10, "Z": -5}})
    s = CncSession(m, SimLink(m, autostart=False))
    s.load("a.nc", "G0 X0 Y0 Z1\nM3 S9000\nG1 Z-1 F100\nG2 X10 Y0 I5 J0\nM30")
    pts = s.toolpath()["puntos"]
    assert pts[0][3] == -1 and s.toolpath()["perforaciones"] == []      # en router no hay perforaciones
    assert [round(v, 3) for v in pts[1][:3]] == [0, 0, 1] and pts[1][3] == 0
    assert pts[-1][:3] == [10, 0, -1] and pts[-1][3] == 1 and pts[-1][4] == 4
    assert s.toolpath()["limites"]["X"] == [-10, 290]


# --- grbl en vivo con un controlador falso ----------------------------------------------------------
class LiveFakeGrbl:
    """grbl simulado para hilos: ok por línea, estado con «?», tiempo real de un byte, $#, $H, $J."""

    def __init__(self, rx=128):
        self.rx = rx
        self.lock = threading.Lock()
        self.out: list[bytes] = [b"Grbl 1.1h ['$' for help]\r\n"]
        self.queue: list[str] = []
        self.received: list[str] = []
        self.realtime: list[int] = []
        self.state = "Idle"
        self.max_used = 0

    def write(self, data: bytes):
        with self.lock:
            if len(data) == 1 and (data in b"?!~\x18" or data[0] >= 0x80):
                self.realtime.append(data[0])
                if data == b"?":
                    self.out.append(f"<{self.state}|MPos:5.000,6.000,-7.000|FS:300,9000|Ov:110,100,90|A:SF|"
                                    f"WCO:1.000,1.000,-2.000>\r\n".encode())
                elif data == b"!":
                    self.state = "Hold"
                elif data == b"~":
                    self.state = "Idle"
                return
            for line in data.decode("ascii").split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line == "$#":
                    self.out += [b"[G54:1.000,1.000,-2.000]\r\n", b"[G92:0.000,0.000,0.000]\r\n", b"ok\r\n"]
                    continue
                self.queue.append(line)
                self.max_used = max(self.max_used, sum(len(q) + 1 for q in self.queue))
                assert self.max_used <= self.rx

    def readline(self) -> bytes:
        with self.lock:
            if self.out:
                return self.out.pop(0)
            if self.queue:
                line = self.queue.pop(0)
                self.received.append(line)
                if line == "M0":
                    self.state = "Hold"
                return b"error:15\r\n" if line.startswith("$J=") and "X999" in line else b"ok\r\n"
        time.sleep(0.001)
        return b""

    def close(self):
        pass


def wait(cond, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_grbl_link_streams_and_reports_status():
    port = LiveFakeGrbl(rx=64)
    m = Machine()
    link = GrblLink(port, m, rx_buffer=64, poll=0.02)
    try:
        assert wait(lambda: link.snapshot()["estado"] == "Idle")
        snap = link.snapshot()
        assert snap["mpos"] == {"X": 5, "Y": 6, "Z": -7} and snap["wco"] == {"X": 1, "Y": 1, "Z": -2}
        assert snap["ov"] == {"avance": 110, "rapido": 100, "husillo": 90}
        assert snap["husillo_dir"] == "cw" and snap["refrigerante"]
        assert link.offsets()["G54"] == {"X": 1, "Y": 1, "Z": -2}
        lines = [f"G1 X{i}.5 Y{i}.25 F900" for i in range(300)]
        link.start(lines)
        assert wait(lambda: link.snapshot()["trabajo"]["terminado"])
        assert port.received[-300:] == [s.replace(" ", "") for s in lines]
        assert 40 < port.max_used <= 64
        link.override("avance", "+10")
        link.jog({"X": 1.5}, 600)
        assert port.received[-1] == "$J=G91 G21 X1.5000 F600"
        link.jog_cancel()
        assert 0x91 in port.realtime and 0x85 in port.realtime
        from hope.cnc.backends.base import BackendError
        with pytest.raises(BackendError, match="recorrido"):
            link.jog({"X": 999}, 600)
    finally:
        link.close()


def test_real_machine_session_needs_operator_approval(guardian):
    port = LiveFakeGrbl()
    m = Machine()
    link = GrblLink(port, m, poll=0.02)
    try:
        assert wait(lambda: link.snapshot()["estado"] == "Idle")
        with pytest.raises(SessionError, match="guardián"):
            CncSession(m, link).reset()
        s = CncSession(m, link, guardian)
        r = s.reset()
        assert "aprobar" in r and not s.enabled and r["aprobar"]["tipo"] == "manual"
        assert s.approve(r["aprobar"]["id"])["habilitada"] and s.enabled
        s.load("p.nc", PROG)
        r = s.cycle_start()
        assert r["aprobar"]["tipo"] == "programa" and not any(x.startswith("G1") for x in port.received)
        s.approve(r["aprobar"]["id"])
        assert wait(lambda: "M0" in port.received)
        assert port.received[1] == "G0X10Y10"            # en G54: el G54 real (1, 1, -2) lo aplica grbl
        assert wait(lambda: link.snapshot()["estado"] == "Hold")
        s.cycle_start()                                  # reanuda tras M0
        assert wait(lambda: link.snapshot()["trabajo"]["terminado"])
        assert 0x7E in port.realtime                     # «~»
        # otra corrida del mismo programa pide otra aprobación; cancelar la deniega
        assert wait(lambda: link.snapshot()["estado"] == "Idle")
        r2 = s.cycle_start()
        assert "aprobar" in r2
        s.cancel()
        rid = r2["aprobar"]["id"]
        assert s.pending is None and rid not in guardian.pending()
        assert json.loads(Path(guardian.config.approvals_path).read_text())[rid]["status"] == "DENIED"
    finally:
        link.close()


# --- servidor -------------------------------------------------------------------------------
@pytest.fixture
def server():
    from hope.cnc.app.server import serve
    m = Machine()
    link = SimLink(m, speed=50)
    s = CncSession(m, link)
    srv, _, _ = serve(s, port=0)
    port = srv.server_address[1]
    yield s, port
    srv.shutdown()
    link.close()


def req(port, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    return r.status, r.read()


def action(port, name, params=None):
    st, raw = req(port, "POST", "/api/accion", {"accion": name, "params": params or {}},
                  {"Content-Type": "application/json", "X-Hope": "1"})
    return st, json.loads(raw)


def test_server_pages_and_security(server):
    s, port = server
    st, page = req(port, "GET", "/")
    assert st == 200 and b"Cycle Start" in page and b"pathview.js" in page
    assert req(port, "GET", "/cnc.js")[0] == 200
    assert req(port, "GET", "/../server.py")[0] == 404
    assert req(port, "GET", "/api/estado", host="evil.example:80")[0] == 403
    st, _ = req(port, "POST", "/api/accion", {"accion": "reset"}, {"Content-Type": "application/json"})
    assert st == 403 and not s.enabled                    # sin X-Hope: no
    st, _ = req(port, "POST", "/api/accion", {"accion": "reset"}, {"Content-Type": "text/plain", "X-Hope": "1"})
    assert st == 403
    assert action(port, "borrar_todo")[0] == 400


def test_server_full_cycle(server):
    s, port = server
    assert action(port, "reset")[1]["habilitada"]
    st, r = action(port, "cargar", {"nombre": "p.nc", "texto": "G0 X10 Y10\nM3 S9000\nG1 Z-1 F3000\nG1 X50\nM5\nM30"})
    assert st == 200 and r["informe"]["runnable"]
    _, prog = req(port, "GET", "/api/programa")
    assert json.loads(prog)["lineas"][0] == "G0 X10 Y10"
    _, path = req(port, "GET", "/api/trayectoria")
    assert len(json.loads(path)["puntos"]) >= 4
    assert action(port, "ciclo")[1]["en_marcha"]
    assert wait(lambda: s.state()["estado"]["trabajo"]["terminado"], 10)
    est = json.loads(req(port, "GET", "/api/estado")[1])
    assert est["trabajo_pos"]["X"] == 50 and est["programa"]["linea_actual"] == 6
    st, r = action(port, "mdi", {"linea": "G0 X9999"})
    assert st == 400 and "recorrido" in r["mensaje"]
    cfg = json.loads(req(port, "GET", "/api/config")[1])
    assert set(cfg["ejes"]) == {"X", "Y", "Z"}
    assert json.loads(req(port, "GET", "/api/origenes")[1])["ok"]


# --- botones de Mach3: Single BLK, Block Delete, M1, Set Next Line, Run From Here, recientes ------------
MACH3 = "G0 X10 Y10\nM3 S9000\nG1 Z-1 F600\nG1 X40\nG1 Y40\n/G1 X99\nM1\nG1 X10\nM5\nM30"


def emitted(s):
    return [t for t, _ in s.program["emitidas"]]


def test_block_delete_and_optional_stop_toggles():
    s, _ = sim_session()
    s.load("p.nc", MACH3)
    assert "G1 X99" in emitted(s) and not any(t.startswith("M0") for t in emitted(s))   # como Mach3: apagados
    s.toggle("borrado_bloque")
    s.toggle("parada_opcional")
    assert "G1 X99" not in emitted(s) and any(t.startswith("M0") for t in emitted(s))
    assert s.state()["opciones"] == {"bloque_a_bloque": False, "borrado_bloque": True, "parada_opcional": True}
    with pytest.raises(SessionError):
        s.toggle("no_existe")


def test_single_block_runs_one_line_per_cycle_start():
    s, link = sim_session()
    s.reset()
    s.load("p.nc", MACH3)
    s.toggle("bloque_a_bloque")
    s.cycle_start()
    seen = []
    for _ in range(12):
        run_sim(link, 20)
        st = s.state()
        if st["estado"]["trabajo"]["terminado"]:
            break
        assert st["estado"]["estado"] == "Hold"
        seen.append(st["programa"]["linea_actual"])
        s.cycle_start()
    assert seen[:5] == [1, 2, 3, 4, 5] and s.state()["estado"]["trabajo"]["terminado"]


def test_set_next_line_and_run_from_here():
    s, link = sim_session()
    s.reset()
    s.load("p.nc", MACH3)
    s.set_next_line(5)
    lines, mapping = s._lines_to_run()
    assert lines[1] == "G1 Y40" and len(lines) == len(mapping)          # sin preparación
    s.set_next_line(5, from_here=True)
    lines, _ = s._lines_to_run()
    assert lines[1:6] == ["G53 G0 Z0", "G0 X40.0000 Y10.0000", "M3 S9000", "G1 Z-1.0000 F600", "F600"]
    assert lines[6] == "G1 Y40"
    s.cycle_start()
    run_sim(link, 60)
    st = s.state()
    assert st["estado"]["trabajo"]["terminado"] and link.pos["X"] == pytest.approx(10)
    assert st["linea_siguiente"] == 0                                     # se vuelve a empezar desde arriba
    with pytest.raises(SessionError, match="fuera"):
        s.set_next_line(99)


def test_run_from_here_in_plasma_goes_back_to_the_pierce():
    from hope.cnc.machine import load_machine
    m = load_machine("configs/cnc_plasma.yaml")
    s = CncSession(m, SimLink(m, autostart=False))
    s.load("b.nc", "G0 X50 Y50 Z10\nM3\nG1 X100 F3000\nG1 Y100\nM5\nM30")
    s.set_next_line(4, from_here=True)                     # a mitad del corte: hay que volver a perforar
    lines, _ = s._lines_to_run()
    assert any(t.startswith("G38.2") for t in lines) and "M3" in lines
    assert lines.index("M3") < lines.index("G1 Y100")


def test_recent_files_persist(tmp_path):
    m = Machine()
    s = CncSession(m, SimLink(m, autostart=False), recent_dir=str(tmp_path / "rec"))
    s.load("a.nc", "G0 X1\nM30")
    s.load("b.nc", "G0 X2\nM30")
    s.load("a.nc", "G0 X1\nM30")
    names = [r["nombre"] for r in s.recent_files()["recientes"]]
    assert names == ["a.nc", "b.nc"]
    s2 = CncSession(m, SimLink(m, autostart=False), recent_dir=str(tmp_path / "rec"))
    sha_b = s2.recent_files()["recientes"][1]["sha"]
    assert s2.load_recent(sha_b)["informe"]["runnable"] and s2.program["nombre"] == "b.nc"
    assert "G21" in s2.state()["modal"] and "G90" in s2.state()["modal"]


def test_grbl_single_block_sends_one_line_per_resume():
    port = LiveFakeGrbl()
    link = GrblLink(port, Machine(), poll=0.02)
    try:
        assert wait(lambda: link.snapshot()["estado"] == "Idle")
        link.set_single(True)
        link.start(["G1 X1 F100", "G1 X2", "G1 X3"])
        assert wait(lambda: link.snapshot()["trabajo"]["confirmadas"] == 1)
        time.sleep(0.1)
        assert port.received[-1] == "G1X1F100" and link.snapshot()["trabajo"]["enviadas"] == 1
        link.resume()
        assert wait(lambda: link.snapshot()["trabajo"]["confirmadas"] == 2)
        link.set_single(False)
        link.resume()
        assert wait(lambda: link.snapshot()["trabajo"]["terminado"])
    finally:
        link.close()


def test_server_mach3_actions(server):
    _, port = server
    action(port, "reset")
    action(port, "cargar", {"nombre": "p.nc", "texto": MACH3})
    assert action(port, "opcion", {"nombre": "bloque_a_bloque"})[1]["opciones"]["bloque_a_bloque"]
    assert action(port, "linea_siguiente", {"linea": 4, "desde_aqui": True})[1]["desde_aqui"]
    rec = json.loads(req(port, "GET", "/api/recientes")[1])["recientes"]
    assert rec[0]["nombre"] == "p.nc"
    assert action(port, "abrir_reciente", {"sha": rec[0]["sha"]})[1]["ok"]
    assert req(port, "GET", "/logo.svg")[0] == 200
