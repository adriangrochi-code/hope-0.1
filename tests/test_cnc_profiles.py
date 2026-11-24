"""Slicer-style profiles of TISMA CNC (hope/cnc/profiles.py): inheritance, saving, cutting feed
calculation (plasma table, chip load), applying them to the G-code and use from the session and server."""

import json
from pathlib import Path

import pytest
import yaml

from hope.cnc.dialect import emit_text
from hope.cnc.gcode import interpret
from hope.cnc.links import SimLink
from hope.cnc.profiles import (
    ProfileError,
    ProfileStore,
    default_selection,
    effective,
    fix_selection,
    machine_profile_from_config,
)
from hope.cnc.session import CncSession


@pytest.fixture
def store(tmp_path):
    return ProfileStore(user_dir=tmp_path / "perfiles")


def feeds(prog):
    return [round(o.feed) for o in prog.ops if o.kind in ("feed", "arc")]


def test_system_profiles_load_and_defaults(store):
    for kind in ("maquina", "material", "consumible", "proceso"):
        assert store.names(kind)
    assert default_selection(store) == {"maquina": "Router genérico", "material": "MDF",
                                        "consumible": 'Fresa plana 1/8" 2 filos',
                                        "proceso": "Fresado: avance del material"}
    plasma = default_selection(store, "Mesa de plasma 1500×3000")
    assert plasma["material"] == "Acero al carbono (plasma)" and plasma["consumible"] == "Antorcha 45 A"
    assert plasma["espesor"] == 3.0
    assert all("plasma" in p["tipos"] for p in store.list("material", "plasma"))   # filters by machine
    for f in Path("hope/cnc/perfiles").rglob("*.json"):
        assert max(map(ord, f.name)) < 128                                                   # USB sticks, old Windows


def test_inheritance_and_marks(store):
    child = store.resolve("proceso", "Plasma: calidad (85 %)")
    assert child["avance_pct"] == 85 and child["perforacion_auto"] is True and "predeterminado" not in child
    assert store.own_keys("proceso", "Plasma: calidad (85 %)") == ["avance_pct"]
    m = store.resolve("maquina", "Router chico 300×180")
    assert m["ejes"]["X"]["max"] == 300 and m["controlador"]["tipo"] == "sim"   # the rest comes from the parent


def test_user_profiles_save_diff_delete_import(store):
    with pytest.raises(ProfileError, match="sistema"):
        store.save("proceso", "Plasma: tabla de corte", {"avance_pct": 90})
    data = store.resolve("proceso", "Plasma: tabla de corte")
    data["avance_pct"] = 92
    out = store.save("proceso", "Mi plasma", data, inherits="Plasma: tabla de corte")
    assert out == {"nombre": "Mi plasma", "hereda": "Plasma: tabla de corte", "avance_pct": 92}   # only what changes
    saved = json.loads(next((store.user_dir / "proceso").glob("*.json")).read_text(encoding="utf-8"))
    assert saved == out
    store.save("proceso", "Mi plasma lento", {"avance_pct": 50}, inherits="Mi plasma")
    with pytest.raises(ProfileError, match="padre"):
        store.delete("proceso", "Mi plasma")
    store.delete("proceso", "Mi plasma lento")
    store.delete("proceso", "Mi plasma")
    with pytest.raises(ProfileError, match="sistema"):
        store.delete("proceso", "Plasma: tabla de corte")
    imp = store.import_text("material", json.dumps({"nombre": "Chapa galvanizada", "hereda": "Acero al carbono (plasma)",
                                                    "nota": "probar"}))
    assert imp["hereda"] == "Acero al carbono (plasma)"
    with pytest.raises(ProfileError):
        store.import_text("material", "no es json")
    bad = store.resolve("material", "Acero al carbono (plasma)")
    bad["plasma"]["tabla"][0]["altura_corte"] = 9
    with pytest.raises(ProfileError, match="altura de corte"):
        store.save("material", "Malo", bad)
    with pytest.raises(ValueError):
        store.save("maquina", "Rota", {"tipo": "impresora", "ejes": {}})


def test_plasma_cut_chart_drives_everything(store):
    sel = default_selection(store, "Mesa de plasma 1500×3000")
    m, s = effective(store, sel)
    assert m.job_feed == 3800 and m.job_feed_mode == "reemplazar"
    assert (m.plasma.pierce_height, m.plasma.cut_height, m.plasma.pierce_delay) == (3.8, 1.5, 0.3)
    assert m.plasma.auto_pierce and m.plasma.probe and s["kerf"] == 1.5 and not s["avisos"]
    m85, _ = effective(store, {**sel, "proceso": "Plasma: calidad (85 %)"})
    assert m85.job_feed == 3230
    thick, _ = effective(store, {**sel, "espesor": 10})
    assert thick.job_feed == 700 and thick.plasma.pierce_height == 4.5 and thick.plasma.pierce_delay == 0.8
    _, near = effective(store, {**sel, "espesor": 7})
    assert any("se usa 6 mm" in w for w in near["avisos"])
    cam, _ = effective(store, {**sel, "proceso": "Plasma: el CAM trae todo"})
    assert cam.job_feed is None and not cam.plasma.auto_pierce
    small, sw = effective(store, {**sel, "maquina": "Mesa de plasma 1250×1250"})
    assert not small.plasma.probe and any("no lo tiene" in w for w in sw["avisos"])
    # in the G-code: the pierce uses the table and the cut goes at the table's speed
    p = interpret("G0 X10 Y10 Z10\nM3\nG1 X100 F1000\nG2 X100 Y10 I-45 J0\nM5\nM30", m)
    assert p.ok, p.diags
    assert feeds(p)[-2:] == [3800, 3800]
    dwell = next(o for o in p.ops if o.kind == "dwell")
    assert dwell.value == 0.3
    assert "F3800" in emit_text(p, m)


def test_router_chipload_feed_and_rpm(store):
    sel = default_selection(store)
    m, _ = effective(store, sel)
    # MDF Vc 400 m/min with Ø3.175 → 40107 rpm, capped at the router's 24000
    assert m.job_feed == pytest.approx(0.06 * 2 * 24000) and m.job_rpm == 24000
    m6, w6 = effective(store, {**sel, "consumible": "Fresa plana 6 mm 2 filos"})
    # 0.12·2·21221 = 5093 > 5000 mm/min: the RPM drop to 5000 / (0.12·2) = 20833 to keep the chip load
    assert m6.job_feed == pytest.approx(5000) and m6.job_rpm == 20833 and w6["avisos"]
    small, ws = effective(store, {**sel, "maquina": "Router chico 300×180"})
    assert small.job_feed == pytest.approx(0.06 * 2 * 10000) and small.job_rpm == 10000 and not ws["avisos"]
    capped, wc = effective(store, {**sel, "maquina": "Router chico 300×180", "consumible": "Fresa plana 6 mm 2 filos"})
    # 0.12·2·10000 = 2400 > 1500: instead of rubbing at 10000 rpm, the RPM drop to keep the chip load
    assert capped.job_feed == 1500 and capped.job_rpm == 6250 and any("no roce" in w for w in wc["avisos"])
    limit, _ = effective(store, {**sel, "proceso": "Fresado: limitar al material"})
    prog_mode, _ = effective(store, {**sel, "proceso": "Fresado: avance del programa (Mach3)"})
    assert limit.job_feed_mode == "limitar" and prog_mode.job_feed is None
    text = "G0 X10 Y10 Z5\nM3\nG1 Z-1 F300\nG1 X50 F5000\nG1 X60 F500\nM5\nM30"
    p = interpret(text, m)
    assert feeds(p) == [1152, 2880, 2880]                  # plunge: 40 % of the cut (MDF)
    assert "M3 S24000" in emit_text(p, m)                  # added the missing S
    assert feeds(interpret(text, limit)) == [300, 2880, 500]       # limit: 300 < 1152 is respected
    assert feeds(interpret(text, prog_mode)) == [300, 5000, 500]


def test_fix_selection_repairs_bad_choices(store):
    sel = fix_selection(store, {"maquina": "Mesa de plasma 1500×3000", "material": "MDF", "consumible": "nada"})
    assert sel["material"] == "Acero al carbono (plasma)" and sel["consumible"] == "Antorcha 45 A"
    assert fix_selection(store, {"maquina": "no existe"})["maquina"] == "Router genérico"


def test_old_yaml_configs_become_profiles(store):
    cfg = yaml.safe_load(Path("configs/cnc_plasma.yaml").read_text(encoding="utf-8"))
    prof = machine_profile_from_config(cfg)
    store.save("maquina", "Mi plasma YAML", prof)
    m, _ = effective(store, {"maquina": "Mi plasma YAML", "material": "Acero al carbono (plasma)",
                             "consumible": "Antorcha 45 A", "proceso": "Plasma: tabla de corte", "espesor": 6})
    assert m.is_plasma and m.plasma.probe and m.job_feed == 1600 and m.axes["Y"].max == 3000


def test_session_switches_profiles(store, tmp_path):
    sel = default_selection(store)
    m, _ = effective(store, sel)
    s = CncSession(m, SimLink(m, autostart=False), profiles=store, selection=sel,
                   link_factory=lambda mm: SimLink(mm, autostart=False))
    s.load("p.nc", "G0 X10 Y10\nM3\nG1 Z-1 F300\nG1 X50 F900\nM5\nM30")
    assert "MDF" in s.program["informe"]["perfil"]
    assert "F2880" in "\n".join(t for t, _ in s.program["emitidas"])
    r = s.choose_profile("maquina", "Mesa de plasma 1500×3000")
    assert r["seleccion"]["material"] == "Acero al carbono (plasma)" and s.m.is_plasma
    assert any(t.startswith("G38.2") for t, _ in s.program["emitidas"])          # the program was checked again
    s.choose_profile(thickness=6)
    assert s.m.job_feed == 1600 and s.state()["perfiles"]["resumen"]["espesor"] == 6
    assert json.loads((store.user_dir / "seleccion.json").read_text(encoding="utf-8"))["espesor"] == 6
    data = store.resolve("proceso", "Plasma: tabla de corte")
    data["avance_pct"] = 50
    s.profile_save("proceso", "Mitad", data, "Plasma: tabla de corte")
    s.choose_profile("proceso", "Mitad")
    assert s.m.job_feed == 800
    detail = s.profile_detail("proceso", "Mitad")
    assert detail["propios"] == ["avance_pct"] and detail["padre"]["avance_pct"] == 100
    s.profile_delete("proceso", "Mitad")
    assert s.selection["proceso"] == "Plasma: tabla de corte"


def test_server_profile_api(tmp_path):
    import http.client

    from hope.cnc.app.server import serve
    st = ProfileStore(user_dir=tmp_path / "p")
    sel = default_selection(st)
    m, _ = effective(st, sel)
    link = SimLink(m, speed=20)
    s = CncSession(m, link, profiles=st, selection=sel, link_factory=lambda mm: SimLink(mm, speed=20))
    srv, _, _ = serve(s, port=0)
    port = srv.server_address[1]

    def call(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        h = {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", "X-Hope": "1"}
        c.request(method, path, body=json.dumps(body).encode() if body else None, headers=h)
        r = c.getresponse()
        return r.status, json.loads(r.read())
    try:
        st_, lst = call("GET", "/api/perfiles")
        assert st_ == 200 and lst["activos"] and lst["seleccion"]["maquina"] == "Router genérico"
        st_, det = call("GET", "/api/perfil?tipo=material&nombre=MDF")
        assert st_ == 200 and det["datos"]["fresado"]["rpm"] == 18000 and det["esquema"]
        assert call("GET", "/api/perfil?tipo=material&nombre=nada")[0] == 404
        st_, r = call("POST", "/api/accion", {"accion": "perfil_elegir", "params": {"tipo": "material", "nombre": "Madera dura"}})
        assert st_ == 200 and r["resumen"]["carga_viruta"] == 0.04
        st_, r = call("POST", "/api/accion", {"accion": "perfil_guardar", "params": {
            "tipo": "material", "nombre": "Pino", "hereda": "Madera dura", "datos": {"fresado": {"rpm": 15000}}}})
        assert st_ == 200 and r["perfil"] == {"nombre": "Pino", "hereda": "Madera dura", "fresado": {"rpm": 15000}}
    finally:
        srv.shutdown()
        link.close()


# --- feeds and speeds calculator ----------------------------------------------------------------------

def test_rpm_from_cutting_speed_like_the_calculators():
    """RPM = Vc × 1000 / (π × D): the metric «SMM × 318 / D» and the imperial «SFM × 3.82 / D(in)»
    are the same formula with the constants worked out."""
    import math

    from hope.cnc.profiles import speeds_rpm
    assert speeds_rpm(300, 6, None, 0, 60000) == pytest.approx(300 * 1000 / (math.pi * 6))
    assert speeds_rpm(300, 6, None, 0, 60000) == pytest.approx(300 * 318.31 / 6, rel=1e-4)
    sfm, d_in = 1000, 0.25                                          # 1000 SFM, 1/4" end mill
    assert speeds_rpm(sfm * 0.3048, d_in * 25.4, None, 0, 60000) == pytest.approx(sfm * 3.82 / d_in, rel=2e-3)
    assert speeds_rpm(400, 3.175, None, 0, 24000) == 24000          # capped at the spindle maximum
    assert speeds_rpm(None, 6, 18000, 0, 24000) == 18000            # no Vc: the material's fixed RPM
    assert speeds_rpm(50, 6, None, 8000, 24000) == 8000             # never below the spindle minimum
    assert speeds_rpm(None, 6, None, 0, 24000) is None


def test_feed_chip_thinning_and_anti_rubbing():
    import math

    from hope.cnc.profiles import CHIP_THINNING_MAX, feeds
    r = feeds(18000, 0.05, 2, 6, 1e9, 0)
    assert r["avance"] == pytest.approx(18000 * 2 * 0.05) and not r["avisos"]       # RPM × flutes × chip load
    assert feeds(18000, 0.05, 2, 6, 1e9, 0, 50)["avance"] == pytest.approx(1800)   # ae = D/2: no correction
    r15 = feeds(18000, 0.05, 2, 6, 1e9, 0, 15)
    ae = 0.9
    assert r15["carga_efectiva"] == pytest.approx(0.05 * 6 / (2 * math.sqrt(6 * ae - ae * ae)))
    assert feeds(18000, 0.05, 2, 6, 1e9, 0, 1)["carga_efectiva"] == pytest.approx(0.05 * CHIP_THINNING_MAX)
    slow = feeds(24000, 0.1, 2, 6, 3000, 5000)       # 4800 mm/min wanted, machine gives 3000
    assert slow["avance"] == 3000 and slow["rpm"] == pytest.approx(15000) and "no roce" in slow["avisos"][0]
    stuck = feeds(24000, 0.1, 2, 6, 1000, 8000)      # would need 5000 rpm, spindle minimum is 8000
    assert stuck["avance"] == 1000 and stuck["rpm"] == 24000 and "rozar" in stuck["avisos"][0]


def test_stock_profiles_recommend_passes(store):
    sel = default_selection(store)
    _, s = effective(store, sel)
    assert s["paso_lateral"] == pytest.approx(3.175 * 0.4, abs=0.01) and s["profundidad"] == pytest.approx(3.175, abs=0.01)
    assert s["remocion"] > 0 and "pasada lat." in s["texto"]
    ad, _ = effective(store, {**sel, "proceso": "Fresado: desbaste adaptativo (pasada 15 %)"})
    base, _ = effective(store, sel)
    assert ad.job_feed > base.job_feed                                # chip thinning raises the feed
