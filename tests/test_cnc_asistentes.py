"""Machining wizards (asistentes.py): every program is valid and its geometry is the requested one.

The checks run the generated G-code through the real interpreter and measure where the tool centre
goes: pocket walls at size − tool radius, cut-outs exactly one tool radius outside the line, tabs
only on the last passes, holes where they belong.
"""

import math

import pytest

from hope.cnc.asistentes import WIZARDS, Tooling, WizardError, generate
from hope.cnc.check import check_program
from hope.cnc.gcode import interpret, linearize
from hope.cnc.machine import load_machine

T = Tooling(diameter=6, feed=1200, plunge=400, rpm=18000, stepover_pct=40, stepdown=2.5)


def _run(kind, params):
    m = load_machine("configs/cnc_maquina.yaml")
    p = interpret(generate(kind, params, T), m)
    assert p.ok, [str(d) for d in p.diags]
    assert check_program(p, m)["runnable"]
    g54 = m.offset("G54")
    pts = []
    for o in p.ops:
        if o.kind in ("feed", "arc"):
            seq = linearize(o, 0.005) if o.kind == "arc" else [o.end]
            pts += [(q["X"] - g54["X"], q["Y"] - g54["Y"], q["Z"] - g54["Z"]) for q in seq]
    return p, pts


@pytest.mark.parametrize("kind", sorted(WIZARDS))
def test_every_wizard_makes_a_valid_program_with_defaults(kind):
    _run(kind, {"x": 50, "y": 40})


def test_rect_pocket_walls_at_size():
    _, pts = _run("cajera_rect", {"x": 50, "y": 40, "ancho": 40, "alto": 20, "profundidad": 6})
    xs = [x for x, _, _ in pts]
    ys = [y for _, y, _ in pts]
    assert min(xs) == pytest.approx(50 - 20 + 3) and max(xs) == pytest.approx(50 + 20 - 3)    # wall − radius
    assert min(ys) == pytest.approx(40 - 10 + 3) and max(ys) == pytest.approx(40 + 10 - 3)
    assert min(z for _, _, z in pts) == pytest.approx(-6)


def test_circular_pocket_and_helical_entry():
    p, pts = _run("cajera_circ", {"x": 0, "y": 0, "diametro": 30, "profundidad": 5})
    assert max(math.hypot(x, y) for x, y, _ in pts) == pytest.approx(12, abs=1e-3)          # 15 − 3
    helix = [o for o in p.ops if o.kind == "arc" and abs(o.end["Z"] - o.start["Z"]) > 1e-9]
    assert len(helix) == 2                                                                   # one per depth


def test_cut_out_rectangle_runs_one_radius_outside_with_tabs():
    _, pts = _run("recorte_rect", {"x": 10, "y": 10, "ancho": 80, "alto": 50, "profundidad": 6,
                                   "puentes": 1, "alto_puente": 1.5})

    def dist(x, y):                                # distance from the rectangle [10,90]×[10,60]
        dx = max(10 - x, 0, x - 90)
        dy = max(10 - y, 0, y - 60)
        return math.hypot(dx, dy)
    assert all(dist(x, y) == pytest.approx(3, abs=1e-3) for x, y, _ in pts)                  # always 3 mm out
    bottom = [z for _, _, z in pts]
    assert min(bottom) == pytest.approx(-6)
    tab_top = [z for _, _, z in pts if abs(z - (-6 + 1.5)) < 1e-6]
    assert tab_top                                                                           # the tabs exist
    # the shallow pass (−2) is not affected by the tabs (they are 4.5 mm below the top)
    levels = sorted({round(z, 3) for _, _, z in pts})
    assert levels == [-6.0, -4.5, -4.0, -2.0]


def test_cut_out_disc_with_tabs():
    _, pts = _run("recorte_circ", {"x": 60, "y": 60, "diametro": 60, "profundidad": 6, "puentes": 4,
                                   "alto_puente": 1.5})
    assert all(math.hypot(x - 60, y - 60) == pytest.approx(33, abs=1e-3) for x, y, _ in pts)   # 30 + 3
    on_tab = [(x, y) for x, y, z in pts if abs(z + 4.5) < 1e-6]
    angles = sorted({round(math.degrees(math.atan2(y - 60, x - 60)) % 360) for x, y in on_tab})
    assert len(angles) >= 4                                                                  # 4 tabs around


def test_drilling_patterns():
    p, _ = _run("agujeros_circulo", {"x": 60, "y": 60, "diametro": 60, "cantidad": 6, "profundidad": 8})
    holes = {(round(o.end["X"], 3), round(o.end["Y"], 3)) for o in p.ops if o.kind == "feed" and o.source}
    assert len(holes) == 6
    p, _ = _run("agujeros_grilla", {"x": 10, "y": 10, "columnas": 3, "filas": 2, "paso_x": 20, "paso_y": 15,
                                    "profundidad": 8, "picado": 3})
    pecks = [o for o in p.ops if o.kind == "feed" and o.source == "G83"]
    assert len(pecks) >= 6 * 3                                                               # 8 mm / 3 per peck


def test_facing_covers_the_edges():
    _, pts = _run("planeado", {"x": 0, "y": 0, "ancho": 100, "alto": 60, "profundidad": 1})
    xs = [x for x, _, _ in pts]
    ys = [y for _, y, _ in pts]
    assert min(xs) < -3 and max(xs) > 103 and min(ys) == pytest.approx(0) and max(ys) == pytest.approx(60)
    rows = sorted({round(y, 3) for y in ys})
    # rows never further apart than the stepover (40 % of D); zip instead of pairwise (3.10+)
    assert max(b - a for a, b in zip(rows, rows[1:])) <= 6 * 0.4 + 1e-9  # noqa: RUF007


def test_bad_data_is_explained():
    with pytest.raises(WizardError, match="más grande"):
        generate("cajera_rect", {"ancho": 4, "alto": 20}, T)
    with pytest.raises(WizardError, match="mayor que 0"):
        generate("cajera_circ", {"diametro": 0}, T)
    with pytest.raises(WizardError, match="número"):
        generate("planeado", {"ancho": "mucho"}, T)


def test_session_uses_the_profiles_and_refuses_plasma():
    from hope.cnc.links import SimLink
    from hope.cnc.profiles import ProfileStore, default_selection, effective
    from hope.cnc.session import CncSession, SessionError
    store = ProfileStore(None)
    sel = {**default_selection(store), "consumible": "Fresa plana 6 mm 2 filos"}
    m, _ = effective(store, sel)
    s = CncSession(m, SimLink(m, autostart=False), profiles=store, selection=sel,
                   link_factory=lambda mm: SimLink(mm, autostart=False))
    forms = s.wizards()
    assert [a["tipo"] for a in forms["asistentes"]] == sorted(WIZARDS) and "Ø6" in forms["herramienta"]
    r = s.run_wizard("cajera_circ", {"x": 100, "y": 100, "diametro": 40, "profundidad": 4})
    assert r["ok"] and s.program["nombre"] == "asistente_cajera_circ.nc" and "fresa Ø6 mm" in s.program["texto"]
    with pytest.raises(SessionError, match="más grande"):
        s.run_wizard("cajera_rect", {"ancho": 3, "alto": 3})
    plasma, _ = effective(store, default_selection(store, "Mesa de plasma 1500×3000"))
    with pytest.raises(SessionError, match="fresadora o router"):
        CncSession(plasma, SimLink(plasma, autostart=False)).run_wizard("planeado", {})
