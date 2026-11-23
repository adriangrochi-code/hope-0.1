"""Optimizaciones de TISMA CNC: tienen que dar EXACTAMENTE el mismo resultado que el camino lento.

* El camino rápido del intérprete (líneas «G1 X Y Z F») contra el camino completo, operación por
  operación, con todos los ejemplos y un programa con ciclos, arcos, subprogramas y cambios de origen.
* Los tiempos por línea de la sesión (un solo plan) contra el método anterior (re-interpretar lo
  emitido y planificar de nuevo).
* Todos los dialectos emiten la misma cantidad de líneas (de eso depende «Correr desde aquí»).
"""

from pathlib import Path

import pytest

from hope.cnc import gcode
from hope.cnc.dialect import DIALECTS, emit
from hope.cnc.gcode import interpret
from hope.cnc.machine import as_controller, load_machine
from hope.cnc.planner import plan
from hope.cnc.session import _cumulative_times

EJEMPLOS = Path("hope/cnc/ejemplos")
MIXTO = """%
(programa con un poco de todo)
G21 G90 G17 G54
G10 L2 P1 X10 Y5 Z-20
T1 M6
S12000 M3
G0 Z5
X0 Y0
G1 Z-1 F300
X20 F900
Y20
G2 X0 Y20 I-10 J0
G91 G1 X5
G90
G81 X30 Y30 Z-3 R2 F200
X40
G80
G0 Z5
M98 P100 L2
G4 P0.5
G61
G1 X1 Y1 F500
G64
M5
M30
O100
G0 X50 Y50
G1 Z-1 F200
G0 Z5
M99
"""


def _casos():
    out = [("configs/cnc_maquina.yaml", MIXTO)]
    for f in sorted(EJEMPLOS.glob("*.nc")):
        maq = ("configs/cnc_plasma.yaml" if "plasma" in f.name else
               "perfil:Router con 4º eje (A)" if "4_ejes" in f.name else "configs/cnc_maquina.yaml")
        out.append((maq, f.read_text(encoding="utf-8")))
    return out


def _maquina(maq):
    """Máquina de un caso: un YAML de configs/ o «perfil:Nombre» (perfil del sistema)."""
    if maq.startswith("perfil:"):
        from hope.cnc.profiles import ProfileStore, default_selection, effective
        store = ProfileStore(None)
        return effective(store, default_selection(store, maq[7:]))[0]
    return load_machine(maq)


def _firma(prog):
    return [(o.kind, o.line, o.start, o.end, o.feed, o.exact_stop, o.value, o.source, o.frame) for o in prog.ops], \
        [str(d) for d in prog.diags]


@pytest.mark.parametrize("maq,texto", _casos())
def test_camino_rapido_igual_al_completo(maq, texto, monkeypatch):
    m = _maquina(maq)
    rapido = _firma(interpret(texto, m))
    monkeypatch.setattr(gcode, "_FAST_KEYS", frozenset())        # sin camino rápido
    completo = _firma(interpret(texto, m))
    assert rapido == completo


@pytest.mark.parametrize("maq,texto", _casos())
def test_tiempos_por_linea_iguales_al_metodo_anterior(maq, texto):
    m = _maquina(maq)
    prog = interpret(texto, m)
    pl = plan(prog, m)
    ref = emit(prog, m, "grbl")
    for d in DIALECTS:
        origins: list[int] = []
        out = emit(prog, m, d, origins)
        assert len(out) == len(origins) == len(ref), d              # mismo largo en todos los dialectos
        cum = _cumulative_times(prog, pl, out, origins)
        # método anterior: re-interpretar la versión grbl y planificar otra vez
        mc = as_controller(m)
        pl2 = plan(interpret("\n".join(t for t, _ in ref), mc), mc)
        old = [0.0]
        for i in range(1, len(out) + 1):
            old.append(old[-1] + pl2.time_by_line.get(i, 0.0))
        dwell = {i + 1: float(t.split("P", 1)[1]) for i, (t, _) in enumerate(ref) if t.startswith("G4 P")}
        acc = 0.0
        for i in range(1, len(old)):
            acc += dwell.get(i, 0.0)
            old[i] += acc
        assert len(cum) == len(old)
        assert max(abs(a - b) for a, b in zip(cum, old)) < 1e-6, d


def test_frame_compartido_hasta_que_cambia_el_origen():
    m = load_machine("configs/cnc_maquina.yaml")
    prog = interpret("G0 X1\nG0 X2\nG10 L2 P1 X5\nG0 X3\nG0 X4", m)
    moves = [o for o in prog.ops if o.kind == "rapid"]
    assert moves[0].frame is moves[1].frame                          # mismo objeto: sin copias
    assert moves[2].frame is moves[3].frame and moves[1].frame is not moves[2].frame
    assert moves[1].frame["X"] != moves[2].frame["X"]                # el cambio de G54 se ve


def test_medir_da_un_informe():
    from hope.cnc.medir import informe_texto, medir
    r = medir(2000)
    assert r["listo"] and r["lineas"] > 1900 and r["etapas_s"][0][0] == "intérprete"
    assert "100 000 líneas" in informe_texto(r)


# --- errores que encontró la revisión de la documentación -------------------------------------------

def test_g92_y_g10_l20_descuentan_g52():
    """Con un G52 activo, «G92 X0» y «G10 L20 P1 X0» tienen que dejar la posición actual en X0."""
    m = load_machine("configs/cnc_maquina.yaml")
    for fijar in ("G92 X0", "G10 L20 P1 X0"):
        it = gcode.Interpreter(m)
        it.run(f"G0 X30\nG52 X5\n{fijar}")
        assert abs(it.work("X")) < 1e-9, fijar


def test_marlin_no_confunde_m03_con_una_pausa():
    from hope.cnc.links import PAUSE_CODE
    for pausa in ("M0", "M00", "M1", "M01", "M0 cambiar fresa", "m0"):
        assert PAUSE_CODE.match(pausa), pausa
    for otra in ("M03 S1000", "M05", "M3", "M104 S200", "M10", "M1.5"):
        assert not PAUSE_CODE.match(otra), otra


# --- 4º eje (ejes rotativos) --------------------------------------------------------------------------

def test_cuarto_eje_informe_y_vista_enrollada():
    """El ejemplo del cilindro: sin errores, el informe separa mm de grados y la vista enrolla el
    trabajo sobre el cilindro (todos los puntos de corte a radio 17–19 del eje X)."""
    import math

    from hope.cnc.check import check_program, report_text
    from hope.cnc.links import SimLink
    from hope.cnc.session import CncSession
    m = _maquina("perfil:Router con 4º eje (A)")
    texto = (EJEMPLOS / "cilindro_4_ejes.nc").read_text(encoding="utf-8")
    prog = interpret(texto, m)
    assert prog.ok, [str(d) for d in prog.diags]
    r = check_program(prog, m)
    assert r["runnable"] and abs(r["giro_grados"]["A"] - 7200) < 1e-6      # las ranuras no giran
    assert "Giro: A 7200°" in report_text(r)
    assert abs(r["avance_mm"] - (80 + 11 + 8 * (13 + 70))) < 0.01            # mm lineales, sin los grados
    s = CncSession(m, SimLink(m))
    s.load("cilindro.nc", texto)
    tp = s.toolpath()
    assert tp["rotativos"] == ["A"]
    corte = [p for p in tp["puntos"] if p[3] == 1]
    radios = [math.hypot(p[1], p[2]) for p in corte]
    assert min(radios) > 16.99 and max(radios) < 19.01
    hel = [p for p in corte if 19 - 1e-3 < math.hypot(p[1], p[2])]          # 4 decimales de redondeo
    assert len(hel) > 20 * 360 / 5 - 5                                      # la hélice se parte cada 5°


def test_eje_inactivo_no_existe():
    from hope.cnc.machine import machine_from_dict
    m = machine_from_dict({"ejes": {"X": {}, "Y": {}, "Z": {}, "A": {"activo": False, "vel_max": 3600}}})
    assert m.axis_names == ("X", "Y", "Z")
    assert not interpret("G1 A10 F100", m).ok                             # «eje A no configurado»
