"""Pruebas de la fachada hope.api: herramientas certificadas, formatos de llamada y bucle agnóstico."""

import json

import pytest

from hope import api
from hope.api import exactas


def test_version():
    assert api.VERSION.count(".") == 2


@pytest.mark.parametrize("expr,esperado", [("0.1+0.2", "0.3"), ("2**-3", "0.125"), ("sqrt(4)", "2.0")])
def test_calc_cert_intervalo(expr, esperado):
    r = exactas.calc_cert(expr)
    assert r.ok and r.verificado
    a, b = (float(x) for x in r.certificado["intervalo"])
    assert a <= float(esperado) <= b


@pytest.mark.parametrize("malo", ["__import__('os')", "x.y", "10**10**10", "(lambda: 1)()", "open(1)",
                                  "2**(3*4)", "'a'*3", "[1,2]"])
def test_expresiones_peligrosas_bloqueadas(malo):
    with pytest.raises(exactas.ExpresionInvalida):
        exactas.calc_cert(malo)


def test_equivalencia():
    assert exactas.equivalencia("sin(x)^2+cos(x)^2", "1").valor is True
    r = exactas.equivalencia("x^2", "x^3")
    assert r.valor is False and r.verificado


def test_raiz_e_integral():
    r = exactas.raiz("x^2-2", 0, 2)
    assert r.verificado and r.valor.startswith("1.41421356")
    assert not exactas.raiz("x^2+1", -1, 1).ok
    r = exactas.integral("exp(-x^2)", "0", "1")
    assert r.verificado and r.valor.startswith("0.7468241328")


def test_lp_certificado_dual():
    pytest.importorskip("scipy")
    r = exactas.lp({"c": [3, 2], "A_ub": [[1, 1], [1, 3]], "b_ub": [4, 6], "maximizar": True})
    assert r.ok and r.verificado and abs(r.valor["objetivo"] - 12) < 1e-9
    assert r.certificado["brecha_relativa"] < 1e-6


def test_smt():
    pytest.importorskip("z3")
    r = exactas.smt("(declare-const x Int)(declare-const y Int)(assert (= (+ x y) 10))(assert (= (- x y) 2))")
    assert r.verificado and r.valor == {"x": "6", "y": "4"}
    assert exactas.smt("(declare-const x Int)(assert (> x 2))(assert (< x 1))").salida == "insatisfacible"


def test_grafo():
    r = exactas.grafo({"op": "camino", "aristas": [["a", "b", 1], ["b", "c", 2], ["a", "c", 5]],
                       "origen": "a", "destino": "c"})
    assert r.verificado and r.valor["largo"] == 3
    r = exactas.grafo({"op": "flujo", "aristas": [["s", "a", 3], ["s", "b", 2], ["a", "t", 2], ["b", "t", 3],
                                                  ["a", "b", 1]], "origen": "s", "destino": "t"})
    assert r.verificado and r.valor == 5


# --- formatos ------------------------------------------------------------------------------------
def test_extraer_todas_las_sintaxis():
    texto = ('Paso ⟦calc: 2 + 2⟧ y después <tool_call>{"name": "calc_cert", "arguments": {"expr": "pi"}}</tool_call>'
             ' y <tool_call>\n<function=raiz>\n<parameter=f>\nx^2-2\n</parameter>\n<parameter=a>\n0\n</parameter>'
             '\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>')
    ll = api.extraer_llamadas(texto)
    assert [c.sintaxis for c in ll] == ["hope", "hermes", "qwen_xml"]
    assert ll[1].argumentos == {"expr": "pi"}
    assert ll[2].argumentos == {"f": "x^2-2", "a": 0, "b": 2}


def test_bloque_json_y_openai():
    ll = api.extraer_llamadas('```json\n{"name": "calc_cert", "arguments": {"expr": "1/3"}}\n```')
    assert ll and ll[0].sintaxis == "json"
    msg = {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                                "function": {"name": "calc_cert", "arguments": '{"expr": "e"}'}}]}
    ll = api.desde_openai(msg)
    assert ll[0].argumentos == {"expr": "e"} and ll[0].id == "c1"


def test_json_invalido_no_rompe():
    ll = api.extraer_llamadas("<tool_call>{no es json</tool_call>")
    assert ll and ll[0].errores


# --- registro ------------------------------------------------------------------------------------
def test_herramientas_especificaciones_y_busqueda():
    h = api.Herramientas()
    specs = h.especificaciones()
    nombres = {s["function"]["name"] for s in specs}
    assert {"calc_cert", "lp", "smt", "grafo", "calc", "unidades"} <= nombres
    for s in specs:
        json.dumps(s)
        assert s["function"]["parameters"]["type"] == "object"
    assert h.buscar("maximizar ganancia con restricciones de recursos", k=3)[0].nombre == "lp"
    assert "tool_call" in h.instrucciones("hermes")


def test_ejecutar_exacta_y_catalogo():
    h = api.Herramientas()
    r = h.ejecutar("calc_cert", {"expr": "1/3", "cifras": 10})
    assert r.ok and r.verificado
    r = h.ejecutar("calc", "220 * 0,05")
    assert r.ok and r.salida.strip() == "11"
    assert not h.ejecutar("no_existe", {}).ok
    assert not h.ejecutar("raiz", {"f": "x"}).ok             # faltan argumentos
    assert len(h.registro) == 4


def test_ejecutar_texto_responde_en_la_misma_sintaxis():
    h = api.Herramientas()
    out = h.ejecutar_texto('<tool_call>{"name": "calc_cert", "arguments": {"expr": "2+2"}}</tool_call>')
    assert out[0][2].startswith("<tool_response>")
    out = h.ejecutar_texto("⟦calc: 6 * 7⟧")
    assert out[0][2].startswith(" → 42")


def test_bucle_agnostico_con_modelo_simulado():
    respuestas = iter(['<tool_call>{"name": "calc_cert", "arguments": {"expr": "6*7"}}</tool_call>',
                       "El resultado es 42."])
    r = api.responder(lambda _: next(respuestas), "¿Cuánto es 6 por 7?")
    assert r.texto == "El resultado es 42." and r.vueltas == 2
    assert r.llamadas[0]["verificado"] is True and r.todo_verificado


def test_criticar():
    out = api.criticar("Una resistencia de 220 ohm con 0,05 A: ¿qué tensión?", "V = 220 * 0,05 = 12", "12")
    assert out["veredicto"] and isinstance(out["problemas"], list)
