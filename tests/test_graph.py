"""Grafo de conocimiento y correlaciones (hope/memory/graph.py, eng_tools.correlations)."""

from hope.memory.knowledge import KnowledgeBase, Notebook
from hope.runtime.toolcalls import ToolRunner
from hope.tools.builtin import default_catalog
from hope.tools.eng_tools import correlations


def test_auto_links_mentions_and_expansion_brings_related_definitions():
    kb = KnowledgeBase()
    nb = Notebook(kb)
    ohm = nb.write("ley de Ohm", "V = I · R: la tensión es la corriente por la resistencia")
    pot = nb.write("potencia eléctrica", "P = V · I; con la ley de Ohm también P = I² · R")
    nb.write("receta de pan", "harina, agua, sal y levadura; amasar y hornear")
    # la nota de potencia menciona «ley de Ohm» → enlace de dependencia
    assert any(nid == ohm for nid, _t, _c in kb.store.neighbors(pot))
    related = [nid for nid, _a in kb.graph.expand([pot])]
    assert ohm in related and all(kb.store.get(n).concept != "receta de pan" for n in related)


def test_hebbian_co_use_patterns_and_decay():
    kb = KnowledgeBase()
    nb = Notebook(kb)
    a = nb.write("divisor de tensión", "Vout = Vin · R2 / (R1 + R2)")
    b = nb.write("resistencias en serie", "Req = R1 + R2")
    c = nb.write("historia de Roma", "fundada según la tradición en el 753 a. C.")
    for _ in range(4):
        kb.graph.co_used([a, b])
    p = kb.graph.patterns()
    assert p["usados_juntos"][0][:2] in (("divisor de tensión", "resistencias en serie"),
                                         ("resistencias en serie", "divisor de tensión"))
    assert any({"divisor de tensión", "resistencias en serie"} <= set(g) for g in p["grupos"])
    assert all("historia de Roma" not in g for g in p["grupos"])
    before = p["usados_juntos"][0][2]
    kb.graph.decay(0.5)
    assert kb.graph.patterns()["usados_juntos"][0][2] < before
    del c


def test_context_includes_related_and_feedback_reinforces():
    kb = KnowledgeBase()
    nb = Notebook(kb)
    nb.write("ley de Ohm", "V = I · R")
    kb.learn_solution("potencia", "¿Qué potencia disipa un resistor de 10 ohm con 2 A?",
                      "Paso 1: P = I² · R (ley de Ohm) = 4 · 10 = 40\nResultado: 40 W", "40 W", verified=True)
    ctx = kb.context("potencia", "¿Qué potencia disipa un resistor de 20 ohm con 1 A?")
    assert "Relacionado:" in ctx.text and "ley de Ohm" in ctx.text
    kb.feedback(ctx.used, helped=True)
    assert kb.graph.patterns()["usados_juntos"]


def test_correlations_tool():
    csv_text = "x,y,z,nombre\n1,2,9,a\n2,4,7,b\n3,6,8,c\n4,8,1,d\n5,10,3,e\n"
    r = correlations(csv_text)
    assert r["n"] == 5 and r["columnas"] == ["x", "y", "z"]
    top = r["pares"][0]
    assert {top["a"], top["b"]} == {"x", "y"} and top["pearson"] == 1.0 and top["fuerza"] == "fuerte"
    out = ToolRunner(default_catalog()).run_text("⟦correlacion: " + csv_text + "⟧")
    assert "x~y: r=1.0" in out
