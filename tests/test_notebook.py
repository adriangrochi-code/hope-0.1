"""Cuaderno de notas con repaso espaciado (hope/memory/knowledge.py, Notebook)."""

from hope.epistemic import EpistemicStatus
from hope.memory.knowledge import DAY, KnowledgeBase, Notebook
from hope.runtime.toolcalls import ToolRunner
from hope.tools.builtin import default_catalog


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def test_write_lookup_version_and_spaced_repetition_until_automated():
    clock = Clock()
    nb = Notebook(KnowledgeBase(), automate_after=3, clock=clock)
    a = nb.write("ley de Ohm", "V = I · R", source="maestro", verified=True)
    assert nb.write("Ley de  Ohm", "V = I · R") == a                       # sin duplicados
    nb.write("ley de Ohm", "V = I · R: la tensión es corriente por resistencia", source="maestro")
    assert "corriente por resistencia" in nb.lookup("ley de ohm").text   # versión nueva
    assert any(h.get("op") == "version" for h in nb.store.history(a))     # la anterior queda registrada
    nb.write("potencia eléctrica", "P = V · I")
    assert nb.lookup("cuánta potencia eléctrica disipa").concept == "potencia eléctrica"   # búsqueda aproximada
    assert nb.due() == []                                                 # nada vence el primer día
    clock.t += 1.1 * DAY
    due = nb.due()
    assert [n.concept for n in due][0] == "ley de Ohm"                    # lo más consultado primero
    texts = nb.practice_texts(due)
    assert texts[0].startswith("¿Qué es «ley de Ohm»?") and "V = I" in texts[0]
    n = nb.record_recall(a, False)
    assert n.interval_days == 1.0 and n.streak == 0
    for expected in (2.0, 4.0, 8.0):
        clock.t = n.due + 1
        n = nb.record_recall(a, True)
        assert n.interval_days == expected
    assert n.automated and nb.store.get(a).status == EpistemicStatus.VALIDADO_CONDICIONAL
    clock.t = n.due + 1
    assert a not in [x.id for x in nb.due()]                              # automatizada: no se repasa
    assert nb.stats()["automatizadas"] == 1


def test_notes_as_tools():
    kb = KnowledgeBase()
    r = ToolRunner(default_catalog(), kb)
    assert r.run_text("⟦anotar: divisor de tensión | Vout = Vin · R2 / (R1 + R2)⟧").endswith("anotado")
    assert "R2 / (R1 + R2)" in r.run_text("⟦nota: divisor de tensión⟧")
    assert "no hay nota" in r.run_text("⟦nota: termodinámica cuántica⟧")
