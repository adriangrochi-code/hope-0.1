import torch

from hope.core.hn1 import build_hn1
from hope.epistemic import EpistemicStatus as S
from hope.memory.embed import HashingEmbedder
from hope.memory.retrieval import retrieve
from hope.memory.store import Level, MemoryStore, Relation
from hope.memory.working import ProjectState, WorkingItem, WorkingMemory

DOC_ES = """La ley de Ohm establece que la tensión es igual a la corriente multiplicada por la resistencia.

Un resistor de 220 ohmios con una corriente de 0,05 amperios tiene una caída de 11 voltios.

La potencia disipada en un resistor es el producto de la tensión por la corriente."""

DOC_EN = """A voltage regulator keeps the output voltage constant when the input or the load change.
Linear regulators dissipate the difference between input and output as heat."""


def _store():
    st = MemoryStore()
    st.add_document(DOC_ES, source="apunte_electronica.md", passage_chars=120)
    st.add_document(DOC_EN, source="regulators.txt", passage_chars=200)
    return st


def test_documents_are_deduplicated_and_kept_exact():
    st = _store()
    doc = st.add_document(DOC_ES, source="copia.md")
    assert st.stats()["documents"] == 2
    i = DOC_ES.index("0,05")
    assert st.source_text(doc, i, 4) == "0,05"           # el número exacto, desde el original


def test_hybrid_retrieval_finds_relevant_passage_with_source():
    st = _store()
    res = retrieve(st, "caida de tension en un resistor de 220 ohmios")
    assert res.found
    top = res.items[0]
    assert "220 ohmios" in top.engram.content
    assert top.source["doc_id"].startswith("doc_")
    exact = st.source_text(**top.source)
    assert exact == top.engram.content
    # Robusto a acentos y errores menores gracias a los n-gramas de caracteres.
    assert "regulator" in retrieve(st, "voltaje regulador output constant").items[0].engram.content


def test_nothing_relevant_means_not_found():
    st = _store()
    assert not retrieve(st, "receta de empanadas tucumanas", min_score=0.5).found


def test_facts_contradictions_are_kept_not_overwritten():
    st = MemoryStore()
    a = st.remember_fact("LM7805", "tensión de salida", "5 V", source="hoja de datos")
    b = st.remember_fact("LM7805", "tensión de salida", "12 V", source="foro", confidence=0.3)
    assert b.contradicts == [a.engram.id]
    assert {e.content for e in st.lookup("lm7805", "Tensión de salida")} == {"5 V", "12 V"}
    assert st.lookup("LM7805", "tensión de salida")[0].content == "5 V"        # más confiable primero
    st.update_status(b.engram.id, S.REFUTADO, evidence="medición propia")
    assert [e.content for e in st.lookup("LM7805", "tensión de salida")] == ["5 V"]
    assert [h["op"] for h in st.history(b.engram.id)] == ["create", "status"]
    again = st.remember_fact("LM7805", "tensión de salida", "5 V", source="otra hoja")
    assert again.engram.id == a.engram.id and again.contradicts == []


def test_contradiction_is_flagged_in_retrieval():
    st = MemoryStore()
    a = st.remember_fact("agua", "punto de ebullición al nivel del mar", "100 °C", source="libro")
    st.remember_fact("agua", "punto de ebullición al nivel del mar", "90 °C", source="error")
    res = retrieve(st, "agua punto de ebullición nivel del mar", min_score=0.0)
    flagged = [r for r in res.items if r.contradicted_by]
    assert flagged and any(r.engram.id == a.engram.id for r in flagged)


def test_verified_causality_requires_validation():
    st = MemoryStore()
    x = st.add_engram(kind="concepto", level=Level.SEMANTICA, concept="calor", content="temperatura alta")
    y = st.add_engram(kind="concepto", level=Level.SEMANTICA, concept="falla", content="resistor quemado")
    st.relate(x, y, Relation.CAUSALIDAD_HIPOTETICA)
    try:
        st.relate(x, y, Relation.CAUSALIDAD_VERIFICADA)
        assert False, "debió rechazarse"
    except ValueError:
        pass
    st.relate(x, y, Relation.CAUSALIDAD_VERIFICADA, status=S.VALIDADO)


def test_consolidation_promotes_and_archives_without_deleting():
    st = MemoryStore()
    used = st.add_engram(kind="experiencia", level=Level.EPISODICA, concept="depuración",
                         content="el error era la polaridad", status=S.RESPALDADO)
    for _ in range(3):
        st.touch(used)
    stale = st.add_engram(kind="nota", level=Level.EPISODICA, concept="ruido", content="nada útil",
                          importance=0.1, status=S.INFERIDO)
    st.db.execute("UPDATE engrams SET created = 0 WHERE id = ?", (stale,))
    rep = st.consolidate()
    assert rep == {"promoted_to_semantic": 1, "archived": 1}
    assert st.get(used).level == Level.SEMANTICA
    assert st.get(stale).archived                      # archivado, no borrado


def test_persistence(tmp_path):
    path = tmp_path / "mem.sqlite"
    st = MemoryStore(path)
    st.remember_fact("hope", "límite de parámetros", "200M", source="usuario")
    st.close()
    st2 = MemoryStore(path)
    assert st2.lookup("hope", "límite de parámetros")[0].content == "200M"
    assert st2.vector_search("hope límite parámetros")[0][1] > 0


def test_working_memory_is_bounded():
    wm = WorkingMemory(capacity=2)
    wm.add(WorkingItem("a", "x", 0.9))
    wm.add(WorkingItem("b", "y", 0.1))
    evicted = wm.add(WorkingItem("c", "z", 0.5))
    assert evicted.key == "b" and [i.key for i in wm.items()] == ["a", "c"]


def test_project_state_detects_changed_files(tmp_path):
    f = tmp_path / "circuito.py"
    f.write_text("R = 220\n")
    ps = ProjectState(objective="analizar el regulador", next_steps=["medir tensión de salida"])
    ps.decide("usar regulador lineal", "la corriente es baja")
    ps.track_file(f)
    ps.save(tmp_path / "estado.json")
    loaded = ProjectState.load(tmp_path / "estado.json")
    assert loaded.validate() == [] and loaded.decisions[0]["justification"] == "la corriente es baja"
    f.write_text("R = 330\n")
    assert loaded.validate() and "cambió" in loaded.validate()[0]


def test_embedder_tolerates_accents_and_typos():
    e = HashingEmbedder()
    v = e.embed(["tensión de salida", "tension de salida", "tensoin de salda", "receta de cocina"])
    assert v[0] @ v[1] > 0.99 and v[0] @ v[2] > v[0] @ v[3]


def test_memory_feeds_the_core():
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    model.eval()
    st = _store()
    texts = retrieve(st, "ley de Ohm resistor").texts()
    mem = model.encode_memory(texts)
    assert mem.shape == (len(texts), model.cfg.d)
    x = torch.tensor([list("calcular la tension".encode())])
    with torch.no_grad():
        base = model(x)["logits"]
        assert torch.allclose(base, model(x, memory=mem)["logits"])     # compuerta cerrada al inicio
        model.memory.gate.fill_(1.0)
        assert not torch.allclose(base, model(x, memory=mem)["logits"])  # abierta: la memoria influye
