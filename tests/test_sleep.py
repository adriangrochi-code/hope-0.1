from hope.epistemic import EpistemicStatus as S
from hope.memory.pesam import ExperienceLog
from hope.memory.store import Level, MemoryStore
from hope.sleep.ideas import Idea, IdeaStore
from hope.sleep.sca import SleepBudget, SleepCycle, wake_review


def _memory():
    m = MemoryStore()
    m.add_engram(kind="procedimiento", level=Level.PROCEDIMENTAL, concept="ley de ohm",
                 content="Tema: ley de ohm\nProblema: tensión en un resistor\nPaso 1: V = I · R", status=S.RESPALDADO)
    m.add_engram(kind="procedimiento", level=Level.PROCEDIMENTAL, concept="potencia eléctrica",
                 content="Tema: potencia\nProblema: potencia en un resistor\nPaso 1: P = I · V con V = I · R",
                 status=S.RESPALDADO)
    m.add_engram(kind="procedimiento", level=Level.PROCEDIMENTAL, concept="suma",
                 content="Tema: suma\nProblema: 17 + 25\nPaso 1: 7 + 5 = 12", status=S.RESPALDADO)
    return m


def _pesam(tmp_path):
    p = ExperienceLog(tmp_path / "pesam.jsonl")
    fails = [{"problema": f"¿Qué tensión cae con {r} ohm y 0.05 A?", "esperado": f"{r * 0.05:g} V",
              "respuesta_hope": "1 W", "tema": "ley de ohm", "tipos_error": ["dimension"]} for r in (100, 220, 330)]
    p.record("evaluacion", S.OBSERVADO, results={}, failures=fails)
    return p


def test_dream_produces_only_hypotheses_and_never_touches_weights(tmp_path):
    ideas = IdeaStore(tmp_path / "ideas.jsonl")
    usage = {"iterations": {"weight_per_iteration": [0.95, 0.03, 0.02],
                            "gain_of_each_extra_iteration": [0.0, 1e-5, 0.0]},
             "expert_usage": [0.3, 0.3, 0.38, 0.02],
             "idle_unit_fraction (Taylor < 10% de la mediana)": {"regions.0.experts.1.fc2": 0.4}}
    sca = SleepCycle(_memory(), ideas, _pesam(tmp_path), usage_report=usage,
                     answerer=lambda topic, p: ("Resultado: 11 V", "11 V"))
    rep = sca.run()
    assert set(rep) == {"ligero", "asociativo", "analitico", "creativo", "evolutivo"}
    assert rep["analitico"]["recurrent_error_kinds"] == {"dimension": 3}
    assert rep["analitico"]["now_solved"] == 1                      # 220 ohm · 0,05 A = 11 V
    modes = {i.mode for i in ideas.all()}
    assert {"analitico", "creativo", "evolutivo"} <= modes
    assert all(i.status == S.GENERADO_EN_SUENO for i in ideas.all())
    assert any("iteración 2" in i.description for i in ideas.all())
    assert any("especialista 3" in i.description for i in ideas.all())
    rels = sca.memory.db.execute("SELECT status FROM relations").fetchall()
    assert all(r[0] == "GENERADO_EN_SUENO" for r in rels)


def test_ideas_are_deduplicated_and_bounded(tmp_path):
    ideas = IdeaStore(tmp_path / "ideas.jsonl")
    sca = SleepCycle(_memory(), ideas, budget=SleepBudget(max_ideas=2))
    sca.run(("creativo",))
    assert len(ideas.all()) == 2
    sca2 = SleepCycle(_memory(), ideas, budget=SleepBudget(max_ideas=50))
    sca2.run(("creativo",))
    assert len(ideas.all()) == 3                                     # 3 combinaciones posibles, sin repetir


def test_interrupt_and_resume(tmp_path):
    ideas = IdeaStore(tmp_path / "ideas.jsonl")
    state = tmp_path / "sueno.json"
    calls = iter([False, False, True])
    sca = SleepCycle(_memory(), ideas, state_path=state)
    rep = sca.run(should_stop=lambda: next(calls, False))
    assert rep["interrupted_at"] == "analitico" and state.exists()      # ligero y asociativo ya hechos
    rep2 = SleepCycle(sca.memory, ideas, state_path=state).run()
    assert "asociativo" not in rep2 and "analitico" in rep2 and "evolutivo" in rep2 and not state.exists()


def test_wake_review_needs_evidence(tmp_path):
    ideas = IdeaStore(tmp_path / "ideas.jsonl")
    a = ideas.add(Idea("d", "creativo", "combinar A y B", "h", "r", proposed_tests=[{"tipo": "t"}]))
    b = ideas.add(Idea("d", "creativo", "combinar A y C", "h", "r", proposed_tests=[{"tipo": "t"}]))
    c = ideas.add(Idea("d", "creativo", "combinar B y C", "h", "r", proposed_tests=[{"tipo": "otro"}]))
    results = {a.idea_id: ("respalda", "exp_1", ""), b.idea_id: ("refuta", "exp_2", "contraejemplo")}
    counts = wake_review(ideas, {"t": lambda idea, test: results[idea.idea_id]})
    assert counts == {"respalda": 1, "refuta": 1}
    reloaded = {i.idea_id: i for i in IdeaStore(tmp_path / "ideas.jsonl").all()}
    assert reloaded[a.idea_id].status == S.RESPALDADO
    assert reloaded[b.idea_id].status == S.REFUTADO                   # se conserva, no se borra
    assert reloaded[c.idea_id].status == S.GENERADO_EN_SUENO          # sin prueba disponible: sigue igual
