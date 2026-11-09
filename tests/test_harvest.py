import json

from guardian.core import Guardian, GuardianConfig
from hope.teacher.client import MockTeacher, TeacherConfig, OpenAICompatibleTeacher
from hope.teacher.cycle import Topic
from hope.teacher.harvest import Harvester
from hope.tools.builtin import default_catalog


def _rows(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def test_harvest_separa_por_estado_y_reanuda(tmp_path):
    topics = [Topic("suma", 1), Topic("ley de Ohm y potencia eléctrica", 2)]
    h = Harvester(MockTeacher(error_rate=0.5, seed=1), default_catalog(), tmp_path, log=lambda *_: None)
    stats = h.run(topics, rounds=3, per_topic=2)
    ok, bad = _rows(tmp_path / "episodios_verificados.jsonl"), _rows(tmp_path / "episodios_refutados.jsonl")
    assert stats["total"]["pedidos"] == 12 and len(ok) + len(bad) == 12 and ok and bad
    assert all(r["estado"] == "RESPALDADO" and r["modelo"] == "mock" and r["tema"] in ("suma", topics[1].name)
               and r["texto_entrenamiento"].startswith("Tema:") for r in ok)
    assert all(r["estado"] == "REFUTADO" and r["peso"] == 0.0 for r in bad)
    texts = (tmp_path / "textos_entrenamiento.txt").read_text(encoding="utf-8")
    assert texts.count("Problema:") == len(ok)
    # Reanudable: un nuevo cosechador lee lo anterior y sigue sumando.
    h2 = Harvester(MockTeacher(seed=2), default_catalog(), tmp_path, log=lambda *_: None)
    assert sum(len(v) for v in h2.seen.values()) == 12
    assert h2.run(topics[:1], rounds=1)["total"]["pedidos"] == 13


def test_harvest_registra_invalidos(tmp_path):
    class Broken(MockTeacher):
        def _complete(self, messages):
            return "no es JSON", {}
    h = Harvester(Broken(), default_catalog(), tmp_path, log=lambda *_: None)
    stats = h.run([Topic("suma", 1)], per_topic=2)
    assert stats["por_tema"]["suma"]["INVALIDO"] == 2 and len(_rows(tmp_path / "invalidos.jsonl")) == 2


def test_harvest_respeta_guardian(tmp_path):
    budget = tmp_path / "b.yaml"
    budget.write_text("max_total_params: 1000\n")
    g = Guardian(GuardianConfig(str(tmp_path), [], str(budget), str(tmp_path / "a.jsonl"), str(tmp_path / "ap.json")))
    remote = OpenAICompatibleTeacher(TeacherConfig(endpoint="https://api.example.com/v1"), g)
    h = Harvester(remote, default_catalog(), tmp_path / "out", log=lambda *_: None)
    assert h.run([Topic("suma", 1)])["total"]["pedidos"] == 0


def test_no_think_agrega_sufijo(monkeypatch):
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "<think>\n\n</think>\nhola"}}]}).encode()

    def fake(req, timeout):
        sent.update(json.loads(req.data))
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake)
    t = OpenAICompatibleTeacher(TeacherConfig(no_think=True))
    assert t.ask([{"role": "user", "content": "hola"}]).text == "hola"
    assert sent["messages"][-1]["content"].endswith("/no_think")


def test_reverificar_reclasifica(tmp_path):
    from hope.teacher.harvest import reverify
    h = Harvester(MockTeacher(error_rate=0.5, seed=1), default_catalog(), tmp_path, log=lambda *_: None)
    h.run([Topic("suma", 1)], rounds=4)
    bad = _rows(tmp_path / "episodios_refutados.jsonl")
    assert bad
    # Se «arregla» un refutado a mano: al re-verificar pasa a verificados.
    ok_before = len(_rows(tmp_path / "episodios_verificados.jsonl"))
    r = bad[0]
    for st in r["episodio"]["pasos"]:
        st["verificacion"]["esperado"] = st["resultado_intermedio"] = str(eval(st["verificacion"]["expr"]))
    r["episodio"]["resultado"] = r["episodio"]["pasos"][-1]["resultado_intermedio"]
    (tmp_path / "episodios_refutados.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [r] + bad[1:]) + "\n", encoding="utf-8")
    assert reverify(tmp_path, default_catalog()) == {"REFUTADO→RESPALDADO": 1}
    assert len(_rows(tmp_path / "episodios_verificados.jsonl")) == ok_before + 1


def test_parse_episode_repara_cierres_y_sobrantes():
    import pytest
    from hope.teacher.episode import EpisodeParseError, parse_episode
    ok = '{"problema": "p", "pasos": [{"n": 1}], "resultado": "1", "errores": ["a")]}'
    assert parse_episode(ok)["reparacion_json"] is True
    extra = '{"problema": "p", "pasos": [{"n": 1}], "resultado": "1"}}'
    assert parse_episode(extra)["resultado"] == "1"
    clean = parse_episode('{"problema": "p", "pasos": [{"n": 1}], "resultado": "(a)"}')
    assert "reparacion_json" not in clean
    with pytest.raises(EpisodeParseError):
        parse_episode('{"problema": "p", "pasos": [{"n": 1}] "resultado": "1"}')


def test_teacher_score_lee_nota_cortada():
    from hope.data.quality import teacher_score

    class Short(MockTeacher):
        def _complete(self, messages):
            return '<think>\n\n</think>\n{"nota": 4, "mot', {}
    assert teacher_score(Short(), "texto") == 4.0

    class Full(MockTeacher):
        def _complete(self, messages):
            return '{"nota": 2, "motivo": "flojo"}', {}
    assert teacher_score(Full(), "texto") == 2.0

    class Bad(MockTeacher):
        def _complete(self, messages):
            return "no sé", {}
    assert teacher_score(Bad(), "texto") is None
