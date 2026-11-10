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


# ---------------------------------------------------------------------------
# Modo continuo (pedido del usuario: Qwen desde el principio, varios maestros)
# ---------------------------------------------------------------------------
def _corpus(tmp_path):
    d = tmp_path / "corpus" / "wiki_es"
    d.mkdir(parents=True)
    t = " ".join(f"El río número {i} lleva {i} + 3 = {i + 3} litros de agua por segundo hacia el mar." for i in range(40))
    (d / "train.bin").write_bytes(("\x00" + t).encode() * 20)
    return d


def test_continuo_varios_maestros_pasajes_y_bins(tmp_path):
    import yaml
    from hope.teacher.harvest import continuo, load_teachers
    from hope.data.corpus import ByteData
    y = tmp_path / "maestros.yaml"
    y.write_text(yaml.safe_dump({"maestros": [{"id": "qwen", "model": "mock"}, {"id": "phi", "model": "mock"},
                                              {"id": "apagado", "model": "mock", "activo": False}]}))
    maestros = load_teachers(y)
    assert [m for m, _ in maestros] == ["qwen", "phi"]
    out = tmp_path / "maestro"
    h = Harvester(maestros[0][1], default_catalog(), out, log=lambda *_: None)
    topics = [Topic("suma", 1), Topic("ley de Ohm y potencia eléctrica", 2)]
    stats = continuo(h, maestros, topics, corpus=[_corpus(tmp_path)], paralelo=3, lote=4, max_calls_dia=12,
                     max_hours=0.002)
    assert stats["total"]["pedidos"] == 12                        # el límite del día corta
    rows = _rows(out / "episodios_verificados.jsonl")
    assert {r["maestro"] for r in rows} == {"qwen", "phi"}         # los dos explican (rotación)
    pas = [r for r in rows if r.get("pasaje_sha")]
    assert pas and all(r["tema"].startswith("pasaje: wiki_es") for r in pas)
    tb, vb = out / "train.bin", out / "val.bin"
    data = tb.read_bytes() + vb.read_bytes()
    assert data.count(b"\x00") == len(rows) and b"Problema:" in data      # formato del corpus
    assert not list(out.glob("*.tmp"))                                     # escritura atómica
    ByteData(tb)                                                           # lo lee el entrenamiento
    # Reanudable: otro cosechador recuerda los pasajes ya explicados y sigue sumando.
    h2 = Harvester(maestros[0][1], default_catalog(), out, log=lambda *_: None)
    assert {r["pasaje_sha"] for r in pas} <= h2.pasajes_vistos
    continuo(h2, maestros, topics, paralelo=2, lote=2, max_calls_dia=4, max_hours=0.002)
    assert len(_rows(out / "episodios_verificados.jsonl")) == 16


def test_rotacion_prefiere_al_que_mas_verifica():
    from collections import Counter
    from hope.teacher.harvest import elegir_maestro
    ms = [("a", None), ("b", None)]
    por = {"a": Counter(RESPALDADO=9, REFUTADO=1), "b": Counter(RESPALDADO=2, REFUTADO=8)}
    usados = Counter()
    for _ in range(20):
        mid, _t = elegir_maestro(ms, por, usados)
        usados[mid] += 1
    assert usados["a"] > usados["b"] > 0


def test_bins_incluyen_lo_de_la_app_y_respetan_inferidos(tmp_path):
    from hope.teacher.harvest import escribir_bins
    (tmp_path / "episodios.jsonl").write_text(json.dumps({"aceptado": True, "estado": "RESPALDADO", "sha": "ab" * 32,
                                                          "texto": "Tema: app\nResultado: 1"}) + "\n")
    (tmp_path / "episodios_inferidos.jsonl").write_text(json.dumps({"texto_entrenamiento": "Tema: inferido"}) + "\n")
    escribir_bins(tmp_path)
    todo = (tmp_path / "train.bin").read_bytes() + (tmp_path / "val.bin").read_bytes()
    assert b"Tema: app" in todo and b"Tema: inferido" in todo
    escribir_bins(tmp_path, incluir_inferidos=False)
    todo = (tmp_path / "train.bin").read_bytes() + (tmp_path / "val.bin").read_bytes()
    assert b"Tema: app" in todo and b"Tema: inferido" not in todo


def test_cli_continuo(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    t = tmp_path / "t.yaml"
    t.write_text("model: mock\n")
    r = subprocess.run([sys.executable, "-m", "hope.teacher.harvest", "--continuo", "--teacher", str(t),
                        "--out", str(tmp_path / "o"), "--corpus", str(_corpus(tmp_path)), "--max-calls-dia", "5",
                        "--max-hours", "0.001", "--progreso"], cwd=repo, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "PROGRESO" in r.stdout and "mañana" in r.stdout
    assert (tmp_path / "o" / "train.bin").exists()
