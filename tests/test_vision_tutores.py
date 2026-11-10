"""Tutores de imágenes locales con verificación cruzada y modo tutor (tutores simulados)."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from hope.data.vision import GENERATORS, ImageStore, ImageTextSource
from hope.teacher.vision_tutores import (ETAPAS, MockDetector, MockOCR, MockScorer, MockVLM, TutorInfo, TutorPanel,
                                         VisionTutorSession, answers_agree, load_panel, problems_from_store)


def _dataset(n=8, seed=0, size=64):
    """Imágenes procedurales con una anotación «verdadera» al estilo del tutor (con nombre_en)."""
    rng = random.Random(seed)
    imgs, truth = [], {}
    for k in range(n):
        kind = ["formas", "texto", "barras", "escena"][k % 4]
        arr, ann = GENERATORS[kind](rng, size)
        for o in ann.get("objetos", []):
            o["nombre_en"] = {"círculo": "circle", "cuadrado": "square", "triángulo": "triangle"}[o["nombre"].split()[0]]
        ann["preguntas"] = [{k2: v for k2, v in q.items() if k2 in ("pregunta", "respuesta", "razonamiento")}
                            for q in ann["preguntas"]]
        ann.setdefault("description_en", ann.get("descripcion_es", ""))
        imgs.append(arr)
        truth[arr.tobytes()] = ann
    return imgs, (lambda img: truth[np.asarray(img).tobytes()])


def _panel(tmp_path, oracle, **kw):
    tutors = {"principal": kw.pop("principal", MockVLM(oracle)), "segundo": kw.pop("segundo", MockVLM(oracle, seed=1)),
              "florence": MockDetector(oracle), "ocr": kw.pop("ocr", MockOCR(oracle)), "verificador": MockScorer(oracle)}
    return TutorPanel(tutors, ImageStore(tmp_path / "vis", 64), **kw)


def test_agreement():
    assert answers_agree("3", "3 figuras") and not answers_agree("23.81", "25 mA")
    assert answers_agree("23.81", "23.8") and not answers_agree("4", "5")
    assert answers_agree("círculo rojo", "El círculo rojo.") and not answers_agree("rojo", "azul")
    assert not answers_agree("negro9", "negro") and not answers_agree("arriba a la izquierda", "arriba")


def test_staged_panel_one_model_per_turn_and_all_verified(tmp_path):
    imgs, oracle = _dataset()
    panel = _panel(tmp_path, oracle)
    seen = []
    while (e := panel.siguiente_etapa(imgs)) is not None:          # un modelo por turno, en orden
        rep = panel.run_stage(e, imgs)
        assert rep["hechas"] == len(imgs) and not rep["errores"]
        seen.append(e)
    assert seen == list(ETAPAS)
    assert panel.run_stage("principal", imgs)["saltadas"] == len(imgs)    # caché: no se repite
    rep = panel.cerrar(imgs)
    # todo lo correcto se confirma; los valores de un gráfico sin texto no los puede confirmar el OCR,
    # así que esas anotaciones quedan «parcial» (sin el campo datos) y no «verificado»
    assert rep["dudoso"] == 0 and rep["verificado"] + rep["parcial"] == len(imgs) and rep["verificado"] >= 2
    for e in panel.store.entries():
        assert e["anotacion"].get("preguntas") or e["anotacion"].get("texto_visible")
        assert "datos" not in e["anotacion"] or e["anotacion"]["tipo"] == "texto"
    src = ImageTextSource.from_store(panel.store)
    assert len(src.items) >= 2 * len(imgs)
    assert all(e["fuente"] == "tutores" for e in panel.store.entries())


def test_wrong_principal_is_caught_and_not_trained(tmp_path):
    imgs, oracle = _dataset(12, seed=3)
    panel = _panel(tmp_path, oracle, principal=MockVLM(oracle, error_rate=1.0))
    rep = panel.annotate(imgs)
    assert rep["verificado"] == 0
    stored = [e["anotacion"] for e in panel.store.entries()]
    assert all(not a.get("preguntas") and not a.get("texto_visible") for a in stored)   # lo equivocado no entra
    dud = [json.loads(line) for line in (panel.store.root / "dudosas.jsonl").read_text(encoding="utf-8").splitlines()]
    assert dud and all(d["dudas"] for d in dud)
    assert any("pregunta" in x for d in dud for x in d["dudas"])


def test_missing_checks_and_restricted_license(tmp_path):
    imgs, oracle = _dataset(4)
    alone = TutorPanel({"principal": MockVLM(oracle)}, ImageStore(tmp_path / "a", 64))
    rep = alone.annotate(imgs)
    assert rep["verificado"] == 0 and rep["dudoso"] == len(imgs)          # sin verificación no hay verificado
    restricted = MockVLM(oracle, info=TutorInfo("principal", "Qwen2.5-VL-3B-Instruct", "Qwen Research", "revisar"))
    p2 = _panel(tmp_path / "b", oracle, principal=restricted)
    rep2 = p2.annotate(imgs)
    assert rep2["solo_verificacion"] == len(imgs) and p2.store.entries() == []
    noisy = _panel(tmp_path / "c", oracle, ocr=MockOCR(oracle, noise=" ### basura ilegible ###"))
    noisy.tutors["florence"] = MockDetector(lambda img: {**oracle(img), "texto_visible": "otra cosa"})
    noisy.annotate(imgs)
    assert all("texto_visible" not in e["anotacion"] or len(e["anotacion"]["texto_visible"]) < 3
               for e in noisy.store.entries())


def test_tutor_mode_hint_retry_and_training_items(tmp_path):
    imgs, oracle = _dataset(8, seed=5)
    probs = [(img, q["pregunta"], str(q["respuesta"]), q.get("razonamiento", ""))
             for img in imgs for q in oracle(img)["preguntas"][:1]]
    calls = {}

    def hope(img, prompt):                       # Hope simulado: falla el primer intento, acierta con pista
        key = np.asarray(img).tobytes()
        calls[key] = calls.get(key, 0) + 1
        q = next(q for q in oracle(img)["preguntas"] if q["pregunta"] in prompt)
        return f"Respuesta: {q['respuesta']}" if "Pista del tutor" in prompt else "Respuesta: nada"

    leaky = MockVLM(oracle, leaky=True)
    ses = VisionTutorSession(hope, leaky)
    rep = ses.run(probs)
    assert rep["correctas_al_primero"] == 0 and rep["correctas_tras_pista"] == 1.0
    # la pista con la respuesta se reemplaza (respuestas de una letra no cuentan como filtración: tutor.py)
    assert rep["pistas_filtradas"] == sum(1 for p in probs if len(p[2]) >= 3 or p[2].isdigit())
    assert all(le.pista and le.respuesta not in le.pista for le in ses.lessons if len(le.respuesta) > 1)
    assert rep["muestras"] == 2 * len(probs)
    assert all(it.image.shape == (3, 64, 64) for it in ses.items)
    assert any("Intento anterior" in it.prompt for it in ses.items)


def test_problems_from_verified_store_and_config_loads(tmp_path):
    imgs, oracle = _dataset(4)
    panel = _panel(tmp_path, oracle)
    panel.annotate(imgs)
    probs = problems_from_store(panel.store)
    assert probs and all(len(p) == 4 for p in probs)
    cfg = Path(__file__).resolve().parents[1] / "configs" / "tutores_vision.yaml"
    real = load_panel(cfg, almacen=tmp_path / "real")
    assert set(real.tutors) == set(ETAPAS)
    assert all(t.info.train_ok for t in real.tutors.values())          # solo licencias que lo permiten
    assert real.tutors["principal"].info.licencia == "Apache-2.0"
    with pytest.raises(ValueError):
        TutorPanel({}, ImageStore(tmp_path / "z", 64))
