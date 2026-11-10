"""Calidad en imágenes: generadores verificables, banco sellado, mosaicos, contrastiva, currículo
y respuestas que llaman herramientas (ARQUITECTURA_HN1 §3.10, §10.1.4)."""

import random
import re
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from hope.core.hn1 import HN1Model, preset
from hope.data.vision import (DOC_KINDS, GENERATORS, LEVELS, RES_CLOSE, RES_OPEN, RESERVED_SEEDS, ImageTextSource,
                              augment, circuit_image, encode_pair, procedural, samples_from_annotation, strip_results)
from hope.eval import vision_bench as vb
from hope.runtime.toolcalls import CALL_RE, ToolRunner
from hope.tools.builtin import default_catalog
from hope.tools.eng_tools import circuit


def tiny(**kw):
    return replace(preset("hn1-tiny"), vision=True, vis_d=32, vis_layers=1, vis_heads=2, vis_image=32,
                   vis_patch=8, **kw)


def test_every_generator_gives_exact_tasks_and_scales():
    for kind, gen in GENERATORS.items():
        for size in (64, 224):
            for sd in range(6):
                arr, ann = gen(random.Random(sd), size)
                assert arr.shape == (size, size, 3) and arr.dtype == np.uint8, kind
                qs = ann["preguntas"]
                assert (qs or ann.get("texto_visible")) and all(q.get("tarea") in vb.TASKS for q in qs), (kind, qs)
                assert all(str(q["respuesta"]).strip() for q in qs)
                assert len(samples_from_annotation(ann)) >= 2
        assert kind in LEVELS


def test_circuit_answers_match_the_simulator():
    for sd in range(25):
        _, ann = circuit_image(random.Random(sd), 128)
        req = float(ann["preguntas"][0]["respuesta"])
        sim = circuit("V1 a 0 1000; " + ann["datos"]["netlist"])
        assert abs(1000 / sim["corrientes_fuentes"]["V1"] - req) <= 0.006 * req + 0.01
        cur = ann["preguntas"][1]
        v = float(re.search(r"aplican (\d+) V", cur["pregunta"]).group(1))
        assert abs(float(cur["respuesta"]) - v / req * 1000) <= 0.006 * float(cur["respuesta"]) + 0.01


def test_tool_calls_in_answers_are_real_and_their_results_are_not_trained():
    runner = ToolRunner(default_catalog())
    seen = Counter()
    for arr, ann in procedural(60, seed=3, size=64, kinds=tuple(GENERATORS)):
        for prompt, answer in samples_from_annotation(ann):
            for m in re.finditer(f"(⟦[^⟧]*⟧){RES_OPEN}(.*?){RES_CLOSE}", answer, re.DOTALL):
                call = CALL_RE.fullmatch(m.group(1))
                seen[call.group(1)] += 1
                assert m.group(2) == " → " + runner.run(call.group(1), call.group(2))   # igual que el runtime
            x, w = encode_pair(prompt, answer, 2048)
            trained = bytes(int(b) for b, wi in zip(x, w) if wi).decode("utf-8")
            assert " → " not in trained and RES_OPEN not in trained
            assert trained.rstrip("\x00") == re.sub(f"{RES_OPEN}.*?{RES_CLOSE}", "", answer, flags=re.DOTALL)
    assert seen["calc"] > 5 and seen["circuito"] > 0


def test_reserved_seeds_documents_and_augment():
    with pytest.raises(ValueError):
        next(procedural(1, seed=RESERVED_SEEDS.start + 5))
    sizes = {k: a.shape[0] for a, k in ((a, ann["generador"]) for a, ann in
                                       procedural(40, 1, 32, tuple(GENERATORS), doc_scale=2))}
    assert all(s == (64 if k in DOC_KINDS else 32) for k, s in sizes.items())
    arr, _ = GENERATORS["formas"](random.Random(0), 64)
    out = augment(arr, random.Random(1))
    assert out.shape == arr.shape and out.dtype == np.uint8


def test_source_groups_sizes_downscales_without_tiles_and_curriculum():
    src = ImageTextSource.from_procedural(40, seed=2, size=32, kinds=tuple(GENERATORS), doc_scale=2, tiles=True)
    rng = np.random.default_rng(0)
    shapes = {src.batch(4, 64, rng)[0].shape[-1] for _ in range(40)}
    assert shapes == {32, 64}                                         # nunca mezcla tamaños en un lote
    flat = ImageTextSource.from_procedural(40, seed=2, size=32, kinds=tuple(GENERATORS), doc_scale=2, tiles=False)
    assert {it.image.shape[-1] for it in flat.items} == {32}
    cur = ImageTextSource.from_procedural(60, seed=2, size=32, kinds=tuple(GENERATORS)).curriculum(100, warm=0.6)
    lv = {it.prompt: it.level for it in cur.items}
    assert cur.max_level() == 1
    for _ in range(5):
        cur.batch(4, 64, rng)
    assert cur.max_level() == 1
    for _ in range(95):
        cur.batch(4, 64, rng)
    assert cur.max_level() == 4 and max(lv.values()) == 4


def test_tiling_has_no_new_params_keeps_function_and_sees_more():
    torch.manual_seed(0)
    m = HN1Model(tiny(vis_tiling=True)).eval()
    flat = HN1Model(tiny())
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in flat.parameters())
    v = m.vision.encoder.tokens_per_image()
    doc = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    assert m.encode_image(doc).shape == (2, 5 * v, m.cfg.d)                    # miniatura + 2×2 mosaicos
    assert m.encode_image(doc[:, :, :32, :32]).shape == (2, v, m.cfg.d)       # imagen normal: igual que antes
    assert m.encode_image(torch.randint(0, 256, (1, 3, 64, 48), dtype=torch.uint8)).shape[1] == 12   # interpola
    x = torch.randint(0, 256, (2, 30))
    assert torch.equal(m(x)["logits"], m(x, image=doc)["logits"])            # sigue empezando neutra
    # los mosaicos llevan la posición de la imagen entera: dos mosaicos iguales dan fichas distintas
    same = torch.full((1, 3, 64, 64), 128, dtype=torch.uint8)
    tok = m.encode_image(same)[0, v:].view(4, v, -1)
    assert not torch.allclose(tok[0], tok[3])


def test_contrastive_loss_trains_the_encoder_without_new_params():
    torch.manual_seed(0)
    m = HN1Model(tiny())
    n0 = sum(p.numel() for p in m.parameters())
    x = torch.randint(0, 256, (4, 40))
    w = torch.zeros(4, 40)
    w[:, 20:] = 1
    img = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    base, st0 = m.loss(x, image=img, weights=w)
    assert st0["contrastiva"] == 0.0
    loss, st = m.loss(x, image=img, weights=w, contrastive=0.5)
    assert st["contrastiva"] > 0 and loss.item() > base.item()
    m.zero_grad()
    m.contrastive_loss(x, m.encode_image(img), w).backward()
    assert m.vision.encoder.stem.weight.grad.abs().sum() > 0
    assert sum(p.numel() for p in m.parameters()) == n0
    # respuestas iguales cuentan como positivas: con todas iguales la pérdida es menor que con todas distintas
    xs = x.clone()
    xs[:, 20:] = x[0, 20:]
    zi = m.encode_image(img)
    assert m.contrastive_loss(xs, zi, w).item() != m.contrastive_loss(x, zi, w).item()


def test_bench_is_balanced_deterministic_and_scores():
    a = vb.build(8, 32)
    b = vb.build(8, 32)
    assert Counter(i.tarea for i in a) == dict.fromkeys(vb.TASKS, 8)
    assert [i.answer for i in a] == [i.answer for i in b] and torch.equal(a[3].image, b[3].image)
    with pytest.raises(ValueError):
        vb.build(2, 32, seed=1)
    train = ImageTextSource.from_procedural(80, seed=0, size=32, kinds=tuple(GENERATORS), doc_scale=2)
    assert vb.contamination(a, (it.image for it in train.items)) == 0
    assert vb.contamination(a, [a[0].image]) == 1
    it = vb.BenchItem(None, "", "23.81", "", "razonamiento", "x", tol_rel=0.01)
    assert vb.score(it, vb.extract_answer("leo…\nRespuesta: 23.9 mA")) == 1.0
    assert vb.score(it, vb.extract_answer("Respuesta: 25")) == 0.0
    clock = vb.BenchItem(None, "", "6:45", "", "valores", "reloj")
    assert vb.score(clock, vb.extract_answer("La corta…\nRespuesta: 6:45.")) == 1.0
    txt = vb.BenchItem(None, "", "hola mundo", "", "texto", "texto_transcribir")
    assert vb.score(txt, "hola mundo") == 1.0 and 0.85 < vb.score(txt, "hola mumdo") < 0.95
    assert vb.extract_answer(f"⟦calc: 2 + 2⟧{RES_OPEN} → 4{RES_CLOSE}.\nRespuesta: 4") == "4"
    assert strip_results(f"a{RES_OPEN}b{RES_CLOSE}") == "ab"


def test_bench_sealed_roundtrip_and_evaluate(tmp_path):
    from hope.eval.sealed import SealError, SealedRegistry
    items = vb.build(3, 32)
    path = vb.save(items, tmp_path / "banco.pt")
    reg = SealedRegistry(tmp_path / "sellado")
    reg.seal("banco", path)
    back = vb.load(path, registry=reg, name="banco")
    assert [i.reference for i in back] == [i.reference for i in items]
    torch.manual_seed(0)
    m = HN1Model(tiny(vis_tiling=True))
    rep = vb.evaluate(m, back, tools=True, max_new=24)
    assert set(rep["tareas"]) == set(vb.TASKS) and rep["preguntas"] == len(items)
    assert all(np.isfinite(v["bpb"]) for v in rep["tareas"].values())
    assert "| contar |" in vb.table(rep)
    path.write_bytes(path.read_bytes() + b"x")                            # retocar el banco
    with pytest.raises(SealError):
        vb.load(path, registry=reg, name="banco")
