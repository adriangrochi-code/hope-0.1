"""Zona visual, crecimiento 200M→500M, datos imagen–texto y anotación por el maestro."""

import json
import random
from dataclasses import replace

import numpy as np
import torch

from hope.bench.plan import budget, cost, meta_model
from hope.core.hn1 import HN1Config, HN1Model, build_hn1, preset
from hope.data.vision import (ImageStore, ImageTextItem, ImageTextSource, bar_chart, encode_pair, procedural,
                              samples_from_annotation)
from hope.growth.hn1_ops import AddVisualZone, plan_growth_to
from hope.teacher.vision import (ImageAnnotator, MockVisionTeacher, _png, audit_teacher, parse_annotation,
                                 select_diverse)
from hope.train.continual import Mixture, Source, Trainer, TrainConfig


def tiny_vision(**kw):
    return replace(preset("hn1-tiny"), vision=True, vis_d=32, vis_layers=1, vis_heads=2, vis_image=32,
                   vis_patch=8, **kw)


def test_new_visual_zone_starts_neutral_and_tokens_are_compressed():
    torch.manual_seed(0)
    m = HN1Model(tiny_vision()).eval()
    x = torch.randint(0, 256, (2, 30))
    img = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    assert m.vision.encoder.tokens_per_image() == 4                  # (32/8/2)²
    assert m.encode_image(img).shape == (2, 4, m.cfg.d)
    assert torch.equal(m(x)["logits"], m(x, image=img)["logits"])  # salida en cero: no cambia nada
    for xa in m.vision.xattn.values():
        torch.nn.init.normal_(xa.out.weight, std=0.1)
    assert not torch.allclose(m(x)["logits"], m(x, image=img)["logits"])
    # otra resolución: las posiciones se interpolan
    assert m.encode_image(torch.randint(0, 256, (1, 3, 64, 48), dtype=torch.uint8)).shape == (1, 12, m.cfg.d)


def test_growth_200m_style_to_500m_style_preserves_function():
    torch.manual_seed(0)
    m, reg = build_hn1("hn1-tiny")
    m.eval()
    c = m.cfg
    target = HN1Config(**{**c.to_dict(), "pattern": list(c.pattern) + ["ssm", "attn"],
                          "zone_kinds": list(c.zone_kinds) * 2, "expert_hidden": 48, "vision": True,
                          "vis_d": 32, "vis_layers": 1, "vis_heads": 2, "vis_image": 32, "vis_patch": 8})
    x = torch.randint(0, 256, (2, 40))
    img = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    before = m(x)["logits"]
    ops = plan_growth_to(m, target)
    names = [type(o).__name__ for o in ops]
    assert names.count("AppendBackboneBlock") == 2 and names[-1] == "AddVisualZone"
    n0 = sum(p.numel() for p in m.parameters())
    added = 0
    for op in ops:
        planned = op.plan(m).param_delta
        rec = op.apply(m, reg)
        assert rec.preserving and rec.param_delta == planned
        added += planned
    assert sum(p.numel() for p in m.parameters()) == n0 + added
    assert torch.allclose(m(x)["logits"], before, atol=1e-5)
    assert torch.allclose(m(x, image=img)["logits"], before, atol=1e-5)
    assert reg.find_by_path("vision") is not None and m.cfg.vision and len(m.cfg.pattern) == len(c.pattern) + 2
    # la generación con imagen da lo mismo por los dos caminos
    assert m.generate("hola", 12, image=img[0]) == m.generate_reference("hola", 12, image=img[0])


def test_presets_500m_and_compute_plan():
    big, small = cost(meta_model("hn1-500m")), cost(meta_model("hn1-200m"))
    assert 430e6 < big.total_params <= 500e6
    assert 1.0 < big.flops_per_byte / small.flops_per_byte < 1.6     # crece el total, no tanto el cómputo
    assert big.tokens_per_image == 49 and big.flops_per_image > 0
    b = budget(big, 6, 12.0, image_share=0.3)
    assert b["bytes_texto"] > 0 and b["imagenes"] > 0
    with __import__("pytest").raises(ValueError):
        AddVisualZone().plan(HN1Model(tiny_vision()))


def test_procedural_images_and_samples():
    for arr, ann in procedural(6, seed=3, size=64):
        assert arr.shape == (64, 64, 3) and arr.dtype == np.uint8
        pairs = samples_from_annotation(ann)
        assert len(pairs) >= 3 and all(p.startswith("⟦imagen⟧") for p, _ in pairs)
    x, w = encode_pair("pregunta\n", "respuesta", 40)
    assert w.sum() == len("respuesta") + 1 and w[:9].sum() == 0


def _oracle_for(n, seed, size):
    truth = {_png(arr, size): ann for arr, ann in procedural(n, seed, size)}
    return lambda png: truth[png]


def test_annotator_dedup_cache_budget_and_teacher_audit(tmp_path):
    store = ImageStore(tmp_path / "vis", size=32)
    imgs = [arr for arr, _ in procedural(12, seed=1, size=64)]
    imgs.append(imgs[0].copy())                                        # duplicado exacto
    near = imgs[1].copy()
    near[0, 0] = 255 - near[0, 0]                                      # casi igual
    imgs.append(near)
    teacher = MockVisionTeacher()
    ann = ImageAnnotator(teacher, store, workers=2)
    rep = ann.annotate(imgs, budget=6)
    assert rep.annotated == 6 and rep.failed == 0
    assert rep.duplicates + rep.cached >= 1
    calls = teacher.calls
    rep2 = ImageAnnotator(teacher, store).annotate(imgs[:12], budget=None)
    assert rep2.cached >= 6 and teacher.calls - calls == rep2.annotated   # lo ya anotado no se vuelve a pedir
    src = ImageTextSource.from_store(store)
    img, x, w = src.batch(4, 64, np.random.default_rng(0))
    assert img.shape == (4, 3, 32, 32) and img.dtype == torch.uint8 and w.sum() > 0
    # verificación del maestro con imágenes cuya verdad se conoce
    good = audit_teacher(MockVisionTeacher(oracle=_oracle_for(10, 5, 64)), n=10, seed=5, size=64)
    bad = audit_teacher(MockVisionTeacher(oracle=_oracle_for(10, 5, 64), error_rate=1.0), n=10, seed=5, size=64)
    assert good["precision"] == 1.0 and bad["precision"] == 0.0
    assert parse_annotation('blabla {"descripcion_es": "x"} fin')["preguntas"] == []


def test_select_diverse_covers_clusters():
    rng = np.random.default_rng(0)
    feats = np.concatenate([rng.normal(c, 0.01, (50, 2)) for c in (0, 5, 10)])
    picked = select_diverse(feats, 3)
    assert {int(i) // 50 for i in picked} == {0, 1, 2}


def test_trainer_learns_to_use_the_image(tmp_path):
    torch.manual_seed(0)
    cfg = replace(preset("hn1-tiny"), d=64, heads=4, vision=True, vis_d=64, vis_layers=2, vis_heads=4,
                  vis_image=64, vis_patch=8)
    m = HN1Model(cfg)
    rng = random.Random(0)
    items = []
    for _ in range(300):
        a, ann = bar_chart(rng, 64)
        items.append(ImageTextItem(torch.from_numpy(a.copy()).permute(2, 0, 1).contiguous(),
                                   "⟦imagen⟧\nDatos:\n", json.dumps(ann["datos"], sort_keys=True)))
    src = ImageTextSource(items)
    tr = Trainer(m, TrainConfig(lr=2e-3, batch=16, warmup=20, precision="fp32"), device="cpu")
    text = Source.from_texts("t", ["hola mundo " * 50], 1.0)
    tr.train(Mixture([text]), 400, vision=src, vision_frac=0.9, vision_seq=40, eval_sources=[])
    img, x, w = src.batch(128, 40, np.random.default_rng(1))
    with torch.no_grad():
        right, _ = m.loss(x, image=img, weights=w)
        wrong, _ = m.loss(x, image=img[torch.randperm(len(img))], weights=w)
    assert right.item() < wrong.item() - 0.1, (right.item(), wrong.item())
    # el checkpoint de un modelo que creció se puede retomar
    m2, reg = build_hn1("hn1-tiny")
    for op in plan_growth_to(m2, replace(m2.cfg, vision=True, vis_d=32, vis_layers=1, vis_heads=2,
                                         vis_image=32, vis_patch=8, zone_kinds=m2.cfg.zone_kinds + ("mlp",))):
        op.apply(m2, reg)
    t2 = Trainer(m2, TrainConfig(precision="fp32"), device="cpu")
    t2.save(tmp_path / "ck")
    back = Trainer.resume(tmp_path / "ck", TrainConfig(precision="fp32"), device="cpu").model
    assert sum(p.numel() for p in back.parameters()) == sum(p.numel() for p in m2.parameters())
