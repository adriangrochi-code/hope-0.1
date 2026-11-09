"""Mejoras sin parámetros: global incremental, generación especulativa, multi-byte, Muon, WSD, calidad."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from hope.core.genome import Genome
from hope.core.hn1 import HN1Model, build_registry, preset, stable_loop
from hope.core.speculative import SpecStats, generate_speculative, ngram_draft
from hope.data.quality import QualityClassifier, filter_docs, rules
from hope.growth.ops import AddExpert
from hope.plasticity.dynamic import AddBackboneConnection
from hope.train.continual import Mixture, Source, Trainer, TrainConfig

BASE = preset("hn1-tiny")
CONFIGS = {
    "base": BASE,
    "ssm": replace(BASE, pattern=("ssm", "ssm", "attn", "ssm")),
    "convergencia": replace(BASE, halting="convergencia"),
    "ventana": replace(BASE, pattern=("swa", "ssm", "attn"), window=8, sinks=2, kv_heads=2),
    "sesgo": replace(BASE, balance="sesgo", shared_expert=True, top_k=3),
    "vision": replace(BASE, vision=True, vis_d=32, vis_layers=1, vis_heads=2, vis_image=32, vis_patch=8),
    "multibyte": replace(BASE, multibyte=2),
    # bucle estable: reinyección de la entrada, sin step_embed, más iteraciones que al entrenar
    "bucle_estable": stable_loop(BASE, mean_iters=2.0, train_max=4, infer_max=6, tol=1e-3),
    "bucle_senoidal": replace(BASE, core_inject="convexa", core_step="senoidal"),
    # parches = tokens de Qwen (regla causal; tokenizador chico guardado, sin red)
    "tokens": replace(BASE, patching="tokens", tokenizer=str(Path(__file__).parent / "data" / "tokenizador_mini.json")),
}


def _wake(m):
    """Saca de cero las salidas que arrancan en cero, para que todo influya en la prueba."""
    with torch.no_grad():
        for mod in m.modules():
            out = getattr(mod, "out", None)
            if isinstance(out, torch.nn.Linear) and out.weight.abs().sum() == 0:
                torch.nn.init.normal_(out.weight, std=0.05)
        for z in m.regions:
            z.router.weight.normal_(0, 1.0)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_incremental_global_and_speculative_match_reference(name):
    torch.manual_seed(0)
    cfg = CONFIGS[name]
    m = HN1Model(cfg).eval()
    _wake(m)
    if name == "base":                              # conectoma dinámico + especialista por genoma
        AddBackboneConnection(0, 1).apply(m, build_registry(m))
        for c in m.connections.values():
            c.gate.data.fill_(0.5)
        AddExpert(0, genome=Genome(("conv", "scan", "gelu"), 16)).apply(m, build_registry(m))
        m.regions[0].expert_mask.zero_()
        for e in m.regions[0].experts:
            if hasattr(e, "fc2"):
                torch.nn.init.normal_(e.fc2.weight, std=0.1)
    # la parte global incremental (parche por parche) da lo mismo que recalcular todo
    patches = torch.randn(1, 30, cfg.d_local)
    with torch.no_grad():
        full = m._global(patches)[0]
        cache = {}
        inc = torch.cat([m._global(patches[:, i:i + 1], cache=cache)[0] for i in range(30)], dim=1)
    assert (full - inc).abs().max() < 1e-4
    img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8) if cfg.vision else None
    mem = torch.randn(3, cfg.d)
    for prompt in ["hola mundo, esto es una prueba. ", "x = 1\ny = 2\nz = x + y\n" * 3]:
        ref = m.generate_reference(prompt, 40, image=img, memory=mem)
        assert m.generate(prompt, 40, image=img, memory=mem) == ref
        st = SpecStats()
        assert generate_speculative(m, prompt, 40, image=img, memory=mem, stats=st) == ref
        assert st.generated == len(ref) and st.passes <= len(ref) + 1      # +1: la pasada del byte de parada


def test_ngram_draft():
    seq = list(b"abc def abc de")
    assert bytes(ngram_draft(seq, 3)) == b"f a"
    assert ngram_draft(list(b"xyz"), 3) == []
    assert bytes(ngram_draft(list(b"zz qq"), 2, reference=b"qq rr")) == b" r"


def test_multibyte_heads_train_and_add_few_params():
    torch.manual_seed(0)
    plain, mb = HN1Model(BASE), HN1Model(replace(BASE, multibyte=3))
    extra = sum(p.numel() for p in mb.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 3 * (BASE.d_local * BASE.d_local + BASE.d_local + BASE.d_local)
    x = torch.randint(0, 256, (2, 50))
    loss, st = mb.loss(x)
    assert st["multibyte_ce"] > 0
    loss.backward()
    assert mb.mb_heads[0][1].weight.grad is not None


@pytest.mark.parametrize("opt", ["adamw", "muon"])
def test_optimizers_and_wsd_schedule(opt):
    torch.manual_seed(0)
    text = Source.from_texts("t", ["el gato come pescado. el perro come carne. " * 30], 1.0)
    tr = Trainer(HN1Model(BASE), TrainConfig(lr=2e-3, warmup=5, batch=8, seq_bytes=64, precision="fp32",
                                             optimizer=opt, schedule="wsd"), device="cpu")
    before = tr.bpb(text, windows=8, seq=64)
    tr.train(Mixture([text]), 60, eval_sources=[])
    assert tr.opt.param_groups[0]["lr"] == pytest.approx(2e-3)       # fase estable: tasa constante
    tr.anneal(Mixture([text]), 20, eval_sources=[])
    assert tr.opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)   # enfriamiento completo
    assert tr.bpb(text, windows=8, seq=64) < before - 1.0


def test_muon_in_low_precision_and_rebuild():
    from hope.growth.ops import rebuild_optimizer
    torch.manual_seed(0)
    tr = Trainer(HN1Model(BASE), TrainConfig(precision="bf16", optimizer="muon", batch=4, seq_bytes=32),
                 device="cpu")
    text = Source.from_texts("t", ["abc " * 100], 1.0)
    tr.train(Mixture([text]), 5, eval_sources=[])
    assert next(tr.model.parameters()).dtype == torch.bfloat16
    new = rebuild_optimizer(tr.opt, tr.model)
    assert type(new).__name__ == "Muon"


def test_weight_ema_in_trainer(tmp_path):
    torch.manual_seed(0)
    text = Source.from_texts("t", ["el gato come pescado. el perro come carne. " * 30], 1.0)
    tr = Trainer(HN1Model(stable_loop(BASE, train_max=4, infer_max=6)),
                 TrainConfig(lr=2e-3, warmup=5, batch=4, seq_bytes=64, precision="bf16", ema=0.9), device="cpu")
    tr.train(Mixture([text]), 30, eval_sources=[])
    assert tr.ema_updates == 30
    name, p = next((n, q) for n, q in tr.model.named_parameters() if q.dim() == 2)
    assert tr.ema[name].dtype == torch.float32                     # copia fp32 aunque los pesos sean bf16
    assert not torch.allclose(tr.ema[name], p.float())
    trained = p.detach().clone()
    with tr.ema_weights():
        assert torch.allclose(p.float(), tr.ema[name].to(p.dtype).float())
        ema_bpb = tr.bpb(text, windows=4, seq=64)
    assert torch.equal(p, trained)                                 # al salir vuelven los pesos entrenados
    assert ema_bpb < 8.0
    tr.save(tmp_path / "ck")
    again = Trainer.resume(tmp_path / "ck", TrainConfig(precision="bf16", ema=0.9), device="cpu")
    assert again.ema_updates == 30 and torch.allclose(again.ema[name], tr.ema[name])


def test_quality_rules_and_classifier():
    good = ("La ley de Ohm establece que la tensión en un resistor es igual a la corriente por la "
            "resistencia. Con 2 A y 10 ohm la tensión es 20 V. Se verifica midiendo con un multímetro. "
            "La potencia disipada es el producto de la tensión por la corriente, en este caso 40 W.")
    spam = "click here buy now cheap cheap cheap!!! $$$ " * 20
    code = "\n".join(f"def f{i}(a, b):\n    return a * {i} + b" for i in range(12))
    assert rules(good).keep and not rules(spam).keep and rules(code).keep and rules(code).is_code
    rng = np.random.default_rng(0)
    goods = [good.replace("2 A", f"{i} A") for i in range(20)]
    bads = ["comprá ya oferta oferta descuento gratis ganá premio " * 5 + str(i) for i in range(20)]
    clf = QualityClassifier(dim=4096)
    clf.fit(goods + bads, [4.0] * 20 + [0.5] * 20)
    s = clf.score([good, bads[0]])
    assert s[0] > s[1] + 1.0
    kept = [d for d, _ in filter_docs(goods[:5] + [spam], clf, threshold=2.0)]
    assert len(kept) == 5 and filter_docs.last_report.rule_drops
    del rng


def test_fast_tools_checkpointing_accum_int8_compile(tmp_path):
    from hope.core.fast import compile_model, find_batch_size, quantize_int8
    torch.manual_seed(0)
    text = Source.from_texts("t", ["la casa es azul y el cielo también. " * 40], 1.0)
    # checkpointing + acumulación: mismos gradientes que sin ellos (lote efectivo igual)
    m1 = HN1Model(BASE)
    m2 = HN1Model(BASE)
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, 256, (4, 40))
    m1.train()
    m2.train()
    m2.grad_checkpointing = True
    l1, _ = m1.loss(x, full_depth=True)
    l2, _ = m2.loss(x, full_depth=True)
    l1.backward()
    l2.backward()
    g1 = torch.cat([p.grad.flatten() for p in m1.parameters() if p.grad is not None])
    g2 = torch.cat([p.grad.flatten() for p in m2.parameters() if p.grad is not None])
    assert torch.allclose(g1, g2, atol=1e-5)
    tr = Trainer(HN1Model(BASE), TrainConfig(batch=2, grad_accum=3, seq_bytes=32, precision="fp32",
                                             checkpointing=True), device="cpu")
    tr.train(Mixture([text]), 3, eval_sources=[])
    assert tr.step == 3
    # int8: mitad de memoria que bf16 / un cuarto que fp32, salida casi igual
    m = HN1Model(BASE).eval()
    ref = m(x)["logits"]
    info = quantize_int8(m)
    assert info["capas"] > 10 and info["bytes_despues"] < info["bytes_antes"] / 3
    assert (m(x)["logits"] - ref).abs().max() < 0.1
    # compilación: si no está disponible en el backend, el modelo sigue funcionando
    m3 = HN1Model(BASE).eval()
    compile_model(m3)
    tr3 = Trainer(m3, TrainConfig(precision="fp32"), device="cpu")
    tr3.save(tmp_path / "ck")                     # guardar con bloques compilados no falla
    calls = []

    def step(b):
        calls.append(b)
        if b > 16:
            raise RuntimeError("XPU out of memory")
    assert find_batch_size(step, start=4) == 16 and calls == [4, 8, 16, 32]


def test_long_run_program_resumes_and_asks_before_growing(tmp_path):
    import json

    import yaml

    from guardian.core import Guardian, GuardianConfig
    from hope.train.run import run
    data = tmp_path / "d"
    data.mkdir()
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (data / f"{split}.bin").write_bytes(("el sol sale por el este. " * 400).encode())
    budget = tmp_path / "budget.yaml"
    budget.write_text("max_total_params: 100000000\nmode: USER_APPROVAL_REQUIRED\n"
                      "consolidation: {max_auto_param_fraction: 0.10}\n")
    gcfg = {"workdir": str(tmp_path), "protected_paths": [], "budget_path": str(budget),
            "audit_path": str(tmp_path / "audit.jsonl"), "approvals_path": str(tmp_path / "ap.json")}
    (tmp_path / "g.yaml").write_text(yaml.safe_dump(gcfg))
    plan = {"salida": str(tmp_path / "run"), "precision": "fp32", "secuencia": 64, "warmup": 2,
            "compilar": False, "checkpointing": False, "checkpoint_min": 0.01, "guardian": str(tmp_path / "g.yaml"),
            "etapas": [{"nombre": "a", "preset": "hn1-tiny", "horas": 0.0004, "lote": 4,
                        "fuentes": {"t": f"{data}:1.0"}},
                       {"nombre": "b", "crecer_a": {"pattern": ["ssm", "attn", "ssm"],
                                                    "zone_kinds": list(BASE.zone_kinds) + ["mlp"]}},
                       {"nombre": "c", "horas": 0.0003, "lote": 4, "enfriar": True,
                        "fuentes": {"t": f"{data}:1.0"}}]}
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))
    run(str(tmp_path / "plan.yaml"), device="cpu")                     # para en el crecimiento
    st = json.loads((tmp_path / "run" / "estado.json").read_text())
    assert st["etapas"]["a"]["hecha"] and not st["etapas"]["b"]["hecha"]
    req = st["etapas"]["b"]["solicitud"]
    Guardian(GuardianConfig(**gcfg)).approve(req, "prueba")
    st["etapas"]["b"]["aprobacion"] = req
    (tmp_path / "run" / "estado.json").write_text(json.dumps(st))
    run(str(tmp_path / "plan.yaml"), device="cpu")                     # crece, enfría y termina
    st = json.loads((tmp_path / "run" / "estado.json").read_text())
    assert all(e["hecha"] for e in st["etapas"].values())
    log = [json.loads(x) for x in (tmp_path / "run" / "registro.jsonl").read_text().splitlines()]
    assert any("crecimiento" in r for r in log) and log[-1]["etapa"] == "c"
    assert (tmp_path / "run" / "estable_antes_de_enfriar" / "model.pt").exists()
    del rng
