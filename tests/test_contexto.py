"""Contexto largo con menos memoria: YaRN, caché comprimida, Quest, segmentos con caché, datos largos,
conversión de un modelo entrenado y banco de contexto (ARQUITECTURA_HN1 §3.9.2)."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from hope.core.hn1 import (HN1Model, long_context, patch_boundaries, preset, set_context_scale,
                           to_long_context)
from hope.core.kvcache import (KVOptions, attention_caches, configure_kv, dequantize, kv_bytes_per_position,
                               quantize, quest_select)
from hope.core.layers import CausalSelfAttention, rope_freqs, yarn_mscale

BASE = preset("hn1-tiny")
LARGO = long_context(BASE, local_window=32, global_window=8)
CFGS = {
    "base": BASE,
    "convergencia": replace(BASE, halting="convergencia"),
    "ventana_gqa": replace(BASE, pattern=("swa", "ssm", "attn"), window=8, sinks=2, kv_heads=2),
    "multibyte": replace(BASE, multibyte=2),
    "largo": LARGO,
}
PROMPT = "el gato come pescado y el perro come carne. " * 12


def _wake(m):
    with torch.no_grad():
        for mod in m.modules():
            out = getattr(mod, "out", None)
            if isinstance(out, torch.nn.Linear) and out.weight.abs().sum() == 0:
                torch.nn.init.normal_(out.weight, std=0.05)
        for z in m.regions:
            z.router.weight.normal_(0, 1.0)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    m = HN1Model(replace(cfg)).eval()       # copia: configure_kv y set_context_scale anotan en model.cfg
    _wake(m)
    return m


def _segments(m, x, seg):
    end = patch_boundaries(x, m.cfg.max_patch)[0]
    st, outs, a = m.new_segment_state(), [], 0
    while a < x.shape[1]:
        b = min(a + seg, x.shape[1])
        if b < x.shape[1]:
            b = a + int(end[0, a:b].nonzero()[-1]) + 1
        outs.append(m.head(m.segment_forward(x[:, a:b], end[:, a:b], st)["hidden"]))
        a = b
    return torch.cat(outs, 1)


# --- YaRN -----------------------------------------------------------------------------------
def test_yarn_frequencies_and_temperature():
    base = rope_freqs(64)
    y = rope_freqs(64, yarn=(8.0, 512.0))
    assert torch.allclose(y[:4], base[:4])                       # frecuencias altas: sin tocar
    assert torch.allclose(y[-1], base[-1] / 8)                   # las que no dan una vuelta: ÷ escala
    assert ((y <= base + 1e-12) & (y >= base / 8 - 1e-12)).all()
    assert yarn_mscale(1.0) == 1.0 and yarn_mscale(16.0) == pytest.approx((0.1 * math.log(16) + 1) ** 2)


def test_yarn_scale_is_selectable_adds_no_params_and_skips_windowed_layers():
    m = _model(replace(BASE, pattern=("swa", "ssm", "attn"), window=8))
    n = sum(p.numel() for p in m.parameters())
    x = torch.randint(0, 256, (1, 300))
    with torch.no_grad():
        ref = m(x)["logits"]
        set_context_scale(m, 4.0, "yarn", 64)
        out = m(x)["logits"]
        set_context_scale(m, 1.0)
        assert torch.equal(m(x)["logits"], ref)
    assert sum(p.numel() for p in m.parameters()) == n and not torch.allclose(out, ref)
    set_context_scale(m, 4.0, "yarn", 64)
    assert m.backbone[0].mixer.rope_yarn is None                    # ventana: no se escala
    assert m.backbone[2].mixer.rope_yarn == (4.0, 64 / 4.5) and m.backbone[2].mixer.q_scale > 1
    assert m.encoder[0].mixer.rope_yarn == (4.0, 64.0)              # parte local: largo en bytes
    m2 = HN1Model(replace(BASE, rope_scale=4.0, rope_kind="yarn"))  # elegible por configuración
    assert m2.backbone[1].mixer.rope_yarn is not None and m2.cfg.rope_kind == "yarn"
    with pytest.raises(ValueError):
        set_context_scale(m, 2.0, "otro")


# --- segmentos con caché (Transformer-XL) ------------------------------------------------------
@pytest.mark.parametrize("name", list(CFGS))
def test_segment_forward_matches_full_forward(name):
    m = _model(CFGS[name])
    x = torch.randint(97, 123, (1, 400))
    x[0, ::5] = 32
    with torch.no_grad():
        assert (_segments(m, x, 97) - m(x)["logits"]).abs().max() < 1e-4


def test_long_loss_cache_equals_full_loss_and_backward_per_segment():
    m = _model(BASE)
    x = torch.randint(97, 123, (2, 301))
    x[:, ::6] = 32
    with torch.no_grad():
        full, st_full = m.loss(x)
        seg, st_seg = m.long_loss(x, 64, memoria="cache")
    assert st_seg["segmentos"] >= 10 and st_seg["ce"] == pytest.approx(st_full["ce"], abs=1e-4)
    # backward por segmento (memoria de un segmento) = mismo gradiente que sumar todo y hacer backward
    m.zero_grad()
    loss, _ = m.long_loss(x, 64, memoria="cache")
    loss.backward()
    g1 = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    m.long_loss(x, 64, memoria="cache", backward_scale=1.0)
    for n, p in m.named_parameters():
        if n in g1:
            assert torch.allclose(p.grad, g1[n], atol=1e-5), n
    assert m.memory is not None
    _, st = m.long_loss(x, 64, memoria="ambas")
    assert math.isfinite(st["ce"])


def test_chunked_loss_matches_and_checkpointed_core():
    m = _model(replace(BASE, multibyte=2))
    x = torch.randint(0, 256, (2, 200))
    full, st = m.loss(x)
    m.loss_chunk = 37
    chunked, st2 = m.loss(x)
    assert st2["ce"] == pytest.approx(st["ce"], abs=1e-5) and st2["multibyte_ce"] == pytest.approx(st["multibyte_ce"], abs=1e-5)
    chunked.backward()
    m.train()
    m.grad_checkpointing = True
    loss, _ = m.loss(x, full_depth=True)
    loss.backward()
    assert m.core.block.mixer.qkv.weight.grad is not None


# --- generación ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(CFGS))
def test_chunked_prefill_and_windowed_local_generation_are_exact(name):
    m = _model(CFGS[name])
    ref = m.generate_reference(PROMPT, 30, max_context=4096)
    assert m.generate(PROMPT, 30, max_context=4096) == ref
    assert m.generate(PROMPT, 30, max_context=4096, prefill_chunk=100) == ref


def test_options_off_keep_plain_cache_and_eviction_is_exact():
    m = _model(BASE)
    assert all(getattr(mod, "kv_opts", None) is None for mod in m.modules())
    st = {}
    m.generate(PROMPT, 5, max_context=4096, kv_stats=st)
    assert st["kv_local"] > 0 and st["kv_global"] > 0
    m = _model(LARGO)                                          # ventana local + descarte
    ref = m.generate_reference(PROMPT, 30, max_context=4096)
    st = {}
    assert m.generate(PROMPT, 30, max_context=4096, kv_stats=st) == ref
    # la parte local guarda a lo sumo ventana + sinks posiciones por capa
    lw = LARGO.local_window + LARGO.sinks
    per = kv_bytes_per_position(LARGO.local_heads, LARGO.d_local // LARGO.local_heads, dtype_bytes=4)
    assert st["kv_local"] <= (LARGO.enc_layers + LARGO.dec_layers) * lw * per


def test_quantized_cache_close_and_smaller():
    m = _model(BASE)
    x = torch.randn(1, 2, 50, 16)
    for bits, tol in ((8, 0.02), (4, 0.2)):
        parts = quantize(x, bits, 8)
        y = dequantize(parts, bits, 8, 16, torch.float32)
        assert (y - x).abs().max() / x.abs().max() < tol
    sizes = {}
    for bits in (16, 8, 4):
        configure_kv(m, bits=bits)
        st = {}
        out = m.generate(PROMPT, 20, max_context=4096, kv_stats=st)
        sizes[bits] = st["kv_local"] + st["kv_global"] + st["kv_nucleo"]
        if bits == 8:
            configure_kv(m)
            assert sum(a == b for a, b in zip(out, m.generate(PROMPT, 20, max_context=4096))) >= 15
    assert sizes[8] < 0.3 * sizes[16] and sizes[4] < 0.7 * sizes[8]      # fp32 → int8 → int4 (CPU)
    with pytest.raises(ValueError):
        KVOptions(bits=3)


def test_measured_cache_bytes_match_formula():
    m = _model(LARGO)
    for bits in (16, 8, 4):
        configure_kv(m, bits=bits, evict=True)
        state = m.new_segment_state()
        x = torch.randint(97, 123, (1, 200))
        x[0, ::5] = 32
        with torch.no_grad():
            m.segment_forward(x, patch_boundaries(x, m.cfg.max_patch)[0], state)
        for c in attention_caches(state):
            stored = c["kq"].shape[2]
            dh = c["kq"].shape[3] * (2 if bits == 4 else 1)
            kvh = c["kq"].shape[1]
            got = sum(v.numel() * v.element_size() for k, v in c.items() if k not in ("jpos", "n"))
            assert got == stored * kv_bytes_per_position(kvh, dh, bits, LARGO.kv_group, dtype_bytes=4)


def test_quest_bound_selection_and_exactness():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 64, 8)
    cache = {}
    from hope.core.kvcache import page_bounds_update
    page_bounds_update(cache, k, 0, 8)
    # la cota de cada página es ≥ el mayor q·k real de la página (cabezas de un mismo grupo)
    qg = q[0, :, 0].view(2, 2, 8)
    real = torch.einsum("grd,gtd->grt", qg, k[0]).view(2, 2, 8, 8).amax(-1).amax(1)    # [Hkv, P]
    bound = torch.maximum(qg[:, :, None] * cache["pmin"][0][:, None], qg[:, :, None] * cache["pmax"][0][:, None]).sum(-1).amax(1)
    assert (bound >= real - 1e-5).all()
    tok, valid = quest_select(q, cache["pmin"], cache["pmax"], 60, 8, 3)
    assert tok.shape == (1, 2, 24) and not valid[0, 0, -4:].any()            # página final parcial
    assert (tok[0, :, :8] == torch.arange(8)).all()                           # primera página siempre
    # integrado a generate: con k páginas que cubren todo, exacto; con pocas, sigue generando
    m = _model(BASE)
    ref = m.generate_reference(PROMPT, 30, max_context=4096)
    configure_kv(m, quest_pages=1000, page=4)
    assert m.generate(PROMPT, 30, max_context=4096) == ref
    configure_kv(m, quest_pages=3, page=4)
    assert len(m.generate(PROMPT, 30, max_context=4096)) == 30
    from hope.core.batched import generate_batch
    from hope.core.speculative import generate_speculative
    out = m.generate(PROMPT, 10, max_context=4096)                          # lote y especulativa: sin cambiar
    assert generate_batch(m, [PROMPT], 10, max_context=4096)[0] == out
    assert generate_speculative(m, PROMPT, 10, max_context=4096) == out


# --- conversión de un modelo entrenado --------------------------------------------------------
def test_to_long_context_converts_trained_model():
    torch.manual_seed(0)
    m = HN1Model(BASE).eval()
    _wake(m)
    # cabezas de clave/valor iguales dentro de cada grupo: la conversión GQA no cambia nada
    for blk in [*m.backbone, m.core.block]:
        mod = blk.mixer
        if isinstance(mod, CausalSelfAttention):
            w = mod.qkv.weight.data
            h, dh = mod.heads, mod.dh
            for part in (slice(h * dh, 2 * h * dh), slice(2 * h * dh, 3 * h * dh)):
                blockw = w[part].view(h, dh, -1)
                w[part] = blockw[::4].repeat_interleave(4, 0).reshape(h * dh, -1)
    x = torch.randint(97, 123, (1, 30))
    with torch.no_grad():
        ref = m(x)["logits"]
    n0 = sum(p.numel() for p in m.parameters())
    info = to_long_context(m, kv_heads=1, local_window=64, global_window=64)
    with torch.no_grad():
        assert (m(x)["logits"] - ref).abs().max() < 1e-4
    assert info["parametros_despues"] < n0 and m.cfg.local_kind == "swa" and m.cfg.kv_heads == 1
    assert HN1Model(m.cfg).load_state_dict(m.state_dict()) is not None        # el checkpoint recarga
    assert preset("hn1-200m-largo").kv_heads == 4 and "swa" in preset("hn1-500m-largo").pattern


# --- datos largos ---------------------------------------------------------------------------
def test_long_docs_build_and_sampling(tmp_path):
    from hope.data.largos import LongDocSource, build_long, group_repos, iter_docs, source_for
    src = tmp_path / "src"
    src.mkdir()
    docs = [("⟦repo: a⟧ ⟦archivo: x.py⟧\n" + "a" * 300).encode(), ("⟦repo: a⟧ ⟦archivo: y.py⟧\n" + "b" * 300).encode(),
            ("⟦repo: b⟧ ⟦archivo: z.py⟧\n" + "c" * 100).encode(), b"corto"]
    (src / "train.bin").write_bytes(b"\x00".join(docs) + b"\x00")
    assert list(iter_docs(src / "train.bin", block=7)) == docs
    grouped = list(group_repos(iter_docs(src / "train.bin")))
    assert len(grouped) == 3 and b"x.py" in grouped[0] and b"y.py" in grouped[0]
    summ = build_long(tmp_path / "out", "repos", {"train": group_repos(iter_docs(src / "train.bin"))}, min_bytes=200)
    assert summ["train"]["docs"] == 1 and summ["train"]["cortos_descartados"] == 2
    s = source_for("repos", tmp_path / "out" / "repos", 1.0)
    assert isinstance(s, LongDocSource)
    x = s.sample(4, 128, np.random.default_rng(0))
    assert x.shape == (4, 128) and (x != 0).all()                            # dentro del documento


# --- banco de contexto ------------------------------------------------------------------------
def test_bench_examples_and_scoring(tmp_path):
    from hope.eval.contexto_bench import (estimate, make_examples, max_context, memory_table, rank,
                                          write_training_data)
    exs = make_examples(lengths=(600, 1200), depths=(0.1, 0.9), n=1)
    assert len(exs) == 3 * 2 * 2
    for e in exs:
        assert abs(len(e.contexto.encode()) - e.largo) < 120 and e.respuesta in e.contexto
        assert e.respuesta not in e.distractores and len(e.distractores) == 3
        if e.tarea == "aguja":
            body = e.contexto.replace(f"La clave secreta es {e.respuesta}.", "")
            assert not any(ch.isdigit() for ch in body)
    deep = [e for e in exs if e.tarea == "aguja" and e.profundidad == 0.9][0]
    assert deep.contexto.index(deep.respuesta) > 0.7 * len(deep.contexto)
    m = _model(LARGO)
    e = exs[0]
    r_full, r_cache = rank(m, e, "completo"), rank(m, e, "cache", segment=200)
    assert r_full["logp"] == pytest.approx(r_cache["logp"], abs=1e-3)
    for mode in ("cache-int8", "resumen", "ventana"):
        assert rank(m, e, mode, segment=200)["ok"] in (0, 1)
    summ = write_training_data(tmp_path / "sint", 20_000, lengths=(600,))
    assert summ["train"]["docs"] > 10
    cfg = preset("hn1-200m-largo")
    e1, e2 = estimate(cfg, 32_768), estimate(preset("hn1-200m"), 32_768)
    assert e1["kv_por_byte"] < e2["kv_por_byte"] / 10
    assert max_context(cfg, 16e9 * 0.9, n_params=176_000_000) > 131_072
    assert len(memory_table(presets=("hn1-tiny",), kv=(16,), contexts=(4096,))) == 1


def test_trainer_segment_cache_and_plan_stages(tmp_path):
    import json

    import yaml

    from hope.train.continual import Mixture, Source, Trainer, TrainConfig
    from hope.train.run import run
    text = Source.from_texts("t", ["el río fluye hacia el mar. " * 200], 1.0)
    tr = Trainer(HN1Model(BASE), TrainConfig(batch=2, seq_bytes=256, segment_bytes=64, precision="fp32",
                                             segment_memory="cache"), device="cpu")
    rep = tr.train(Mixture([text]), 3, eval_sources=[text])
    assert rep.steps == 3 and all(math.isfinite(v) for v in rep.bpb_after.values())
    data = tmp_path / "d"
    data.mkdir()
    for split in ("train", "val"):
        (data / f"{split}.bin").write_bytes(("el sol sale por el este. " * 400).encode())
    plan = {"salida": str(tmp_path / "run"), "precision": "fp32", "secuencia": 64, "warmup": 2,
            "compilar": False, "checkpointing": True, "checkpoint_min": 0.01,
            "etapas": [{"nombre": "a", "preset": "hn1-tiny", "horas": 0.0002, "lote": 2, "fuentes": {"t": f"{data}:1.0"}},
                       {"nombre": "opcional", "activa": False, "horas": 1, "fuentes": {"t": f"{data}:1.0"}},
                       {"nombre": "convertir", "convertir_contexto": {"kv_heads": 2, "local_window": 64,
                                                                      "global_window": 16}},
                       {"nombre": "largo", "horas": 0.0003, "lote": 2, "secuencia": 256, "segmento": 64,
                        "memoria_segmentos": "cache", "rope": {"tipo": "yarn", "escala": 4, "bytes_entrenados": 64},
                        "fuentes": {"t": f"{data}:1.0"}}]}
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))
    run(str(tmp_path / "plan.yaml"), device="cpu")
    st = json.loads((tmp_path / "run" / "estado.json").read_text())
    assert st["etapas"]["largo"]["hecha"] and not st["etapas"]["opcional"]["hecha"]
    tr2 = Trainer.resume(tmp_path / "run" / "largo", TrainConfig(precision="fp32"), device="cpu")
    assert tr2.model.cfg.kv_heads == 2 and tr2.model.cfg.rope_kind == "yarn" and tr2.model.cfg.rope_scale == 4
