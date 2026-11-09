"""Parches = tokens de Qwen (``hope/core/tokpatch.py``) y destilación (``hope/train/distill.py``).

Sin red: usa ``tests/data/tokenizador_mini.json`` (BPE a nivel de bytes de 768 tokens con la misma
pre-tokenización que Qwen2.5). La prueba con el tokenizador real de Qwen corre solo si ya está en
la caché (``~/.cache/hope/tokenizers``)."""

import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from hope.core.batched import Conversation, generate_batch
from hope.core.hn1 import HN1Model, preset
from hope.core.speculative import SpecStats, generate_speculative
from hope.core.tokpatch import TokenPatcher, _complete_utf8, cache_dir

MINI = str(Path(__file__).parent / "data" / "tokenizador_mini.json")
TEXTS = [
    "El desarrollo de las desigualdades sociales en 1910-1920. María Hortensia, ¿cómo estás?",
    "def f(x):\n    return x  +  1   # comentario\n\n\nclass A:\n\tpass\n",
    "it's   don't  I'll\t\t  tabs\n\n  \n  fin",
    "日本語のテキストと emojis 🙂🙂 y acentos: áéíóú ñ ü",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa b " + " " * 70 + "c",
    "http://ejemplo.com/una/ruta?con=parametros&x=1234567890",
]


@pytest.fixture(scope="module")
def patcher():
    return TokenPatcher(MINI)


def _wake(m):
    with torch.no_grad():
        for mod in m.modules():
            out = getattr(mod, "out", None)
            if isinstance(out, torch.nn.Linear) and out.weight.abs().sum() == 0:
                torch.nn.init.normal_(out.weight, std=0.05)
        for z in m.regions:
            z.router.weight.normal_(0, 1.0)


def _model(**kw):
    torch.manual_seed(0)
    m = HN1Model(replace(preset("hn1-tiny"), patching="tokens", tokenizer=MINI, **kw)).eval()
    _wake(m)
    return m


def test_complete_utf8():
    assert _complete_utf8(b"ab") == 2
    assert _complete_utf8("á".encode()[:1]) == 0
    assert _complete_utf8(b"x" + "😀".encode()[:3]) == 1
    assert _complete_utf8(b"x" + "😀".encode()) == 5
    assert _complete_utf8(b"\xff") == 1                    # byte inválido: se trata como carácter


@pytest.mark.parametrize("text", TEXTS + ["byte inválido \xff\xfe fin"])
def test_boundaries_are_causal(patcher, text):
    """Lo confirmado al leer un prefijo no cambia al leer más: la frontera solo depende de lo visto,
    y cada frontera se confirma después de su último byte."""
    raw = text.encode("utf-8", "surrogateescape") if "\xff" not in text else text.encode("latin-1")
    ends, commits = patcher.boundaries(raw)
    assert all(c >= e + 1 for e, c in zip(ends, commits))
    assert ends == sorted(set(ends)) and commits == sorted(commits)
    for n in range(1, len(raw) + 1):
        e_n, c_n = patcher.boundaries(raw[:n])
        keep = [(e, c) for e, c in zip(ends, commits) if c <= n - 1]
        assert list(zip(e_n, c_n)) == keep, n


@pytest.mark.parametrize("text", TEXTS[:4])
def test_matches_full_tokenization(patcher, text):
    """Las fronteras causales son las de la tokenización completa (salvo el último pre-token, que
    todavía no está confirmado)."""
    raw = text.encode()
    causal, _ = patcher.boundaries(raw)
    ref = patcher.reference_ends(raw)
    enc = patcher.tok.encode(text, add_special_tokens=False)
    pos, full = 0, []
    for t in enc.tokens:                                     # ByteLevel: un carácter por byte
        pos += len(t)
        full.append(pos - 1)
    assert ref == full
    assert set(causal) <= set(full)
    last_piece = len(patcher.pieces(text)[-1].encode())
    assert [e for e in full if e < len(raw) - last_piece] == causal


def test_long_pretoken_is_bounded(patcher):
    raw = b"x" * 300
    ends, commits = patcher.boundaries(raw)
    gaps = np.diff([-1] + ends)
    assert len(ends) > 5 and gaps.max() <= patcher.max_open


def test_tensors_ctx(patcher):
    x = torch.tensor([list(TEXTS[0].encode()[:60]), list(TEXTS[1].encode()[:60])])
    end, pid, ctx = patcher.tensors(x)
    assert end[:, -1].all()
    assert (ctx <= pid).all()                                # nunca el parche propio ni uno futuro
    assert (ctx[:, 1:] >= ctx[:, :-1]).all()
    assert int(ctx.max()) > 3


def test_pickle_and_deepcopy(patcher):
    p2 = pickle.loads(pickle.dumps(patcher))
    assert p2.boundaries(TEXTS[0].encode()) == patcher.boundaries(TEXTS[0].encode())
    m = _model()
    m2 = copy.deepcopy(m)
    assert m2.token_patcher is m.token_patcher or m2.token_patcher.spec == MINI


def test_model_is_causal():
    m = _model()
    x = torch.tensor([list(TEXTS[0].encode()[:70])])
    with torch.no_grad():
        full = m(x)["logits"]
        for n in range(1, x.shape[1], 3):
            assert (m(x[:, :n])["logits"] - full[:, :n]).abs().max() < 1e-4, n


@pytest.mark.parametrize("cfg", [{}, {"pattern": ("ssm", "ssm", "attn")}, {"halting": "convergencia"},
                                 {"multibyte": 2}])
def test_generation_matches_reference(cfg):
    """generate (caché local + GlobalCache), la caché por parche, el lote, la conversación y la
    especulativa dan lo mismo que recalcular todo en cada byte."""
    m = _model(**cfg)
    mem = torch.randn(3, m.cfg.d)
    for prompt in ["hola mundo, esto es una prueba de desigualdades. ", "x = 1\ny = 2\n", "María"]:
        ref = m.generate_reference(prompt, 40, memory=mem)
        assert m.generate(prompt, 40, memory=mem) == ref
        assert m.generate_patch_cache(prompt, 40, memory=mem) == ref
        st = SpecStats()
        assert generate_speculative(m, prompt, 40, memory=mem, stats=st) == ref
        assert generate_batch(m, [prompt, prompt], 40, memory=mem) == [ref, ref]
    conv = Conversation(m)
    a = conv.say("hola ", 12)
    b = conv.say(" y vos", 12)
    assert m.generate("hola " + a.decode("latin-1") + " y vos", 12) == b or \
        m.generate(b"hola " + a + b" y vos", 12) == b


def test_sliding_window_matches_reference():
    m = _model()
    prompt = TEXTS[0] * 2
    assert m.generate(prompt, 20, max_context=64) == m.generate_reference(prompt, 20, max_context=64)


def test_training_step_and_checkpoint(tmp_path):
    from hope.train.continual import Mixture, Source, Trainer, TrainConfig
    m = _model()
    tr = Trainer(m, TrainConfig(batch=2, seq_bytes=64, precision="fp32", optimizer="muon", schedule="wsd",
                                warmup=1), device="cpu")
    data = np.frombuffer(("\n".join(TEXTS) * 20).encode(), dtype=np.uint8).copy()
    rep = tr.train(Mixture([Source("t", data, 1.0)]), 3, eval_sources=[])
    assert rep.steps == 3 and np.isfinite(rep.train_bpb_last)
    tr.save(tmp_path / "ck")
    tr2 = Trainer.resume(tmp_path / "ck", TrainConfig(precision="fp32", optimizer="muon"), device="cpu")
    assert tr2.model.cfg.patching == "tokens"
    assert tr2.model.generate("hola", 8) == tr.model.generate("hola", 8)


# ---------------------------------------------------------------------------
# Destilación
# ---------------------------------------------------------------------------
def _fake_teacher(tmp_path, corpus: bytes, patcher, k: int = 4, vocab: int = 768, seed: int = 0):
    """Logits sintéticos con el formato de ``build_teacher_logits``: en cada final de token, el
    token siguiente real con prob. alta y 3 distractores."""
    rng = np.random.default_rng(seed)
    ends, ids = [], []
    start = 0
    for doc in corpus.split(b"\x00"):
        if doc:
            from hope.train.distill import _token_ends
            i, e = _token_ends(patcher, doc)
            ends += [start + x for x in e[:-1]]
            ids += i[1:]
        start += len(doc) + 1
    n = len(ends)
    top = np.zeros((n, k), dtype=np.uint32)
    top[:, 0] = ids
    top[:, 1:] = rng.integers(0, vocab, size=(n, k - 1))
    logp = np.log(np.tile(np.array([0.7, 0.1, 0.1, 0.05][:k]), (n, 1))).astype(np.float16)
    freq = np.zeros(vocab)
    np.add.at(freq, top.reshape(-1), np.exp(logp.astype(np.float64)).reshape(-1))
    f = tmp_path / "maestro"
    f.mkdir()
    np.save(f / "ends.npy", np.asarray(ends, dtype=np.int64))
    np.save(f / "ids.npy", top)
    np.save(f / "logp.npy", logp)
    np.save(f / "freq.npy", freq)
    (f / "meta.json").write_text(json.dumps({"entries": n, "k": k}))
    return f


def test_token_ends_cover_document(patcher):
    from hope.train.distill import _token_ends
    doc = TEXTS[0].encode()
    ids, ends = _token_ends(patcher, doc)
    assert ends[-1] == len(doc) - 1 and len(ids) == len(ends)
    assert patcher.tok.decode(ids) == TEXTS[0]


def test_distillation_loss_learns(tmp_path, patcher):
    from hope.train.distill import Distiller, TeacherLogits
    corpus = ("\x00".join(TEXTS[:3]) * 6).encode()
    t = TeacherLogits(_fake_teacher(tmp_path, corpus, patcher), vocab_k=64)
    assert 0 < t.vocab_coverage(64) <= 1
    dist = Distiller(t, weight=1.0)
    m = _model(distill_vocab=64)
    assert m.distill_head.weight.shape == (65, m.cfg.d)
    arr = np.frombuffer(corpus, dtype=np.uint8)
    starts = np.array([0, 37])
    x = torch.from_numpy(np.stack([arr[s:s + 81] for s in starts]).astype(np.int64))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    kls = []
    for _ in range(25):
        loss, st = m.loss(x, aux=dist.for_batch(starts))
        opt.zero_grad()
        loss.backward()
        assert m.distill_head.weight.grad is not None and m.to_global.weight.grad.abs().sum() > 0
        opt.step()
        kls.append(st["kl"])
    assert st["parches_alineados"] > 0.8
    assert kls[-1] < 0.5 * kls[0]


def test_distill_lookup_alignment(tmp_path, patcher):
    from hope.train.distill import TeacherLogits
    corpus = "\x00".join(TEXTS[:2]).encode()
    t = TeacherLogits(_fake_teacher(tmp_path, corpus, patcher), vocab_k=32)
    found, idx = t.lookup(np.array([int(t.ends[3]), int(t.ends[3]) + 1, 10**9]))
    assert found.tolist() == [True, int(t.ends[4]) == int(t.ends[3]) + 1, False]
    assert idx[0] == 3


@pytest.mark.skipif(not (cache_dir() / "Qwen--Qwen2.5-0.5B" / "tokenizer.json").exists(),
                    reason="tokenizador de Qwen no descargado")
def test_real_qwen_agreement():
    p = TokenPatcher("Qwen/Qwen2.5-0.5B")
    text = ("La tokenización de Qwen corta las palabras en piezas frecuentes; el parcheo causal "
            "debería coincidir casi siempre. def suma(a, b):\n    return a + b\n") * 3
    a = p.agreement(text.encode())
    assert a["precision"] == 1.0 and a["recall"] > 0.95
