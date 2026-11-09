"""Generación en lote y caché de conversación: misma salida que ``generate`` (§3.9.1)."""

from dataclasses import replace

import pytest
import torch

from hope.core.batched import BatchStats, Conversation, generate_batch
from hope.core.hn1 import HN1Model, preset

BASE = preset("hn1-tiny")
CONFIGS = {
    "base": BASE,
    "ssm": replace(BASE, pattern=("ssm", "ssm", "attn", "ssm")),
    "convergencia": replace(BASE, halting="convergencia"),
    "ventana": replace(BASE, pattern=("swa", "ssm", "attn"), window=8, sinks=2, kv_heads=2),
    "sesgo": replace(BASE, balance="sesgo", shared_expert=True, top_k=3),
    "vision": replace(BASE, vision=True, vis_d=32, vis_layers=1, vis_heads=2, vis_image=32, vis_patch=8),
    "multibyte": replace(BASE, multibyte=2),
    "parches_cortos": replace(BASE, kv_heads=1, max_patch=4),
    "relacional_ventana": BASE,                      # ventana corta del especialista relacional (abajo)
}
PROMPTS = ["hola mundo, esto es una prueba. ", "x = 1\ny = 2\nz = x + y\n" * 3, "a",
           "hola mundo, esto es una prueba. ", "superextraordinariamente", "¿qué tensión cae? V = I * R"]


def _wake(m):
    """Saca de cero las salidas que arrancan en cero, para que todo influya en la prueba."""
    with torch.no_grad():
        for mod in m.modules():
            out = getattr(mod, "out", None)
            if isinstance(out, torch.nn.Linear) and out.weight.abs().sum() == 0:
                torch.nn.init.normal_(out.weight, std=0.05)
        for z in m.regions:
            z.router.weight.normal_(0, 1.0)
        if m.memory is not None:
            m.memory.gate.fill_(0.5)


def _model(name):
    torch.manual_seed(0)
    cfg = CONFIGS[name]
    m = HN1Model(cfg).eval()
    _wake(m)
    if name == "relacional_ventana":
        for e in m.regions[0].experts:
            if e.kind == "relacional":
                e.window = 3
    img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8) if cfg.vision else None
    return m, img, torch.randn(3, cfg.d)


def _trained_tiny():
    torch.manual_seed(0)
    m = HN1Model(BASE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.tensor([list(("la ley de ohm dice que V = I * R. " * 6).encode())])
    for _ in range(40):
        loss, _ = m.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


@pytest.mark.parametrize("name", list(CONFIGS))
def test_batch_matches_generate(name):
    m, img, mem = _model(name)
    st = BatchStats()
    got = generate_batch(m, PROMPTS, 40, image=img, memory=mem, stats=st)
    for p, g in zip(PROMPTS, got):
        assert g == m.generate(p, 40, image=img, memory=mem), (name, p)
    assert st.prefills == len(PROMPTS) - 1                      # el prompt repetido se procesa una vez
    assert st.rows_per_step > 1                                  # varias secuencias por pasada
    assert st.patches_per_global_call > 1                        # parte global compartida (largos distintos)


@pytest.mark.parametrize("name", ["base", "ssm", "convergencia", "ventana", "vision"])
def test_conversation_matches_generate_of_whole_history(name):
    m, img, mem = _model(name)
    conv = Conversation(m, image=img, memory=mem)
    turns = ["Usuario: hola, ¿cómo andás?\nHope:", "\nUsuario: contame de la ley de Ohm.\nHope:", "\nUsuario: gracias"]
    history = b""
    for i, t in enumerate(turns):
        out = conv.say(t, 30 + 5 * i)
        history += t.encode()
        assert out == m.generate(history, 30 + 5 * i, image=img, memory=mem), (name, i)
        history += out
    assert bytes(conv.history) == history
    assert conv.stats.fed_bytes < conv.stats.history_bytes / 2


def test_stops_differ_per_sequence_and_conversation_keeps_state():
    m = _trained_tiny()
    prompts = ["la ley de ", "V = I", "x", "Problema: ¿qué tensión cae?\n", "la ley de ", "superextraordinariamente"]
    for kw in ({"stop": b" "}, {"stop_seq": b"R."}, {"stop_after_line": b"la"}, {"stop": b"e"}, {}):
        got = generate_batch(m, prompts, 60, **kw)
        assert got == [m.generate(p, 60, **kw) for p in prompts], kw
    # la conversación: parada por byte (no forma parte del texto), por secuencia y por largo
    conv = Conversation(m)
    history = b""
    for t, kw in [("la ley de ", {"stop": b" "}), ("ohm dice", {"stop_seq": b"R"}), (" que ", {}),
                  ("", {"stop": b"\x01"}), ("V", {"stop": b"="})]:
        out = conv.say(t, 25, **kw)
        history += t.encode()
        assert out == m.generate(history, 25, **kw), (t, kw)
        history += out


def test_fallbacks_long_context_and_swa_local():
    m = _trained_tiny()
    prompts = ["la ley de ohm " * 3, "V = I", "la ley de ohm dice que"]
    st = BatchStats()
    got = generate_batch(m, prompts, 30, max_context=40, stats=st)
    assert got == [m.generate(p, 30, max_context=40) for p in prompts]
    assert st.fallbacks == 2                                    # no entran en max_context
    # conversación que se pasa de la ventana y después vuelve a entrar con la caché al día
    conv = Conversation(m, max_context=64)
    history = b""
    for t, n in [("la ley de ", 20), ("ohm dice que V = I * R y la potencia ", 20), ("P", 5)]:
        out = conv.say(t, n)
        history += t.encode()
        assert out == m.generate(history, n, max_context=64)
        history += out
    assert conv.stats.fallbacks >= 1
    torch.manual_seed(0)
    swa = HN1Model(replace(BASE, local_kind="swa", window=8, sinks=2)).eval()
    assert generate_batch(swa, ["hola mundo ", "abc"], 20) == [swa.generate(p, 20) for p in ["hola mundo ", "abc"]]
    c = Conversation(swa)
    assert c.say("hola ", 10) == swa.generate("hola ", 10)


def test_same_prompt_shares_global_passes_and_sampling_works():
    m, _, _ = _model("ssm")
    st = BatchStats()
    got = generate_batch(m, ["el sol sale por el este. "] * 8, 40, stats=st)
    assert len(set(got)) == 1 and got[0] == m.generate("el sol sale por el este. ", 40)
    assert st.prefills == 1 and st.patches_per_global_call >= 7        # 8 parches por pasada global
    torch.manual_seed(1)
    samples = generate_batch(m, ["el sol sale por el este. "] * 8, 30, temperature=1.0, stop=b"")
    assert all(len(s) == 30 for s in samples) and len(set(samples)) > 1
