import time

import torch

from hope.core.hn1 import build_hn1


def _trained_tiny():
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.tensor([list(("la ley de ohm dice que V = I * R. " * 6).encode())])
    for _ in range(40):
        loss, _ = model.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def test_cached_generation_matches_reference_exactly():
    model = _trained_tiny()
    for prompt in ("la ley de ", "V = I", "Problema: ¿qué tensión cae?\n"):
        fast = model.generate(prompt, max_new=60)
        slow = model.generate_reference(prompt, max_new=60)
        assert fast == slow, (prompt, fast, slow)


def test_cached_generation_respects_context_window():
    model = _trained_tiny()
    assert model.generate("la ley de ohm " * 3, max_new=30, max_context=24) == \
        model.generate_reference("la ley de ohm " * 3, max_new=30, max_context=24)


def test_cached_generation_is_faster():
    torch.manual_seed(0)
    model, _ = build_hn1("hn1-tiny")
    prompt = "Tema: ley de Ohm\nProblema: ¿Qué tensión cae en un resistor de 220 ohm con 0.05 A?\n" * 2
    t0 = time.perf_counter()
    model.generate_reference(prompt, max_new=80, stop=b"\x01")
    slow = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.generate(prompt, max_new=80, stop=b"\x01")
    fast = time.perf_counter() - t0
    assert fast < slow
