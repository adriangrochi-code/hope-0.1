"""Refuerzo con recompensas verificables (hope/train/rlvr.py)."""

import torch

from hope.core.hn1 import HN1Model, preset
from hope.train.rlvr import GRPO, RLConfig, default_extract, problems_from_generators, verified_reward


def test_grpo_increases_reward():
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny"))
    rl = GRPO(m, RLConfig(group=6, temperature=1.0, max_new=12, lr=3e-3, kl_weight=0.0))
    # Recompensa verificable de juguete: fracción de bytes «a» en la respuesta.
    reward = lambda i, t: t.count("a") / max(len(t), 1)  # noqa: E731
    first = [rl.step(["x: "] * 2, reward)["recompensa_media"] for _ in range(3)]
    for _ in range(25):
        rl.step(["x: "] * 2, reward)
    last = [rl.step(["x: "] * 2, reward)["recompensa_media"] for _ in range(3)]
    assert sum(last) / 3 > sum(first) / 3 + 0.1


def test_verified_reward_and_generator_problems():
    probs = problems_from_generators(["aritmetica"], 3)
    assert len(probs) == 3 and all(p["respuesta"] for p in probs)
    p = {"problema": "¿Cuánto es 6 * 7?", "respuesta": "42"}
    match = lambda a, b: a is not None and a.strip() == b  # noqa: E731
    assert verified_reward(p, "Paso 1: 6 * 7 = 42\nResultado: 42", default_extract, match) == 1.0
    assert verified_reward(p, "Paso 1: 6 * 7 = 41\nResultado: 42", default_extract, match) == 0.2
    assert verified_reward(p, "Resultado: 41", default_extract, match) == 0.0
