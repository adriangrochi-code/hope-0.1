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


def test_cost_aware_reward_prefers_the_calculator():
    from hope.runtime.toolcalls import ToolRunner
    from hope.tools.builtin import default_catalog
    from hope.train.rlvr import CostWeights, Rollout, efficient_reward, masked_logprob, rollout_with_tools
    p = {"problema": "¿Cuánto es 6 * 7?", "respuesta": "42"}
    match = lambda a, b: a is not None and a.strip() == b  # noqa: E731
    by_hand = Rollout(b"P: ", [(("Paso 1: 6 * 7 = 6 + 6 + 6 + 6 + 6 + 6 + 6 = 42\n" * 2 + "Resultado: 42").encode(),
                                True)], 0.8, 0.0, 0)
    tool = Rollout(b"P: ", [("⟦calc: 6 * 7⟧".encode(), True), (" → 42".encode(), False),
                            ("\nResultado: 42".encode(), True)], 0.2, 0.01, 1)
    wrong_fast = Rollout(b"P: ", [(b"Resultado: 41", True)], 0.05, 0.0, 0)
    q = lambda r: verified_reward(p, r.text, default_extract, match)  # noqa: E731
    w = CostWeights()
    r_hand, r_tool, r_wrong = (efficient_reward(q(x), x, w) for x in (by_hand, tool, wrong_fast))
    assert r_tool > r_hand > r_wrong                     # correcta y barata > correcta y cara > rápida y mal
    assert tool.model_bytes < by_hand.model_bytes
    # los bytes del resultado de la herramienta no cuentan como generados por Hope
    m = HN1Model(preset("hn1-tiny"))
    _, n = masked_logprob(m, tool)
    assert n == tool.model_bytes
    roll = rollout_with_tools(m, "P: ", ToolRunner(default_catalog()), temperature=1.0, max_new=20)
    assert roll.model_bytes <= 20 and roll.seconds > 0
