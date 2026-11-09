"""Refuerzo con recompensas verificables (hope/train/rlvr.py)."""

import torch

from hope.core.hn1 import HN1Model, preset
from hope.train.rlvr import GRPO, RLConfig, default_extract, problems_from_generators, verified_reward


def test_grpo_increases_reward():
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny"))
    rl = GRPO(m, RLConfig(group=6, temperature=1.0, max_new=12, curriculum=(), lr=3e-3, kl_weight=0.0))
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


def test_group_logprobs_in_one_pass_match_single():
    from hope.train.rlvr import completion_logprob, token_logprobs
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny")).eval()
    comps = [b"abc def", b"x", b"hola, mundo. 42"]
    lp, mask = token_logprobs(m, b"P: ", comps)
    for i, c in enumerate(comps):
        single, n = completion_logprob(m, b"P: ", c)
        assert n == len(c) == int(mask[i].sum())
        assert abs(float((lp[i] * mask[i]).sum().detach()) - float(single.detach())) < 1e-3


def test_dr_grpo_loss_is_unbiased_by_length_and_std():
    from hope.train.rlvr import group_advantage, policy_loss
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    adv = group_advantage(r)                                       # centrada, sin dividir por el desvío
    assert torch.allclose(adv, torch.tensor([0.75, -0.25, -0.25, -0.25]))
    assert torch.allclose(group_advantage(r * 0.1), adv * 0.1)     # la escala de la recompensa importa
    cfg = RLConfig(kl_weight=0.0, max_new=100)
    lp = torch.full((2, 50), -1.0)
    mask = torch.ones(2, 50)
    mask[1, 10:] = 0                                               # respuesta corta (10 bytes) y larga (50)
    loss, kl = policy_loss(lp, lp, mask, torch.tensor([-1.0, -1.0]), cfg, 100)
    # constante: cada byte pesa lo mismo, así que la incorrecta larga pesa 5 veces más que la corta
    assert torch.allclose(loss, torch.tensor([-0.5, -0.1])) and torch.allclose(kl, torch.zeros(2))
    old, _ = policy_loss(lp, lp, mask, torch.tensor([-1.0, -1.0]), RLConfig(loss_norm="largo"), 100)
    assert torch.allclose(old, torch.tensor([-1.0, -1.0]))         # GRPO original: igual por muestra


def test_zero_variance_groups_are_refilled_and_pool_keeps_the_edge():
    from hope.train.rlvr import ProblemPool
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny"))
    calls = []

    def sampler(model, prompt, n, *, max_new, temperature, stop):  # interfaz ``sampler`` (lote)
        calls.append((prompt, n, max_new))
        k = int(prompt.decode().split()[-1])
        return [b"a" * (j + 1) if j < k else b"b" for j in range(n)]
    rl = GRPO(m, RLConfig(group=4, max_new=8, curriculum=(), kl_weight=1e-3), sampler=sampler)
    reward = lambda p, t: 1.0 if t.startswith("a") else 0.0       # noqa: E731
    # k = 0: todas mal (sin varianza); k = 4: todas bien; k = 2: mitad y mitad
    pool = ProblemPool([{"k": k} for k in (0, 4, 2, 1, 3)], lambda p: f"P {p['k']}", reward, seed=0, cooldown=100)
    st = rl.step_pool(pool, 2)
    assert st["grupos"] >= 2 and rl.report.refilled_groups == rl.report.skipped_groups
    assert all(n == 4 for _, n, _ in calls)                        # el grupo entero en una llamada
    for _ in range(4):
        rl.step_pool(pool, 2)
    states = {pool.problems[i]["k"]: s["estado"] for i, s in enumerate(pool.stats)}
    assert states[4] in ("retirado", "activo") and states[0] in ("postergado", "activo")
    seen = [i for i, s in enumerate(pool.stats) if s["vistos"]]
    assert all(pool.problems[i]["k"] in (1, 2, 3) for i in pool.edge(0.1, 0.8)) and seen
    assert "kl" in st


def test_length_curriculum_grows_when_answers_are_cut():
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny"))
    sampler = lambda model, prompt, n, *, max_new, temperature, stop: [b"a" * max_new] * (n - 1) + [b"b"]  # noqa: E731
    rl = GRPO(m, RLConfig(group=4, max_new=4, curriculum=(8, 16), curriculum_truncation=0.5), sampler=sampler)
    reward = lambda i, t: float(t.startswith("a"))                 # noqa: E731
    sizes = [rl.step(["x: "], reward)["max_new"] for _ in range(4)]
    assert sizes == [4, 8, 16, 16] and rl.report.max_new_history == sizes
    assert RLConfig().max_new == 512 and RLConfig().curriculum == (1024, 2048) and RLConfig().kl_weight == 1e-3


def test_default_sampler_falls_back_to_serial():
    from hope.train.rlvr import default_sampler
    torch.manual_seed(0)
    m = HN1Model(preset("hn1-tiny")).eval()
    out = default_sampler(m, b"hola ", 3, max_new=5, temperature=1.0, stop=b"\x00")
    assert len(out) == 3 and all(len(o) <= 5 for o in out)
