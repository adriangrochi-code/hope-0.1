import torch

from hope.council.agents import PROFILES, Agent, hope_solver
from hope.council.coordinator import AgentUsageMemory, Coordinator
from hope.council.server import RemoteAgent, serve

P = "¿Qué tensión cae en un resistor de 220 ohm con 0.05 A?"
SCRIPTS = {
    "innovador": ("Paso 1: 220 / 0,05 = 4400\nResultado: 4400 V", "4400 V"),       # idea original, pero mal
    "conservador": ("Paso 1: 220 * 0,05 = 11\nResultado: 11 V", "11 V"),
    "optimizado": ("Paso 1: 220 * 0,05 = 11\nResultado: 11 V", "11 V"),
    "veloz": ("Resultado: 12 V", "12 V"),
}


def _agents():
    return [Agent(PROFILES[n], lambda topic, problem, p, n=n: SCRIPTS[n]) for n in SCRIPTS]


def test_coordinator_verifies_and_prefers_verified_consensus():
    co = Coordinator(_agents())
    res = co.solve("ley de Ohm", P)
    assert res.answer == "11 V"
    assert res.chosen.verdict == "ok" and res.chosen.agreement > 0
    bad = {s.candidate.agent: s.verdict for s in res.ranking}
    assert bad["innovador"] == "errores" and bad["veloz"] == "errores"


def test_routing_learns_which_agents_work_per_topic(tmp_path):
    mem = AgentUsageMemory(tmp_path / "agentes.json")
    co = Coordinator(_agents(), memory=mem)
    for _ in range(5):
        co.solve("ley de Ohm", P, expected="11 V")
    chosen = mem.select("ley de Ohm", list(SCRIPTS), k=2)
    assert set(chosen) == {"conservador", "optimizado"}
    assert AgentUsageMemory(tmp_path / "agentes.json").stats["ley de Ohm"]["veloz"]["reward"] == 0


def test_agents_can_run_as_separate_servers():
    local = _agents()
    servers = [serve(a) for a in local]
    try:
        remote = [RemoteAgent(f"http://127.0.0.1:{s.server_port}") for s in servers]
        assert remote[0].profile.name == "innovador"
        res = Coordinator(remote).solve("ley de Ohm", P)
        assert res.answer == "11 V" and res.chosen.candidate.meta["remote"].startswith("http://127.0.0.1")
    finally:
        for s in servers:
            s.shutdown()


def test_real_hope_core_runs_the_four_personalities():
    torch.manual_seed(0)
    from hope.core.hn1 import build_hn1
    from hope.tools.builtin import default_catalog
    model, _ = build_hn1("hn1-tiny")
    solver = hope_solver(model, default_catalog())
    agents = [Agent(PROFILES[n], solver) for n in PROFILES]
    for a in agents:
        a.profile.max_bytes = 40
    res = Coordinator(agents, parallel=False).solve("ley de Ohm", P)
    assert len(res.ranking) == 4 and res.chosen is not None


def test_single_gpu_council_shares_one_model_and_switches_adapters():
    torch.manual_seed(0)
    from hope.core.hn1 import build_hn1
    from hope.council.agents import build_local_council
    from hope.plasticity.adapters import MultiLoRALinear, adapter_names, use_adapter
    model, _ = build_hn1("hn1-tiny")
    n_params_before = sum(p.numel() for n, p in model.named_parameters() if "lora_" not in n)
    agents = build_local_council(model)
    assert sorted(adapter_names(model)) == sorted(PROFILES)
    assert sum(p.numel() for n, p in model.named_parameters() if "lora_" not in n) == n_params_before
    x = torch.tensor([list(b"un solo modelo")])
    model.eval()
    with torch.no_grad():
        base = model(x)["logits"]
        for m in model.modules():
            if isinstance(m, MultiLoRALinear):
                m.lora_b["innovador"].data.normal_(0, 0.5)
        use_adapter(model, "innovador")
        assert not torch.allclose(model(x)["logits"], base)              # cada agente tiene su «parte»
        use_adapter(model, "veloz")
        assert torch.allclose(model(x)["logits"], base)                  # el adaptador del veloz sigue en cero
        use_adapter(model, None)
    for a in agents:
        a.profile.max_bytes = 30
    res = Coordinator(agents).solve("ley de Ohm", P)
    assert len(res.ranking) == 4
    assert all(m.active is None for m in model.modules() if isinstance(m, MultiLoRALinear))   # vuelve al base


def test_core_iterations_are_applied_per_agent():
    torch.manual_seed(0)
    from hope.core.hn1 import build_hn1
    model, _ = build_hn1("hn1-tiny")
    a = model.generate("x", max_new=20, n_iters=1)
    b = model.generate_reference("x", max_new=20, n_iters=1)
    assert a == b
    assert model(torch.tensor([[1, 2, 3]]), n_iters=1)["iters_used"] == 1
