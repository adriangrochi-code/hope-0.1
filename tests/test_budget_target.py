"""Presupuesto de parámetros como objetivo (hope/growth/budget_target.py; ARQUITECTURA_HN1 §6.5)."""

import shutil

import pytest
import torch

from guardian.cli import load as load_guardian, set_limit
from hope.core.hn1 import HN1Model, build_hn1, preset
from hope.growth.budget_target import (apply_plan, load_model, measure, plan_to_budget, save_model)
from hope.growth.hn1_ops import plan_growth_to

VISION = dict(vis_d=32, vis_layers=1, vis_heads=2, vis_image=32, vis_patch=8)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def _batches(seed=0, n=3):
    g = torch.Generator().manual_seed(seed)
    text = ("la resistencia limita la corriente; def suma(a, b): return a + b\n" * 8).encode()
    base = torch.tensor(list(text[:97]))
    return [torch.stack([base, torch.randint(0, 256, (97,), generator=g)]) for _ in range(n)]


def _guardian(tmp_path, ceiling, mode="USER_APPROVAL_REQUIRED"):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(exist_ok=True)
    shutil.copy("configs/growth_budget.yaml", cfg_dir / "growth_budget.yaml")
    (tmp_path / "guardian.yaml").write_text(
        f"workdir: {tmp_path}\nprotected_paths: [{cfg_dir}]\nbudget_path: {cfg_dir / 'growth_budget.yaml'}\n"
        f"audit_path: {tmp_path / 'audit.jsonl'}\napprovals_path: {tmp_path / 'approvals.json'}\n")
    g = load_guardian(str(tmp_path / "guardian.yaml"))
    set_limit(g, ceiling, by="prueba", mode=mode)
    return g


def _logits(m, batches):
    m.eval()
    with torch.no_grad():
        return [m(x[:, :-1])["logits"] for x in batches]


@pytest.mark.parametrize("factor", [1.3, 2.0, 3.5])
def test_grow_preserves_function_and_never_overshoots(factor):
    torch.manual_seed(0)
    m, reg = build_hn1("hn1-tiny")
    b = _batches()
    before = _logits(m, b)
    n0 = _n(m)
    target = int(factor * n0)
    plan = plan_to_budget(m, target, vision_cfg=VISION, max_flops_ratio=3.0)
    assert plan.direccion == "crecer" and plan.total_despues <= target and _n(m) == n0     # planificar no toca
    res = apply_plan(m, plan, reg, user_approved=True, batches=b)
    assert res.hecho and _n(m) == plan.total_despues <= target
    assert res.diferencia_max_salida < 1e-5
    after = _logits(m, b)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(before, after))
    # lo más cerca posible: no queda lugar ni para la operación más chica (una tanda de 64 unidades)
    assert target - plan.total_despues < 0.05 * target or plan.avisos


def test_policy_order_experts_first_and_measured_efficiency():
    m, _ = build_hn1("hn1-tiny")
    plan = plan_to_budget(m, int(4 * _n(m)), vision_cfg=VISION, max_flops_ratio=3.0)
    cats = [s.categoria for s in plan.pasos]
    assert cats == sorted(cats, key=["especialistas", "bloques", "ensanchar", "vision"].index)
    assert cats[0] == "especialistas"
    eff = {s.categoria: s.param_delta / max(s.flops_por_byte_delta, 1e-9) for s in plan.pasos}
    assert eff["especialistas"] > eff["bloques"]          # dispersos: más capacidad por cómputo
    r = plan.resumen()
    assert r["velocidad"]["despues"]["flops_por_byte"] >= r["velocidad"]["antes"]["flops_por_byte"]
    assert r["velocidad"]["despues"]["flops_por_byte"] <= 3.0 * r["velocidad"]["antes"]["flops_por_byte"] * 1.01


def test_compute_cap_limits_blocks():
    m, _ = build_hn1("hn1-tiny")
    plan = plan_to_budget(m, int(10 * _n(m)), max_flops_ratio=1.3, block_flops_ratio=1.3, vision_cfg=VISION)
    assert plan.velocidad_despues["flops_por_byte"] <= 1.3 * plan.velocidad_antes["flops_por_byte"] * 1.001


def test_hardware_memory_is_respected():
    from hope.bench.hardware import PROFILES, training_fit
    with torch.device("meta"):
        m = HN1Model(preset("hn1-200m"))
    plan = plan_to_budget(m, 10**9, "gtx1060")
    assert plan.total_despues < 10**9
    assert training_fit(plan.total_despues, PROFILES["gtx1060"])["entra_con_offload"]
    assert plan.memoria_despues["modo"] == "offload"
    only_gpu = plan_to_budget(m, 10**9, "gtx1060", offload=False)
    assert only_gpu.total_despues < plan.total_despues and only_gpu.memoria_despues["entra"]


def test_guardian_ceiling_and_approval(tmp_path):
    torch.manual_seed(0)
    m, reg = build_hn1("hn1-tiny")
    n0 = _n(m)
    g = _guardian(tmp_path, int(1.5 * n0))
    plan = plan_to_budget(m, 3 * n0, guardian=g, vision_cfg=VISION)
    assert plan.techo == int(1.5 * n0) and plan.total_despues <= plan.techo and plan.avisos
    assert all(s.requiere_aprobacion for s in plan.pasos if abs(s.param_delta) > 0.1 * s.total_antes)
    res = apply_plan(m, plan, reg, guardian=g)
    assert not res.hecho and res.pendiente and _n(m) == n0                 # nada cambia sin aprobación
    g.approve(res.pendiente, "prueba")
    res = apply_plan(m, plan, reg, guardian=g, approval=res.pendiente)
    assert res.pasos_aplicados >= 1 and _n(m) <= plan.techo
    # Hope no puede subir el techo: solo la persona con guardian.cli limite
    from guardian.core import Action
    assert not g.decide(Action("budget.change", {"max_total_params": 10**12})).allowed


def test_plan_without_approval_is_refused():
    m, _ = build_hn1("hn1-tiny")
    plan = plan_to_budget(m, 2 * _n(m))
    with pytest.raises(PermissionError):
        apply_plan(m, plan)


def test_reduce_removes_least_used_and_loss_matches_report():
    torch.manual_seed(0)
    m, reg = build_hn1("hn1-tiny")
    b = _batches()
    # Crecer primero (bloques agregados con salida en cero = identidad) y después reducir.
    grow = plan_to_budget(m, int(1.6 * _n(m)), max_flops_ratio=3.0, max_experts=10)
    apply_plan(m, grow, reg, user_approved=True, unmask_new=True)
    usage = measure(m, b)["uso"][0]
    n1 = _n(m)
    target = int(0.85 * n1)
    plan = plan_to_budget(m, target, batches=b)
    assert plan.direccion == "reducir" and _n(m) == n1                    # planificar no toca
    removed = [r for s in plan.pasos for r in s.retiros if r[0] == "especialista"]
    blocks = [r for s in plan.pasos for r in s.retiros if r[0] == "bloque"]
    assert blocks                                                          # los agregados casi identidad salen
    if removed:
        kept = [u for i, u in enumerate(usage) if ("especialista", 0, i) not in removed]
        assert max(usage[e] for _, _, e in removed) <= min(kept) + 1e-9   # salen los menos usados
    res = apply_plan(m, plan, user_approved=True, batches=b)
    assert res.hecho and _n(m) == plan.total_despues <= target
    measured = measure(m, b)["bpb"]
    assert abs(measured - plan.bpb_despues) < 1e-3                         # la pérdida es la informada
    assert len(m.backbone) >= m.cfg.base_blocks and len(m.cfg.pattern) == len(m.backbone)


def test_reduce_used_experts_reports_real_loss():
    """Una reducción fuerte saca especialistas que sí se usan: la pérdida cambia y es la informada."""
    torch.manual_seed(1)
    m, _ = build_hn1("hn1-tiny")
    b = _batches(seed=1)
    n0 = _n(m)
    plan = plan_to_budget(m, int(0.93 * n0), batches=b, min_experts=3)
    assert plan.pasos and plan.total_despues <= int(0.93 * n0)
    res = apply_plan(m, plan, user_approved=True, batches=b)
    after = measure(m, b)["bpb"]
    assert abs(after - plan.bpb_despues) < 1e-3 and abs(res.bpb_medido[-1] - plan.bpb_despues) < 1e-3
    assert plan.bpb_despues != plan.bpb_antes
    assert sum(int((z.expert_mask == 0).sum()) for z in m.regions) >= 3


def test_round_trip_with_checkpoints(tmp_path):
    torch.manual_seed(0)
    m, _ = build_hn1("hn1-tiny")
    b = _batches()
    ref = _logits(m, b)
    n0 = _n(m)
    save_model(m, tmp_path / "a")
    m1, ck = load_model(tmp_path / "a")
    up = plan_to_budget(m1, 2 * n0, max_flops_ratio=3.0)
    apply_plan(m1, up, user_approved=True)
    save_model(m1, tmp_path / "b", ck)
    m2, ck2 = load_model(tmp_path / "b")
    assert _n(m2) == up.total_despues
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(ref, _logits(m2, b)))
    down = plan_to_budget(m2, n0, batches=b)
    apply_plan(m2, down, user_approved=True, batches=b)
    save_model(m2, tmp_path / "c", ck2)
    m3, _ = load_model(tmp_path / "c")
    assert _n(m3) <= n0
    # Todo lo agregado estaba enmascarado o era identidad: volver a ≤ n0 no pierde nada.
    assert all(torch.allclose(x, y, atol=1e-4) for x, y in zip(ref, _logits(m3, b)))
    from hope.train.continual import Trainer, TrainConfig
    tr = Trainer.resume(tmp_path / "c", TrainConfig(precision="fp32"), "cpu")      # lo retoma el entrenamiento
    assert _n(tr.model) == _n(m3)


@pytest.mark.parametrize("name,low,high", [("hn1-700m", 650e6, 700e6), ("hn1-1b", 950e6, 1e9)])
def test_intermediate_presets_reachable_from_200m(name, low, high):
    from hope.bench.plan import cost, meta_model
    from hope.core.hn1 import build_registry
    c = cost(meta_model(name))
    assert low < c.total_params <= high
    assert c.flops_per_byte / cost(meta_model("hn1-200m")).flops_per_byte < 2.0
    t = preset(name)
    assert (t.d, t.d_local, t.enc_layers, t.dec_layers) == (1024, 512, 2, 3)
    with torch.device("meta"):
        m = HN1Model(preset("hn1-200m"))
    reg = build_registry(m)
    for op in plan_growth_to(m, name):
        op.apply(m, reg)
    assert m.cfg.pattern == t.pattern and m.vision is not None
    assert m.vision.at_blocks == t.vision_blocks()
    assert len(m.regions[0].experts) == len(t.zone_kinds)
    assert abs(_n(m) - c.total_params) < 0.02 * c.total_params
    assert plan_growth_to(m, name) == []
    # también desde hn1-500m
    with torch.device("meta"):
        m5 = HN1Model(preset("hn1-500m"))
    ops = plan_growth_to(m5, name)
    assert ops and not any(type(o).__name__ == "AddVisualZone" for o in ops)


def test_budget_cli_shows_plan(tmp_path, capsys):
    from hope.growth.budget_target import main
    m, _ = build_hn1("hn1-tiny")
    save_model(m, tmp_path / "ck")
    g = _guardian(tmp_path, 10 * _n(m))
    assert main(["--checkpoint", str(tmp_path / "ck"), "--objetivo", str(3 * _n(m)), "--perfil", "a770",
                 "--guardian", str(tmp_path / "guardian.yaml")]) == 0
    out = capsys.readouterr().out
    assert "Parámetros:" in out and "Memoria" in out and "Velocidad" in out and "agregar especialistas" in out
    assert main(["--checkpoint", str(tmp_path / "ck"), "--objetivo", str(3 * _n(m)),
                 "--guardian", str(tmp_path / "guardian.yaml"), "--aplicar", "--salida", str(tmp_path / "out")]) == 3
    pending = list(g.pending())
    assert pending


def test_run_stage_presupuesto_with_approvals(tmp_path):
    """Etapa ``presupuesto:`` de hope.train.run: se detiene en cada paso grande hasta que el usuario aprueba."""
    import json

    import yaml

    from hope.train.run import run
    m, _ = build_hn1("hn1-tiny")
    n0 = _n(m)
    g = _guardian(tmp_path, 10 * n0)
    out = tmp_path / "salida"
    plan = {"salida": str(out), "precision": "fp32", "compilar": False, "guardian": str(tmp_path / "guardian.yaml"),
            "etapas": [{"nombre": "presupuesto", "preset": "hn1-tiny", "presupuesto": str(2 * n0)}]}
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    for _ in range(6):
        run(str(path), "cpu")
        st = json.loads((out / "estado.json").read_text(encoding="utf-8"))
        es = st["etapas"]["presupuesto"]
        if es["hecha"]:
            break
        g.approve(es["solicitud"], "prueba")
        es["aprobacion"] = es["solicitud"]
        (out / "estado.json").write_text(json.dumps(st), encoding="utf-8")
    assert es["hecha"]
    grown, _ = load_model(out / "presupuesto")
    assert n0 < _n(grown) <= 2 * n0
