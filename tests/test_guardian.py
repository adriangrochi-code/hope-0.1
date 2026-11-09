import json

import pytest

from guardian.core import Action, Guardian, GuardianConfig, Level


@pytest.fixture
def guardian(tmp_path):
    budget = tmp_path / "configs" / "growth_budget.yaml"
    budget.parent.mkdir()
    budget.write_text(open("configs/growth_budget.yaml").read().replace(
        "mode: EXPERIMENT_ONLY", "mode: AUTO_CONSOLIDATION_WITHIN_LIMIT"))
    work = tmp_path / "runs"
    work.mkdir()
    cfg = GuardianConfig(workdir=str(work), protected_paths=[str(budget.parent)],
                         budget_path=str(budget), audit_path=str(tmp_path / "g" / "audit.jsonl"),
                         approvals_path=str(tmp_path / "g" / "approvals.json"))
    return Guardian(cfg)


def _structural(kind, delta, total=150_000_000, level=4):
    return Action(kind, {"change": {"description": "x", "param_delta": delta, "plasticity_level": level},
                         "current_total": total})


def test_unknown_and_forbidden_actions_are_denied(guardian):
    assert not guardian.decide(Action("algo.nuevo")).allowed
    for kind in ("budget.change", "safety.modify", "audit.modify"):
        d = guardian.decide(Action(kind))
        assert not d.allowed and not d.needs_human


def test_ceiling_cannot_be_approved(guardian):
    d = guardian.decide(_structural("structural.experiment", 60_000_000))
    assert not d.allowed and not d.needs_human           # invariante: ni siquiera queda pendiente


def test_small_consolidation_is_automatic_big_needs_human(guardian):
    small = guardian.decide(_structural("structural.consolidate", 1_000_000))
    assert small.allowed and small.level == Level.SUPERVISED
    big = guardian.decide(_structural("structural.consolidate", 30_000_000))   # > 10%
    assert not big.allowed and big.needs_human


def test_human_approval_flow(guardian):
    action = Action("network.access", {"host": "example.org"})
    d = guardian.decide(action)
    assert d.needs_human
    rid = d.pending_request
    assert not guardian.decide(action, approval_id=rid).allowed       # todavía no aprobado
    guardian.approve(rid, approver="usuario")
    other = Action("network.access", {"host": "otro.org"})
    assert not guardian.decide(other, approval_id=rid).allowed        # la aprobación es para esta acción
    assert guardian.decide(action, approval_id=rid).allowed
    assert not guardian.decide(action, approval_id=rid).allowed       # se usa una sola vez


def test_file_write_rules(guardian, tmp_path):
    inside = str(tmp_path / "runs" / "x.txt")
    assert guardian.decide(Action("file.write", {"path": inside})).level == Level.INTERNAL
    assert not guardian.decide(Action("file.write", {"path": guardian.config.budget_path})).allowed
    assert guardian.decide(Action("file.write", {"path": str(tmp_path / "fuera.txt")})).needs_human


def test_tool_outside_sandbox_needs_human(guardian):
    assert guardian.decide(Action("tool.execute", {"tool": "python", "sandboxed": True})).allowed
    assert guardian.decide(Action("tool.execute", {"tool": "python", "sandboxed": False})).needs_human


def test_audit_chain_detects_tampering(guardian):
    guardian.decide(Action("teacher.query", {"q": "ley de Ohm"}))
    guardian.decide(Action("algo.nuevo"))
    assert guardian.audit.verify() == (True, None)
    lines = guardian.audit.path.read_text().splitlines()
    first = json.loads(lines[0])
    first["data"]["allowed"] = not first["data"]["allowed"]
    lines[0] = json.dumps(first)
    guardian.audit.path.write_text("\n".join(lines) + "\n")
    assert guardian.audit.verify() == (False, 0)
