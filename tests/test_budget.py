from hope.growth.budget import (ConsolidationPolicy, GrowthBudget, GrowthMode, ProposedChange,
                                check_change, needs_human_approval)


def _check(budget, delta, total=100_000, experimental=True, level=4, protected=False):
    return check_change(budget, current_total=total, current_active=total, current_regions=2,
                        current_connections=0,
                        change=ProposedChange("x", delta, plasticity_level=level,
                                              touches_protected=protected),
                        experimental=experimental)


def test_ceiling_is_hard():
    b = GrowthBudget(max_total_params=100_500, mode=GrowthMode.AUTO_CONSOLIDATION_WITHIN_LIMIT)
    assert _check(b, 400).allowed
    assert not _check(b, 600).allowed


def test_modes():
    assert not _check(GrowthBudget(10**9, mode=GrowthMode.EXPANSION_DISABLED), 10).allowed
    assert not _check(GrowthBudget(10**9, mode=GrowthMode.PAUSED), -10).allowed
    exp_only = GrowthBudget(10**9, mode=GrowthMode.EXPERIMENT_ONLY)
    assert _check(exp_only, 10).allowed
    assert not _check(exp_only, 10, experimental=False).allowed


def test_important_changes_always_need_human():
    b = GrowthBudget(10**9, mode=GrowthMode.AUTO_CONSOLIDATION_WITHIN_LIMIT,
                     consolidation=ConsolidationPolicy(max_auto_param_fraction=0.10))
    small = _check(b, 9_000, experimental=False)              # 9% < 10%
    assert small.allowed and not needs_human_approval(b, small)
    big = _check(b, 15_000, experimental=False)               # 15% > 10%
    assert needs_human_approval(b, big)
    assert needs_human_approval(b, _check(b, 10, experimental=False, level=5))
    assert needs_human_approval(b, _check(b, 10, experimental=False, protected=True))


def test_yaml_roundtrip():
    b = GrowthBudget.from_yaml(open("configs/growth_budget.yaml").read())
    assert b.max_total_params == 200_000_000
    assert b.mode == GrowthMode.EXPERIMENT_ONLY
