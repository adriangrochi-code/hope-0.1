import shutil

import pytest

from guardian.cli import load, parse_count, set_limit
from guardian.core import Action


@pytest.mark.parametrize("text,value", [("1B", 10**9), ("1b", 10**9), ("1.5B", 1_500_000_000), ("500M", 500 * 10**6),
                                        ("200 millones", 200 * 10**6), ("1.000.000.000", 10**9),
                                        ("1_000_000_000", 10**9), ("5 mil millones", 5 * 10**9)])
def test_parse_count(text, value):
    assert parse_count(text) == value


def test_parse_count_rejects_garbage():
    with pytest.raises(ValueError):
        parse_count("mucho")


def test_raise_limit_to_1b(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    shutil.copy("configs/growth_budget.yaml", cfg_dir / "growth_budget.yaml")
    (tmp_path / "guardian.yaml").write_text(
        f"workdir: {tmp_path}\nprotected_paths: [{cfg_dir}]\nbudget_path: {cfg_dir / 'growth_budget.yaml'}\n"
        f"audit_path: {tmp_path / 'audit.jsonl'}\napprovals_path: {tmp_path / 'approvals.json'}\n")
    g = load(str(tmp_path / "guardian.yaml"))
    change = {"change": {"description": "x", "param_delta": 400_000_000}, "current_total": 200_000_000}
    assert not g.decide(Action("structural.experiment", change)).allowed       # supera 200M

    warnings = set_limit(g, parse_count("1B"), by="usuario", note="probar 1B")
    assert g.budget().max_total_params == 10**9
    assert any("GB" in w for w in warnings)                                   # 1B no entra en 16 GB para entrenar
    assert g.decide(Action("structural.experiment", change)).allowed           # ahora sí, en copia experimental
    text = (cfg_dir / "growth_budget.yaml").read_text()
    assert "# Este archivo pertenece al usuario" in text                       # se conservan los comentarios
    entry = [e for e in g.audit.entries() if e["event"] == "budget.changed_by_human"][-1]
    assert entry["data"]["old"]["max_total_params"] == 200_000_000 and entry["data"]["by"] == "usuario"
    assert g.audit.verify() == (True, None)
    # Hope sigue sin poder cambiarlo por su cuenta.
    assert not g.decide(Action("budget.change", {"max_total_params": 10**10})).allowed
