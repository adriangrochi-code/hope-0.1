import pytest
import torch

from hope.core.toy import build_toy_model
from hope.growth.ops import AddExpert
from hope.lineage.store import STABLE, LineageError, LineageStore


def test_experiment_promote_and_rollback(tmp_path):
    torch.manual_seed(0)
    store = LineageStore(tmp_path)
    model, reg = build_toy_model()
    v0 = store.commit(model, reg, branch=STABLE, message="inicial")
    x = torch.randint(0, 64, (2, 8))
    ref = model(x).detach()

    with pytest.raises(LineageError):
        store.commit(model, reg, branch=STABLE, message="directo a stable")

    br = store.experiment("mas-capacidad")
    AddExpert(0).apply(model, reg)
    v1 = store.commit(model, reg, branch=br, message="agregar especialista",
                      meta={"diagnostico": "dx_0", "op": "add_expert"})
    assert store.head(STABLE) == v0

    with pytest.raises(LineageError):
        store.promote(br, approval_ref="")
    store.promote(br, approval_ref="eval_123")
    assert store.head(STABLE) == v1
    assert [v.id for v in store.history(STABLE)] == [v1, v0]

    store.rollback(STABLE, v0, reason="regresión detectada")
    restored, reg0 = store.load(store.head(STABLE))
    assert torch.equal(restored(x).detach(), ref)
    assert reg0.report(restored).total < reg.report(model).total
    kinds = [e["event"] for e in store.events()]
    assert kinds[-1] == "rollback" and "promote" in kinds


def test_rollback_only_within_lineage(tmp_path):
    store = LineageStore(tmp_path)
    model, reg = build_toy_model()
    store.commit(model, reg, branch=STABLE, message="v0")
    a = store.experiment("a")
    va = store.commit(model, reg, branch=a, message="a1")
    b = store.experiment("b")
    store.commit(model, reg, branch=b, message="b1")
    with pytest.raises(LineageError):
        store.rollback(b, va, reason="no es ancestro")


def test_corruption_is_detected(tmp_path):
    store = LineageStore(tmp_path)
    model, reg = build_toy_model()
    v = store.commit(model, reg, branch=STABLE, message="v0")
    p = tmp_path / "versions" / v / "model.pt"
    p.write_bytes(p.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(LineageError):
        store.load(v)
