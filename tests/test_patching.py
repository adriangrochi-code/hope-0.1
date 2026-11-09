import numpy as np
import torch

from hope.core.hn1 import HN1Model, patch_boundaries, preset
from hope.core.patching import context_hash, fit_entropy_table

TEXT = ("la resistencia del circuito se calienta cuando la corriente aumenta. " * 40
        + "El regulador mantiene la tensión de salida constante. " * 40).encode()


def test_context_hash_is_causal():
    x = torch.tensor([list(b"abcdefghij")])
    y = x.clone()
    y[0, 6:] = 0
    assert torch.equal(context_hash(x, 3, 65536)[:, :6], context_hash(y, 3, 65536)[:, :6])


def test_entropy_patching_puts_boundaries_where_text_is_hard():
    data = np.frombuffer(TEXT, dtype=np.uint8)
    table = fit_entropy_table(data, order=3, buckets=4096)
    x = torch.from_numpy(data[:200].astype(np.int64)).unsqueeze(0)
    t = torch.from_numpy(table.astype(np.float32))
    end, _ = patch_boundaries(x, 16, entropy=(t, 2.0, 3))
    # Texto muy repetido: casi todo es predecible, así que los parches son largos.
    assert 200 / int(end.sum()) > 4


def test_hn1_with_entropy_patching_fits_trains_and_stays_causal(tmp_path):
    torch.manual_seed(0)
    cfg = preset("hn1-tiny")
    cfg.patching, cfg.entropy_buckets = "entropia", 4096
    model = HN1Model(cfg)
    info = model.fit_patcher(np.frombuffer(TEXT, dtype=np.uint8), target_bytes_per_patch=5.0)
    x = torch.tensor([list(TEXT[:120])])
    out = model(x)
    assert abs(1 / out["patches_per_byte"] - 5.0) < 2.0
    y = x.clone()
    y[:, 70:] = 7
    model.eval()
    with torch.no_grad():
        assert (model(x)["logits"][:, :70] - model(y)["logits"][:, :70]).abs().max() < 1e-5
    loss, _ = model.loss(x)
    loss.backward()
    torch.save(model.state_dict(), tmp_path / "m.pt")
    m2 = HN1Model(cfg)
    m2.load_state_dict(torch.load(tmp_path / "m.pt"))
    assert float(m2.entropy_threshold) == info["threshold_bits"] or abs(float(m2.entropy_threshold) - info["threshold_bits"]) < 1e-4
    assert model.generate("la resistencia", max_new=20) == model.generate_reference("la resistencia", max_new=20)
