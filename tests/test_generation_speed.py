from hope.bench.generation_speed import bytes_per_token, measure
from hope.core.hn1 import build_hn1
from hope.device import select_device


def test_speed_report_has_token_equivalents():
    model, _ = build_hn1("hn1-tiny")
    bpt, how = bytes_per_token(None)
    assert bpt == 4.0 and "aproximado" in how
    rep = measure(model, device=select_device("cpu"), bpt=bpt, prompt_bytes=128, gen_bytes=8, reasoning=True)
    assert rep["generation_tokens_equiv_per_s"] * 4 == rep["generation_bytes_per_s"]
    assert rep["reasoning_iterations"] >= 1 and rep["reasoning_tokens_equiv"] > 0
