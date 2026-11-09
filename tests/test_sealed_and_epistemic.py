import pytest

from hope.epistemic import EpistemicRecord, EpistemicStatus as S, Evidence, EvidenceRequired
from hope.eval.sealed import SealError, SealedRegistry, contamination_report


def test_sealed_sets(tmp_path):
    data = tmp_path / "ohm.jsonl"
    data.write_text('{"q": "V=IR, I=2A, R=3ohm, V=?"}\n')
    reg = SealedRegistry(tmp_path / "sealed")
    reg.seal("ohm_v1", data, max_decision_reads=2)
    with pytest.raises(SealError):
        reg.open("ohm_v1", purpose="training", requester="trainer")
    reg.open("ohm_v1", purpose="evaluation", requester="asg", decision=True)
    assert not reg.is_worn("ohm_v1")
    reg.open("ohm_v1", purpose="evaluation", requester="asg", decision=True)
    assert reg.is_worn("ohm_v1")
    data.write_text('{"q": "modificado"}\n')
    with pytest.raises(SealError):
        reg.open("ohm_v1", purpose="evaluation", requester="asg")


def test_contamination():
    train = ["la ley de ohm dice que la tension es igual a la corriente por la resistencia del circuito"]
    evals = ["pregunta nueva sobre potencia", train[0]]
    rep = contamination_report(train, evals, n=8)
    assert rep["indices"] == [1]


def test_epistemic_promotion_requires_evidence():
    r = EpistemicRecord(S.GENERADO_EN_SUENO)
    r.transition(S.HIPOTESIS)                       # mismo nivel: no requiere evidencia
    with pytest.raises(EvidenceRequired):
        r.transition(S.RESPALDADO)
    r.transition(S.PROBADO, Evidence("exp_1"))
    with pytest.raises(EvidenceRequired):
        r.transition(S.VALIDADO, Evidence("exp_1", independent=False))
    r.transition(S.VALIDADO, Evidence("tve_7", independent=True))
    r.transition(S.REFUTADO)                        # bajar siempre está permitido
    assert [h[1] for h in r.history] == ["HIPOTESIS", "PROBADO", "VALIDADO", "REFUTADO"]
