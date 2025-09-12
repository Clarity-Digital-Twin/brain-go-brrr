import pytest

from brain_go_brrr.infra.eval.nedc_wrapper import NEDCClinicalEvaluator


@pytest.mark.unit
@pytest.mark.synth
def test_nedc_proxy_metrics_simple_case():
    evalr = NEDCClinicalEvaluator()
    refs = [(0.0, 10.0), (20.0, 30.0)]
    preds = [(0.0, 10.0), (100.0, 110.0)]  # 1 TP, 1 FA
    metrics = evalr.compute_all_metrics(preds, refs, duration_hours=1.0)

    assert 0.0 <= metrics["sensitivity"] <= 1.0
    assert metrics["sensitivity"] == pytest.approx(0.5, rel=1e-3)
    # 1 FA over 1 hour -> 24 FA per 24h
    assert metrics["fa_24h"] == pytest.approx(24.0, rel=1e-3)
    # TP=1, FP=1, FN=1 => precision=0.5, recall=0.5 => F1=0.5
    assert metrics["taes_f1"] == pytest.approx(0.5, rel=1e-3)

