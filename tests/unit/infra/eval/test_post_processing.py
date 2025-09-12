import numpy as np
import pytest

from brain_go_brrr.infra.eval.post_processing import AdvancedPostProcessor


@pytest.mark.unit
@pytest.mark.synth
def test_hysteresis_basic_detection():
    # Use smaller min_duration to avoid filtering out short test events
    p = AdvancedPostProcessor(hysteresis=(0.3, 0.7), min_duration_sec=0.01, fs=256)
    probs = np.array([0.1, 0.2, 0.8, 0.65, 0.4, 0.2, 0.1, 0.9, 0.85, 0.2], dtype=float)
    events = p.apply(probs)
    # Should detect two events (roughly 2-5 and 7-9 indices)
    assert len(events) == 2
    # Confidence should be within [0, 1]
    for _, _, conf in events:
        assert 0.0 <= conf <= 1.0


@pytest.mark.unit
@pytest.mark.synth
def test_gap_merge_and_min_duration():
    fs = 256
    # Two events with a small gap that should merge
    probs = np.zeros(1024, dtype=float)
    probs[100:200] = 0.9
    probs[201:260] = 0.9  # 1 sample gap (<< merge_gap_sec)
    p = AdvancedPostProcessor(
        hysteresis=(0.5, 0.7), merge_gap_sec=0.05, min_duration_sec=0.1, fs=fs
    )
    events = p.apply(probs)
    assert len(events) == 1
    start_sec, end_sec, _ = events[0]
    # Duration should be >= min_duration
    assert (end_sec - start_sec) >= 0.1
