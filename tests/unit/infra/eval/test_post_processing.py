import numpy as np
import pytest

from brain_go_brrr.infra.eval.post_processing import AdvancedPostProcessor


@pytest.mark.unit
@pytest.mark.synth
def test_hysteresis_basic_detection():
    # Create longer array with clear events to test hysteresis
    fs = 256
    p = AdvancedPostProcessor(
        hysteresis=(0.3, 0.7), min_duration_sec=0.01, merge_gap_sec=0.01, fs=fs
    )

    # Build array with two clear high-probability regions
    probs = np.zeros(fs * 2, dtype=float)  # 2 seconds
    probs[10:30] = 0.8  # First event
    probs[100:150] = 0.9  # Second event

    events = p.apply(probs)
    # Should detect two events
    assert len(events) == 2
    # Confidence should be within [0, 1]
    for _, _, conf in events:
        assert 0.7 <= conf <= 1.0  # Should match high threshold


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


@pytest.mark.unit
@pytest.mark.synth
def test_merge_boundary_inclusive():
    fs = 100
    # Two events separated by exactly merge_gap -> should merge
    probs = np.zeros(400, dtype=float)
    probs[10:20] = 0.9
    probs[30:40] = 0.9
    gap_sec = (30 - 20) / fs
    p = AdvancedPostProcessor(
        hysteresis=(0.5, 0.7), merge_gap_sec=gap_sec, min_duration_sec=0.0, fs=fs
    )
    events = p.apply(probs)
    assert len(events) == 1


@pytest.mark.unit
@pytest.mark.synth
def test_min_duration_boundary_inclusive():
    fs = 50
    # One event with length exactly min_duration -> should be kept
    probs = np.zeros(200, dtype=float)
    start, end = 10, 20
    probs[start:end] = 0.9
    min_dur = (end - start) / fs
    p = AdvancedPostProcessor(
        hysteresis=(0.5, 0.7), merge_gap_sec=0.0, min_duration_sec=min_dur, fs=fs
    )
    events = p.apply(probs)
    assert len(events) == 1
