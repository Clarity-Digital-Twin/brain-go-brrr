from pathlib import Path

import numpy as np
import pytest

from brain_go_brrr.infra.data.tusz_detection_dataset import _events_to_mask, _parse_tse


@pytest.mark.unit
@pytest.mark.synth
def test_parse_tse_and_events_to_mask(tmp_path: Path):
    """Test that TSE parser correctly filters seizure-only annotations."""
    tse = tmp_path / "rec.tse"
    tse.write_text(
        """
# Test TSE file with mixed annotations
0.0 1.0 seiz
5.0 7.5 FNSZ
10.0 10.5 seiz
15.0 20.0 background
25.0 30.0 artifact
35.0 40.0 focal_seizure
""".strip()
    )

    events = _parse_tse(tse)
    # Should only return lines with 'seiz' in label (case-insensitive)
    assert len(events) == 3  # seiz, seiz, focal_seizure
    assert events[0] == (0.0, 1.0)   # seiz
    assert events[1] == (10.0, 10.5)  # seiz
    assert events[2] == (35.0, 40.0)  # focal_seizure

    # Convert to mask at 256 Hz over duration 45 s
    fs = 256
    dur = 45.0
    mask = _events_to_mask(events, dur, fs)
    assert mask.dtype == np.bool_
    # Check approximate number of positive samples: 1.0 + 0.5 + 5.0 = 6.5 s
    assert mask.sum() == pytest.approx(int(6.5 * fs), rel=0.01)


@pytest.mark.unit
@pytest.mark.synth
def test_parse_tse_edge_cases(tmp_path: Path):
    """Test TSE parser handles edge cases correctly."""
    
    # Test 1: Empty file
    tse_empty = tmp_path / "empty.tse"
    tse_empty.write_text("")
    assert _parse_tse(tse_empty) == []
    
    # Test 2: Only comments
    tse_comments = tmp_path / "comments.tse"
    tse_comments.write_text("# Only comments\n# No data")
    assert _parse_tse(tse_comments) == []
    
    # Test 3: No seizure labels
    tse_no_seiz = tmp_path / "no_seizures.tse"
    tse_no_seiz.write_text("""
0.0 10.0 background
10.0 20.0 artifact
20.0 30.0 FNSZ
30.0 40.0 other
    """.strip())
    assert _parse_tse(tse_no_seiz) == []
    
    # Test 4: Lines without labels (should be ignored)
    tse_no_labels = tmp_path / "no_labels.tse"
    tse_no_labels.write_text("""
0.0 10.0
10.0 20.0
20.0 30.0 seizure
    """.strip())
    events = _parse_tse(tse_no_labels)
    assert len(events) == 1
    assert events[0] == (20.0, 30.0)
    
    # Test 5: Mixed case seizure labels
    tse_mixed_case = tmp_path / "mixed_case.tse"
    tse_mixed_case.write_text("""
0.0 10.0 SEIZURE
10.0 20.0 Seizure
20.0 30.0 SeIzUrE
30.0 40.0 focal_SEIZURE
    """.strip())
    events = _parse_tse(tse_mixed_case)
    assert len(events) == 4  # All should be detected
    
    # Test 6: Invalid numbers (should skip)
    tse_invalid = tmp_path / "invalid.tse"
    tse_invalid.write_text("""
0.0 10.0 seizure
not_a_number 20.0 seizure
30.0 also_not seizure
40.0 50.0 seizure
    """.strip())
    events = _parse_tse(tse_invalid)
    assert len(events) == 2  # Only valid lines
    assert events[0] == (0.0, 10.0)
    assert events[1] == (40.0, 50.0)
    
    # Test 7: Start >= End (should skip)
    tse_bad_range = tmp_path / "bad_range.tse"
    tse_bad_range.write_text("""
10.0 5.0 seizure
20.0 20.0 seizure
30.0 40.0 seizure
    """.strip())
    events = _parse_tse(tse_bad_range)
    assert len(events) == 1  # Only valid range
    assert events[0] == (30.0, 40.0)
