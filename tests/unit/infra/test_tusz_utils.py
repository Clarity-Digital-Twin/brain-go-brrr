from pathlib import Path

import numpy as np
import pytest

from brain_go_brrr.infra.data.tusz_detection_dataset import (
    _events_to_mask,
    _parse_csv,
    _parse_tse,
)
from brain_go_brrr.infra.data.tusz_labels import (
    TUSZ_EPILEPTIC_CODES,
    is_seizure_label,
    merge_intervals,
)


@pytest.mark.unit
@pytest.mark.synth
def test_tusz_seizure_codes():
    """Test that epileptic TUSZ seizure codes are recognized."""
    # All epileptic seizure codes should be recognized
    for code in TUSZ_EPILEPTIC_CODES:
        assert is_seizure_label(code)
        assert is_seizure_label(code.upper())
        assert is_seizure_label(f"prefix_{code}_suffix")

    # Non-seizure codes should not be recognized
    assert not is_seizure_label("bckg")
    assert not is_seizure_label("background")
    assert not is_seizure_label("artf")
    assert not is_seizure_label("artifact")
    assert not is_seizure_label("eyem")
    assert not is_seizure_label("eye_movement")

    # Generic seizure labels should still work
    assert is_seizure_label("seizure")
    assert is_seizure_label("focal_seizure")
    assert is_seizure_label("generalized_seizure")


@pytest.mark.unit
def test_merge_intervals():
    xs = [(0.0, 1.0), (0.5, 2.0), (3.0, 3.5), (3.5, 4.0), (5.0, 6.0)]
    merged = merge_intervals(xs)
    assert merged == [(0.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    # With gap merge of 0.2, (4.0,5.0) gap should merge
    xs2 = [(0.0, 1.0), (1.1, 2.0)]
    merged2 = merge_intervals(xs2, gap=0.2)
    assert merged2 == [(0.0, 2.0)]


@pytest.mark.unit
@pytest.mark.synth
def test_parse_tse_with_tusz_codes(tmp_path: Path):
    """Test TSE parsing with actual TUSZ seizure codes."""
    tse_file = tmp_path / "test.tse"
    tse_file.write_text(
        """
0.0 10.0 bckg
10.0 20.0 fnsz
20.0 30.0 artf
30.0 40.0 gnsz
40.0 50.0 eyem
50.0 60.0 cpsz
60.0 70.0 background
70.0 80.0 tcsz
    """.strip()
    )

    events = _parse_tse(tse_file)
    # Should only get the actual seizure events
    assert len(events) == 4
    assert events[0] == (10.0, 20.0)  # fnsz
    assert events[1] == (30.0, 40.0)  # gnsz
    assert events[2] == (50.0, 60.0)  # cpsz
    assert events[3] == (70.0, 80.0)  # tcsz


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
    # Should return FNSZ (TUSZ code) and lines with 'seiz' in label
    assert len(events) == 4  # seiz, FNSZ, seiz, focal_seizure
    assert events[0] == (0.0, 1.0)  # seiz
    assert events[1] == (5.0, 7.5)  # FNSZ (TUSZ seizure code)
    assert events[2] == (10.0, 10.5)  # seiz
    assert events[3] == (35.0, 40.0)  # focal_seizure

    # Convert to mask at 256 Hz over duration 45 s
    fs = 256
    dur = 45.0
    mask = _events_to_mask(events, dur, fs)
    assert mask.dtype == np.bool_
    # Check approximate number of positive samples: 1.0 + 2.5 + 0.5 + 5.0 = 9.0 s
    assert mask.sum() == pytest.approx(int(9.0 * fs), rel=0.01)


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
    tse_no_seiz.write_text(
        """
0.0 10.0 background
10.0 20.0 artifact
20.0 30.0 FNSZ
30.0 40.0 other
    """.strip()
    )
    assert _parse_tse(tse_no_seiz) == []

    # Test 4: Lines without labels (should be ignored)
    tse_no_labels = tmp_path / "no_labels.tse"
    tse_no_labels.write_text(
        """
0.0 10.0
10.0 20.0
20.0 30.0 seizure
    """.strip()
    )
    events = _parse_tse(tse_no_labels)
    assert len(events) == 1
    assert events[0] == (20.0, 30.0)

    # Test 5: Mixed case seizure labels
    tse_mixed_case = tmp_path / "mixed_case.tse"
    tse_mixed_case.write_text(
        """
0.0 10.0 SEIZURE
10.0 20.0 Seizure
20.0 30.0 SeIzUrE
30.0 40.0 focal_SEIZURE
    """.strip()
    )
    events = _parse_tse(tse_mixed_case)
    assert len(events) == 4  # All should be detected

    # Test 6: Invalid numbers (should skip)
    tse_invalid = tmp_path / "invalid.tse"
    tse_invalid.write_text(
        """
0.0 10.0 seizure
not_a_number 20.0 seizure
30.0 also_not seizure
40.0 50.0 seizure
    """.strip()
    )
    events = _parse_tse(tse_invalid)
    assert len(events) == 2  # Only valid lines
    assert events[0] == (0.0, 10.0)
    assert events[1] == (40.0, 50.0)

    # Test 7: Start >= End (should skip)
    tse_bad_range = tmp_path / "bad_range.tse"
    tse_bad_range.write_text(
        """
10.0 5.0 seizure
20.0 20.0 seizure
30.0 40.0 seizure
    """.strip()
    )
    events = _parse_tse(tse_bad_range)
    assert len(events) == 1  # Only valid range
    assert events[0] == (30.0, 40.0)


@pytest.mark.unit
@pytest.mark.synth
def test_parse_tse_coalesces_duplicate_events(tmp_path: Path):
    """TSE with repeated seizure lines should merge overlaps/touching intervals."""
    tse = tmp_path / "dupe.tse"
    tse.write_text(
        """
10.0 20.0 fnsz
10.0 20.0 fnsz
19.5 30.0 focal_seizure
30.0 40.0 gnsz
        """.strip()
    )
    events = _parse_tse(tse)
    # (10,20) and (19.5,30) should merge to (10,30) with gap=0.0 due to overlap
    assert events == [(10.0, 30.0), (30.0, 40.0)]


@pytest.mark.unit
@pytest.mark.synth
def test_parse_csv_tusz_format(tmp_path: Path):
    """Test CSV parsing with actual TUSZ CSV format."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        """
channel,start_time,stop_time,label,confidence
FP1-F7,0.0,10.0,bckg,1.0
FP1-F7,10.0,20.0,fnsz,1.0
F7-T3,20.0,30.0,artf,1.0
F7-T3,30.0,40.0,gnsz,1.0
T3-T5,40.0,50.0,eyem,1.0
T3-T5,50.0,60.0,mysz,1.0
T5-O1,60.0,70.0,absz,1.0
    """.strip()
    )

    events = _parse_csv(csv_file)
    # Should only get the actual seizure events
    assert len(events) == 4
    assert events[0] == (10.0, 20.0)  # fnsz
    assert events[1] == (30.0, 40.0)  # gnsz
    assert events[2] == (50.0, 60.0)  # mysz
    assert events[3] == (60.0, 70.0)  # absz


@pytest.mark.unit
@pytest.mark.synth
def test_parse_csv_coalesces_duplicate_channel_events(tmp_path: Path):
    """TUSZ repeats same event per-channel; merged intervals must coalesce."""
    csv_file = tmp_path / "dupe.csv"
    # Two channels annotate the same seizure interval; should merge to one
    csv_file.write_text(
        """
channel,start_time,stop_time,label,confidence
Fp1-F7,10.0,20.0,fnsz,1.0
F7-T3,10.0,20.0,fnsz,0.9
Fp1-F7,30.0,40.0,gnsz,1.0
F7-T3,30.0,40.0,gnsz,1.0
        """.strip()
    )

    events = _parse_csv(csv_file)
    assert events == [(10.0, 20.0), (30.0, 40.0)]


@pytest.mark.unit
@pytest.mark.synth
def test_parse_csv_edge_cases(tmp_path: Path):
    """Test CSV parser handles edge cases correctly with same seizure-only semantics."""
    # Test 1: Comma-delimited CSV with TUSZ codes
    csv_comma = tmp_path / "comma.csv"
    csv_comma.write_text(
        """
0.0,10.0,bckg
10.0,20.0,fnsz
20.0,30.0,artf
30.0,40.0,focal_seizure
40.0,50.0,gnsz
50.0,60.0
    """.strip()
    )
    events = _parse_csv(csv_comma)
    assert len(events) == 3  # Only seizure lines
    assert events[0] == (10.0, 20.0)  # fnsz
    assert events[1] == (30.0, 40.0)  # focal_seizure
    assert events[2] == (40.0, 50.0)  # gnsz

    # Test 2: Space-delimited CSV (like TSE)
    csv_space = tmp_path / "space.csv"
    csv_space.write_text(
        """
0.0 10.0 background
10.0 20.0 seizure_type_1
20.0 30.0 artifact
30.0 40.0 focal_seizure
    """.strip()
    )
    events = _parse_csv(csv_space)
    assert len(events) == 2
    assert events[0] == (10.0, 20.0)
    assert events[1] == (30.0, 40.0)

    # Test 3: Mixed delimiters
    csv_mixed = tmp_path / "mixed.csv"
    csv_mixed.write_text(
        """
0.0, 10.0, background
10.0  20.0  seizure_type_1
20.0,30.0,artifact
30.0 40.0 focal_seizure
    """.strip()
    )
    events = _parse_csv(csv_mixed)
    assert len(events) == 2
    assert events[0] == (10.0, 20.0)
    assert events[1] == (30.0, 40.0)

    # Test 4: Empty file
    csv_empty = tmp_path / "empty.csv"
    csv_empty.write_text("")
    assert _parse_csv(csv_empty) == []

    # Test 5: Comments and headers
    csv_headers = tmp_path / "headers.csv"
    csv_headers.write_text(
        """
# start,end,label
# This is a comment
0.0,10.0,background
10.0,20.0,seizure
    """.strip()
    )
    events = _parse_csv(csv_headers)
    assert len(events) == 1
    assert events[0] == (10.0, 20.0)

    # Test 6: Non-existent file
    csv_missing = tmp_path / "missing.csv"
    assert _parse_csv(csv_missing) == []

    # Test 7: Case-insensitive seizure detection
    csv_case = tmp_path / "case.csv"
    csv_case.write_text(
        """
0.0,10.0,SEIZURE
10.0,20.0,Seizure
20.0,30.0,SeIzUrE
30.0,40.0,focal_SEIZURE
    """.strip()
    )
    events = _parse_csv(csv_case)
    assert len(events) == 4  # All should be detected

    # Test 8: Invalid numbers (should skip)
    csv_invalid = tmp_path / "invalid.csv"
    csv_invalid.write_text(
        """
0.0,10.0,seizure
not_a_number,20.0,seizure
30.0,also_not,seizure
40.0,50.0,seizure
    """.strip()
    )
    events = _parse_csv(csv_invalid)
    assert len(events) == 2  # Only valid lines
    assert events[0] == (0.0, 10.0)
    assert events[1] == (40.0, 50.0)

    # Test 9: Start >= End (should skip)
    csv_bad_range = tmp_path / "bad_range.csv"
    csv_bad_range.write_text(
        """
10.0,5.0,seizure
20.0,20.0,seizure
30.0,40.0,seizure
    """.strip()
    )
    events = _parse_csv(csv_bad_range)
    assert len(events) == 1  # Only valid range
    assert events[0] == (30.0, 40.0)
