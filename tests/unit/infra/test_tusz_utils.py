from pathlib import Path

import numpy as np
import pytest

from brain_go_brrr.infra.data.tusz_detection_dataset import _events_to_mask, _parse_tse


@pytest.mark.unit
@pytest.mark.synth
def test_parse_tse_and_events_to_mask(tmp_path: Path):
    tse = tmp_path / "rec.tse"
    tse.write_text(
        """
0.0 1.0 seiz
5.0 7.5 FNSZ
10.0 10.5 seiz
""".strip()
    )

    events = _parse_tse(tse)
    assert len(events) == 3
    assert events[0] == (0.0, 1.0)

    # Convert to mask at 256 Hz over duration 12 s
    fs = 256
    dur = 12.0
    mask = _events_to_mask(events, dur, fs)
    assert mask.dtype == np.bool_
    # Check approximate number of positive samples: 1.0 + 2.5 + 0.5 = 4.0 s
    assert mask.sum() == pytest.approx(int(4.0 * fs), rel=0.01)
