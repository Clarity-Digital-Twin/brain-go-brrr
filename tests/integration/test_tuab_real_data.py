"""Real data tests for TUAB dataset.

These tests ONLY run with real TUAB data and are marked @data.
They require --run-data flag and actual dataset availability.
"""


import mne
import pytest


@pytest.mark.integration
@pytest.mark.data  # Requires REAL data only
class TestTUABRealData:
    """Tests that verify TUAB real dataset characteristics."""

    def test_real_tuab_exists(self, tuab_sample_path):
        """Test that real TUAB data exists and has expected properties."""
        # This will skip if no real data available
        assert tuab_sample_path.exists()
        assert tuab_sample_path.suffix == ".edf"

        # Real TUAB files are typically larger than synthetic
        file_size_mb = tuab_sample_path.stat().st_size / (1024 * 1024)
        assert file_size_mb > 1.0, "Real TUAB files should be >1MB"

    def test_real_tuab_channel_names(self, tuab_sample_path):
        """Test that real TUAB data has old channel naming (T3/T4/T5/T6)."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=False, verbose=False)

        # Real TUAB uses old naming convention
        ch_names_upper = [ch.upper() for ch in raw.ch_names]

        # Check for at least some old names (real data characteristic)
        old_names_found = any(name in ch_names_upper for name in ["T3", "T4", "T5", "T6"])
        assert old_names_found, "Real TUAB should have old channel names"

    def test_real_tuab_sampling_rate(self, tuab_sample_path):
        """Test real TUAB sampling rate variations."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=False, verbose=False)

        # Real TUAB can have various sampling rates (250, 256, 512 Hz)
        valid_rates = [250, 256, 512]
        assert raw.info["sfreq"] in valid_rates, f"Unexpected sampling rate: {raw.info['sfreq']}"

    def test_real_tuab_duration(self, tuab_sample_path):
        """Test that real TUAB recordings have realistic durations."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=False, verbose=False)

        duration_minutes = raw.times[-1] / 60
        # Real clinical recordings are typically 10-30 minutes
        assert duration_minutes >= 5, "Real TUAB should be at least 5 minutes"

    def test_real_tuab_channel_count(self, tuab_sample_path):
        """Test real TUAB channel count matches clinical standard."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=False, verbose=False)

        # Real TUAB typically has 19-21 EEG channels
        n_channels = len(raw.ch_names)
        assert 19 <= n_channels <= 25, f"Unexpected channel count: {n_channels}"
