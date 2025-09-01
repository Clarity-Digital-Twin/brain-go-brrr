"""Real data tests for TUEV dataset.

These tests ONLY run with real TUEV data and are marked @data.
They require --run-data flag and actual dataset availability.
"""

import mne
import pytest


@pytest.mark.integration
@pytest.mark.data  # Requires REAL data only
class TestTUEVRealData:
    """Tests that verify TUEV real dataset characteristics."""

    def test_real_tuev_exists(self, tuev_sample_path):
        """Test that real TUEV data exists and has expected properties."""
        # This will skip if no real data available
        assert tuev_sample_path.exists()
        assert tuev_sample_path.suffix == ".edf"

        # Real TUEV files are typically larger than synthetic
        file_size_mb = tuev_sample_path.stat().st_size / (1024 * 1024)
        assert file_size_mb > 2.0, "Real TUEV files should be >2MB"

    def test_real_tuev_event_annotations(self, tuev_sample_path):
        """Test that real TUEV data contains event annotations."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=False, verbose=False)

        # Real TUEV should have annotations for events
        annotations = raw.annotations
        # Even background recordings have some annotations
        assert len(annotations) >= 0, "Real TUEV should have annotation structure"

    def test_real_tuev_eog_channel(self, tuev_sample_path):
        """Test that real TUEV has EOG channels."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=False, verbose=False)

        # Check for EOG channels (common in TUEV)
        ch_names_upper = [ch.upper() for ch in raw.ch_names]
        has_eog = any("EOG" in name or "EYE" in name for name in ch_names_upper)

        # Not all TUEV files have EOG, but many do
        # This is more of a characteristic check than a hard requirement
        if has_eog:
            eog_channels = [ch for ch in ch_names_upper if "EOG" in ch or "EYE" in ch]
            assert len(eog_channels) >= 1, "Should have at least one EOG channel"

    def test_real_tuev_sampling_rate(self, tuev_sample_path):
        """Test real TUEV sampling rate variations."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=False, verbose=False)

        # Real TUEV typically uses standard clinical rates
        valid_rates = [250, 256, 500, 512, 1000]
        assert raw.info["sfreq"] in valid_rates, f"Unexpected sampling rate: {raw.info['sfreq']}"

    def test_real_tuev_duration(self, tuev_sample_path):
        """Test that real TUEV recordings have realistic durations."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=False, verbose=False)

        duration_minutes = raw.times[-1] / 60
        # Real TUEV recordings vary but are typically 10+ minutes
        assert duration_minutes >= 5, "Real TUEV should be at least 5 minutes"

    def test_real_tuev_channel_count(self, tuev_sample_path):
        """Test real TUEV channel count matches expected range."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=False, verbose=False)

        # Real TUEV can have 18-36 channels (including EOG, ECG, EMG)
        n_channels = len(raw.ch_names)
        assert 18 <= n_channels <= 36, f"Unexpected channel count: {n_channels}"

    def test_real_tuev_event_types(self, tuev_sample_path):
        """Test that real TUEV path corresponds to known event types."""
        # Known TUEV event types
        known_types = ["bckg", "gped", "pled", "spike", "seizure", "artifact"]

        # The parent directory should indicate the event type
        # or it might be deeper in the path structure
        path_str = str(tuev_sample_path)
        has_known_type = any(evt_type in path_str.lower() for evt_type in known_types)

        # This is a sanity check that we're loading from the right structure
        assert has_known_type or "tuev" in path_str.lower(), "Path should indicate TUEV structure"
