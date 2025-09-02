"""Smoke test for TUEV dataset integration.

This test verifies that TUEV (TUH EEG Events) data can be loaded
and processed through our pipeline for event detection.
"""

import mne
import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.synth  # Can run with synthetic data
class TestTUEVSmoke:
    """Basic smoke tests for TUEV dataset integration."""

    def test_tuev_fixture_loads(self, tuev_sample_path):
        """Test that TUEV fixture provides a valid EDF file."""
        assert tuev_sample_path.exists()
        assert tuev_sample_path.suffix == ".edf"

    def test_tuev_file_readable(self, tuev_sample_path):
        """Test that TUEV EDF file can be read with MNE."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=False, verbose=False)
        assert raw is not None
        assert raw.info["sfreq"] > 0
        assert len(raw.ch_names) > 0

    def test_tuev_channel_mapping(self, tuev_sample_path):
        """Test TUEV channel mapping and standard 10-20 channels."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=False, verbose=False)

        # TUEV uses standard 10-20 system plus EOG
        expected_channels = [
            "FP1",
            "FP2",
            "F7",
            "F3",
            "FZ",
            "F4",
            "F8",
            "T3",
            "T7",  # Either old or new naming
            "C3",
            "CZ",
            "C4",
            "T4",
            "T8",  # Either old or new naming
            "T5",
            "P7",  # Either old or new naming
            "P3",
            "PZ",
            "P4",
            "T6",
            "P8",  # Either old or new naming
            "O1",
            "O2",
        ]

        ch_names_upper = [ch.upper() for ch in raw.ch_names]

        # Check that we have most standard channels
        found_count = 0
        for ch in expected_channels:
            if ch in ch_names_upper:
                found_count += 1

        # Should have at least 15 of the standard channels
        assert found_count >= 15, f"Only found {found_count} standard channels"

    def test_tuev_sampling_rate(self, tuev_sample_path):
        """Test TUEV data can be resampled to 256Hz if needed."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=True, verbose=False)
        original_sfreq = raw.info["sfreq"]

        if original_sfreq != 256:
            raw.resample(256, verbose=False)

        assert raw.info["sfreq"] == 256

    def test_tuev_channel_count(self, tuev_sample_path):
        """Test TUEV has expected channel count."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=False, verbose=False)

        # TUEV typically has 20-25 channels (including EOG/ECG)
        # Synthetic has 22, real might vary
        assert 18 <= len(raw.ch_names) <= 30, f"Unexpected channel count: {len(raw.ch_names)}"

    def test_tuev_data_shape(self, tuev_sample_path):
        """Test TUEV data has correct shape for event detection."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=True, verbose=False)

        # Get data
        data = raw.get_data()

        # Check shape
        n_channels, n_samples = data.shape
        assert n_channels > 0
        assert n_samples > 0

        # Check duration is reasonable (at least 30 seconds for events)
        duration = n_samples / raw.info["sfreq"]
        assert duration >= 30, f"Recording too short for events: {duration}s"

    def test_tuev_data_range(self, tuev_sample_path):
        """Test TUEV data is in expected voltage range."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=True, verbose=False)

        # Get data
        data = raw.get_data()

        # EEG should be in microvolts range (1e-6 to 1e-3 V)
        data_abs = np.abs(data)
        max_val = np.max(data_abs)
        mean_val = np.mean(data_abs)

        # Check reasonable ranges
        assert max_val < 1.0, f"Data too large: max={max_val}V"
        assert max_val > 1e-8, f"Data too small: max={max_val}V"
        assert mean_val < 1e-3, f"Mean too large: {mean_val}V"

    def test_tuev_event_detection_shape(self, tuev_sample_path):
        """Test TUEV data shape for event detection windows."""
        raw = mne.io.read_raw_edf(tuev_sample_path, preload=True, verbose=False)

        # Resample to 256Hz if needed
        if raw.info["sfreq"] != 256:
            raw.resample(256, verbose=False)

        # Event detection typically uses 1-2 second windows
        window_size = 256 * 2  # 512 samples for 2 seconds
        data = raw.get_data()

        if data.shape[1] >= window_size:
            # Extract windows
            n_windows = data.shape[1] // window_size
            assert n_windows > 0

            # Get first window
            window = data[:, :window_size]
            assert window.shape[1] == 512

    def test_tuev_with_synthetic_event(self, tuev_sample_path):
        """Test synthetic TUEV data contains simulated event."""
        import os

        # Only run this for synthetic data
        if os.environ.get("BGB_ALLOW_SYNTH_TUEV") != "1":
            pytest.skip("Only testing synthetic event detection")

        raw = mne.io.read_raw_edf(tuev_sample_path, preload=True, verbose=False)
        data = raw.get_data()

        # Synthetic data has a spike at 60 seconds
        if raw.info["sfreq"] == 256:
            event_time = 60 * 256
            if data.shape[1] > event_time + 128:
                # Check for amplitude increase around event
                baseline = np.mean(np.abs(data[:, :event_time]))
                event_region = np.mean(np.abs(data[:, event_time : event_time + 128]))

                # Event should have higher amplitude
                assert event_region > baseline * 1.5

    @pytest.mark.slow
    def test_tuev_with_eegpt_windows(self, tuev_sample_path):
        """Test TUEV data can be windowed for EEGPT processing."""
        pytest.importorskip("torch")

        raw = mne.io.read_raw_edf(tuev_sample_path, preload=True, verbose=False)

        # Resample to 256Hz if needed
        if raw.info["sfreq"] != 256:
            raw.resample(256, verbose=False)

        # EEGPT uses 4-second windows
        window_size = 256 * 4  # 1024 samples
        data = raw.get_data()

        # Calculate number of windows
        n_windows = (data.shape[1] - window_size) // (window_size // 2) + 1

        if n_windows > 0:
            # Extract first window
            window = data[:, :window_size]
            assert window.shape[1] == 1024

            # Would pass to EEGPT for feature extraction
            # EEGPT expects (batch, channels, samples)
            # and outputs (batch, 2048) features

    def test_tuev_dataconfig_integration(self):
        """Test that DataConfig properly resolves TUEV paths."""
        from brain_go_brrr.application.config import DataConfig

        config = DataConfig()

        # Test version property
        version = config.tuev_version
        assert version is not None
        # Version can be empty for versionless layouts or start with "v" for versioned
        assert version == "" or version.startswith("v")

        # Test sample file getter
        # This might return None if no data mounted
        sample = config.get_tuev_sample_file()
        # Don't assert it exists - just that method works
        assert sample is None or sample.suffix == ".edf"
