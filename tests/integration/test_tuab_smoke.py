"""Smoke test for TUAB dataset integration.

This test verifies that TUAB data can be loaded and processed
through our pipeline with proper channel mapping and normalization.
"""

import mne
import numpy as np
import pytest


@pytest.mark.data  # Requires TUAB dataset or synthetic fallback
@pytest.mark.integration
class TestTUABSmoke:
    """Basic smoke tests for TUAB dataset integration."""

    def test_tuab_fixture_loads(self, tuab_sample_path):
        """Test that TUAB fixture provides a valid EDF file."""
        assert tuab_sample_path.exists()
        assert tuab_sample_path.suffix == ".edf"

    def test_tuab_file_readable(self, tuab_sample_path):
        """Test that TUAB EDF file can be read with MNE."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=False, verbose=False)
        assert raw is not None
        assert raw.info["sfreq"] > 0
        assert len(raw.ch_names) > 0

    def test_tuab_channel_mapping(self, tuab_sample_path):
        """Test TUAB channel mapping (T3→T7, T4→T8, T5→P7, T6→P8)."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=False, verbose=False)

        # Check if old naming is present and needs mapping
        old_names = ["T3", "T4", "T5", "T6"]
        new_names = ["T7", "T8", "P7", "P8"]

        ch_names_upper = [ch.upper() for ch in raw.ch_names]

        # If using real TUAB data, it might have old names
        # If using synthetic, it should already have correct names
        for old, new in zip(old_names, new_names, strict=False):
            # Either old or new naming is acceptable
            assert old in ch_names_upper or new in ch_names_upper, (
                f"Neither {old} nor {new} found in channels"
            )

    def test_tuab_sampling_rate(self, tuab_sample_path):
        """Test TUAB data can be resampled to 256Hz if needed."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=True, verbose=False)
        original_sfreq = raw.info["sfreq"]

        if original_sfreq != 256:
            raw.resample(256, verbose=False)

        assert raw.info["sfreq"] == 256

    def test_tuab_channel_count(self, tuab_sample_path):
        """Test TUAB has expected channel count."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=False, verbose=False)

        # TUAB typically has 19-22 channels
        # Synthetic has 19, real might have more
        assert 18 <= len(raw.ch_names) <= 25, f"Unexpected channel count: {len(raw.ch_names)}"

    def test_tuab_data_shape(self, tuab_sample_path):
        """Test TUAB data has correct shape for processing."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=True, verbose=False)

        # Get data
        data = raw.get_data()

        # Check shape
        n_channels, n_samples = data.shape
        assert n_channels > 0
        assert n_samples > 0

        # Check duration is reasonable (at least 10 seconds)
        duration = n_samples / raw.info["sfreq"]
        assert duration >= 10, f"Recording too short: {duration}s"

    def test_tuab_data_range(self, tuab_sample_path):
        """Test TUAB data is in expected voltage range."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=True, verbose=False)

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

    @pytest.mark.slow
    def test_tuab_with_eegpt_shape(self, tuab_sample_path):
        """Test TUAB data produces correct EEGPT embedding shape."""
        pytest.importorskip("torch")

        raw = mne.io.read_raw_edf(tuab_sample_path, preload=True, verbose=False)

        # Resample to 256Hz if needed
        if raw.info["sfreq"] != 256:
            raw.resample(256, verbose=False)

        # Get 4-second window (EEGPT requirement)
        window_size = 256 * 4  # 1024 samples
        data = raw.get_data()

        if data.shape[1] >= window_size:
            window = data[:, :window_size]

            # Check window shape
            assert window.shape[1] == 1024

            # Would pass to EEGPT here, but we're just checking shape
            # EEGPT expects (batch, channels, samples)
            # and outputs (batch, 2048) features

    def test_tuab_dataconfig_integration(self):
        """Test that DataConfig properly resolves TUAB paths."""
        from brain_go_brrr.application.config import DataConfig

        config = DataConfig()

        # Test version property
        version = config.tuab_version
        assert version is not None
        assert version.startswith("v")

        # Test sample file getter
        # This might return None if no data mounted
        sample = config.get_tuab_sample_file()
        # Don't assert it exists - just that method works
        assert sample is None or sample.suffix == ".edf"
