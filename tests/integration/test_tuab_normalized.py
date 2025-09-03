"""TUAB integration tests using NORMALIZED data (the RIGHT way).

Tests the CONTRACT, not raw EDF headers.
Uses the SSOT preprocessor, not raw MNE reads.
"""


import numpy as np
import pytest

from brain_go_brrr.infra.preprocessing.mne_preprocessor import TUABPreprocessor


class TestTUABNormalized:
    """Test TUAB data using the SSOT normalizer."""

    def test_tuab_normalized_contract(self, tuab_sample_path):
        """Test TUAB normalized output meets strict contract."""
        # Use the SSOT preprocessor!
        preprocessor = TUABPreprocessor()

        # Process returns (epochs, info_dict)
        epochs, info = preprocessor.process_raw(tuab_sample_path)

        # STRICT assertions on NORMALIZED output

        # 1. 18 or 19 channels (TUAB standard - Oz is optional)
        assert len(epochs.ch_names) in [18, 19], (
            f"Expected 18-19 channels, got {len(epochs.ch_names)}"
        )

        # 2. Modern naming (T7 not T3)
        required_channels = ["T7", "T8", "P7", "P8"]
        for ch in required_channels:
            assert ch in epochs.ch_names, f"Missing required channel: {ch}"

        # 3. NO old naming
        forbidden_channels = ["T3", "T4", "T5", "T6"]
        for ch in forbidden_channels:
            assert ch not in epochs.ch_names, f"Old naming found: {ch}"

        # 4. Sampling rate EXACTLY 256Hz
        assert epochs.info["sfreq"] == 256, f"Expected 256Hz, got {epochs.info['sfreq']}"

        # 5. Voltage in REASONABLE range (microvolts)
        # Get epoch data instead of raw continuous
        epoch_data = epochs.get_data()  # (n_epochs, n_channels, n_times)

        # Robust quantile check (99.9th percentile)
        q999 = np.quantile(np.abs(epoch_data), 0.999)

        # EEG should be in microvolts range after preprocessing
        # 1e-7 V = 0.1 µV (noise floor)
        # 5e-3 V = 5000 µV (very large but possible after filtering)
        assert 1e-7 <= q999 <= 5e-3, f"Voltage out of range: {q999}V"

        # Mean should be much smaller
        mean_abs = np.mean(np.abs(epoch_data))
        assert mean_abs < 1e-4, f"Mean voltage too high: {mean_abs}V"

        # 6. Epochs are 4 seconds
        epoch_duration = epochs.times[-1] - epochs.times[0]
        assert abs(epoch_duration - 4.0) < 0.1, f"Epoch duration {epoch_duration}s, expected 4s"

        # 7. Info dict shows Autoreject worked
        assert info["n_epochs_after"] > 0, "No epochs survived Autoreject"
        assert info["n_rejected"] >= 0, "Autoreject info missing"

    def test_tuab_preprocessor_handles_bad_channels(self, tuab_sample_path):
        """Test that preprocessor handles missing/bad channels gracefully."""
        preprocessor = TUABPreprocessor()

        try:
            epochs, info = preprocessor.process_raw(tuab_sample_path)
            # If it succeeds, we should have valid output
            assert epochs is not None
            assert len(epochs) > 0
        except ValueError as e:
            # Only acceptable error is too few channels
            assert "Too few standard channels" in str(e)
            pytest.skip(f"File has insufficient channels: {e}")

    def test_tuab_epochs_consistent(self, tuab_sample_path):
        """Test that epochs are consistent shape."""
        preprocessor = TUABPreprocessor()
        epochs, _ = preprocessor.process_raw(tuab_sample_path)

        # Get all epoch data
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)

        # All epochs same shape
        n_epochs, n_channels, n_times = data.shape
        assert n_channels in [18, 19], f"Expected 18-19 channels per epoch, got {n_channels}"

        # 4 seconds at 256Hz = 1024 samples
        expected_samples = int(4.0 * 256)
        # Allow small tolerance for edge effects
        assert abs(n_times - expected_samples) <= 2, (
            f"Expected ~{expected_samples} samples, got {n_times}"
        )

        # Each epoch should be independent (no NaN contamination)
        for i in range(n_epochs):
            assert not np.any(np.isnan(data[i])), f"Epoch {i} contains NaN"
            assert not np.any(np.isinf(data[i])), f"Epoch {i} contains Inf"
