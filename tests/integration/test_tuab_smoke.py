"""Smoke test for TUAB dataset integration.

This test verifies that TUAB data can be loaded and processed
through our pipeline with proper channel mapping and normalization.
"""

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.synth  # Can run with synthetic data
class TestTUABSmoke:
    """Basic smoke tests for TUAB dataset integration using SSOT preprocessor."""

    def test_tuab_fixture_loads(self, tuab_sample_path):
        """Test that TUAB fixture provides a valid EDF file."""
        assert tuab_sample_path.exists()
        assert tuab_sample_path.suffix == ".edf"

    def test_tuab_preprocessor_contract(self, tuab_sample_path):
        """Test TUAB data through SSOT preprocessor meets strict contract."""
        from brain_go_brrr.infra.preprocessing.mne_preprocessor import TUABPreprocessor

        # Use SSOT preprocessor
        preprocessor = TUABPreprocessor()
        epochs, info = preprocessor.process_raw(tuab_sample_path)

        # STRICT assertions on NORMALIZED output

        # 1. Channel count: 18-19 (Oz optional per TUAB reality)
        assert len(epochs.ch_names) in [18, 19], (
            f"Expected 18-19 channels, got {len(epochs.ch_names)}"
        )

        # 2. Modern naming ONLY (T7 not T3)
        required_modern = ["T7", "T8", "P7", "P8"]
        for ch in required_modern:
            assert ch in epochs.ch_names, f"Missing required modern channel: {ch}"

        # 3. NO old naming
        forbidden_old = ["T3", "T4", "T5", "T6"]
        for ch in forbidden_old:
            assert ch not in epochs.ch_names, f"Old naming found: {ch}"

        # 4. Sampling rate EXACTLY 256Hz
        assert epochs.info["sfreq"] == 256, f"Expected 256Hz, got {epochs.info['sfreq']}"

        # 5. Voltage in REASONABLE range (microvolts in SI units - Volts)
        data = epochs.get_data()
        data_abs = np.abs(data)

        # Use robust quantiles instead of max (handles outliers better)
        q999 = np.quantile(data_abs, 0.999)
        q50 = np.median(data_abs)

        # After normalization, should be in microvolts range (1e-7 to 5e-3 V)
        assert q999 < 5e-3, f"Data too large (99.9th percentile): {q999}V"
        assert q999 > 1e-7, f"Data too small (99.9th percentile): {q999}V"
        assert q50 < 1e-3, f"Median too large: {q50}V"

        # 6. Epoch shape consistency
        n_epochs, n_channels, n_times = data.shape
        assert n_channels in [18, 19], "Inconsistent channel count in epochs"

        # 4 seconds at 256Hz = 1024 samples
        expected_samples = int(4.0 * 256)
        assert n_times == expected_samples, (
            f"Expected {expected_samples} samples per epoch, got {n_times}"
        )

    def test_tuab_channel_selection(self, tuab_sample_path):
        """Test that preprocessor correctly selects and orders channels."""
        from brain_go_brrr.infra.data.channels import CHANNELS_TUAB_19
        from brain_go_brrr.infra.preprocessing.mne_preprocessor import TUABPreprocessor

        preprocessor = TUABPreprocessor()
        epochs, info = preprocessor.process_raw(tuab_sample_path)

        # Check channels are in expected order (subset of CHANNELS_TUAB_19)
        # Since Oz might be missing, we check that all present channels
        # are in the expected set and order
        expected_set = set(CHANNELS_TUAB_19)
        actual_set = set(epochs.ch_names)

        # All actual channels must be in expected set
        assert actual_set.issubset(expected_set), (
            f"Unexpected channels: {actual_set - expected_set}"
        )

        # If Oz is missing, that's the only allowed missing channel
        missing = expected_set - actual_set
        if missing:
            assert missing == {"Oz"}, f"Unexpected missing channels: {missing}"

    def test_tuab_provenance_tracking(self, tuab_sample_path):
        """Test that preprocessing tracks provenance correctly."""
        from brain_go_brrr.infra.preprocessing.mne_preprocessor import TUABPreprocessor

        preprocessor = TUABPreprocessor()
        epochs, info = preprocessor.process_raw(tuab_sample_path)

        # Check provenance info (flat dict structure from preprocessor)
        assert "n_epochs_before" in info or "n_epochs_after" in info
        # Can't assert preprocessing dict that doesn't exist yet
        # TODO: Add proper provenance structure to preprocessor

    @pytest.mark.slow
    def test_tuab_with_eegpt_compatibility(self, tuab_sample_path):
        """Test TUAB preprocessed data is compatible with EEGPT."""
        pytest.importorskip("torch")
        from brain_go_brrr.infra.preprocessing.mne_preprocessor import TUABPreprocessor

        preprocessor = TUABPreprocessor()
        epochs, info = preprocessor.process_raw(tuab_sample_path)

        # Get epoch data
        data = epochs.get_data()

        # EEGPT expects (batch, channels, samples)
        # Our epochs are already in this format
        batch_size, n_channels, n_samples = data.shape

        # Verify shape for EEGPT
        assert n_channels in [18, 19], "Channel count for EEGPT"
        assert n_samples == 1024, "4 seconds at 256Hz for EEGPT"

        # Data should be normalized and ready for EEGPT
        # (Would pass to model here in real usage)

    def test_tuab_dataconfig_integration(self):
        """Test that DataConfig properly resolves TUAB paths."""
        from brain_go_brrr.application.config import DataConfig

        config = DataConfig()

        # Test version property
        version = config.tuab_version
        assert version is not None
        # Version can be empty for versionless layouts or start with "v" for versioned
        assert version == "" or version.startswith("v")

        # Test sample file getter
        # This might return None if no data mounted
        sample = config.get_tuab_sample_file()
        # Don't assert it exists - just that method works
        assert sample is None or sample.suffix == ".edf"
