"""Smoke test for TUEV dataset integration.

This test verifies that TUEV data can be loaded and processed
through our pipeline with proper channel mapping and normalization.
"""

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.synth  # Can run with synthetic data
class TestTUEVSmoke:
    """Basic smoke tests for TUEV dataset integration using SSOT preprocessor."""

    def test_tuev_fixture_loads(self, tuev_sample_path):
        """Test that TUEV fixture provides a valid EDF file."""
        assert tuev_sample_path.exists()
        assert tuev_sample_path.suffix == ".edf"

    def test_tuev_preprocessor_contract(self, tuev_sample_path):
        """Test TUEV data through SSOT preprocessor meets strict contract."""
        from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor
        
        # Use SSOT preprocessor
        preprocessor = TUEVPreprocessor()
        epochs, info = preprocessor.process_raw(tuev_sample_path)
        
        # STRICT assertions on NORMALIZED output
        
        # 1. Channel count: EXACTLY 20 for TUEV
        assert len(epochs.ch_names) == 20, (
            f"Expected exactly 20 channels, got {len(epochs.ch_names)}"
        )
        
        # 2. Modern naming ONLY (T7 not T3)
        required_modern = ["T7", "T8", "P7", "P8"]
        for ch in required_modern:
            assert ch in epochs.ch_names, f"Missing required modern channel: {ch}"
        
        # 3. NO old naming
        forbidden_old = ["T3", "T4", "T5", "T6"]
        for ch in forbidden_old:
            assert ch not in epochs.ch_names, f"Old naming found: {ch}"
        
        # 4. TUEV specifics: HAS Fz, NO Fpz, HAS Oz
        assert "FZ" in epochs.ch_names, "TUEV must have Fz"
        assert "FPZ" not in epochs.ch_names, "TUEV must NOT have Fpz"
        assert "OZ" in epochs.ch_names, "TUEV must have Oz"
        
        # 5. Sampling rate EXACTLY 256Hz
        assert epochs.info["sfreq"] == 256, f"Expected 256Hz, got {epochs.info['sfreq']}"
        
        # 6. Voltage in REASONABLE range (microvolts in SI units - Volts)
        data = epochs.get_data()
        data_abs = np.abs(data)
        
        # Use robust quantiles instead of max (handles outliers better)
        q999 = np.quantile(data_abs, 0.999)
        q50 = np.median(data_abs)
        
        # After normalization, should be in microvolts range (1e-7 to 5e-3 V)
        assert q999 < 5e-3, f"Data too large (99.9th percentile): {q999}V"
        assert q999 > 1e-7, f"Data too small (99.9th percentile): {q999}V"
        assert q50 < 1e-3, f"Median too large: {q50}V"
        
        # 7. Epoch shape consistency
        n_epochs, n_channels, n_times = data.shape
        assert n_channels == 20, f"Inconsistent channel count in epochs"
        
        # 1 second at 256Hz = 256 samples (TUEV uses 1s windows for events)
        expected_samples = int(1.0 * 256)
        assert n_times == expected_samples, (
            f"Expected {expected_samples} samples per epoch, got {n_times}"
        )

    def test_tuev_channel_selection(self, tuev_sample_path):
        """Test that preprocessor correctly selects and orders channels."""
        from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor
        from brain_go_brrr.infra.data.channels import CHANNELS_TUEV_20
        
        preprocessor = TUEVPreprocessor()
        epochs, info = preprocessor.process_raw(tuev_sample_path)
        
        # Check channels match EXACTLY (TUEV is stricter than TUAB)
        expected_set = set(CHANNELS_TUEV_20)
        actual_set = set(epochs.ch_names)
        
        # Must match exactly
        assert actual_set == expected_set, (
            f"Channel mismatch. Missing: {expected_set - actual_set}, "
            f"Extra: {actual_set - expected_set}"
        )
        
        # Check order preservation (important for models)
        for i, expected_ch in enumerate(CHANNELS_TUEV_20):
            if i < len(epochs.ch_names):
                assert epochs.ch_names[i] == expected_ch, (
                    f"Channel order mismatch at position {i}: "
                    f"expected {expected_ch}, got {epochs.ch_names[i]}"
                )

    def test_tuev_provenance_tracking(self, tuev_sample_path):
        """Test that preprocessing tracks provenance correctly."""
        from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor
        
        preprocessor = TUEVPreprocessor()
        epochs, info = preprocessor.process_raw(tuev_sample_path)
        
        # Check provenance info
        assert "preprocessing" in info
        assert "channel_count" in info["preprocessing"]
        assert info["preprocessing"]["channel_count"] == 20
        assert info["preprocessing"]["sampling_rate"] == 256
        assert info["preprocessing"]["dataset"] == "TUEV"
        assert info["preprocessing"]["window_size"] == 1.0  # TUEV uses 1s windows

    @pytest.mark.slow
    def test_tuev_event_detection_shape(self, tuev_sample_path):
        """Test TUEV preprocessed data shape for event detection."""
        pytest.importorskip("torch")
        from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor
        
        preprocessor = TUEVPreprocessor()
        epochs, info = preprocessor.process_raw(tuev_sample_path)
        
        # Get epoch data
        data = epochs.get_data()
        
        # Event detection expects (batch, channels, samples)
        batch_size, n_channels, n_samples = data.shape
        
        # Verify shape for event detection
        assert n_channels == 20, "Exactly 20 channels for TUEV"
        assert n_samples == 256, "1 second at 256Hz for event detection"
        
        # Data should be normalized and ready for event detection model
        # (Would pass to model here in real usage)

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