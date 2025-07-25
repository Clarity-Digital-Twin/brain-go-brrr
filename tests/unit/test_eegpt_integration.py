"""Test suite for EEGPT model integration.

Following TDD approach - tests written before implementation.
Based on EEGPT paper specifications and architecture diagrams.
"""

from pathlib import Path

import mne
import numpy as np
import pytest
import torch

from brain_go_brrr.models.eegpt_model import (
    EEGPTConfig,
    EEGPTModel,
    extract_features_from_raw,
    preprocess_for_eegpt,
)


class TestEEGPTConfig:
    """Test EEGPT configuration handling."""

    def test_default_config(self):
        """Test default EEGPT configuration matches paper specs."""
        config = EEGPTConfig()

        # From EEGPT paper specifications
        assert config.sampling_rate == 256  # Hz
        assert config.window_duration == 4.0  # seconds
        assert config.window_samples == 1024  # 4s * 256Hz
        assert config.patch_size == 64  # 250ms patches
        assert config.n_patches_per_window == 16  # 1024 / 64
        assert config.max_channels == 58
        assert config.model_size == "large"  # 10M parameters

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid window duration
        config = EEGPTConfig(window_duration=3.7)  # Would give 947.2 samples
        with pytest.raises(ValueError, match="Window duration must result in integer samples"):
            _ = config.window_samples  # This triggers the validation

        # Test invalid patch size
        config = EEGPTConfig(patch_size=100)  # 1024 not divisible by 100
        with pytest.raises(ValueError, match="Patch size must divide window samples"):
            _ = config.n_patches_per_window  # This triggers the validation


class TestEEGPTModel:
    """Test EEGPT model loading and inference."""

    @pytest.fixture
    def model_path(self):
        """Path to pretrained EEGPT model."""
        # Use relative path from project root
        project_root = Path(__file__).parent.parent.parent
        return project_root / "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"

    @pytest.fixture
    def eegpt_model(self, model_path):
        """Initialize EEGPT model."""
        return EEGPTModel(checkpoint_path=model_path)

    def test_model_loading(self, model_path):
        """Test model loads from checkpoint."""
        model = EEGPTModel(checkpoint_path=model_path)

        assert model is not None
        assert model.encoder is not None
        assert model.device in [torch.device("cpu"), torch.device("cuda")]
        assert model.is_loaded is True

    def test_model_architecture(self, eegpt_model):
        """Test model architecture matches paper specifications."""
        # Based on EEGPT architecture diagram (page 3)
        assert hasattr(eegpt_model.encoder, "patch_embed")
        assert hasattr(eegpt_model.encoder, "blocks")  # Transformer blocks
        assert hasattr(eegpt_model.encoder, "norm")  # Layer normalization

        # Check for summary tokens (S=4 from paper)
        assert eegpt_model.n_summary_tokens == 4

    def test_preprocessing_pipeline(self):
        """Test preprocessing matches EEGPT requirements."""
        # Create synthetic EEG data
        sfreq = 100  # Original sampling rate
        n_channels = 19
        duration = 10  # seconds

        data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6  # µV scale converted to V
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)  # MNE expects data in volts

        # Preprocess for EEGPT
        processed = preprocess_for_eegpt(raw)

        # Check preprocessing results
        assert processed.info["sfreq"] == 256  # Resampled to 256 Hz
        assert processed.get_data().shape[0] <= 58  # Max 58 channels

        # Check data is in reasonable range (MNE uses V internally)
        # Our input was µV, MNE converts to V, so values should be small
        data_v = processed.get_data()  # In volts
        assert np.abs(data_v).max() < 0.001  # Should be < 1mV when in V

    def test_window_extraction(self, eegpt_model):
        """Test 4-second window extraction."""
        # Create 20-second recording
        sfreq = 256
        n_channels = 19
        duration = 20

        data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6  # V scale (50 µV)

        # Extract windows
        windows = eegpt_model.extract_windows(data, sfreq)

        # Should have 5 non-overlapping 4-second windows
        assert len(windows) == 5
        assert windows[0].shape == (n_channels, 1024)  # 4s * 256Hz

    def test_feature_extraction(self, eegpt_model):
        """Test feature extraction from windows."""
        # Create single 4-second window
        window = np.random.randn(19, 1024) * 50e-6  # (channels, samples) in V scale (50 µV)
        channel_names = [f"EEG{i:03d}" for i in range(19)]

        # Extract features
        features = eegpt_model.extract_features(window, channel_names)

        # Check output shape - should be summary tokens
        assert features.shape[0] == 4  # 4 summary tokens
        assert features.shape[1] > 0  # Feature dimension

    def test_abnormality_prediction(self, eegpt_model):
        """Test abnormality score prediction."""
        # Create test EEG data
        sfreq = 256
        n_channels = 19
        duration = 20  # 20 seconds

        data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6  # V scale (50 µV)
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Get abnormality prediction
        result = eegpt_model.predict_abnormality(raw)

        # Check result structure
        assert "abnormal_probability" in result
        assert "confidence" in result
        assert "window_scores" in result

        # Check value ranges
        assert 0 <= result["abnormal_probability"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert len(result["window_scores"]) == 5  # 5 windows in 20s

    def test_channel_adaptation(self, eegpt_model):
        """Test adaptive spatial filter for different channel configurations."""
        # Test with different channel counts
        for n_channels in [10, 19, 32, 58]:
            window = np.random.randn(n_channels, 1024) * 50e-6
            channel_names = [f"EEG{i:03d}" for i in range(n_channels)]

            # Should handle variable channel counts
            features = eegpt_model.extract_features(window, channel_names)
            assert features is not None
            assert features.shape[0] == 4  # Always 4 summary tokens

    def test_batch_processing(self, eegpt_model):
        """Test batch processing of multiple windows."""
        # Create batch of windows
        batch_size = 8
        windows = np.random.randn(batch_size, 19, 1024) * 50e-6

        # Process batch
        features = eegpt_model.extract_features_batch(windows)

        # Check output shape
        assert features.shape[0] == batch_size
        assert features.shape[1] == 4  # Summary tokens

    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_pipeline(self, model_path):
        """Test complete pipeline from raw EEG to abnormality score."""
        # Load Sleep-EDF test file
        edf_path = Path(
            "/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"
        )

        if not edf_path.exists():
            pytest.skip("Sleep-EDF data not available")

        # Load only 1 minute of EEG data for testing
        # (Full file is 22 hours - way too much for a unit test!)
        raw = mne.io.read_raw_edf(edf_path, preload=False)

        # Crop to first minute
        raw.crop(tmin=0, tmax=60)
        raw.load_data()

        # Extract features using EEGPT
        features = extract_features_from_raw(raw, model_path)

        # Check results
        assert "features" in features
        assert "abnormal_probability" in features
        assert "processing_time" in features
        assert features["abnormal_probability"] is not None

        # Performance check: should process 1 minute in < 5 seconds
        assert features["processing_time"] < 5.0, (
            f"Processing too slow: {features['processing_time']:.2f}s"
        )
