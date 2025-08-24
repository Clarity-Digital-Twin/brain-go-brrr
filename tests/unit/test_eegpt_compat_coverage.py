"""Additional tests to boost coverage for eegpt_compat module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from brain_go_brrr.infra.ml_models.eegpt_compat import (
    EEGPTConfig,
    EEGPTModel,
    extract_features_from_raw,
    preprocess_for_eegpt,
)


class TestEEGPTConfigEdgeCases:
    """Test edge cases for EEGPTConfig."""

    def test_config_post_init_calculates_window_samples(self):
        """Test that window_samples is calculated from duration and sampling rate."""
        config = EEGPTConfig(window_duration=2.0, sampling_rate=512)
        assert config.window_samples == 1024

    def test_config_n_patches_per_window_property(self):
        """Test n_patches_per_window property calculation."""
        config = EEGPTConfig(window_samples=1024, patch_size=64)
        assert config.n_patches_per_window == 16

    def test_config_validation_raises_on_indivisible_samples(self):
        """Test config raises error when window_samples not divisible by patch_size."""
        # 1.5 * 256 = 384, which is NOT divisible by 64 (384 = 64 * 6 = 384, wait that's divisible)
        # 1.2 * 256 = 307.2 -> 307, which is NOT divisible by 64
        with pytest.raises(ValueError, match="must be divisible"):
            EEGPTConfig(window_duration=1.2, sampling_rate=256, patch_size=64)


class TestEEGPTModelEdgeCases:
    """Test edge cases for EEGPTModel."""

    def test_model_with_dict_config(self):
        """Test model initialization with dictionary config."""
        config_dict = {"device": "cpu", "sampling_rate": 128}
        model = EEGPTModel(config=config_dict, auto_load=False)
        assert model.config.sampling_rate == 128
        assert model.device == torch.device("cpu")

    def test_model_with_object_config(self):
        """Test model initialization with object config."""
        from brain_go_brrr.application.config import ModelConfig

        config = ModelConfig()
        config.model_path = Path("/tmp/test.ckpt")
        model = EEGPTModel(config=config, auto_load=False)
        assert model.config is not None

    def test_extract_windows_method(self):
        """Test extract_windows method for backward compatibility."""
        model = EEGPTModel(auto_load=False)
        data = np.random.randn(20, 2048).astype(np.float32)  # 8 seconds
        windows = model.extract_windows(data, sampling_rate=256)
        assert len(windows) == 2  # Two 4-second windows
        assert windows[0].shape == (20, 1024)

    def test_extract_features_batch_with_numpy(self):
        """Test extract_features_batch with numpy input."""
        model = EEGPTModel(auto_load=False)
        model.is_loaded = True

        # Mock encoder
        model.encoder = MagicMock()
        model.encoder.extract_features = MagicMock(return_value=torch.zeros(4, 768))

        windows = np.random.randn(4, 20, 1024).astype(np.float32)
        features = model.extract_features_batch(windows)

        assert features.shape == (4, 768)
        assert features.dtype == np.float32

    def test_extract_features_batch_with_torch(self):
        """Test extract_features_batch with torch input."""
        model = EEGPTModel(auto_load=False)
        model.is_loaded = True

        # Mock encoder
        model.encoder = MagicMock()
        model.encoder.extract_features = MagicMock(return_value=torch.zeros(4, 768))

        windows = torch.randn(4, 20, 1024)
        features = model.extract_features_batch(windows)

        assert features.shape == (4, 768)
        assert features.dtype == np.float32

    def test_extract_features_batch_without_encoder(self):
        """Test extract_features_batch fallback when encoder is None."""
        model = EEGPTModel(auto_load=False)
        model.encoder = None

        windows = np.random.randn(4, 20, 1024).astype(np.float32)
        features = model.extract_features_batch(windows)

        # Should return zeros as fallback
        assert features.shape == (4, 768)
        assert np.allclose(features, 0)

    def test_predict_abnormality_method(self):
        """Test predict_abnormality method."""
        import mne

        model = EEGPTModel(auto_load=False)
        model.is_loaded = True

        # Mock encoder to return consistent features
        model.encoder = MagicMock()
        model.encoder.extract_features = MagicMock(return_value=torch.ones(1, 512))

        # Create test raw
        sfreq = 256
        data = np.random.randn(20, 2048) * 1e-6  # 8 seconds
        ch_names = [f"CH{i}" for i in range(20)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        result = model.predict_abnormality(raw)

        assert "abnormal_probability" in result
        assert "confidence" in result
        assert "window_scores" in result
        assert "n_windows_processed" in result
        assert result["n_windows_processed"] > 0

    def test_cleanup_method(self):
        """Test cleanup method for CUDA memory."""
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            model = EEGPTModel(device="cuda", auto_load=False)
            model.cleanup()
            mock_empty_cache.assert_called_once()

    def test_get_cached_channel_ids(self):
        """Test _get_cached_channel_ids method."""
        model = EEGPTModel(auto_load=False)
        channel_names = ["Fp1", "Fp2", "C3", "C4"]
        ids = model._get_cached_channel_ids(channel_names)
        assert ids == [0, 1, 2, 3]


class TestCompatibilityFunctions:
    """Test standalone compatibility functions."""

    def test_preprocess_for_eegpt_with_filters(self):
        """Test preprocessing with filters."""
        import mne

        # Create test raw
        sfreq = 512
        data = np.random.randn(20, 2048) * 1e-6
        ch_names = [f"CH{i}" for i in range(20)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Preprocess with resampling and filtering
        processed = preprocess_for_eegpt(
            raw,
            sampling_rate=256,
            bandpass=(1.0, 40.0),
            notch=60.0
        )

        assert processed.info["sfreq"] == 256
        assert isinstance(processed, mne.io.BaseRaw)

    def test_preprocess_for_eegpt_with_target_sfreq(self):
        """Test preprocessing with target_sfreq parameter."""
        import mne

        # Create test raw
        sfreq = 512
        data = np.random.randn(20, 2048) * 1e-6
        ch_names = [f"CH{i}" for i in range(20)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Use target_sfreq parameter (legacy name)
        processed = preprocess_for_eegpt(raw, target_sfreq=128)

        assert processed.info["sfreq"] == 128

    def test_extract_features_from_raw(self):
        """Test extract_features_from_raw convenience function."""
        import mne

        # Create test raw
        sfreq = 256
        data = np.random.randn(20, 1024) * 1e-6
        ch_names = [f"CH{i}" for i in range(20)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Mock model
        mock_model = MagicMock()
        mock_model.extract_features = MagicMock(return_value=np.zeros((1, 512), dtype=np.float32))

        with patch("brain_go_brrr.infra.ml_models.eegpt_compat.EEGPTModel", return_value=mock_model):
            features = extract_features_from_raw(raw, model=None)

        assert features.dtype == np.float32
        assert features.shape == (1, 512)

    def test_extract_features_from_raw_with_existing_model(self):
        """Test extract_features_from_raw with provided model."""
        import mne

        # Create test raw
        sfreq = 256
        data = np.random.randn(20, 1024) * 1e-6
        ch_names = [f"CH{i}" for i in range(20)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Create model
        model = EEGPTModel(auto_load=False)
        model.is_loaded = True
        model.encoder = MagicMock()
        model.encoder.extract_features = MagicMock(return_value=torch.zeros(1, 512))

        features = extract_features_from_raw(raw, model=model)

        assert features.dtype == np.float32
        assert features.shape == (1, 512)


class TestStrictModeValidation:
    """Test that strict mode properly validates shapes."""

    def test_wrong_summary_shape_raises(self):
        """Test that wrong summary shape raises ValueError in strict mode."""
        model = EEGPTModel(auto_load=False)
        model.is_loaded = True

        # Mock encoder that returns wrong shape
        model.encoder = MagicMock()
        model.encoder.extract_features = MagicMock(return_value=torch.ones(1, 768))

        data = np.random.randn(20, 1024).astype(np.float32)

        # Should raise ValueError for unexpected shape
        with pytest.raises(ValueError, match="Unexpected summary shape"):
            model.extract_features(data, summary=True)

    def test_wrong_token_shape_raises(self):
        """Test that wrong token shape raises ValueError in strict mode."""
        model = EEGPTModel(auto_load=False)
        model.is_loaded = True

        # Mock encoder that returns wrong token shape
        model.encoder = MagicMock()
        model.encoder.extract_features = MagicMock(return_value=torch.ones(1, 2048))

        data = np.random.randn(20, 1024).astype(np.float32)

        # Should raise ValueError for unexpected shape
        with pytest.raises(ValueError, match="Unexpected token shape"):
            model.extract_features(data, summary=False)
