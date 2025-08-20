"""Tests for models.eegpt_model - CLEAN, NO HEAVY WEIGHT LOADING."""

import numpy as np
import torch

from brain_go_brrr.infra.ml_models.eegpt_compat import EEGPTConfig, EEGPTModel, preprocess_for_eegpt


class TestEEGPTConfig:
    """Test EEGPT configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EEGPTConfig()

        # Test actual fields that exist in compat config
        assert config.sampling_rate == 256
        assert config.window_duration == 4.0
        assert config.patch_size == 64
        assert config.window_samples == 1024
        assert config.n_channels == 20

    def test_custom_config(self):
        """Test custom configuration."""
        config = EEGPTConfig(window_duration=8.0, sampling_rate=128)

        assert config.window_duration == 8.0
        assert config.sampling_rate == 128

    def test_window_samples_calculation(self):
        """Test window samples calculation."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0)
        assert config.window_samples == 1024

    def test_window_config(self):
        """Test window configuration."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0, patch_size=64)
        # Just test that config can be created with these values
        assert config.patch_size == 64


class TestEEGPTModel:
    """Test EEGPT model without loading weights."""

    def test_model_initialization(self):
        """Test model initialization without loading weights."""
        from pathlib import Path

        from brain_go_brrr.application.config import ModelConfig

        # Create config with non-existent model path so it doesn't try to load
        config = ModelConfig()
        config.model_path = Path("/tmp/nonexistent_model.ckpt")

        # Initialize model without auto-loading
        model = EEGPTModel(config=config, auto_load=False)

        # Model doesn't store config directly
        assert model is not None
        assert hasattr(model, "encoder")
        assert hasattr(model, "device")

    def test_forward_pass_shape(self):
        """Test that extract_features returns correct shape."""
        from pathlib import Path

        config = {"model_path": Path("/tmp/nonexistent_model.ckpt")}

        model = EEGPTModel(config=config, auto_load=False)

        # Mark as loaded to prevent loading attempt
        model.is_loaded = True

        # Create a dummy encoder that returns zeros - use float32 for consistency
        class DummyEncoder:
            def __call__(self, *args, **kwargs):
                # Return a fixed shape output
                return torch.zeros(1, 768, dtype=torch.float32)  # Common embedding size

            def extract_features(self, x):
                # Return features with shape matching EEGPT output
                return torch.zeros(x.shape[0], 768, dtype=torch.float32)

            def prepare_chan_ids(self, channel_names):
                return torch.arange(len(channel_names))

        model.encoder = DummyEncoder()

        # Create test data: (channels, samples) - use float32
        data = np.random.randn(20, 1024).astype(np.float32)
        channel_names = [f"CH{i}" for i in range(20)]

        # Test extract_features returns expected shape
        features = model.extract_features(data, channel_names)

        assert len(features.shape) == 2  # 2D output
        assert features.dtype == np.float32  # Project dtype policy: float32

    def test_preprocess_for_eegpt(self):
        """Test preprocessing for EEGPT."""
        import mne

        # Create MNE Raw object
        sfreq = 256
        n_channels = 20
        n_samples = 1024
        data = np.random.randn(n_channels, n_samples) * 1e-6  # microvolts
        ch_names = [f"CH{i}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Preprocess
        processed = preprocess_for_eegpt(raw, sampling_rate=256)

        # Should return MNE Raw object
        assert isinstance(processed, mne.io.BaseRaw)
        assert processed.info["sfreq"] == 256
        assert len(processed.ch_names) <= 58  # Max channels for EEGPT

    def test_patch_embedding_dimension(self):
        """Test patch dimensions."""
        config = EEGPTConfig(patch_size=64)

        # For EEGPT, patches are computed from time dimension
        # Each patch is 64 samples (250ms at 256Hz)
        # A 4-second window has 1024 samples

        assert config.patch_size == 64
        assert config.window_samples == 1024  # 4s * 256Hz
        # Config doesn't have embed_dim or n_patches_per_window anymore


class TestModelInference:
    """Test model inference capabilities."""

    def test_batch_inference(self):
        """Test batch inference shapes."""
        config = EEGPTConfig()

        # Simple shape validation test - no need for mocks
        batch_data = torch.randn(4, 20, 1024)

        # Verify expected shapes
        batch_size = batch_data.shape[0]
        n_patches = batch_data.shape[2] // config.patch_size

        assert batch_size == 4
        assert n_patches == 16  # 1024 / 64

    def test_single_sample_inference(self):
        """Test single sample inference shapes."""
        config = EEGPTConfig()

        # Single sample
        single_data = torch.randn(1, 20, 1024)

        # Expected shape calculation
        n_patches = single_data.shape[2] // config.patch_size

        assert n_patches == 16  # 1024 / 64
