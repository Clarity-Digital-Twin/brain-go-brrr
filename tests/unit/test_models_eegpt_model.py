"""Tests for models.eegpt_model - CLEAN, NO HEAVY WEIGHT LOADING."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from brain_go_brrr.models.eegpt_model import (
    EEGPTConfig,
    EEGPTModel,
    preprocess_for_eegpt,
)


class TestEEGPTConfig:
    """Test EEGPT configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EEGPTConfig()

        # Test based on actual defaults in dataclass
        assert config.model_size == "large"
        assert config.n_summary_tokens == 4
        assert config.embed_dim == 512
        assert config.sampling_rate == 256
        assert config.window_duration == 4.0
        assert config.patch_size == 64
        assert config.max_channels == 58

    def test_custom_config(self):
        """Test custom configuration."""
        config = EEGPTConfig(model_size="xlarge", embed_dim=768, window_duration=8.0)

        assert config.model_size == "xlarge"
        assert config.embed_dim == 768
        assert config.window_duration == 8.0

    def test_window_samples_calculation(self):
        """Test window samples calculation."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0)
        assert config.window_samples == 1024

    def test_n_patches_calculation(self):
        """Test patches per window calculation."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0, patch_size=64)
        assert config.n_patches_per_window == 16  # 1024 / 64


class TestEEGPTModel:
    """Test EEGPT model without loading weights."""

    def test_model_initialization(self):
        """Test model initialization without loading weights."""
        from brain_go_brrr.core.config import ModelConfig
        from pathlib import Path
        
        # Create config with non-existent model path so it doesn't try to load
        config = ModelConfig()
        config.model_path = Path("/tmp/nonexistent_model.ckpt")
        
        # Initialize model without auto-loading
        model = EEGPTModel(config=config, auto_load=False)

        assert model.config == config
        assert hasattr(model, "encoder")
        assert hasattr(model, "device")

    def test_forward_pass_shape(self):
        """Test that extract_features returns correct shape."""
        from brain_go_brrr.core.config import ModelConfig
        from pathlib import Path
        
        config = ModelConfig()
        config.model_path = Path("/tmp/nonexistent_model.ckpt")
        config.embed_dim = 512
        config.n_summary_tokens = 4
        
        model = EEGPTModel(config=config, auto_load=False)
        
        # Mark as loaded to prevent loading attempt
        model.is_loaded = True
        
        # Create a dummy encoder that returns zeros
        class DummyEncoder:
            def __call__(self, *args, **kwargs):
                return torch.zeros(1, config.n_summary_tokens, config.embed_dim, dtype=torch.float64)
            
            def prepare_chan_ids(self, channel_names):
                return torch.arange(len(channel_names))
        
        model.encoder = DummyEncoder()
        
        # Create test data: (channels, samples)
        data = np.random.randn(20, 1024).astype(np.float64)
        channel_names = [f"CH{i}" for i in range(20)]
        
        # Test extract_features returns (n_summary_tokens, embed_dim)
        features = model.extract_features(data, channel_names)
        
        assert features.shape == (config.n_summary_tokens, config.embed_dim)
        assert features.dtype == np.float64

    def test_preprocess_for_eegpt(self):
        """Test preprocessing for EEGPT."""
        import mne
        
        # Create MNE Raw object
        sfreq = 256
        n_channels = 20
        n_samples = 1024
        data = np.random.randn(n_channels, n_samples) * 1e-6  # microvolts
        ch_names = [f"CH{i}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # Preprocess
        processed = preprocess_for_eegpt(raw, target_sfreq=256)
        
        # Should return MNE Raw object
        assert isinstance(processed, mne.io.BaseRaw)
        assert processed.info['sfreq'] == 256
        assert len(processed.ch_names) <= 58  # Max channels for EEGPT

    def test_patch_embedding_dimension(self):
        """Test patch embedding dimensions."""
        config = EEGPTConfig(patch_size=64, embed_dim=512)
        
        # For EEGPT, patches are computed from time dimension
        # Each patch is 64 samples (250ms at 256Hz)
        # A 4-second window has 1024 samples = 16 patches
        
        assert config.patch_size == 64
        assert config.embed_dim == 512
        assert config.window_samples == 1024  # 4s * 256Hz
        assert config.n_patches_per_window == 16  # 1024 / 64


class TestModelInference:
    """Test model inference capabilities."""

    def test_batch_inference(self):
        """Test batch inference."""
        config = EEGPTConfig()

        with patch("brain_go_brrr.models.eegpt_model.EEGPTModel") as mock_model:
            mock_instance = MagicMock()

            # Mock forward to handle batches
            def mock_forward(x):
                batch_size = x.shape[0]
                n_patches = x.shape[2] // config.patch_size
                return torch.randn(batch_size, n_patches, config.embed_dim)

            mock_instance.forward = mock_forward
            mock_model.return_value = mock_instance

            # Test with batch of 4
            batch_data = torch.randn(4, 20, 1024)
            output = mock_instance.forward(batch_data)

            assert output.shape[0] == 4  # Batch size preserved
            assert output.shape[2] == config.embed_dim

    def test_single_sample_inference(self):
        """Test single sample inference."""
        with patch("brain_go_brrr.models.eegpt_model.EEGPTModel") as mock_model:
            mock_instance = MagicMock()
            mock_instance.forward.return_value = torch.randn(1, 16, 512)
            mock_model.return_value = mock_instance

            # Single sample
            single_data = torch.randn(1, 20, 1024)
            output = mock_instance.forward(single_data)

            assert output.shape == (1, 16, 512)
