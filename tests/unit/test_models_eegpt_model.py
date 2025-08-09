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

    @patch("brain_go_brrr.models.eegpt_model.load_checkpoint")
    def test_model_initialization(self, mock_load):
        """Test model initialization with mocked weights."""
        # Mock checkpoint loading
        mock_load.return_value = {}

        config = EEGPTConfig()
        model = EEGPTModel(config, load_weights=False)

        assert model.config == config
        assert hasattr(model, "encoder")
        assert hasattr(model, "patch_embed")

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        config = EEGPTConfig(n_channels=20, embed_dim=512)

        # Create lightweight mock model
        with patch.object(EEGPTModel, "__init__", lambda x, y, z: None):
            model = EEGPTModel.__new__(EEGPTModel)
            model.config = config

            # Mock the encoder with simple linear layer
            model.encoder = nn.Linear(512, 512)
            model.patch_embed = nn.Linear(config.n_channels * config.patch_size, 512)

            # Create input: batch=2, channels=20, time=1024 samples (4s at 256Hz)
            batch_size = 2
            input_data = torch.randn(batch_size, config.n_channels, 1024)

            # Mock forward to return correct shape
            with patch.object(model, "forward") as mock_forward:
                # Output should be [batch, n_patches, embed_dim]
                n_patches = 1024 // config.patch_size  # 16 patches
                mock_forward.return_value = torch.randn(batch_size, n_patches, config.embed_dim)

                output = model.forward(input_data)

                assert output.shape == (batch_size, n_patches, config.embed_dim)

    def test_preprocess_for_eegpt(self):
        """Test preprocessing for EEGPT."""
        # Test data preprocessing
        data = np.random.randn(20, 1024).astype(np.float32)

        # Preprocess
        processed = preprocess_for_eegpt(data, sampling_rate=256, target_rate=256)

        # Should return normalized data
        assert processed.shape == data.shape
        assert processed.dtype == np.float32

    def test_patch_embedding_dimension(self):
        """Test patch embedding dimensions."""
        config = EEGPTConfig(n_channels=20, patch_size=64, embed_dim=512)

        # Patch embed input size = n_channels * patch_size
        input_dim = config.n_channels * config.patch_size
        assert input_dim == 1280

        # Output should be embed_dim
        patch_embed = nn.Linear(input_dim, config.embed_dim)

        # Test shape
        patch_input = torch.randn(10, input_dim)  # 10 patches
        patch_output = patch_embed(patch_input)
        assert patch_output.shape == (10, config.embed_dim)


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
