"""
Test EEGPT model loading functionality.
Following TDD approach - test first, then ensure implementation works.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from brain_go_brrr.core.config import Config
from brain_go_brrr.models.eegpt_model import EEGPTConfig, EEGPTModel


class TestEEGPTModelLoading:
    """Test EEGPT model loading and initialization."""

    def test_eegpt_model_initialization_without_checkpoint(self):
        """Test that EEGPTModel can be initialized with a checkpoint path."""
        # Given: A valid config and a checkpoint path (even if file doesn't exist)
        config = EEGPTConfig()
        checkpoint_path = Path("nonexistent_checkpoint.ckpt")

        # When: We initialize the model without auto-loading
        model = EEGPTModel(checkpoint_path=checkpoint_path, config=config, auto_load=False)

        # Then: The model should be initialized successfully
        assert model is not None
        assert model.checkpoint_path == checkpoint_path
        assert model.config == config
        assert model.is_loaded is False  # No model loaded yet

    @patch('brain_go_brrr.models.eegpt_model.torch.load')
    def test_eegpt_model_loading_with_mock_checkpoint(self, mock_torch_load):
        """Test model loading with a mocked checkpoint."""
        # Given: A mock checkpoint file
        mock_checkpoint = {
            'model_state_dict': {'dummy_param': torch.tensor([1.0])},
            'config': {'n_channels': 19, 'seq_len': 1024}
        }
        mock_torch_load.return_value = mock_checkpoint

        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=Path("test.ckpt"), config=config, auto_load=False)

        # When: We load a checkpoint
        checkpoint_path = Path("mock_checkpoint.ckpt")

        # Mock the path exists check
        with patch.object(Path, 'exists', return_value=True):
            result = model.load_checkpoint(checkpoint_path)

        # Then: The loading should succeed
        assert result is True
        mock_torch_load.assert_called_once()

    def test_eegpt_model_loading_with_nonexistent_file(self):
        """Test model loading fails gracefully with non-existent file."""
        # Given: A model and non-existent checkpoint path
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=Path("test.ckpt"), config=config, auto_load=False)
        checkpoint_path = Path("nonexistent_file.ckpt")

        # When: We try to load a non-existent checkpoint
        result = model.load_checkpoint(checkpoint_path)

        # Then: Loading should fail gracefully
        assert result is False

    @patch('brain_go_brrr.models.eegpt_architecture.EEGTransformer')
    def test_model_architecture_initialization(self, mock_transformer):
        """Test that the model architecture is initialized correctly."""
        # Given: A mock transformer
        mock_transformer_instance = MagicMock()
        mock_transformer.return_value = mock_transformer_instance

        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=Path("test.ckpt"), config=config, auto_load=False)

        # When: We initialize the model architecture
        model._initialize_model()

        # Then: The transformer should be initialized with correct parameters
        mock_transformer.assert_called_once()
        assert model.model == mock_transformer_instance

    def test_feature_extraction_requires_loaded_model(self):
        """Test that feature extraction requires a loaded model."""
        # Given: A model without a loaded checkpoint
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=Path("test.ckpt"), config=config, auto_load=False)

        # When: We try to extract features without a loaded model
        dummy_data = torch.randn(1, 19, 1024)  # batch_size=1, channels=19, seq_len=1024

        # Then: It should raise an appropriate error
        with pytest.raises((RuntimeError, ValueError)):
            model.extract_features(dummy_data)

    @patch('brain_go_brrr.models.eegpt_architecture.EEGTransformer')
    def test_feature_extraction_with_loaded_model(self, mock_transformer):
        """Test feature extraction with a properly loaded model."""
        # Given: A model with loaded architecture
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.return_value = torch.randn(1, 308, 512)  # Mock features
        mock_transformer.return_value = mock_transformer_instance

        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=Path("test.ckpt"), config=config, auto_load=False)
        model._initialize_model()

        # When: We extract features from EEG data
        dummy_data = torch.randn(1, 19, 1024)
        features = model.extract_features(dummy_data)

        # Then: Features should be extracted successfully
        assert features is not None
        assert isinstance(features, torch.Tensor)
        mock_transformer_instance.assert_called_once_with(dummy_data)
