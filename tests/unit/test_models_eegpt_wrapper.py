"""Tests for EEGPT wrapper - Clean testing without MNE dependencies."""

from unittest.mock import MagicMock, patch

import torch


class TestEEGPTWrapper:
    """Test EEGPT wrapper basic functionality."""

    @patch("brain_go_brrr.models.eegpt_wrapper.create_eegpt_model")
    def test_wrapper_initialization(self, mock_create):
        """Test wrapper can be initialized."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

        mock_model = MagicMock()
        mock_create.return_value = mock_model

        wrapper = EEGPTWrapper(checkpoint_path=None)
        assert wrapper.model == mock_model
        mock_create.assert_called_once()

    @patch("brain_go_brrr.models.eegpt_wrapper.create_eegpt_model")
    def test_wrapper_forward(self, mock_create):
        """Test wrapper forward pass."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

        # Create a mock model that returns actual tensor
        expected_output = torch.randn(1, 4, 512)  # 4 summary tokens, 512 embed dim
        mock_model = MagicMock()
        mock_model.forward.return_value = expected_output
        mock_create.return_value = mock_model

        wrapper = EEGPTWrapper()

        # Test forward with correct input shape
        x = torch.randn(1, 20, 1024)  # batch, channels, time
        output = wrapper.forward(x)

        assert output is not None
        assert torch.equal(output, expected_output)  # Check we get the expected output
        mock_model.forward.assert_called_once()

    @patch("brain_go_brrr.models.eegpt_wrapper.create_eegpt_model")
    def test_wrapper_normalization(self, mock_create):
        """Test input normalization parameters."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

        mock_model = MagicMock()
        mock_model.forward.return_value = torch.randn(1, 4, 512)
        mock_create.return_value = mock_model

        wrapper = EEGPTWrapper()

        # Test that normalization is enabled by default
        assert wrapper.normalize == True

        # Test setting normalization parameters
        wrapper.set_normalization_params(mean=0.5, std=2.0)
        assert wrapper.input_mean.item() == 0.5
        assert wrapper.input_std.item() == 2.0

        # Test estimating normalization from data
        test_data = torch.randn(1, 20, 1024)
        wrapper.estimate_normalization_params(test_data)
        # After estimation, mean and std should be updated
        assert wrapper.input_mean is not None
        assert wrapper.input_std is not None
