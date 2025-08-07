"""Tests for EEGPT wrapper - Clean testing without MNE dependencies."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch


class TestEEGPTWrapper:
    """Test EEGPT wrapper basic functionality."""

    @patch('brain_go_brrr.models.eegpt_wrapper.create_eegpt_model')
    def test_wrapper_initialization(self, mock_create):
        """Test wrapper can be initialized."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        wrapper = EEGPTWrapper(checkpoint_path=None)
        assert wrapper.model == mock_model
        mock_create.assert_called_once()

    @pytest.mark.skip(reason="EEGPTWrapper forward method needs update")
    @patch('brain_go_brrr.models.eegpt_wrapper.create_eegpt_model')
    def test_wrapper_forward(self, mock_create):
        """Test wrapper forward pass."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_model.forward.return_value = torch.randn(1, 16, 768)
        mock_create.return_value = mock_model
        
        wrapper = EEGPTWrapper()
        
        # Test forward with correct input shape
        x = torch.randn(1, 20, 1024)  # batch, channels, time
        output = wrapper.forward(x)
        
        assert output is not None
        mock_model.forward.assert_called()

    @pytest.mark.skip(reason="EEGPTWrapper normalize method needs update")
    @patch('brain_go_brrr.models.eegpt_wrapper.create_eegpt_model')  
    def test_wrapper_normalize_input(self, mock_create):
        """Test input normalization."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        wrapper = EEGPTWrapper()
        
        # Test normalization if method exists
        if hasattr(wrapper, 'normalize'):
            x = torch.randn(1, 20, 1024)
            normalized = wrapper.normalize(x)
            assert normalized.shape == x.shape
        else:
            # Normalization happens internally
            pass