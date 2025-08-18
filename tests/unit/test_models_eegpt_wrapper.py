"""Tests for EEGPT wrapper - Clean testing without heavy dependencies."""

import torch
import torch.nn as nn


class DummyEEGPTModel(nn.Module):
    """Minimal EEGPT model for testing."""

    def __init__(self):
        """Initialize dummy model with minimal parameters."""
        super().__init__()
        self.linear = nn.Linear(1024, 512)  # Just for having params

    def forward(self, x: torch.Tensor, chan_ids: torch.Tensor | None = None, 
                return_all_temporal: bool = False) -> torch.Tensor:
        """Return fixed-size summary tokens."""
        batch_size = x.shape[0]
        if return_all_temporal:
            # Return all temporal features (16 patches × 4 summary × 512)
            n_patches = x.shape[-1] // 64  # 64 samples per patch
            return torch.zeros((batch_size, n_patches, 4, 512), dtype=x.dtype, device=x.device)
        else:
            # Return 4 summary tokens of 512 dimensions
            return torch.zeros((batch_size, 4, 512), dtype=x.dtype, device=x.device)

    def prepare_chan_ids(self, channel_names: list[str]) -> torch.Tensor:
        """Mock channel ID preparation."""
        return torch.arange(len(channel_names))


class TestEEGPTWrapper:
    """Test EEGPT wrapper basic functionality."""

    def test_wrapper_initialization(self):
        """Test wrapper can be initialized with injected model."""
        from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper

        # Use dependency injection - no monkey-patching needed
        dummy_model = DummyEEGPTModel()
        wrapper = EEGPTWrapper(checkpoint_path=None, model=dummy_model)

        assert wrapper.model is not None
        assert isinstance(wrapper.model, DummyEEGPTModel)
        assert wrapper.model is dummy_model  # Same instance

    def test_wrapper_forward(self):
        """Test wrapper forward pass."""
        from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper

        # Clean dependency injection
        wrapper = EEGPTWrapper(model=DummyEEGPTModel())

        # Test forward with correct input shape
        x = torch.randn(1, 20, 1024)  # batch, channels, time
        output = wrapper.forward(x)

        assert output is not None
        assert output.shape == (1, 4, 512)  # 4 summary tokens, 512 embed dim
        assert output.dtype == x.dtype

    def test_wrapper_normalization(self):
        """Test input normalization parameters."""
        from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper

        wrapper = EEGPTWrapper(model=DummyEEGPTModel())

        # Test that normalization is enabled by default
        assert wrapper.normalize

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
