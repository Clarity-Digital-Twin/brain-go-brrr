"""Tests for probe feature preparation utilities."""

import pytest
import torch

from brain_go_brrr.utils.probe_utils import prepare_probe_features


class TestProbeFeaturePreparation:
    """Test suite for prepare_probe_features function."""

    def test_error_on_single_512_vector(self):
        """Should error on single 512 vector with helpful message."""
        with pytest.raises(ValueError, match="call.*summary=False"):
            prepare_probe_features(torch.randn(512))

    def test_error_on_batch_512(self):
        """Should error on (B, 512) batch with guidance."""
        with pytest.raises(ValueError, match="call.*summary=False"):
            prepare_probe_features(torch.randn(10, 512))

    def test_accept_single_4x512(self):
        """Should accept and convert (4, 512) to (1, 2048)."""
        input_tensor = torch.randn(4, 512)
        result = prepare_probe_features(input_tensor)
        assert result.shape == (1, 2048)
        # Verify it's actually flattened correctly
        expected = input_tensor.view(1, -1)
        torch.testing.assert_close(result, expected)

    def test_accept_batch_4x512(self):
        """Should accept and convert (B, 4, 512) to (B, 2048)."""
        batch_size = 10
        input_tensor = torch.randn(batch_size, 4, 512)
        result = prepare_probe_features(input_tensor)
        assert result.shape == (batch_size, 2048)
        # Verify flattening preserves data
        expected = input_tensor.view(batch_size, -1)
        torch.testing.assert_close(result, expected)

    def test_passthrough_batch_2048(self):
        """Should pass through (B, 2048) unchanged."""
        input_tensor = torch.randn(10, 2048)
        result = prepare_probe_features(input_tensor)
        assert result is input_tensor  # Same object, no copy

    def test_passthrough_single_2048(self):
        """Should pass through (2048,) as (1, 2048)."""
        input_tensor = torch.randn(2048)
        result = prepare_probe_features(input_tensor)
        assert result.shape == (1, 2048)

    def test_error_on_wrong_dimensions(self):
        """Should error on unexpected shapes."""
        # Wrong last dimension
        with pytest.raises(ValueError, match="Expected.*512.*2048"):
            prepare_probe_features(torch.randn(10, 1024))
        
        # Wrong middle dimension for 3D
        with pytest.raises(ValueError, match="Expected.*4 tokens"):
            prepare_probe_features(torch.randn(10, 3, 512))

    def test_preserves_device(self):
        """Should preserve tensor device."""
        if torch.cuda.is_available():
            input_tensor = torch.randn(4, 512).cuda()
            result = prepare_probe_features(input_tensor)
            assert result.device == input_tensor.device

    def test_preserves_dtype(self):
        """Should preserve tensor dtype."""
        for dtype in [torch.float16, torch.float32, torch.float64]:
            input_tensor = torch.randn(4, 512, dtype=dtype)
            result = prepare_probe_features(input_tensor)
            assert result.dtype == dtype

    def test_gradient_flow(self):
        """Should allow gradients to flow through."""
        input_tensor = torch.randn(4, 512, requires_grad=True)
        result = prepare_probe_features(input_tensor)
        assert result.requires_grad
        
        # Verify gradient flows
        loss = result.sum()
        loss.backward()
        assert input_tensor.grad is not None
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))