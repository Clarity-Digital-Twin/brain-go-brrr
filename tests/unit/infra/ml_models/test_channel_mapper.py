"""Unit tests for TUEV channel mapper."""

import pytest
import torch
from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper


def test_channel_mapper_shapes():
    """Test that mapper correctly transforms 23→20 channels."""
    mapper = TUEVChannelMapper()
    
    # Test 3D input (B, C, T)
    x_3d = torch.randn(32, 23, 1024)  # 4s @ 256Hz
    y_3d = mapper(x_3d)
    assert y_3d.shape == (32, 20, 1024), f"Expected (32, 20, 1024), got {y_3d.shape}"
    
    # Test 4D input (B, C, H, T)
    x_4d = torch.randn(32, 23, 1, 1024)
    y_4d = mapper(x_4d)
    assert y_4d.shape == (32, 20, 1, 1024), f"Expected (32, 20, 1, 1024), got {y_4d.shape}"


def test_gradient_flow():
    """Test that gradients flow through the mapper."""
    mapper = TUEVChannelMapper()
    x = torch.randn(1, 23, 256, requires_grad=True)
    y = mapper(x)
    loss = y.mean()
    loss.backward()
    
    assert x.grad is not None, "No gradient on input"
    assert torch.any(x.grad != 0), "Zero gradients on input"
    
    # Check that all mapper parameters have gradients
    for name, param in mapper.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_deterministic_init():
    """Test reproducible initialization with seed."""
    torch.manual_seed(42)
    mapper1 = TUEVChannelMapper()
    
    torch.manual_seed(42)
    mapper2 = TUEVChannelMapper()
    
    # Check weights are identical
    for (n1, p1), (n2, p2) in zip(mapper1.named_parameters(), mapper2.named_parameters()):
        assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
        assert torch.allclose(p1, p2), f"Parameter {n1} not identical"


def test_custom_parameters():
    """Test mapper with custom dropout and channels."""
    mapper = TUEVChannelMapper(in_channels=25, out_channels=19, dropout=0.5)
    
    x = torch.randn(8, 25, 512)
    y = mapper(x)
    assert y.shape == (8, 19, 512), f"Expected (8, 19, 512), got {y.shape}"


def test_eval_mode():
    """Test that mapper behaves correctly in eval mode."""
    mapper = TUEVChannelMapper()
    mapper.eval()
    
    # Run same input twice in eval mode - should be identical
    x = torch.randn(2, 23, 1024)
    with torch.no_grad():
        y1 = mapper(x)
        y2 = mapper(x)
    
    assert torch.allclose(y1, y2), "Outputs differ in eval mode"