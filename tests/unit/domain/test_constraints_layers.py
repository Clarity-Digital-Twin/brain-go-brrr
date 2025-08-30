"""Tests for constrained layers ensuring max-norm is enforced."""

from __future__ import annotations

import torch

from brain_go_brrr.domain.constraints import Conv1dWithConstraint, LinearWithConstraint


def _row_norms(weight: torch.Tensor) -> torch.Tensor:
    # For Linear: weight shape (out, in)
    return weight.norm(p=2, dim=1)


def test_linear_with_constraint_enforces_max_norm():
    layer = LinearWithConstraint(8, 4, max_norm=0.1, do_weight_norm=True)
    # Inflate weights artificially then run forward to trigger renorm
    with torch.no_grad():
        layer.weight.mul_(100.0)

    x = torch.randn(3, 8)
    _ = layer(x)

    norms = _row_norms(layer.weight.data)
    assert torch.all(norms <= 0.1001)  # small numeric tolerance


def test_linear_without_constraint_does_not_renorm():
    layer = LinearWithConstraint(8, 4, max_norm=0.1, do_weight_norm=False)
    with torch.no_grad():
        layer.weight.mul_(100.0)

    x = torch.randn(3, 8)
    _ = layer(x)

    norms = _row_norms(layer.weight.data)
    assert torch.any(norms > 0.1)


def test_conv1d_with_constraint_path():
    conv = Conv1dWithConstraint(8, 16, kernel_size=3, padding=1, max_norm=0.2, do_weight_norm=True)
    with torch.no_grad():
        conv.weight.mul_(50.0)

    x = torch.randn(2, 8, 64)
    _ = conv(x)
    # Check per-output-channel norm across (in_channels, kernel)
    # weight shape: (out_channels, in_channels, kernel_size)
    eps = 1e-5
    norms = conv.weight.data.view(conv.weight.size(0), -1).norm(p=2, dim=1)
    assert torch.all(norms <= 0.2 + eps)
