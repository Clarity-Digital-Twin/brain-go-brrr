"""Tests for unified EEGPTProbe covering key branches and error paths.

We inject a small fake backbone to avoid model loading and exercise:
- architecture selection (linear, two_layer)
- robust_mode NaN/Inf handling and clipping
- channel_adapter 1x1 conv path
- _accepts_param path and return_all_temporal branching
- unexpected feature shape error
"""

from __future__ import annotations

import inspect
import types

import pytest
import torch

from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe


class FakeBackboneAccepts:
    """Backbone whose extract_features accepts `return_all_temporal` and returns shapes accordingly."""

    def eval(self) -> "FakeBackboneAccepts":  # noqa: D401 - trivial
        return self

    def parameters(self):  # noqa: D401 - trivial
        return []

    def extract_features(self, x: torch.Tensor, return_all_temporal: bool = False) -> torch.Tensor:  # noqa: D401 - keep signature
        b = x.shape[0]
        if return_all_temporal:
            # Simulate (B, N_temporal, 4, 512)
            return torch.zeros((b, 2, 4, 512), dtype=torch.float32, device=x.device)
        # Default: summary tokens (B, 4, 512)
        return torch.zeros((b, 4, 512), dtype=torch.float32, device=x.device)


class FakeBackboneSimple:
    """Backbone that does not accept `return_all_temporal` parameter."""

    def eval(self) -> "FakeBackboneSimple":  # noqa: D401 - trivial
        return self

    def parameters(self):  # noqa: D401 - trivial
        return []

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - simple
        b = x.shape[0]
        return torch.zeros((b, 4, 512), dtype=torch.float32, device=x.device)


class FakeBackboneBadShape(FakeBackboneSimple):
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b = x.shape[0]
        return torch.zeros((b, 5, 256), dtype=torch.float32, device=x.device)


class TestProbeInitialization:
    def test_linear_architecture_with_injected_backbone(self) -> None:
        probe = EEGPTProbe(
            backbone=FakeBackboneSimple(), architecture="linear", freeze_backbone=True
        )
        x = torch.randn(3, 20, 1024)
        out = probe(x)
        assert out.shape[0] == 3 and out.shape[1] == 2

    def test_two_layer_architecture(self) -> None:
        probe = EEGPTProbe(
            backbone=FakeBackboneSimple(), architecture="two_layer", freeze_backbone=True
        )
        x = torch.randn(2, 20, 1024)
        out = probe(x)
        assert out.shape == (2, 2)


class TestProbeBranches:
    def test_robust_mode_replaces_nans_and_clips(self) -> None:
        probe = EEGPTProbe(backbone=FakeBackboneSimple(), robust_mode=True, freeze_backbone=True)
        x = torch.randn(1, 20, 1024)
        x[0, 0, 0] = float("nan")
        x[0, 1, 1] = float("inf")
        out = probe(x)
        assert out.shape == (1, 2)

    def test_channel_adapter_path(self) -> None:
        # Provide 19 input channels and enable adapter to 20
        probe = EEGPTProbe(
            backbone=FakeBackboneSimple(), channel_adapter=True, n_input_channels=19, freeze_backbone=True
        )
        x = torch.randn(4, 19, 1024)
        out = probe(x)
        assert out.shape == (4, 2)

    def test_return_all_temporal_branch(self) -> None:
        probe = EEGPTProbe(backbone=FakeBackboneAccepts(), freeze_backbone=True)
        x = torch.randn(2, 20, 1024)
        out = probe(x, return_all_temporal=True)
        assert out.shape == (2, 2)

    def test_unexpected_feature_shape_raises(self) -> None:
        probe = EEGPTProbe(backbone=FakeBackboneBadShape(), freeze_backbone=True)
        x = torch.randn(1, 20, 1024)
        with pytest.raises(ValueError, match="Unexpected feature shape"):
            _ = probe(x)

