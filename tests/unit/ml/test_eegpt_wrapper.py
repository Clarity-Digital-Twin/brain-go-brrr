"""Behavioral tests for EEGPTWrapper and factory.

Minimal fakes to exercise normalization, feature extraction shape logic,
and factory behavior without loading real models.
"""

from __future__ import annotations

import json

import pytest
import torch

import brain_go_brrr.infra.ml_models.eegpt_wrapper as w


class _BackboneAccepts(torch.nn.Module):
    def forward(self, x: torch.Tensor, _chan_ids=None, return_all_temporal: bool = False):
        b = x.shape[0]
        if return_all_temporal:
            return torch.zeros((b, 2, 4, 512), dtype=torch.float32)
        return torch.zeros((b, 4, 512), dtype=torch.float32)


class _BackboneSimple(torch.nn.Module):
    def forward(self, x: torch.Tensor, _chan_ids=None):
        b = x.shape[0]
        return torch.zeros((b, 4, 512), dtype=torch.float32)


def test_create_normalized_eegpt_uses_adjacent_normalization(tmp_path, monkeypatch):
    # Create dummy checkpoint path with adjacent normalization.json
    ckpt = tmp_path / "weights" / "model.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.touch()
    (ckpt.parent / "normalization.json").write_text(json.dumps({"mean": 0.1, "std": 2.0}))

    # Patch backbone factory to return simple backbone to avoid heavy load
    monkeypatch.setattr(w, "create_eegpt_model", lambda _p: _BackboneSimple())

    model = w.create_normalized_eegpt(str(ckpt), normalize=True)
    assert isinstance(model, w.EEGPTWrapper)

    # Verify normalization buffers were set from file
    assert torch.is_tensor(model.input_mean)
    assert torch.is_tensor(model.input_std)
    assert abs(model.input_mean.item() - 0.1) < 1e-6
    assert abs(model.input_std.item() - 2.0) < 1e-6


def test_extract_features_shapes_and_nan_rejection(monkeypatch):
    # Patch to backbone that accepts return_all_temporal via forward signature
    monkeypatch.setattr(w, "create_eegpt_model", lambda _p: _BackboneAccepts())
    wrapper = w.EEGPTWrapper(checkpoint_path=None)

    x = torch.randn(3, 20, 1024)
    # summary=True → (B, 512)
    f_sum = wrapper.extract_features(x, summary=True)
    assert f_sum.shape == (3, 512)

    # summary=False → (B, 4, 512)
    f_tok = wrapper.extract_features(x, summary=False)
    assert f_tok.shape == (3, 4, 512)

    # NaN input should be rejected
    x_nan = x.clone()
    x_nan[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        _ = wrapper.extract_features(x_nan)


def test_forward_branch_without_return_all_temporal(monkeypatch):
    # Patch to backbone without return_all_temporal in signature
    monkeypatch.setattr(w, "create_eegpt_model", lambda _p: _BackboneSimple())
    wrapper = w.EEGPTWrapper(checkpoint_path=None)
    x = torch.randn(2, 20, 1024)
    out = wrapper.forward(x, return_all_temporal=True)
    # Should still return (B, 4, 512) via fallback
    assert out.shape == (2, 4, 512)


def test_estimate_and_override_normalization(monkeypatch):
    monkeypatch.setattr(w, "create_eegpt_model", lambda _p: _BackboneSimple())
    wrapper = w.EEGPTWrapper(checkpoint_path=None)

    # Estimate from data
    data = torch.zeros(2, 20, 10)
    data[:, :, :] = 5.0
    wrapper.estimate_normalization_params(data)
    assert abs(wrapper.input_mean.item() - 5.0) < 1e-6
    assert wrapper.input_std.item() < 1e-6  # zero variance across batch/time

    # Override explicitly via factory params
    model2 = w.create_normalized_eegpt(checkpoint_path=None, normalize=True, mean=0.5, std=3.0)
    assert abs(model2.input_mean.item() - 0.5) < 1e-6
    assert abs(model2.input_std.item() - 3.0) < 1e-6
