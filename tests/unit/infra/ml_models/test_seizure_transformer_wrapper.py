import numpy as np
import pytest
import torch

from brain_go_brrr.infra.ml_models.seizure_transformer_wrapper import (
    SeizureTransformerWrapper,
)

# Require SciPy for these tests to avoid patching internals
scipy = pytest.importorskip("scipy")


class _DummyModel(torch.nn.Module):
    def __init__(self, out_len: int) -> None:
        super().__init__()
        self.out_len = out_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        b = x.shape[0]
        # Return per-timestep predictions in [0, 1]
        return torch.ones((b, self.out_len), dtype=torch.float32, device=x.device)


@pytest.mark.unit
@pytest.mark.synth
def test_wrapper_predict_probabilities():
    n_channels = 19
    fs = 256
    window_samples = 15360
    t = window_samples + 123  # force a padded second window

    eeg = np.zeros((n_channels, t), dtype=np.float32)  # Volts
    model = _DummyModel(out_len=window_samples)
    wrapper = SeizureTransformerWrapper(
        model=model, n_channels=n_channels, fs=fs, window_samples=window_samples
    )

    preds = wrapper.predict(eeg, apply_postprocessing=False)
    assert preds.shape == (t,)
    # DummyModel returns logits=1.0, after sigmoid ≈ 0.731
    expected_prob = 1 / (1 + np.exp(-1.0))  # sigmoid(1.0)
    assert np.allclose(preds, expected_prob, rtol=1e-4)


@pytest.mark.unit
@pytest.mark.synth
def test_wrapper_postprocessing_default_binary():
    n_channels = 19
    fs = 256
    window_samples = 15360
    t = window_samples

    # Create a model that returns high logits (2.0) so sigmoid(2.0) ≈ 0.88 > 0.8 threshold
    class HighLogitModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            return torch.ones((b, window_samples), dtype=torch.float32, device=x.device) * 2.0

    eeg = np.zeros((n_channels, t), dtype=np.float32)
    model = HighLogitModel()
    wrapper = SeizureTransformerWrapper(
        model=model, n_channels=n_channels, fs=fs, window_samples=window_samples
    )
    out = wrapper.predict(eeg)  # default apply_postprocessing=True
    assert out.shape == (t,)
    # Post-processing applies morphological ops which can modify edges
    # Just verify it's binary (0 or 1)
    assert np.all((out == 0) | (out == 1))
    # Since sigmoid(2.0) ≈ 0.88 > 0.8 threshold, most should be 1
    assert np.mean(out) > 0.9


@pytest.mark.unit
def test_strict_weight_loader_raises_on_mismatch():
    # Build a tiny model and a mismatched state dict
    model = _DummyModel(out_len=10)
    bad_state = {"some.other.key": torch.tensor(1)}
    with pytest.raises(RuntimeError):
        SeizureTransformerWrapper._load_weights_strict(model, bad_state)  # type: ignore[arg-type]
