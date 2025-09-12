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
    assert np.allclose(preds, 1.0)


@pytest.mark.unit
@pytest.mark.synth
def test_wrapper_postprocessing_default_binary():
    n_channels = 19
    fs = 256
    window_samples = 15360
    t = window_samples

    eeg = np.zeros((n_channels, t), dtype=np.float32)
    model = _DummyModel(out_len=window_samples)
    wrapper = SeizureTransformerWrapper(
        model=model, n_channels=n_channels, fs=fs, window_samples=window_samples
    )
    out = wrapper.predict(eeg)  # default apply_postprocessing=True
    assert out.shape == (t,)
    # Ones from model thresholded at 0.8 and morphed -> ones
    assert np.allclose(out, 1.0)
