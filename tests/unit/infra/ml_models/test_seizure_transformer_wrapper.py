import numpy as np
import pytest
import torch

from brain_go_brrr.infra.ml_models.seizure_transformer_wrapper import (
    SeizureTransformerWrapper,
)


class _DummyModel(torch.nn.Module):
    def __init__(self, out_len: int) -> None:
        super().__init__()
        self.out_len = out_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # x shape: (B, C, T)
        b = x.shape[0]
        # Return per-timestep predictions in [0, 1]
        return torch.ones((b, self.out_len), dtype=torch.float32, device=x.device)


@pytest.mark.unit
@pytest.mark.synth
def test_wrapper_predict_probabilities_without_scipy(monkeypatch):
    n_channels = 19
    fs = 256
    window_samples = 15360
    t = window_samples + 123  # force a padded second window

    # Dummy EEG (Volts) with correct shape
    eeg = np.zeros((n_channels, t), dtype=np.float32)

    # Build wrapper with dummy model; skip SciPy by monkeypatching preprocess
    model = _DummyModel(out_len=window_samples)
    wrapper = SeizureTransformerWrapper(
        model=model, n_channels=n_channels, fs=fs, window_samples=window_samples
    )
    monkeypatch.setattr(wrapper, "_preprocess_clip", lambda x: x)

    preds = wrapper.predict(eeg, apply_postprocessing=False)
    assert preds.shape == (t,)
    # All ones from dummy model after concatenation and slice
    assert np.allclose(preds, 1.0)

