"""Behavioral tests for EEGPT orchestration pipeline.

Covers the high-level function `predict_abnormality_with_eegpt` without
importing heavy MNE modules by patching preprocessing helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from brain_go_brrr.application.pipeline.eegpt_orchestration import (
    _get_triage_level,
    predict_abnormality_with_eegpt,
)


class FakeRaw:
    """Minimal Raw-like object with only what's needed by orchestration.

    Only provides an `info` dict with sampling frequency.
    """

    def __init__(self, sfreq: float = 256.0) -> None:
        self.info = {"sfreq": sfreq}


class FakeModel:
    """Simple backbone exposing `to`, `eval`, and `extract_features`.

    The returned features control the abnormal probability via sigmoid(mean).
    """

    def __init__(self, feature_value: float) -> None:
        self._value = feature_value

    def to(self, device: str) -> FakeModel:
        return self

    def eval(self) -> FakeModel:
        return self

    def extract_features(self, x: torch.Tensor, summary: bool = False) -> torch.Tensor:
        # Return (B, 4, 512) tensor filled with the configured value
        b = x.shape[0]
        return torch.full((b, 4, 512), fill_value=self._value, dtype=torch.float32, device=x.device)


def _patch_preprocessing(monkeypatch: pytest.MonkeyPatch, n_windows: int = 5) -> None:
    """Patch preprocessing helpers to avoid heavy MNE usage.

    - preprocess_for_eegpt: returns normalized array, shape (20, 1024 * n_windows)
    - validate_eeg_input: returns (True, "Valid") regardless of input
    - extract_windows: returns list of (20, 1024) windows
    - prepare_batch_for_eegpt: stacks windows to torch tensor (B, 20, 1024)
    """
    import brain_go_brrr.application.pipeline.eegpt_orchestration as orch

    def fake_preprocess_for_eegpt(raw: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        # Produce perfectly normalized data shaped for window extraction
        samples = 1024 * n_windows
        data = np.zeros((20, samples), dtype=np.float64)
        return data

    def fake_validate(data: np.ndarray, *args: Any, **kwargs: Any) -> tuple[bool, str]:
        return True, "Valid"

    def fake_extract_windows(
        data: np.ndarray, window_duration: float, sampling_rate: int, overlap: float
    ) -> list[np.ndarray]:
        # Return exactly n_windows of 4s x 256Hz
        windows: list[np.ndarray] = []
        for _ in range(n_windows):
            windows.append(np.zeros((20, 1024), dtype=np.float64))
        return windows

    def fake_prepare_batch(windows: list[np.ndarray], n_channels: int = 20, device: str = "cpu") -> torch.Tensor:
        batch = np.stack(windows, axis=0)
        t = torch.from_numpy(batch).float()
        return t.to(device)

    monkeypatch.setattr(orch, "preprocess_for_eegpt", fake_preprocess_for_eegpt)
    monkeypatch.setattr(orch, "validate_eeg_input", fake_validate)
    monkeypatch.setattr(orch, "extract_windows", fake_extract_windows)
    monkeypatch.setattr(orch, "prepare_batch_for_eegpt", fake_prepare_batch)


class TestTriageMapping:
    def test_get_triage_level(self) -> None:
        # Not abnormal → always routine
        assert _get_triage_level(0.99, False) == "routine"
        # Abnormal confidence mapping
        assert _get_triage_level(0.95, True) == "urgent"
        assert _get_triage_level(0.80, True) == "expedite"
        assert _get_triage_level(0.60, True) == "review"


class TestPredictAbnormality:
    def test_predict_without_probe_abnormal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path without probe – abnormal case.

        Use FakeModel with positive features → sigmoid(mean) ~ 0.73 (>0.5),
        so windows predict abnormal; overall triage should be expedite (0.7<conf≤0.9).
        """
        _patch_preprocessing(monkeypatch, n_windows=40)  # exercise mini-batch stepping

        raw = FakeRaw(sfreq=256.0)
        model = FakeModel(feature_value=1.0)  # sigmoid(1)≈0.73

        result = predict_abnormality_with_eegpt(
            model_or_path=model,
            raw=raw,  # FakeRaw ok because we patched preprocessing/validation
            probe_path=None,
            window_duration=4.0,
            overlap=0.5,
            device="cpu",
        )

        assert set(result.keys()) >= {
            "prediction",
            "confidence",
            "window_predictions",
            "window_confidences",
            "n_windows",
            "triage",
        }
        assert result["n_windows"] == 40
        assert result["prediction"] == "abnormal"
        assert 0.7 <= result["confidence"] <= 0.9
        assert result["triage"] in {"expedite", "urgent"}

    def test_predict_without_probe_normal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path without probe – normal case.

        Use FakeModel with negative features → sigmoid(mean) ~ 0.27, normal prediction.
        """
        _patch_preprocessing(monkeypatch, n_windows=5)

        raw = FakeRaw(sfreq=256.0)
        model = FakeModel(feature_value=-1.0)  # sigmoid(-1)≈0.27

        result = predict_abnormality_with_eegpt(
            model_or_path=model,
            raw=raw,
            probe_path=None,
            device="cpu",
        )

        assert result["n_windows"] == 5
        assert result["prediction"] == "normal"
        assert 0.0 <= result["confidence"] < 0.5
        assert result["triage"] == "routine"

    def test_predict_with_model_path_calls_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Passing a string/Path should use create_normalized_eegpt factory.

        We patch the factory to return our FakeModel.
        """
        _patch_preprocessing(monkeypatch, n_windows=3)

        # Patch the factory symbol in orchestration module
        called = {"count": 0}

        def fake_factory(path: str) -> FakeModel:
            called["count"] += 1
            return FakeModel(feature_value=-1.0)

        import brain_go_brrr.application.pipeline.eegpt_orchestration as orch

        monkeypatch.setattr(orch, "create_normalized_eegpt", fake_factory)

        raw = FakeRaw(sfreq=256.0)

        result = predict_abnormality_with_eegpt(
            model_or_path=Path("/dummy/checkpoint.ckpt"),
            raw=raw,
            probe_path=None,
            device="cpu",
        )

        assert called["count"] == 1
        assert result["n_windows"] == 3
        assert result["prediction"] == "normal"

    def test_missing_probe_path_is_ignored(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Providing a non-existent probe_path should be safely ignored.

        Function should still run without attempting to load probe.
        """
        _patch_preprocessing(monkeypatch, n_windows=2)

        raw = FakeRaw(sfreq=256.0)
        model = FakeModel(feature_value=0.0)  # sigmoid(0)=0.5 -> prediction False

        # Non-existent path
        probe_path = tmp_path / "does_not_exist.pt"
        assert not probe_path.exists()

        result = predict_abnormality_with_eegpt(
            model_or_path=model,
            raw=raw,
            probe_path=probe_path,
            device="cpu",
        )

        assert result["n_windows"] == 2
        # Threshold is > 0.5; sigmoid(0)=0.5 should yield normal
        assert result["prediction"] == "normal"
