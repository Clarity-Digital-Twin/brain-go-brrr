"""Centralized mock utilities for EEGPT and other models.

This module provides consistent mocking for EEGPT model methods to ensure
tests run without requiring the actual 400MB model checkpoint.

Mock return shapes:
- extract_features (single window): (n_summary_tokens, 512)
- extract_features (batch): (batch_size, n_summary_tokens, 512)
- extract_features_batch: (batch_size, n_summary_tokens, 512)
- process_recording: dict with 'features', 'n_windows', 'window_times'

Where n_summary_tokens defaults to 4 (configurable via model.config).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


def mock_eegpt_model_loading(monkeypatch):
    """Mock EEGPT model loading and inference methods.

    This function is called automatically by the mock_eegpt_model fixture
    in conftest.py unless EEGPT_MODEL_PATH environment variable is set.

    Args:
        monkeypatch: pytest monkeypatch fixture
    """

    def mock_load_model(self, checkpoint_path=None):
        """Mock model loading - sets is_loaded flag."""
        self.model = MagicMock()
        self.is_loaded = True
        return True

    def mock_extract_features(self, eeg_data, ch_names=None):
        """Mock feature extraction with correct output shapes.

        Args:
            eeg_data: Input EEG data
                - 2D array (channels, time): single window
                - 3D array (batch, channels, time): batch of windows
            ch_names: Channel names (unused in mock)

        Returns:
            np.ndarray: Features with shape:
                - Single window: (n_summary_tokens, 512)
                - Batch: (batch_size, n_summary_tokens, 512)
        """
        n_summary_tokens = getattr(self.config, "n_summary_tokens", 4)

        if eeg_data.ndim == 2:
            # Single window, return (n_summary_tokens, 512)
            return np.random.randn(n_summary_tokens, 512).astype(np.float32)
        else:
            # Batch, return (batch_size, n_summary_tokens, 512)
            batch_size = eeg_data.shape[0]
            return np.random.randn(batch_size, n_summary_tokens, 512).astype(np.float32)

    def mock_extract_features_batch(self, batch_data):
        """Mock batch feature extraction.

        Args:
            batch_data: 3D array (batch_size, channels, time)

        Returns:
            np.ndarray: Features with shape (batch_size, n_summary_tokens, 512)
        """
        batch_size = batch_data.shape[0]
        n_summary_tokens = getattr(self.config, "n_summary_tokens", 4)
        return np.random.randn(batch_size, n_summary_tokens, 512).astype(np.float32)

    def mock_process_recording(
        self,
        raw=None,
        data=None,
        sampling_rate=256,
        overlap=0.5,
        ch_names=None,
        batch_size=32,
        **kwargs,
    ):
        """Mock processing a full MNE Raw recording.

        Args:
            raw: MNE Raw object (preferred)
            data: Raw numpy array (alternative API)
            sampling_rate: Sampling rate if data is provided
            overlap: Window overlap fraction (0-1)
            ch_names: Channel names (unused in mock)
            batch_size: Batch size for processing (unused in mock)
            **kwargs: Additional arguments (ignored)

        Returns:
            dict: Results containing:
                - features: (n_windows, n_summary_tokens, 512) array
                - n_windows: Number of windows extracted
                - window_times: List of (start, end) tuples for each window
                - processing_complete: True (for memory test)
                - abnormal_probability: 0.15 (for other tests)
                - confidence: 0.85 (for other tests)
        """
        # Handle both raw and data arguments
        if raw is not None:
            duration = raw.times[-1]
        elif data is not None:
            # Assume data is (n_channels, n_samples)
            duration = data.shape[1] / sampling_rate
        else:
            # Default duration
            duration = 20 * 60  # 20 minutes

        window_size = 4.0  # EEGPT uses 4-second windows
        step_size = window_size * (1 - overlap)
        n_windows = max(1, int((duration - window_size) / step_size) + 1)
        n_summary_tokens = getattr(self.config, "n_summary_tokens", 4)

        return {
            "features": np.random.randn(n_windows, n_summary_tokens, 512).astype(np.float32),
            "n_windows": n_windows,
            "window_times": [
                (i * step_size, min(i * step_size + window_size, duration))
                for i in range(n_windows)
            ],
            "processing_complete": True,
            "abnormal_probability": 0.15,
            "confidence": 0.85,
        }

    def mock_predict_abnormality(self, raw):
        """Mock abnormality prediction."""
        # Calculate number of windows
        duration = raw.times[-1] if hasattr(raw, "times") else 20.0
        window_size = 4.0
        overlap = 0.5
        step_size = window_size * (1 - overlap)
        n_windows = max(1, int((duration - window_size) / step_size) + 1)

        # Generate mock window scores
        window_scores = [0.15 + 0.1 * np.random.rand() for _ in range(n_windows)]

        return {
            "abnormal_probability": 0.15,
            "confidence": 0.85,
            "window_scores": window_scores,
            "n_windows": n_windows,
            "used_streaming": False,
        }

    # Apply all mocks
    monkeypatch.setattr("brain_go_brrr.models.eegpt_model.EEGPTModel.load_model", mock_load_model)
    monkeypatch.setattr(
        "brain_go_brrr.models.eegpt_model.EEGPTModel.extract_features", mock_extract_features
    )
    monkeypatch.setattr(
        "brain_go_brrr.models.eegpt_model.EEGPTModel.extract_features_batch",
        mock_extract_features_batch,
    )
    monkeypatch.setattr(
        "brain_go_brrr.models.eegpt_model.EEGPTModel.process_recording",
        mock_process_recording,
    )
    monkeypatch.setattr(
        "brain_go_brrr.models.eegpt_model.EEGPTModel.predict_abnormality",
        mock_predict_abnormality,
    )


@pytest.fixture
def mock_sleep_probe():
    """Mock sleep staging linear probe."""
    from unittest.mock import Mock

    import torch

    probe = Mock()
    probe.predict_stage.return_value = (["N2"], torch.tensor([0.85]))
    probe.predict_stage.side_effect = None  # Can be overridden in tests
    return probe


@pytest.fixture
def mock_abnormality_probe():
    """Mock abnormality detection linear probe."""
    from unittest.mock import Mock

    import torch

    probe = Mock()
    probe.predict_abnormal_probability.return_value = torch.tensor([0.15])
    probe.detect_abnormality.return_value = {
        "abnormal": False,
        "confidence": 0.85,
        "probabilities": {"normal": 0.85, "abnormal": 0.15},
    }
    return probe
