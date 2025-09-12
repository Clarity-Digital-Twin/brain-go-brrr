"""Integration tests for SeizureTransformer wrapper with real TUSZ data.

These tests require both TUSZ data and the wu_2025 package to be available.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from brain_go_brrr.infra.data.tusz_detection_dataset import (
    TUSZDetectionDataset,
    WindowConfig,
)
from brain_go_brrr.infra.eval.post_processing import AdvancedPostProcessor
from brain_go_brrr.infra.ml_models.seizure_transformer_wrapper import (
    SeizureTransformerWrapper,
)


@pytest.mark.data
@pytest.mark.integration
@pytest.mark.slow
def test_wrapper_on_real_tusz_recording():
    """Test SeizureTransformer wrapper on actual TUSZ data."""
    data_root_env = os.environ.get("BGB_DATA_ROOT")
    if not data_root_env:
        pytest.skip("BGB_DATA_ROOT not set")
    tusz_root = Path(data_root_env) / "datasets/tusz/edf"
    if not tusz_root.exists():
        pytest.skip("TUSZ data dir not found under BGB_DATA_ROOT")

    # Try to import wu_2025 (skip if not available)
    try:
        import wu_2025  # noqa: F401
    except ImportError:
        pytest.skip("wu_2025 package not available")

    # Load one dev recording
    cfg = WindowConfig(
        fs=256,
        window_sec=60.0,  # SeizureTransformer expects 60s
        stride_sec=60.0,  # No overlap for simplicity
    )

    ds = TUSZDetectionDataset(root_dir=tusz_root, split="dev", cfg=cfg)

    if len(ds) == 0:
        pytest.skip("No windows available in dataset")

    # Get first window
    x, y_true = ds[0]
    eeg = x.numpy()  # Shape: (19, 15360)

    # Create wrapper (no weights, just architecture test)
    wrapper = SeizureTransformerWrapper(
        n_channels=19,
        fs=256,
        window_samples=15360,
    )

    # Test inference
    with torch.no_grad():
        # Raw probabilities
        probs = wrapper.predict(eeg, apply_postprocessing=False)
        assert probs.shape == (15360,)
        assert probs.dtype == np.float32
        assert np.all((probs >= 0) & (probs <= 1))

        # Binary predictions (default)
        binary = wrapper.predict(eeg)
        assert binary.shape == (15360,)
        assert np.all((binary == 0) | (binary == 1))


@pytest.mark.data
@pytest.mark.integration
def test_wrapper_preprocessing_on_tusz():
    """Test that preprocessing pipeline handles TUSZ data correctly."""
    data_root_env = os.environ.get("BGB_DATA_ROOT")
    if not data_root_env:
        pytest.skip("BGB_DATA_ROOT not set")
    tusz_root = Path(data_root_env) / "datasets/tusz/edf"
    if not tusz_root.exists():
        pytest.skip("TUSZ data dir not found under BGB_DATA_ROOT")

    # Load a short window
    cfg = WindowConfig(fs=256, window_sec=4.0, stride_sec=4.0)
    ds = TUSZDetectionDataset(root_dir=tusz_root, split="dev", cfg=cfg)

    if len(ds) == 0:
        pytest.skip("No windows available")

    x, _ = ds[0]
    eeg = x.numpy()  # (19, 1024)

    # Pad to 60s for wrapper
    eeg_padded = np.zeros((19, 15360), dtype=np.float32)
    eeg_padded[:, :1024] = eeg

    # Create wrapper and test preprocessing
    wrapper = SeizureTransformerWrapper()

    # Should handle preprocessing without errors
    preprocessed = wrapper._preprocess_clip(eeg_padded)

    # Check output
    assert preprocessed.shape == eeg_padded.shape
    assert preprocessed.dtype == np.float32
    # Should be filtered (different from input)
    assert not np.allclose(preprocessed, eeg_padded)


@pytest.mark.data
@pytest.mark.integration
def test_postprocessor_on_tusz_predictions():
    """Test advanced post-processor with realistic seizure patterns."""
    # Create synthetic but realistic predictions
    fs = 256
    duration_sec = 120  # 2 minutes
    n_samples = fs * duration_sec

    # Simulate predictions with seizure-like patterns
    probs = np.zeros(n_samples, dtype=np.float32)

    # Add a clear seizure (10-20 seconds)
    probs[fs * 10 : fs * 20] = 0.9 + 0.1 * np.random.rand(fs * 10)

    # Add some noise/artifacts
    probs += 0.1 * np.random.rand(n_samples)

    # Add a brief spike that should be filtered
    probs[fs * 30 : fs * 30 + 100] = 0.85

    # Apply post-processing
    processor = AdvancedPostProcessor(
        hysteresis=(0.3, 0.7),
        merge_gap_sec=2.0,
        min_duration_sec=2.0,
        fs=fs,
    )

    events = processor.apply(probs)

    # Should detect the main seizure
    assert len(events) >= 1

    # Main seizure should be around 10s duration
    main_event = max(events, key=lambda e: e[1] - e[0])
    duration = main_event[1] - main_event[0]
    assert 8.0 < duration < 12.0

    # Short spike should be filtered out
    for start, end, _ in events:
        assert (end - start) >= 2.0  # Min duration enforced
