"""Integration tests for TUSZ dataset with real data.

These tests require TUSZ data at BGB_DATA_ROOT/datasets/tusz/edf/.
They are marked with @pytest.mark.data and will be skipped unless
--run-data is passed and BGB_DATA_ROOT is set.
"""

import os
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mne")
from brain_go_brrr.infra.data.tusz_detection_dataset import (
    TUSZDetectionDataset,
    WindowConfig,
)


@pytest.mark.data
@pytest.mark.integration
def test_tusz_dataset_loads_real_dev_recording():
    """Test loading actual TUSZ dev set recording."""
    data_root_env = os.environ.get("BGB_DATA_ROOT")
    if not data_root_env:
        pytest.skip("BGB_DATA_ROOT not set")
    tusz_root = Path(data_root_env) / "datasets/tusz/edf"
    if not tusz_root.exists():
        pytest.skip("TUSZ data dir not found under BGB_DATA_ROOT")

    # Use small window for quick test
    cfg = WindowConfig(
        fs=256,
        window_sec=4.0,  # 4s windows
        stride_sec=2.0,  # 50% overlap
        positive_fraction=0.2,
    )

    ds = TUSZDetectionDataset(
        root_dir=tusz_root,
        split="dev",
        cfg=cfg,
    )

    # Should have discovered recordings
    assert len(ds._records) > 0, "No recordings found in dev set"

    # Should have built index
    assert len(ds) > 0, "No windows indexed"

    # Load first window
    x, y = ds[0]

    # Check shapes
    assert x.shape[0] == 19  # Standard 19 channels
    assert x.shape[1] == int(cfg.window_sec * cfg.fs)  # 1024 samples
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32

    # Check label
    assert y.dtype == torch.int64
    assert y.item() in [0, 1]

    # Check data range (should be in Volts, typical EEG range)
    data_np = x.numpy()
    assert np.abs(data_np).max() < 1e-3, "Data likely not in Volts"
    assert np.abs(data_np).max() > 1e-8, "Data suspiciously small"


@pytest.mark.data
@pytest.mark.integration
def test_tusz_dataset_channel_aliasing():
    """Test that old channel names (T3/T4/T5/T6) are properly aliased."""
    data_root_env = os.environ.get("BGB_DATA_ROOT")
    if not data_root_env:
        pytest.skip("BGB_DATA_ROOT not set")
    tusz_root = Path(data_root_env) / "datasets/tusz/edf"
    if not tusz_root.exists():
        pytest.skip("TUSZ data dir not found under BGB_DATA_ROOT")

    cfg = WindowConfig(fs=256, window_sec=4.0, stride_sec=4.0)
    ds = TUSZDetectionDataset(root_dir=tusz_root, split="dev", cfg=cfg)

    # The dataset should handle T3→T7, T4→T8, T5→P7, T6→P8 aliasing
    # Load a window and verify we get 19 channels
    if len(ds) > 0:
        x, _ = ds[0]
        assert x.shape[0] == 19, "Channel aliasing/selection failed"


@pytest.mark.data
@pytest.mark.integration
@pytest.mark.slow
def test_tusz_dataset_seizure_labeling():
    """Test that seizure events from annotations create positive labels."""
    data_root_env = os.environ.get("BGB_DATA_ROOT")
    if not data_root_env:
        pytest.skip("BGB_DATA_ROOT not set")
    tusz_root = Path(data_root_env) / "datasets/tusz/edf"
    if not tusz_root.exists():
        pytest.skip("TUSZ data dir not found under BGB_DATA_ROOT")

    # Use longer windows to increase chance of capturing seizures
    cfg = WindowConfig(
        fs=256,
        window_sec=12.0,  # 12s windows like BiLSTM
        stride_sec=1.0,  # Dense sliding
        positive_fraction=0.2,
    )

    ds = TUSZDetectionDataset(root_dir=tusz_root, split="dev", cfg=cfg)

    # Sample multiple windows to find both positive and negative examples
    positives = 0
    negatives = 0
    max_samples = min(100, len(ds))

    for i in range(max_samples):
        _, y = ds[i]
        if y.item() == 1:
            positives += 1
        else:
            negatives += 1

    # Should have both types (TUSZ has ~7% seizure time on average)
    assert positives > 0, "No positive (seizure) windows found"
    assert negatives > 0, "No negative (background) windows found"

    # Rough sanity check on class balance
    seizure_ratio = positives / max_samples
    assert 0.01 < seizure_ratio < 0.5, f"Unusual seizure ratio: {seizure_ratio:.2%}"


@pytest.mark.data
@pytest.mark.integration
def test_tusz_dataset_deterministic_indexing():
    """Test that dataset indexing is deterministic."""
    data_root_env = os.environ.get("BGB_DATA_ROOT")
    if not data_root_env:
        pytest.skip("BGB_DATA_ROOT not set")
    tusz_root = Path(data_root_env) / "datasets/tusz/edf"
    if not tusz_root.exists():
        pytest.skip("TUSZ data dir not found under BGB_DATA_ROOT")

    cfg = WindowConfig(fs=256, window_sec=4.0, stride_sec=2.0)

    # Create two instances
    ds1 = TUSZDetectionDataset(root_dir=tusz_root, split="dev", cfg=cfg)
    ds2 = TUSZDetectionDataset(root_dir=tusz_root, split="dev", cfg=cfg)

    # Should have same length
    assert len(ds1) == len(ds2)

    # Should return same data for same index
    if len(ds1) > 0:
        x1, y1 = ds1[0]
        x2, y2 = ds2[0]
        assert torch.allclose(x1, x2, atol=1e-6)
        assert y1 == y2
