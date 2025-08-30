"""Behavioral tests for lightweight EEGPT preprocessing helpers.

Covers extract_windows, prepare_batch_for_eegpt, and validate_eeg_input
without importing heavy MNE dependencies.
"""

from __future__ import annotations

import numpy as np
import torch

from brain_go_brrr.domain.preprocessing.eegpt_preprocessing import (
    extract_windows,
    prepare_batch_for_eegpt,
    validate_eeg_input,
)


class TestValidateEEGInput:
    def test_rejects_wrong_dimensionality(self) -> None:
        arr = np.zeros((20, 1024, 2), dtype=np.float64)
        ok, msg = validate_eeg_input(arr)  # type: ignore[arg-type]
        assert not ok and "2D" in msg

    def test_channel_count_bounds(self) -> None:
        too_few = np.zeros((18, 1024), dtype=np.float64)
        ok, msg = validate_eeg_input(too_few)
        assert not ok and "Too few channels" in msg

        too_many = np.zeros((59, 1024), dtype=np.float64)
        ok, msg = validate_eeg_input(too_many)
        # Depending on the code path, either a general mismatch or explicit too-many message
        assert not ok and ("Channel count mismatch" in msg or "Too many channels" in msg)

    def test_sample_tolerance(self) -> None:
        # 1024 expected, 15% deviation should fail with tolerance 10%
        arr = np.zeros((20, 900), dtype=np.float64)
        ok, msg = validate_eeg_input(arr, expected_samples=1024, tolerance=0.1)
        assert not ok and "Sample count mismatch" in msg

        # Within tolerance should pass
        arr2 = np.zeros((20, 950), dtype=np.float64)
        ok2, _ = validate_eeg_input(arr2, expected_samples=1024, tolerance=0.2)
        assert ok2

    def test_nan_inf_and_range(self) -> None:
        nan_arr = np.zeros((20, 1024), dtype=np.float64)
        nan_arr[0, 0] = np.nan
        ok, msg = validate_eeg_input(nan_arr)
        assert not ok and "NaN" in msg

        inf_arr = np.zeros((20, 1024), dtype=np.float64)
        inf_arr[0, 1] = np.inf
        ok, msg = validate_eeg_input(inf_arr)
        assert not ok and "Inf" in msg

        large_arr = np.zeros((20, 1024), dtype=np.float64)
        large_arr[0, 2] = 100.0
        ok, msg = validate_eeg_input(large_arr)
        assert not ok and "out of range" in msg

        valid = np.zeros((20, 1024), dtype=np.float64)
        ok, _ = validate_eeg_input(valid)
        assert ok


class TestExtractWindows:
    def test_no_overlap(self) -> None:
        # 3 windows of 1024 at 256 Hz → 12 seconds total
        n_channels = 20
        win = 1024
        total = 3 * win
        data = np.zeros((n_channels, total), dtype=np.float64)
        windows = extract_windows(data, window_duration=4.0, sampling_rate=256, overlap=0.0)
        assert len(windows) == 3
        for w in windows:
            assert w.shape == (n_channels, win)

    def test_with_overlap(self) -> None:
        # 3 windows with 50% overlap from 6 windows worth of data
        n_channels = 20
        win = 1024
        # 6 windows worth allows 0.5 overlap to produce more steps
        data = np.zeros((n_channels, 6 * win), dtype=np.float64)
        windows = extract_windows(data, window_duration=4.0, sampling_rate=256, overlap=0.5)
        # stride = 512; count = floor((6*1024 - 1024)/512)+1 = floor(5120/512)+1 = 10+1 = 11
        assert len(windows) == 11
        assert all(w.shape == (n_channels, win) for w in windows)


class TestPrepareBatch:
    def test_pad_and_trim_channels(self) -> None:
        # Create three windows with 18, 20, 22 channels to exercise pad/trim
        win = 1024
        w1 = np.zeros((18, win), dtype=np.float64)
        w2 = np.zeros((20, win), dtype=np.float64)
        w3 = np.zeros((22, win), dtype=np.float64)
        batch = prepare_batch_for_eegpt([w1, w2, w3], n_channels=20, device="cpu")
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (3, 20, win)
        # Padded rows in w1 should be zeros
        assert torch.all(batch[0, 18:, :] == 0)
