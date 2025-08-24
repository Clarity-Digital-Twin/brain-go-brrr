"""EEGPT-specific preprocessing functions.

Extracted from eegpt_model.py to proper domain layer.
Note: For basic EEGPT input preparation, use domain.preprocessing.eegpt_prepare.prepare_for_eegpt
This module provides additional preprocessing with filtering and normalization.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from brain_go_brrr._typing import MNERaw

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Standard EEGPT channel names (modern 10-20 system)
EEGPT_CHANNELS = [
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
    "OZ",
]

# Old to modern channel mapping for TUAB compatibility
CHANNEL_MAPPING = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def preprocess_for_eegpt(
    raw: MNERaw,
    target_sfreq: int = 256,
    lowpass: float = 50.0,
    highpass: float = 0.5,
    reference: str = "average",
    channels: list[str] | None = None,
) -> npt.NDArray[np.float64]:
    """Preprocess raw EEG data for EEGPT input.

    Args:
        raw: MNE Raw object with EEG data
        target_sfreq: Target sampling frequency (Hz)
        lowpass: Low-pass filter cutoff (Hz)
        highpass: High-pass filter cutoff (Hz)
        reference: Reference type ('average' or specific channel)
        channels: Specific channels to use (None = use standard EEGPT channels)

    Returns:
        Preprocessed EEG array of shape (n_channels, n_samples)
    """
    # Make a copy to avoid modifying original
    raw = raw.copy()

    # Rename old channel names to modern equivalents
    rename_mapping = {}
    for old_name, new_name in CHANNEL_MAPPING.items():
        if old_name in raw.ch_names:
            rename_mapping[old_name] = new_name
    if rename_mapping:
        raw.rename_channels(rename_mapping)
        logger.info(f"Renamed channels: {rename_mapping}")

    # Pick channels
    if channels is None:
        channels = [ch for ch in EEGPT_CHANNELS if ch in raw.ch_names]

    if len(channels) < 19:
        logger.warning(f"Only {len(channels)} channels available (minimum 19 recommended)")

    raw.pick_channels(channels, ordered=True)

    # Apply filters
    raw.filter(l_freq=highpass, h_freq=lowpass, fir_design='firwin')

    # Resample if needed
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq)

    # Set reference
    if reference == "average":
        raw.set_eeg_reference('average', projection=False)
    else:
        raw.set_eeg_reference(reference, projection=False)

    # Get data
    data = raw.get_data()

    # Z-score normalization per channel
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-6
    data = (data - mean) / std

    return data  # type: ignore[no-any-return]


def extract_windows(
    data: npt.NDArray[np.float64],
    window_duration: float = 4.0,
    sampling_rate: int = 256,
    overlap: float = 0.0,
) -> list[npt.NDArray[np.float64]]:
    """Extract fixed-size windows from continuous EEG data.

    Args:
        data: EEG array of shape (n_channels, n_samples)
        window_duration: Window duration in seconds
        sampling_rate: Sampling rate in Hz
        overlap: Overlap fraction (0.0 to 0.9)

    Returns:
        List of windows, each of shape (n_channels, window_samples)
    """
    n_channels, n_samples = data.shape
    window_samples = int(window_duration * sampling_rate)
    stride_samples = int(window_samples * (1 - overlap))

    windows = []
    start = 0
    while start + window_samples <= n_samples:
        window = data[:, start : start + window_samples]
        windows.append(window)
        start += stride_samples

    logger.info(f"Extracted {len(windows)} windows of {window_duration}s")
    return windows


def prepare_batch_for_eegpt(
    windows: list[npt.NDArray[np.float64]], n_channels: int = 20, device: str = "cpu"
) -> "torch.Tensor":
    """Prepare a batch of windows for EEGPT input.

    Args:
        windows: List of EEG windows
        n_channels: Target number of channels (pad/trim as needed)
        device: Device to place tensor on

    Returns:
        Batch tensor of shape (batch_size, n_channels, n_samples)
    """
    import torch

    batch_list = []
    for window in windows:
        current_channels = window.shape[0]

        # Pad or trim channels
        if current_channels < n_channels:
            # Pad with zeros
            padding = np.zeros((n_channels - current_channels, window.shape[1]))
            window = np.vstack([window, padding])
        elif current_channels > n_channels:
            # Trim to first n_channels
            window = window[:n_channels]

        batch_list.append(window)

    # Stack into batch
    batch = np.stack(batch_list, axis=0)

    # Convert to tensor
    batch_tensor = torch.from_numpy(batch).float()

    if device != "cpu":
        batch_tensor = batch_tensor.to(device)

    return batch_tensor


def validate_eeg_input(
    data: npt.NDArray[np.float64],
    expected_channels: int = 20,
    expected_samples: int = 1024,
    tolerance: float = 0.1,
) -> tuple[bool, str]:
    """Validate EEG input data for EEGPT.

    Args:
        data: EEG data array
        expected_channels: Expected number of channels
        expected_samples: Expected number of samples
        tolerance: Tolerance for sample count (fraction)

    Returns:
        Tuple of (is_valid, message)
    """
    if data.ndim != 2:
        return False, f"Expected 2D array, got {data.ndim}D"

    n_channels, n_samples = data.shape

    # Check against expected channels with tolerance
    if expected_channels and abs(n_channels - expected_channels) > expected_channels * 0.2:
        return False, f"Channel count mismatch: got {n_channels}, expected ~{expected_channels}"

    if n_channels < 19:
        return False, f"Too few channels: {n_channels} (minimum 19)"

    if n_channels > 58:
        return False, f"Too many channels: {n_channels} (maximum 58)"

    sample_diff = abs(n_samples - expected_samples) / expected_samples
    if sample_diff > tolerance:
        return False, f"Sample count mismatch: {n_samples} (expected ~{expected_samples})"

    # Check for NaN/Inf
    if np.isnan(data).any():
        return False, "Data contains NaN values"

    if np.isinf(data).any():
        return False, "Data contains Inf values"

    # Check value range (after normalization should be ~[-5, 5])
    if np.abs(data).max() > 50:
        return False, f"Data values out of range: max={np.abs(data).max():.1f}"

    return True, "Valid"
