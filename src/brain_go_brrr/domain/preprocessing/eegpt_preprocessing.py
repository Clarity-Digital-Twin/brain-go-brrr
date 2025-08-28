"""EEGPT-specific preprocessing functions.

This is the SINGLE source of truth for EEGPT preprocessing.
Consolidates functions from eegpt_prepare.py and eegpt_preprocessing.py.
"""

import logging
from typing import TYPE_CHECKING, Any

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


def prepare_for_eegpt(
    raw: Any,  # MNE Raw object
    target_sfreq: int = 256,
    required_channels: int = 19,
    pad_to_multiple: int = 64,
) -> npt.NDArray[np.float32]:
    """Prepare raw EEG data for EEGPT model input.

    Single entry point for all EEGPT preprocessing:
    1. Resample to target sampling rate
    2. Map old channel names to modern naming
    3. Validate channel count
    4. Pad temporal dimension to multiple of patch size
    5. Validate no NaN/Inf values

    Args:
        raw: MNE Raw object with EEG data
        target_sfreq: Target sampling rate (default 256 Hz)
        required_channels: Minimum required channels (default 19)
        pad_to_multiple: Pad temporal dimension to multiple of this (default 64)

    Returns:
        Preprocessed numpy array of shape (channels, samples)

    Raises:
        ValueError: If data contains NaN/Inf or insufficient channels
    """
    # Step 1: Resample if needed
    current_sfreq = int(raw.info['sfreq'])
    if current_sfreq != target_sfreq:
        logger.info(f"Resampling from {current_sfreq} Hz to {target_sfreq} Hz")
        raw = raw.copy().resample(target_sfreq, npad='auto')

    # Step 2: Map old channel names to modern naming
    current_channels = raw.ch_names
    renamed_channels = []
    for ch in current_channels:
        ch_upper = ch.upper()
        # Apply mapping if needed (T3->T7, T4->T8, T5->P7, T6->P8)
        channel_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
        mapped = channel_map.get(ch_upper, ch_upper)
        renamed_channels.append(mapped)

    # Update channel names if any were mapped
    if renamed_channels != current_channels:
        raw.rename_channels(dict(zip(current_channels, renamed_channels, strict=False)))
        logger.info(f"Mapped channel names: {current_channels[:3]} -> {renamed_channels[:3]}...")

    # Step 3: Validate channel count
    n_channels = len(raw.ch_names)
    if n_channels < required_channels:
        raise ValueError(
            f"Insufficient channels: got {n_channels}, need at least {required_channels}"
        )

    # Step 4: Get data and pad temporal dimension
    data = raw.get_data()
    n_samples = data.shape[1]

    if pad_to_multiple > 0:
        remainder = n_samples % pad_to_multiple
        if remainder != 0:
            pad_amount = pad_to_multiple - remainder
            logger.debug(
                f"Padding {n_samples} samples by {pad_amount} to reach multiple of {pad_to_multiple}"
            )
            # Pad with edge values (repeat last samples)
            data = np.pad(data, ((0, 0), (0, pad_amount)), mode='edge')

    # Step 5: Validate no NaN/Inf
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or Inf values")

    # Ensure float32 for consistency
    data = data.astype(np.float32)

    # Step 6: Final assertions to guarantee contract
    actual_sfreq = int(raw.info['sfreq'])
    assert actual_sfreq == target_sfreq, f"Sampling rate mismatch: {actual_sfreq} != {target_sfreq}"

    n_samples_final = data.shape[1]
    assert (
        n_samples_final % pad_to_multiple == 0
    ), f"Temporal dimension not padded correctly: {n_samples_final} % {pad_to_multiple} != 0"

    logger.debug(f"Prepared EEGPT input: shape {data.shape}, dtype {data.dtype}")

    return data  # type: ignore[no-any-return]


def supports_eegpt(raw: Any) -> bool:
    """Check if raw data supports EEGPT analysis.

    Args:
        raw: MNE Raw object

    Returns:
        True if data meets EEGPT requirements
    """
    # Check channel count
    n_channels = len(raw.ch_names)
    if n_channels < 19:
        logger.debug(f"EEGPT requires 19+ channels, got {n_channels}")
        return False

    # Check sampling rate is reasonable (can be resampled)
    sfreq = raw.info['sfreq']
    if sfreq < 100 or sfreq > 5000:
        logger.debug(f"Unreasonable sampling rate: {sfreq} Hz")
        return False

    return True
