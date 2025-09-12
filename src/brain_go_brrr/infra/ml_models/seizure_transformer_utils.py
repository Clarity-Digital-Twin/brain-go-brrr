"""Seizure Transformer preprocessing and post-processing utilities.

This module implements the EXACT preprocessing pipeline from Wu et al. 2025
as documented in the OSS implementation.

CRITICAL: This preprocessing is REQUIRED for using pretrained weights!
Different preprocessing = degraded performance.
"""

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, iirnotch, lfilter, resample

from brain_go_brrr.infra.data.channels import (
    CHANNEL_ALIASES as _CHANNEL_ALIASES,
)
from brain_go_brrr.infra.data.channels import (
    map_to_canonical,
)


class SeizurePreprocessor:
    """Exact preprocessing from Wu et al. 2025 OSS implementation.

    Pipeline (exact order):
    1. Z-score normalization (per-channel, over full recording)
    2. Resample to 256Hz if needed
    3. Bandpass 0.5-120Hz (order=3, causal)
    4. Notch filters at 1Hz and 60Hz

    Note: This preprocessing is done BEFORE windowing in the OSS code!
    """

    def __init__(self, target_fs: int = 256):
        """Initialize preprocessor with filter coefficients.

        Args:
            target_fs: Target sampling rate (default 256Hz per paper)
        """
        self.fs = target_fs
        self.lowcut = 0.5
        self.highcut = 120.0  # Note: 120Hz, not 100Hz!

        # Pre-compute filter coefficients at target sampling rate
        # Notch filters (Q=30 from OSS) — single pass per reference
        self.notch_1_b, self.notch_1_a = iirnotch(1.0, Q=30, fs=self.fs)
        self.notch_60_b, self.notch_60_a = iirnotch(60.0, Q=30, fs=self.fs)

        # Bandpass coefficients (Butterworth order=3)
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.bp_b, self.bp_a = butter(3, [low, high], btype='band')

    def preprocess(self, eeg: npt.NDArray[np.float32], fs_original: int) -> npt.NDArray[np.float32]:
        """Apply exact preprocessing from Wu et al. 2025.

        Args:
            eeg: Raw EEG data (n_channels, n_samples) in Volts
            fs_original: Original sampling rate

        Returns:
            Preprocessed EEG (n_channels, n_samples_resampled)
        """
        # Ensure float32 for consistency
        eeg = eeg.astype(np.float32)

        # 1. Z-score normalization (per-channel, over full recording)
        # CRITICAL: This is done BEFORE windowing in the OSS code!
        mean = np.mean(eeg, axis=1, keepdims=True)
        std = np.std(eeg, axis=1, keepdims=True)
        std[std == 0] = 1.0  # Avoid division by zero
        eeg = (eeg - mean) / std

        # 2. Resample to 256Hz if needed
        if fs_original != self.fs:
            n_samples_new = int(eeg.shape[1] * self.fs / fs_original)
            eeg_resampled = np.zeros((eeg.shape[0], n_samples_new), dtype=np.float32)
            for ch in range(eeg.shape[0]):
                eeg_resampled[ch] = resample(eeg[ch], n_samples_new).astype(np.float32)
            eeg = eeg_resampled

        # 3. Bandpass filter (0.5-120Hz, order=3, causal)
        # CRITICAL: Use lfilter (causal), not filtfilt (zero-phase)!
        for ch in range(eeg.shape[0]):
            eeg[ch] = lfilter(self.bp_b, self.bp_a, eeg[ch])

        # 4. Notch filters (1Hz, 60Hz) — single pass after bandpass (OSS parity)
        for ch in range(eeg.shape[0]):
            eeg[ch] = lfilter(self.notch_1_b, self.notch_1_a, eeg[ch])
            eeg[ch] = lfilter(self.notch_60_b, self.notch_60_a, eeg[ch])

        return eeg


class SeizurePostProcessor:
    """Exact post-processing from Wu et al. 2025 OSS implementation.

    This is used ONLY for clinical event metrics (FA/24h, sensitivity).
    NOT applied before AUROC computation!
    """

    def __init__(
        self,
        threshold: float = 0.8,
        morph_open_size: int = 5,
        morph_close_size: int = 5,
        min_duration_sec: float = 2.0,
        fs: int = 256,
    ):
        """Initialize post-processor.

        Args:
            threshold: Probability threshold (default 0.8 from paper)
            morph_open_size: Opening kernel size in samples (default 5)
            morph_close_size: Closing kernel size in samples (default 5)
            min_duration_sec: Minimum event duration in seconds (default 2.0)
            fs: Sampling rate for duration calculations
        """
        self.threshold = threshold
        self.morph_open_size = morph_open_size
        self.morph_close_size = morph_close_size
        self.min_duration_sec = min_duration_sec
        self.fs = fs
        self.min_duration_samples = int(min_duration_sec * fs)

    def postprocess(self, probs: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
        """Apply exact OSS post-processing pipeline.

        Args:
            probs: Probability predictions (n_samples,) from model

        Returns:
            Binary predictions after post-processing
        """
        from scipy.ndimage import binary_closing, binary_opening

        # 1. Threshold at 0.8
        binary = (probs > self.threshold).astype(np.int32)

        # 2. Morphological opening (remove short bursts)
        structure = np.ones(self.morph_open_size)
        binary = binary_opening(binary, structure=structure).astype(np.int32)

        # 3. Morphological closing (fill gaps)
        structure = np.ones(self.morph_close_size)
        binary = binary_closing(binary, structure=structure).astype(np.int32)

        # 4. Remove events < 2 seconds
        binary = self._remove_short_events(binary)

        return binary.astype(np.int32)

    def _remove_short_events(self, binary: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Remove seizure events shorter than min_duration."""
        from scipy.ndimage import label

        # Find connected components (seizure events)
        labeled, num_features = label(binary)

        # Check each event's duration
        for i in range(1, num_features + 1):
            event_mask = labeled == i
            event_length: int = int(np.sum(event_mask))
            if event_length < self.min_duration_samples:
                binary[event_mask] = 0

        return binary


def standardize_channel_names(channel_names: list[str]) -> list[str]:
    """Apply channel aliases to standardize names.

    Args:
        channel_names: List of channel names

    Returns:
        List with legacy names replaced by modern equivalents
    """
    return [_CHANNEL_ALIASES.get(ch, ch) for ch in channel_names]


def prepare_channels(
    data: npt.NDArray[np.float32],
    channel_names: list[str],
    target_channels: list[str] | None = None,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Delegate to SSOT mapping in infra.data.channels.map_to_canonical."""
    return map_to_canonical(data, channel_names, target_channels)
