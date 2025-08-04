"""Sliding window extraction for EEG data processing.

Implements 8-second windows with configurable overlap.
"""

import numpy as np


class WindowExtractor:
    """Extract sliding windows from continuous EEG data."""

    def __init__(self, window_seconds: float = 8.0, overlap_seconds: float = 4.0):
        """Initialize window extractor.

        Args:
            window_seconds: Window duration in seconds
            overlap_seconds: Overlap between windows in seconds
        """
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self.stride_seconds = window_seconds - overlap_seconds

    def extract(self, data: np.ndarray, sfreq: float) -> list[np.ndarray]:
        """Extract sliding windows from continuous data.

        Args:
            data: EEG data of shape (n_channels, n_samples)
            sfreq: Sampling frequency in Hz

        Returns:
            List of windows, each of shape (n_channels, window_samples)
        """
        n_channels, n_samples = data.shape
        window_samples = int(self.window_seconds * sfreq)
        stride_samples = int(self.stride_seconds * sfreq)

        # Calculate number of windows
        n_windows = (n_samples - window_samples) // stride_samples + 1

        # Handle case where data is shorter than window
        if n_windows <= 0:
            return []

        # Extract windows
        windows = []
        for i in range(n_windows):
            start = i * stride_samples
            end = start + window_samples

            # Ensure we don't go past the end
            if end > n_samples:
                break

            window = data[:, start:end]
            windows.append(window)

        return windows

    def extract_with_timestamps(
        self, data: np.ndarray, sfreq: float
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """Extract windows with their timestamps.

        Args:
            data: EEG data of shape (n_channels, n_samples)
            sfreq: Sampling frequency in Hz

        Returns:
            Tuple of (windows, timestamps) where timestamps are (start, end) tuples
        """
        windows = self.extract(data, sfreq)
        stride_seconds = self.stride_seconds

        timestamps = []
        for i in range(len(windows)):
            start_time = i * stride_seconds
            end_time = start_time + self.window_seconds
            timestamps.append((start_time, end_time))

        return windows, timestamps


class WindowValidator:
    """Validate extracted windows for quality."""

    def __init__(self, expected_channels: int, expected_samples: int):
        """Initialize validator.

        Args:
            expected_channels: Expected number of channels
            expected_samples: Expected number of samples per window
        """
        self.expected_channels = expected_channels
        self.expected_samples = expected_samples

    def is_valid(self, window: np.ndarray) -> bool:
        """Check if window is valid.

        Args:
            window: Window to validate

        Returns:
            True if valid, False otherwise
        """
        # Check shape
        if window.shape != (self.expected_channels, self.expected_samples):
            return False

        # Check for NaN values
        if np.any(np.isnan(window)):
            return False

        # Check for infinite values
        return not np.any(np.isinf(window))


class BatchWindowExtractor:
    """Extract windows from multiple recordings."""

    def __init__(self, window_seconds: float = 8.0, overlap_seconds: float = 4.0):
        """Initialize batch extractor.

        Args:
            window_seconds: Window duration in seconds
            overlap_seconds: Overlap between windows in seconds
        """
        self.extractor = WindowExtractor(window_seconds, overlap_seconds)

    def extract_batch(
        self, recordings: list[np.ndarray], sfreq: float
    ) -> tuple[list[np.ndarray], list[int]]:
        """Extract windows from multiple recordings.

        Args:
            recordings: List of recordings, each of shape (n_channels, n_samples)
            sfreq: Sampling frequency in Hz

        Returns:
            Tuple of (all_windows, recording_indices) where recording_indices
            maps each window to its source recording
        """
        all_windows = []
        recording_indices = []

        for rec_idx, recording in enumerate(recordings):
            windows = self.extractor.extract(recording, sfreq)
            all_windows.extend(windows)
            recording_indices.extend([rec_idx] * len(windows))

        return all_windows, recording_indices
