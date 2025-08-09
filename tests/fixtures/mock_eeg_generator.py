"""Mock EEG data generators for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mne


class MockEEGGenerator:
    """Generate realistic mock EEG data for testing."""

    # TUAB standard channels with old naming
    TUAB_CHANNELS = [
        "FP1",
        "FP2",
        "F7",
        "F3",
        "FZ",
        "F4",
        "F8",
        "T3",
        "C3",
        "CZ",
        "C4",
        "T4",  # Old naming
        "T5",
        "P3",
        "PZ",
        "P4",
        "T6",  # Old naming
        "O1",
        "O2",
    ]

    @staticmethod
    def create_raw(
        duration: float = 60.0,
        sfreq: int = 256,
        n_channels: int | None = None,
        ch_names: list[str] | None = None,
        add_artifacts: bool = True,
        seed: int | None = None,
    ) -> mne.io.RawArray:
        """Create mock raw EEG data.

        Args:
            duration: Duration in seconds
            sfreq: Sampling frequency
            n_channels: Number of channels (defaults to len(ch_names))
            ch_names: Channel names (defaults to TUAB channels)
            add_artifacts: Whether to add realistic artifacts
            seed: Random seed for reproducibility

        Returns:
            Mock raw EEG data
        """
        if seed is not None:
            np.random.seed(seed)

        # Use TUAB channels by default
        if ch_names is None:
            ch_names = (
                MockEEGGenerator.TUAB_CHANNELS[:n_channels]
                if n_channels
                else MockEEGGenerator.TUAB_CHANNELS
            )

        n_channels = len(ch_names)
        n_samples = int(sfreq * duration)

        # Generate realistic EEG data (50 µV scale)
        data = np.random.randn(n_channels, n_samples) * 50e-6

        # Add 1/f noise for realism
        for ch in range(n_channels):
            freqs = np.fft.fftfreq(n_samples, 1 / sfreq)
            fft = np.fft.fft(data[ch])
            fft[1:] = fft[1:] / np.sqrt(np.abs(freqs[1:]))  # 1/f scaling
            data[ch] = np.real(np.fft.ifft(fft))

        if add_artifacts:
            data = MockEEGGenerator._add_artifacts(data, sfreq)

        # Create MNE Raw object
        import mne

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        return raw

    @staticmethod
    def _add_artifacts(data: np.ndarray, sfreq: int) -> np.ndarray:
        """Add realistic artifacts to EEG data."""
        n_channels, n_samples = data.shape
        duration = n_samples / sfreq

        # Eye blinks (frontal channels)
        if duration > 10:
            n_blinks = int(duration / 15)  # ~4 blinks per minute
            for _ in range(n_blinks):
                blink_time = np.random.uniform(5, duration - 5)
                blink_idx = int(blink_time * sfreq)
                blink_duration = int(0.3 * sfreq)  # 300ms

                # Affect frontal channels more
                frontal_channels = min(4, n_channels)
                data[:frontal_channels, blink_idx : blink_idx + blink_duration] += (
                    np.random.randn(frontal_channels, blink_duration) * 150e-6
                )

        # Muscle artifacts (temporal channels)
        if duration > 20 and n_channels > 8:
            muscle_start = np.random.uniform(10, duration - 10)
            muscle_idx = int(muscle_start * sfreq)
            muscle_duration = int(2 * sfreq)  # 2 seconds

            # Affect temporal channels
            temporal_indices = [7, 11] if n_channels > 11 else [min(7, n_channels - 1)]
            for idx in temporal_indices:
                if idx < n_channels:
                    data[idx, muscle_idx : muscle_idx + muscle_duration] += (
                        np.random.randn(muscle_duration) * 100e-6
                    )

        # Bad channel (flat or extremely noisy)
        if n_channels > 10:
            bad_channel = np.random.randint(0, n_channels)
            if np.random.random() > 0.5:
                data[bad_channel, :] = np.random.randn(n_samples) * 0.1e-6  # Flat
            else:
                data[bad_channel, :] = np.random.randn(n_samples) * 300e-6  # Very noisy

        return data

    @staticmethod
    def create_epochs(
        n_epochs: int = 20,
        epoch_duration: float = 10.0,
        sfreq: int = 256,
        ch_names: list[str] | None = None,
    ) -> mne.Epochs:
        """Create mock epochs directly."""
        # Create longer raw data
        raw = MockEEGGenerator.create_raw(
            duration=(n_epochs + 1) * epoch_duration, sfreq=sfreq, ch_names=ch_names
        )

        # Create epochs
        import mne

        events = mne.make_fixed_length_events(raw, duration=epoch_duration)
        epochs = mne.Epochs(
            raw, events, tmin=0, tmax=epoch_duration, baseline=None, preload=True, verbose=False
        )

        return epochs[:n_epochs]  # Return exact number requested
