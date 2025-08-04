"""Adapter classes for AutoReject integration with windowed data.

Clean adapter pattern - no overengineering, just simple conversion between
sliding windows and MNE epochs for AutoReject compatibility.
"""

import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)


class WindowEpochAdapter:
    """Convert between sliding windows and MNE Epochs.

    Simple adapter that makes our sliding window approach compatible
    with AutoReject's epoch-based processing.
    """

    def __init__(self, window_duration: float = 10.0, window_stride: float = 5.0):
        """Initialize adapter.

        Args:
            window_duration: Window length in seconds
            window_stride: Stride between windows in seconds
        """
        self.window_duration = window_duration
        self.window_stride = window_stride
        self.overlap = 1.0 - (window_stride / window_duration)

    def raw_to_windowed_epochs(self, raw: mne.io.Raw) -> mne.Epochs:
        """Convert raw data to epochs matching our windowing scheme.

        Args:
            raw: Continuous raw EEG data

        Returns:
            Epochs object with windows as epochs

        Raises:
            ValueError: If recording is too short for even one window
        """
        sfreq = raw.info["sfreq"]
        n_samples = raw.n_times
        window_samples = int(self.window_duration * sfreq)
        stride_samples = int(self.window_stride * sfreq)

        # Check if we can extract at least one window
        if n_samples < window_samples:
            raise ValueError(
                f"Recording too short ({n_samples / sfreq:.1f}s) for "
                f"{self.window_duration}s windows"
            )

        # Create events at window start positions
        start_positions = np.arange(0, n_samples - window_samples + 1, stride_samples)

        # MNE events format: [sample_idx, 0, event_id]
        events = np.column_stack(
            [
                start_positions,
                np.zeros(len(start_positions), dtype=int),
                np.ones(len(start_positions), dtype=int),
            ]
        )

        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"window": 1},
            tmin=0,
            tmax=self.window_duration - (1.0 / sfreq),
            baseline=None,
            preload=True,
            verbose=False,
            proj=False,  # Don't apply projections
        )

        logger.debug(f"Created {len(epochs)} epochs from {n_samples / sfreq:.1f}s recording")

        return epochs

    def epochs_to_continuous(
        self, epochs_clean: mne.Epochs, original_raw: mne.io.Raw
    ) -> mne.io.Raw:
        """Reconstruct continuous data from cleaned epochs.

        Handles overlapping windows by averaging in overlap regions.

        Args:
            epochs_clean: Cleaned epochs from AutoReject
            original_raw: Original raw data (for metadata)

        Returns:
            Reconstructed continuous raw data
        """
        data_clean = epochs_clean.get_data()  # (n_epochs, n_channels, n_times)
        n_channels = data_clean.shape[1]
        total_samples = original_raw.n_times

        # Initialize output arrays
        reconstructed = np.zeros((n_channels, total_samples))
        counts = np.zeros(total_samples)

        # Get event times (start of each window)
        events = epochs_clean.events

        # Reconstruct with overlap handling
        for i, (event_sample, _, _) in enumerate(events):
            if i < len(data_clean):  # Ensure we have data
                start_idx = event_sample
                end_idx = start_idx + data_clean.shape[2]

                if end_idx <= total_samples:
                    # Add data and increment counts for averaging
                    reconstructed[:, start_idx:end_idx] += data_clean[i]
                    counts[start_idx:end_idx] += 1

        # Average overlapping regions
        counts[counts == 0] = 1  # Avoid division by zero
        reconstructed /= counts

        # Create new raw object
        raw_clean = mne.io.RawArray(reconstructed, original_raw.info.copy(), verbose=False)

        logger.debug(
            f"Reconstructed {total_samples / original_raw.info['sfreq']:.1f}s of continuous data"
        )

        return raw_clean


class SyntheticPositionGenerator:
    """Generate anatomically valid channel positions for datasets lacking montage info.

    Simple, clean approach - just standard 10-20 positions, no magic.
    """

    # Standard 10-20 positions in meters (from MNE-Python)
    STANDARD_1020_POSITIONS = {
        # Frontal
        "FP1": np.array([-0.0270, 0.0866, 0.0150]),
        "FP2": np.array([0.0270, 0.0866, 0.0150]),
        "F7": np.array([-0.0702, 0.0596, -0.0150]),
        "F3": np.array([-0.0450, 0.0693, 0.0300]),
        "FZ": np.array([0.0000, 0.0732, 0.0450]),
        "F4": np.array([0.0450, 0.0693, 0.0300]),
        "F8": np.array([0.0702, 0.0596, -0.0150]),
        # Temporal
        "T7": np.array([-0.0860, 0.0000, -0.0150]),
        "T3": np.array([-0.0860, 0.0000, -0.0150]),  # Old name, same position
        "T8": np.array([0.0860, 0.0000, -0.0150]),
        "T4": np.array([0.0860, 0.0000, -0.0150]),  # Old name, same position
        # Central
        "C3": np.array([-0.0520, 0.0000, 0.0600]),
        "CZ": np.array([0.0000, 0.0000, 0.0850]),
        "C4": np.array([0.0520, 0.0000, 0.0600]),
        # Parietal
        "P7": np.array([-0.0702, -0.0596, -0.0150]),
        "T5": np.array([-0.0702, -0.0596, -0.0150]),  # Old name, same position
        "P3": np.array([-0.0450, -0.0693, 0.0300]),
        "PZ": np.array([0.0000, -0.0732, 0.0450]),
        "P4": np.array([0.0450, -0.0693, 0.0300]),
        "P8": np.array([0.0702, -0.0596, -0.0150]),
        "T6": np.array([0.0702, -0.0596, -0.0150]),  # Old name, same position
        # Occipital
        "O1": np.array([-0.0270, -0.0866, 0.0150]),
        "OZ": np.array([0.0000, -0.0918, 0.0000]),
        "O2": np.array([0.0270, -0.0866, 0.0150]),
    }

    def add_positions_to_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Add synthetic but anatomically valid positions to raw data.

        Args:
            raw: Raw EEG data without channel positions

        Returns:
            Raw data with added montage
        """
        ch_names = raw.ch_names
        ch_pos = {}

        # Map channels to standard positions
        for ch_name in ch_names:
            ch_upper = ch_name.upper()
            if ch_upper in self.STANDARD_1020_POSITIONS:
                ch_pos[ch_name] = self.STANDARD_1020_POSITIONS[ch_upper]

        if not ch_pos:
            logger.warning(
                f"No standard channel names found in {ch_names[:5]}..., "
                "using evenly spaced positions"
            )
            # Fallback: evenly spaced on a circle
            ch_pos = self._create_circular_positions(ch_names)
        else:
            logger.info(f"Added standard positions for {len(ch_pos)}/{len(ch_names)} channels")

        # Create and set montage
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos)
        raw.set_montage(montage, on_missing="ignore")

        return raw

    def _create_circular_positions(self, ch_names: list) -> dict:
        """Create evenly spaced positions on a circle as fallback.

        Simple circular arrangement at head level.
        """
        n_channels = len(ch_names)
        radius = 0.075  # 7.5 cm - realistic head radius
        z_level = 0.02  # 2 cm above origin

        angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)

        ch_pos = {}
        for i, ch_name in enumerate(ch_names):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            ch_pos[ch_name] = np.array([x, y, z_level])

        return ch_pos
