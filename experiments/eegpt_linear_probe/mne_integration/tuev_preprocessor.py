"""
MNE+Autoreject preprocessing for TUEV dataset.
Extends TUABPreprocessor to handle TUEV's 23-channel format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import mne
import numpy as np

from .preprocessor import TUABPreprocessor

logger = logging.getLogger(__name__)


class TUEVPreprocessor(TUABPreprocessor):
    """MNE+Autoreject preprocessing for TUEV dataset.

    Key differences from TUAB:
    1. 23 input channels (not 20)
    2. Different channel naming convention
    3. Multi-class labels from .lab files
    """

    # TUEV uses TCP montage with 23 channels
    # These are the channels present in TUEV
    TUEV_CHANNELS = [
        'FP1',
        'FP2',
        'FPZ',
        'F7',
        'F3',
        'FZ',
        'F4',
        'F8',
        'A1',
        'T3',
        'C3',
        'CZ',
        'C4',
        'T4',
        'A2',
        'T5',
        'P3',
        'PZ',
        'P4',
        'T6',
        'O1',
        'OZ',
        'O2',
    ]

    # Map TUEV channels to standard 20 (dropping A1, A2, FPZ)
    # Also handle old naming (T3→T7, T4→T8, T5→P7, T6→P8)
    CHANNEL_MAPPING = {
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8',
        'A1': None,  # Drop reference channels
        'A2': None,
        'FPZ': None,  # Drop extra midline channel
    }

    # Final 20 channels we want (standard 10-20 without FPZ)
    STANDARD_CHANNELS = [
        'Fp1',
        'Fp2',
        'F7',
        'F3',
        'Fz',
        'F4',
        'F8',
        'T7',
        'C3',
        'Cz',
        'C4',
        'T8',
        'P7',
        'P3',
        'Pz',
        'P4',
        'P8',
        'O1',
        'O2',
        'Oz',
    ]

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize TUEV preprocessor.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        logger.info("Initialized TUEVPreprocessor for 23→20 channel mapping")

    def _apply_channel_mapping(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply TUEV channel mapping from 23 to 20 channels.

        Handles:
        - Old to modern naming (T3→T7, etc.)
        - Dropping reference channels (A1, A2)
        - Dropping extra midline channel (FPZ)
        - Case normalization

        Args:
            raw: Raw MNE object with 23 channels

        Returns:
            Raw object with 20 standard channels
        """
        import re

        # First normalize channel names (TUEV uses uppercase)
        rename_dict = {}
        channels_to_drop = []

        for ch_name in raw.ch_names:
            # Clean channel name
            clean_name = re.sub(r'^EEG\s+', '', ch_name, flags=re.IGNORECASE)
            clean_name = re.sub(r'-REF$', '', clean_name, flags=re.IGNORECASE)
            clean_name = clean_name.strip().upper()

            # Check if this needs mapping
            if clean_name in self.CHANNEL_MAPPING:
                new_name = self.CHANNEL_MAPPING[clean_name]
                if new_name is None:
                    # Mark for dropping
                    channels_to_drop.append(ch_name)
                else:
                    # Map to new name
                    rename_dict[ch_name] = new_name
            else:
                # Standardize case (FP1 -> Fp1, FZ -> Fz, etc.)
                if clean_name == 'FP1':
                    rename_dict[ch_name] = 'Fp1'
                elif clean_name == 'FP2':
                    rename_dict[ch_name] = 'Fp2'
                elif clean_name == 'FZ':
                    rename_dict[ch_name] = 'Fz'
                elif clean_name == 'CZ':
                    rename_dict[ch_name] = 'Cz'
                elif clean_name == 'PZ':
                    rename_dict[ch_name] = 'Pz'
                elif clean_name == 'OZ':
                    rename_dict[ch_name] = 'Oz'
                elif clean_name == 'FPZ':
                    # Drop FPZ
                    channels_to_drop.append(ch_name)
                else:
                    # Keep channel with proper case
                    for std_name in self.STANDARD_CHANNELS:
                        if clean_name == std_name.upper():
                            rename_dict[ch_name] = std_name
                            break

        # Apply renaming
        if rename_dict:
            logger.info(f"Renaming {len(rename_dict)} channels")
            raw.rename_channels(rename_dict)

        # Drop unwanted channels
        if channels_to_drop:
            logger.info(f"Dropping {len(channels_to_drop)} channels: {channels_to_drop}")
            raw.drop_channels(channels_to_drop)

        # Select and reorder to standard 20 channels
        available_standard = [ch for ch in self.STANDARD_CHANNELS if ch in raw.ch_names]
        missing_channels = [ch for ch in self.STANDARD_CHANNELS if ch not in raw.ch_names]

        # Critical: Enforce minimum channel requirement per SSOT
        if missing_channels:
            # Warn-once logic for missing channels
            if not hasattr(self, '_warned_missing_channels'):
                self._warned_missing_channels = set()

            missing_key = frozenset(missing_channels)
            if missing_key not in self._warned_missing_channels:
                logger.warning(f"Missing standard channels: {missing_channels}")
                self._warned_missing_channels.add(missing_key)

            if len(available_standard) < 19:  # Minimum requirement
                error_msg = (
                    f"Too few standard channels ({len(available_standard)}/20). Need at least 19."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Pick and reorder channels
        raw.pick(available_standard)
        logger.info(f"Selected {len(raw.ch_names)} standard channels from TUEV's 23")

        return raw

    def _apply_channel_mapping_with_tracking(self, raw: mne.io.Raw) -> tuple[mne.io.Raw, list[str]]:
        """Apply channel mapping and return missing channels list."""
        # First get the list of missing channels before processing
        import re

        # Check which standard channels will be missing
        raw_channels_upper = set()
        for ch_name in raw.ch_names:
            clean_name = re.sub(r'^EEG\s+', '', ch_name, flags=re.IGNORECASE)
            clean_name = re.sub(r'-REF$', '', clean_name, flags=re.IGNORECASE)
            clean_name = clean_name.strip().upper()

            # Apply mapping if needed
            if clean_name in self.CHANNEL_MAPPING:
                mapped = self.CHANNEL_MAPPING[clean_name]
                if mapped:
                    raw_channels_upper.add(mapped.upper())
            else:
                # Standardize casing
                if clean_name in ['FP1', 'FP2', 'FZ', 'CZ', 'PZ', 'OZ'] or clean_name != 'FPZ':
                    raw_channels_upper.add(clean_name)

        # Find missing channels
        standard_upper = {ch.upper() for ch in self.STANDARD_CHANNELS}
        missing = list(standard_upper - raw_channels_upper)

        # Now apply the actual mapping
        raw = self._apply_channel_mapping(raw)

        return raw, missing

    def process_raw_with_annotations(
        self, edf_path: Path, annotations: list[dict[str, float | str]], window_overlap: float = 0.0
    ) -> tuple[mne.Epochs, dict[str, int], list[str]]:
        """Process raw EDF file with fixed-grid windows and overlap-based labeling.

        Args:
            edf_path: Path to EDF file
            annotations: List of dicts with 'start', 'end', 'label' keys
            window_overlap: Overlap fraction (0.0 or 0.5)

        Returns:
            Tuple of (clean_epochs, info_dict, window_labels)
        """
        logger.info(f"Processing {edf_path} with {len(annotations)} annotations")

        # Load and preprocess raw data
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        logger.info(f"Loaded {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

        # Apply TUEV-specific channel mapping (23→20) and track missing channels
        raw, missing_channels = self._apply_channel_mapping_with_tracking(raw)

        # Apply standard preprocessing (filtering, resampling, etc.)
        if raw.info['sfreq'] != self.sampling_rate:
            logger.info(f"Resampling from {raw.info['sfreq']} to {self.sampling_rate} Hz")
            raw.resample(self.sampling_rate, npad='auto')

        # Apply MNE preprocessing
        raw = self._apply_mne_preprocessing(raw)

        # Create fixed-grid windows
        windows = self._create_fixed_grid_windows(raw, window_overlap)

        # Label each window based on annotations
        window_labels = []
        for win_start, win_end in windows:
            label = self._label_window(win_start, win_end, annotations)
            window_labels.append(label)

        # Create epochs from fixed grid
        # MNE requires event codes > 0, so we use a single code for all windows
        # Labels are tracked separately in window_labels
        events = []
        window_event_code = 1  # Single event code for all windows

        for i, (win_start, win_end) in enumerate(windows):
            start_sample = int(win_start * raw.info['sfreq'])
            events.append([start_sample, 0, window_event_code])  # All windows get code 1

        events_array = np.array(events, dtype=int)  # MNE requires int dtype for events
        event_id = {"window": window_event_code}  # Simple event_id dict

        # Create epochs
        epochs = mne.Epochs(
            raw,
            events_array,
            event_id=event_id,
            tmin=0,
            tmax=self.window_duration,
            baseline=None,
            preload=True,
            verbose=False,
        )

        n_epochs_before = len(epochs)

        # Apply Autoreject with gentle parameters for TUEV
        epochs_clean, ar_params = self._apply_autoreject_tuev(epochs)
        n_epochs_after = len(epochs_clean)

        # Create info dict with comprehensive metadata
        info = {
            'n_epochs_before': n_epochs_before,
            'n_epochs_after': n_epochs_after,
            'n_rejected': n_epochs_before - n_epochs_after,
            'reject_rate': (n_epochs_before - n_epochs_after) / n_epochs_before
            if n_epochs_before > 0
            else 0,
            'missing_channels': missing_channels,  # Track for QC
            'sfreq_after': raw.info['sfreq'],  # Should be 256 Hz after resampling
            'window_overlap': window_overlap,  # Track overlap used
            'final_channels': list(raw.ch_names),  # Track exact channel order
            'ar_learned_params': ar_params,  # Track AR learned parameters
        }

        # Log warning if reject rate too high (warn-once per file)
        if info['reject_rate'] > 0.15:
            # Create a unique warning key for this file
            warning_key = f"high_reject_{edf_path.name}"
            if not hasattr(self, '_warned_files'):
                self._warned_files = set()

            if warning_key not in self._warned_files:
                logger.warning(
                    f"High reject rate for {edf_path.name}: {info['reject_rate']:.2%}. Consider adjusting AR parameters."
                )
                self._warned_files.add(warning_key)

        return epochs_clean, info, window_labels

    def _create_fixed_grid_windows(
        self, raw: mne.io.Raw, overlap: float = 0.0
    ) -> list[tuple[float, float]]:
        """Create fixed-grid windows with optional overlap.

        Args:
            raw: MNE Raw object
            overlap: Overlap fraction (0.0 = no overlap, 0.5 = 50% overlap)

        Returns:
            List of (start, end) times in seconds
        """
        duration = raw.times[-1]
        step = self.window_duration * (1 - overlap)

        windows = []
        start = 0
        while start + self.window_duration <= duration:
            windows.append((start, start + self.window_duration))
            start += step

        logger.info(f"Created {len(windows)} fixed-grid windows with {overlap*100:.0f}% overlap")
        return windows

    def _label_window(
        self, win_start: float, win_end: float, annotations: list[dict[str, float | str]]
    ) -> str:
        """Label a window using argmax overlap with spike priority.

        Args:
            win_start: Window start time in seconds
            win_end: Window end time in seconds
            annotations: List of annotations with 'start', 'end', 'label' keys

        Returns:
            Label string (one of: 'spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg')
        """
        overlaps = {}

        for ann in annotations:
            # Calculate overlap in seconds
            overlap_start = max(win_start, ann['start'])
            overlap_end = min(win_end, ann['end'])
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                label = ann['label']
                if label not in overlaps:
                    overlaps[label] = 0
                overlaps[label] += overlap_duration

        # Priority override: if spike has sufficient overlap (≥120ms), prioritize it
        spike_priority_threshold = 0.12  # 120ms
        if overlaps.get('spsw', 0) >= spike_priority_threshold:
            return 'spsw'

        # Otherwise use argmax with minimum duration
        minimum_overlap_threshold = 0.10  # 100ms
        if overlaps:
            # Only consider if overlap ≥ minimum threshold
            valid_overlaps = {k: v for k, v in overlaps.items() if v >= minimum_overlap_threshold}

            if valid_overlaps:
                # Find max overlap
                max_overlap = max(valid_overlaps.values())

                # Handle ties with deterministic priority
                priority_order = ['spsw', 'gped', 'pled', 'artf', 'eyem', 'bckg']
                for label in priority_order:
                    if label in valid_overlaps and valid_overlaps[label] == max_overlap:
                        return label

        # Default to background
        return 'bckg'

    def _apply_autoreject_tuev(self, epochs: mne.Epochs) -> tuple[mne.Epochs, dict]:
        """Apply Autoreject with gentle parameters for TUEV spike preservation.

        Args:
            epochs: Input epochs

        Returns:
            Tuple of (clean epochs, learned parameters dict)
        """
        from autoreject import AutoReject

        # Gentle parameters for TUEV
        ar = AutoReject(
            n_interpolate=[1, 2],  # Gentler than TUAB
            consensus=[0.5, 0.7, 0.9],  # Higher thresholds
            cv=3,  # Faster
            thresh_method='bayesian_optimization',
            random_state=42,
            verbose=False,
        )

        epochs_clean = ar.fit_transform(epochs)

        # Collect learned parameters
        ar_params = {}
        if hasattr(ar, 'n_interpolate_'):
            ar_params['n_interpolate'] = ar.n_interpolate_.get('eeg', None)
            logger.info(f"Learned n_interpolate: {ar.n_interpolate_}")
        if hasattr(ar, 'consensus_'):
            ar_params['consensus'] = ar.consensus_.get('eeg', None)
            logger.info(f"Learned consensus: {ar.consensus_}")

        return epochs_clean, ar_params
