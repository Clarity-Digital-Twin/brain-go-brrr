"""
MNE+Autoreject preprocessing for TUAB dataset.
Implements the verified preprocessing pipeline to improve EEGPT from 56% to 87% AUROC.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mne
from autoreject import AutoReject, Ransac

logger = logging.getLogger(__name__)


class TUABPreprocessor:
    """MNE+Autoreject preprocessing for TUAB dataset.

    This preprocessor implements the verified pipeline from our documentation:
    1. Load EDF with MNE
    2. Resample to 256 Hz (EEGPT requirement)
    3. Apply channel mapping (T3→T7, T4→T8, T5→P7, T6→P8)
    4. Bandpass filter (0.5-45 Hz)
    5. Notch filter (60 Hz)
    6. Detect bad channels with RANSAC
    7. Interpolate bad channels
    8. Re-reference to average
    9. Create 4-second epochs
    10. Apply Autoreject with TUAB-specific parameters
    """

    # TUAB channel mapping from old to modern naming
    CHANNEL_MAPPING = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}

    # Standard 20 channels for TUAB (after mapping) - MNE standard casing
    STANDARD_CHANNELS = [
        'Fp1',  # Note: MNE uses 'Fp' not 'FP'
        'Fp2',
        'F7',
        'F3',
        'Fz',  # Note: MNE uses lowercase 'z'
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
        """Initialize preprocessor with configuration.

        Args:
            config: Optional configuration dict with preprocessing parameters
        """
        self.config = config or {}

        # Set defaults from verified documentation
        self.sampling_rate = self.config.get('sampling_rate', 256)
        self.window_duration = self.config.get('window_duration', 4.0)
        self.window_overlap = self.config.get('window_overlap', 0.0)  # Overlap fraction (0.0 to 0.5)
        self.bandpass_low = self.config.get('bandpass_low', 0.5)
        self.bandpass_high = self.config.get('bandpass_high', 45.0)
        # Notch frequency: use config, then raw.info line_freq, then default to 60Hz (US)
        self.notch_freq = self.config.get('notch_freq', None)  # Will be set per file if None

        # TUAB-specific Autoreject parameters (verified)
        self.ar_n_interpolate = self.config.get('ar_n_interpolate', [1, 2, 3, 4])
        self.ar_consensus = self.config.get('ar_consensus', [0.3, 0.5, 0.7])
        self.ar_cv = self.config.get('ar_cv', 5)  # Reduced from default=10 for speed

        logger.info(f"Initialized TUABPreprocessor with config: {self.config}")

    def process_raw(self, edf_path: Path) -> tuple[mne.Epochs, dict[str, int]]:
        """Apply full preprocessing pipeline to raw EDF file.

        Args:
            edf_path: Path to EDF file

        Returns:
            Tuple of (clean_epochs, info_dict) where info_dict contains:
                - n_epochs_before: Number of epochs before Autoreject
                - n_epochs_after: Number of epochs after Autoreject
                - n_rejected: Number of rejected epochs
        """
        logger.info(f"Processing {edf_path}")

        # 1. Load with MNE
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        logger.info(f"Loaded {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

        # 2. Apply channel mapping (T3→T7, etc.)
        raw = self._apply_channel_mapping(raw)

        # 3. Standardize to 256 Hz if needed
        if raw.info['sfreq'] != self.sampling_rate:
            logger.info(f"Resampling from {raw.info['sfreq']} to {self.sampling_rate} Hz")
            raw.resample(self.sampling_rate, npad='auto')

        # 4. Set channel types and montage
        # Only set channels to 'eeg' if they're in our standard set
        # This avoids mis-typing EOG/ECG channels if present
        channel_types = {}
        standard_upper = {ch.upper() for ch in self.STANDARD_CHANNELS}

        for ch in raw.ch_names:
            ch_upper = ch.upper()
            # Check if it's a standard EEG channel
            if ch_upper in standard_upper:
                channel_types[ch] = 'eeg'
            # Detect EOG channels
            elif 'EOG' in ch_upper:
                channel_types[ch] = 'eog'
            # Detect ECG/EKG channels
            elif 'ECG' in ch_upper or 'EKG' in ch_upper:
                channel_types[ch] = 'ecg'
            # Leave others as-is

        if channel_types:
            raw.set_channel_types(channel_types)
            logger.info(f"Set channel types: {len([c for c in channel_types.values() if c == 'eeg'])} EEG, "
                       f"{len([c for c in channel_types.values() if c == 'eog'])} EOG, "
                       f"{len([c for c in channel_types.values() if c == 'ecg'])} ECG")

        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='warn')
        except Exception as e:
            logger.warning(f"Could not set montage: {e}")

        # 5. Apply MNE global preprocessing
        raw = self._apply_mne_preprocessing(raw)

        # 6. Create 4-second epochs
        epochs = self._create_epochs(raw)
        n_epochs_before = len(epochs)

        # 7. Apply Autoreject
        epochs_clean = self._apply_autoreject(epochs)
        n_epochs_after = len(epochs_clean)

        # Create info dict
        info = {
            'n_epochs_before': n_epochs_before,
            'n_epochs_after': n_epochs_after,
            'n_rejected': n_epochs_before - n_epochs_after,
        }

        return epochs_clean, info

    def _apply_channel_mapping(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply TUAB channel mapping from old to modern naming.

        Handles various TUAB channel name formats:
        - 'T3', 'EEG T3-REF', 'T3-REF' -> 'T7'
        - Case insensitive matching

        Args:
            raw: Raw MNE object

        Returns:
            Raw object with renamed channels and only standard 20 channels
        """
        import re

        # Create rename dictionary for various TUAB formats
        rename_dict = {}
        for ch_name in raw.ch_names:
            # Strip common prefixes and suffixes (case insensitive)
            clean_name = re.sub(r'^EEG\s+', '', ch_name, flags=re.IGNORECASE)
            clean_name = re.sub(r'-REF$', '', clean_name, flags=re.IGNORECASE)
            clean_name = clean_name.strip().upper()

            # Check if this matches an old channel name
            for old_name, new_name in self.CHANNEL_MAPPING.items():
                if clean_name == old_name.upper() and new_name not in raw.ch_names:
                    rename_dict[ch_name] = new_name
                    break

        if rename_dict:
            logger.info(f"Renaming channels: {rename_dict}")
            raw.rename_channels(rename_dict)

        # Now standardize channel names to match expected casing
        case_mapping = {}
        for ch_name in raw.ch_names:
            for std_name in self.STANDARD_CHANNELS:
                if ch_name.upper() == std_name.upper() and ch_name != std_name:
                    case_mapping[ch_name] = std_name
                    break

        if case_mapping:
            logger.info(f"Standardizing channel case: {case_mapping}")
            raw.rename_channels(case_mapping)

        # Select and reorder to standard 20 channels
        available_standard = [ch for ch in self.STANDARD_CHANNELS if ch in raw.ch_names]
        missing_channels = [ch for ch in self.STANDARD_CHANNELS if ch not in raw.ch_names]

        if missing_channels:
            logger.warning(f"Missing standard channels: {missing_channels}")
            if len(available_standard) < 19:  # Minimum requirement
                raise ValueError(
                    f"Too few standard channels ({len(available_standard)}/20). Need at least 19."
                )

        # Pick and reorder channels
        if len(available_standard) < 20:
            logger.warning(f"Only {len(available_standard)}/20 standard channels available")

        # Use raw.pick() for better compatibility across MNE versions
        # The order is preserved based on the input list
        raw.pick(available_standard)
        logger.info(f"Selected {len(raw.ch_names)} standard channels")

        return raw

    def _apply_mne_preprocessing(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply MNE global preprocessing steps.

        Args:
            raw: Raw MNE object

        Returns:
            Preprocessed raw object
        """
        # Bandpass filter (0.5-45 Hz for EEGPT)
        logger.info(f"Applying bandpass filter {self.bandpass_low}-{self.bandpass_high} Hz")
        raw.filter(self.bandpass_low, self.bandpass_high, fir_design='firwin', verbose=False)

        # Notch filter for line noise
        # Determine notch frequency: config > raw.info > default 60Hz
        if self.notch_freq is not None:
            notch_freq = self.notch_freq
        elif 'line_freq' in raw.info and raw.info['line_freq'] is not None:
            notch_freq = raw.info['line_freq']
            logger.info(f"Using line frequency from data: {notch_freq} Hz")
        else:
            notch_freq = 60.0  # Default to US standard
            logger.info("No line frequency in data, defaulting to 60 Hz")

        logger.info(f"Applying notch filter at {notch_freq} Hz")
        raw.notch_filter([notch_freq, notch_freq * 2], fir_design='firwin', verbose=False)

        # Detect and annotate muscle artifacts
        try:
            muscle_annot, muscle_scores = mne.preprocessing.annotate_muscle_zscore(
                raw, threshold=4.0, ch_type='eeg', min_length_good=0.2, filter_freq=(110, 140)
            )
            raw.set_annotations(raw.annotations + muscle_annot)
            logger.info(f"Found {len(muscle_annot)} muscle artifact segments")
        except Exception as e:
            logger.warning(f"Could not detect muscle artifacts: {e}")

        # Detect bad channels via RANSAC
        try:
            logger.info("Running RANSAC for bad channel detection")
            # Create temporary epochs for RANSAC
            events = mne.make_fixed_length_events(raw, duration=self.window_duration)
            epochs_temp = mne.Epochs(
                raw,
                events,
                tmin=0,
                tmax=self.window_duration,
                baseline=None,
                preload=True,
                verbose=False,
            )

            ransac = Ransac(n_jobs=1, random_state=42, verbose=False)
            ransac.fit(epochs_temp)

            if ransac.bad_chs_:
                logger.info(f"RANSAC detected bad channels: {ransac.bad_chs_}")
                raw.info['bads'].extend(ransac.bad_chs_)
                raw.interpolate_bads(reset_bads=True)
            else:
                logger.info("No bad channels detected by RANSAC")
        except Exception as e:
            logger.warning(f"RANSAC failed: {e}")

        # Re-reference to average
        logger.info("Setting average reference")
        raw.set_eeg_reference('average', projection=False)

        return raw

    def _create_epochs(self, raw: mne.io.Raw) -> mne.Epochs:
        """Create fixed-length epochs for EEGPT.

        Args:
            raw: Preprocessed raw object

        Returns:
            Epochs object
        """
        # Use make_fixed_length_epochs to avoid off-by-one sample issues
        # This properly handles window boundaries and annotations
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=self.window_duration,
            overlap=self.window_duration * self.window_overlap,  # Configurable overlap
            baseline=None,  # No baseline for EEGPT
            reject_by_annotation=True,  # Respect bad segments from muscle detection
            preload=True,
            verbose=False,
        )

        logger.info(f"Created {len(epochs)} epochs of {self.window_duration}s")
        return epochs

    def _apply_autoreject(self, epochs: mne.Epochs) -> mne.Epochs:
        """Apply Autoreject with TUAB-optimized parameters.

        Args:
            epochs: MNE Epochs object

        Returns:
            Clean epochs after Autoreject
        """
        logger.info("Applying Autoreject with TUAB-specific parameters")

        # TUAB-specific parameters (verified against documentation)
        ar = AutoReject(
            n_interpolate=self.ar_n_interpolate,  # [1, 2, 3, 4] for 20 channels
            consensus=self.ar_consensus,  # [0.3, 0.5, 0.7] for clinical data
            cv=self.ar_cv,  # 5, reduced from default=10
            thresh_method='bayesian_optimization',
            random_state=42,
            n_jobs=1,
            verbose=False,
        )

        # Fit and transform
        epochs_clean = ar.fit_transform(epochs)

        # Log statistics
        n_epochs_before = len(epochs)
        n_epochs_after = len(epochs_clean)
        percent_removed = 100 * (1 - n_epochs_after / n_epochs_before)

        logger.info(f"Autoreject: {n_epochs_before} → {n_epochs_after} epochs")
        logger.info(f"Removed: {n_epochs_before - n_epochs_after} ({percent_removed:.1f}%)")

        # Log learned parameters (they're dicts by channel type)
        if hasattr(ar, 'n_interpolate_'):
            logger.info(f"Learned n_interpolate (EEG): {ar.n_interpolate_.get('eeg', 'N/A')}")
            logger.info(f"Learned consensus (EEG): {ar.consensus_.get('eeg', 'N/A')}")

        return epochs_clean
