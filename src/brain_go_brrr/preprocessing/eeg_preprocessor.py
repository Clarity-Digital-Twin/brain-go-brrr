"""EEG preprocessing pipeline following BioSerenity-E1 specifications.

This module provides a standardized preprocessing pipeline for EEG data
that prepares recordings for abnormality detection using EEGPT.
"""

import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)

try:
    from autoreject import AutoReject

    AUTOREJECT_AVAILABLE = True
except ImportError:
    AUTOREJECT_AVAILABLE = False

# Standard 10-20 channel mapping from EEGPT
# Maps various channel naming conventions to standard indices
CHANNEL_MAPPING = {
    # Frontal
    "FP1": 0,
    "Fp1": 0,
    "FPZ": 1,
    "Fpz": 1,
    "FP2": 2,
    "Fp2": 2,
    "AF7": 3,
    "AF3": 4,
    "AF4": 5,
    "AF8": 6,
    "F7": 7,
    "F5": 8,
    "F3": 9,
    "F1": 10,
    "FZ": 11,
    "Fz": 11,
    "F2": 12,
    "F4": 13,
    "F6": 14,
    "F8": 15,
    # Fronto-temporal
    "FT7": 16,
    "FC5": 17,
    "FC3": 18,
    "FC1": 19,
    "FCZ": 20,
    "FCz": 20,
    "FC2": 21,
    "FC4": 22,
    "FC6": 23,
    "FT8": 24,
    # Temporal/Central
    "T7": 25,
    "T3": 25,  # T3/T7 are same position
    "C5": 26,
    "C3": 27,
    "C1": 28,
    "CZ": 29,
    "Cz": 29,
    "C2": 30,
    "C4": 31,
    "C6": 32,
    "T8": 33,
    "T4": 33,  # T4/T8 are same position
    # Centro-parietal
    "TP7": 34,
    "CP5": 35,
    "CP3": 36,
    "CP1": 37,
    "CPZ": 38,
    "CPz": 38,
    "CP2": 39,
    "CP4": 40,
    "CP6": 41,
    "TP8": 42,
    # Parietal
    "P7": 43,
    "P5": 44,
    "P3": 45,
    "P1": 46,
    "PZ": 47,
    "Pz": 47,
    "P2": 48,
    "P4": 49,
    "P6": 50,
    "P8": 51,
    # Occipital
    "PO7": 52,
    "PO3": 53,
    "POZ": 54,
    "POz": 54,
    "PO4": 55,
    "PO8": 56,
    "O1": 57,
    "OZ": 58,
    "Oz": 58,
    "O2": 59,
    "IZ": 60,
    "Iz": 60,
    # Common alternatives
    "T5": 43,  # P7 in newer nomenclature
    "T6": 51,  # P8 in newer nomenclature
}

# BioSerenity-E1 16-channel montage based on standard 10-20 positions
# Provides good coverage while being computationally efficient
BIOSERENITY_16_CHANNELS = [
    "Fp1",
    "Fp2",  # Frontal polar
    "F3",
    "F4",  # Frontal
    "F7",
    "F8",  # Lateral frontal
    "C3",
    "C4",  # Central
    "T3",
    "T4",  # Temporal (T7/T8 in newer nomenclature)
    "P3",
    "P4",  # Parietal
    "O1",
    "O2",  # Occipital
    "Fz",
    "Cz",  # Midline
]

# Alternative names for compatibility
CHANNEL_ALIASES = {"T3": ["T7"], "T4": ["T8"], "T5": ["P7"], "T6": ["P8"]}


class EEGPreprocessor:
    """EEG preprocessing pipeline following BioSerenity-E1 specifications."""

    def __init__(
        self,
        target_sfreq: int = 128,
        lowpass_freq: float = 45.0,
        highpass_freq: float = 0.5,
        notch_freq: float = 50.0,
        channel_subset_size: int = 16,
        use_standard_montage: bool = True,
        use_autoreject: bool = True,
    ):
        """Initialize preprocessor with filtering parameters.

        Args:
            target_sfreq: Target sampling frequency (128 Hz for BioSerenity-E1)
            lowpass_freq: Low-pass filter cutoff (45 Hz)
            highpass_freq: High-pass filter cutoff (0.5 Hz)
            notch_freq: Notch filter frequency (50/60 Hz)
            channel_subset_size: Number of channels to select (16)
            use_standard_montage: Use BioSerenity-E1 16-channel montage
            use_autoreject: Use Autoreject for artifact rejection (recommended)
        """
        self.target_sfreq = target_sfreq
        self.lowpass_freq = lowpass_freq
        self.highpass_freq = highpass_freq
        self.notch_freq = notch_freq
        self.channel_subset_size = channel_subset_size
        self.use_standard_montage = use_standard_montage
        self.use_autoreject = use_autoreject and AUTOREJECT_AVAILABLE

    def preprocess(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply full preprocessing pipeline.

        Args:
            raw: Input EEG data

        Returns:
            Preprocessed EEG data
        """
        # Make a copy to avoid modifying original
        raw = raw.copy()

        # Apply Autoreject first if available - it handles artifacts before filtering
        if self.use_autoreject:
            raw = self._apply_autoreject(raw)

        # Resample first if needed to avoid Nyquist issues
        raw = self._resample_to_target(raw)

        # Apply filters in order
        raw = self._apply_highpass_filter(raw)
        raw = self._apply_lowpass_filter(raw)
        raw = self._apply_notch_filter(raw)
        raw = self._apply_average_reference(raw)
        raw = self._select_channel_subset(raw)

        return raw

    def _apply_autoreject(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply Autoreject for artifact rejection.

        This uses local+global thresholding to detect bad segments via K-fold CV,
        drastically reducing artifacts before filtering.

        References:
        ----------
        Jas et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG data.
        NeuroImage, 159, 417-429.
        """
        if not AUTOREJECT_AVAILABLE:
            return raw

        # Check if we have valid channel positions
        montage = raw.get_montage()
        if montage is None:
            logger.warning(
                "No channel positions found - using amplitude-based rejection instead of AutoReject"
            )
            return self._amplitude_based_rejection(raw)

        # Check if positions are valid (not all zeros)
        positions = np.array(
            [
                ch["loc"][:3]
                for ch in raw.info["chs"]
                if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH
            ]
        )
        if len(positions) == 0 or np.allclose(positions, 0):
            logger.warning(
                "Channel positions are invalid - using amplitude-based rejection instead of AutoReject"
            )
            return self._amplitude_based_rejection(raw)

        # Check if we have enough EEG channels for AutoReject
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) < 5:  # AutoReject needs at least 5 channels
            logger.warning(
                f"Only {len(eeg_picks)} EEG channels found - using amplitude-based rejection instead of AutoReject"
            )
            return self._amplitude_based_rejection(raw)

        # Create fixed-length epochs for Autoreject (4s windows matching EEGPT)
        # This preserves the temporal structure for artifact detection
        epochs = mne.make_fixed_length_epochs(
            raw, duration=4.0, preload=True, proj=False, verbose=False
        )

        # Apply Autoreject with automatic threshold detection
        # Using n_jobs=-1 to use all available cores
        ar = AutoReject(n_jobs=-1, verbose=False)
        ar.fit(epochs)  # Learn thresholds via cross-validation
        epochs_clean = ar.transform(epochs)  # Apply learned thresholds

        # Convert back to continuous Raw format while preserving metadata
        # Using concatenate_raws with proper Raw conversion
        raw_clean = mne.concatenate_raws(
            [mne.io.RawArray(epoch, epochs.info, verbose=False) for epoch in epochs_clean],
            verbose=False,
        )

        # Preserve all critical metadata from original
        if raw.annotations is not None:
            raw_clean.set_annotations(raw.annotations.copy())

        # Preserve other important info fields
        if "meas_date" in raw.info and raw.info["meas_date"] is not None:
            raw_clean.set_meas_date(raw.info["meas_date"])
        # Note: dig (digitization) info is preserved automatically through epochs
        for key in ["line_freq", "device_info", "subject_info"]:
            if key in raw.info:
                raw_clean.info[key] = raw.info[key]

        return raw_clean

    def _amplitude_based_rejection(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Simple amplitude-based artifact rejection when positions are not available.

        This fallback method uses amplitude thresholds to detect and mark bad channels
        and segments when AutoReject cannot be used due to missing channel positions.
        """
        # Get EEG channels
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(picks) == 0:
            return raw

        data = raw.get_data(picks=picks)

        # Detect high amplitude artifacts (>150 µV)
        threshold = 150e-6
        bad_mask = np.abs(data) > threshold

        # Mark channels as bad if >10% of samples exceed threshold
        bad_channels = []
        for i, ch_idx in enumerate(picks):
            if np.sum(bad_mask[i]) > 0.1 * data.shape[1]:
                bad_channels.append(raw.ch_names[ch_idx])

        if bad_channels:
            raw.info["bads"].extend(bad_channels)
            logger.info(f"Marked {len(bad_channels)} bad channels using amplitude criteria")

        # Also detect and annotate bad segments
        # Find segments where multiple channels exceed threshold
        n_bad_per_sample = np.sum(bad_mask, axis=0)
        bad_segments = n_bad_per_sample > (0.5 * len(picks))  # >50% channels bad

        # Convert to annotations
        sfreq = raw.info["sfreq"]
        bad_starts = np.where(np.diff(np.concatenate([[False], bad_segments, [False]])) == 1)[0]
        bad_ends = np.where(np.diff(np.concatenate([[False], bad_segments, [False]])) == -1)[0]

        if len(bad_starts) > 0 and len(bad_starts) == len(bad_ends):
            onsets = bad_starts / sfreq
            durations = (bad_ends - bad_starts) / sfreq
            descriptions = ["BAD_amplitude"] * len(onsets)

            annotations = mne.Annotations(
                onsets, durations, descriptions, orig_time=raw.info["meas_date"]
            )
            raw.set_annotations(raw.annotations + annotations)
            logger.info(f"Marked {len(onsets)} bad segments using amplitude criteria")
        elif len(bad_starts) != len(bad_ends):
            logger.warning(
                f"Mismatch in bad segment boundaries: {len(bad_starts)} starts, {len(bad_ends)} ends"
            )

        return raw

    def _apply_highpass_filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply 0.5 Hz high-pass filter to remove DC and drift."""
        # Use higher order filter for better attenuation (>40dB requirement)
        raw.filter(
            l_freq=self.highpass_freq,
            h_freq=None,
            picks="eeg",
            method="iir",
            iir_params={"order": 8, "ftype": "butter", "output": "sos"},
            verbose=False,
        )
        return raw

    def _apply_lowpass_filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply 45 Hz low-pass filter to remove high-frequency noise."""
        # Use higher order filter for better attenuation
        raw.filter(
            l_freq=None,
            h_freq=self.lowpass_freq,
            picks="eeg",
            method="iir",
            iir_params={"order": 8, "ftype": "butter", "output": "sos"},
            verbose=False,
        )
        return raw

    def _apply_notch_filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply notch filter to remove powerline interference."""
        raw.notch_filter(freqs=self.notch_freq, picks="eeg", method="iir", verbose=False)
        return raw

    def _resample_to_target(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Resample to target frequency (128 Hz for BioSerenity-E1)."""
        if raw.info["sfreq"] != self.target_sfreq:
            raw.resample(sfreq=self.target_sfreq, verbose=False)
        return raw

    def _apply_average_reference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply average re-referencing."""
        raw.set_eeg_reference("average", projection=False, verbose=False)
        return raw

    def _select_channel_subset(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Select 16-channel subset as per BioSerenity-E1."""
        # If we already have 16 or fewer channels, return as is
        if len(raw.ch_names) <= self.channel_subset_size:
            return raw

        if self.use_standard_montage:
            # Try to use BioSerenity-E1 standard 16-channel montage
            channels_to_find = BIOSERENITY_16_CHANNELS.copy()
            found_channels = []

            # Normalize channel names in raw data for matching
            raw_channels_upper = {ch.upper(): ch for ch in raw.ch_names}

            for target_ch in channels_to_find:
                # Try exact match first
                if target_ch in raw.ch_names:
                    found_channels.append(target_ch)
                # Try uppercase match
                elif target_ch.upper() in raw_channels_upper:
                    found_channels.append(raw_channels_upper[target_ch.upper()])
                # Try aliases (e.g., T3 -> T7)
                elif target_ch in CHANNEL_ALIASES:
                    for alias in CHANNEL_ALIASES[target_ch]:
                        if alias in raw.ch_names:
                            found_channels.append(alias)
                            break
                        elif alias.upper() in raw_channels_upper:
                            found_channels.append(raw_channels_upper[alias.upper()])
                            break

            # If we found enough channels from the standard montage, use them
            if len(found_channels) >= self.channel_subset_size:
                channels_to_keep = found_channels[: self.channel_subset_size]
            else:
                # Fill remaining slots with channels ordered by EEGPT mapping priority
                channels_to_keep = found_channels.copy()

                # Sort remaining channels by their position in CHANNEL_MAPPING
                remaining_channels = [ch for ch in raw.ch_names if ch not in channels_to_keep]
                channel_priorities = []

                for ch in remaining_channels:
                    # Get priority from CHANNEL_MAPPING
                    priority = CHANNEL_MAPPING.get(ch, CHANNEL_MAPPING.get(ch.upper(), 999))
                    channel_priorities.append((priority, ch))

                # Sort by priority and add channels until we reach subset size
                channel_priorities.sort(key=lambda x: x[0])
                for _, ch in channel_priorities:
                    channels_to_keep.append(ch)
                    if len(channels_to_keep) >= self.channel_subset_size:
                        break
        else:
            # Fallback: Use first N channels
            channels_to_keep = raw.ch_names[: self.channel_subset_size]

        # Pick the selected channels
        raw.pick_channels(channels_to_keep, ordered=True)
        return raw

    def extract_windows(
        self, raw: mne.io.BaseRaw, window_duration: float = 16.0, overlap: float = 0.0
    ) -> list[np.ndarray]:
        """Extract windows from EEG data.

        Args:
            raw: Preprocessed EEG data
            window_duration: Window size in seconds (16s for BioSerenity-E1)
            overlap: Overlap between windows (0-1)

        Returns:
            List of window arrays
        """
        data = raw.get_data()
        sfreq = raw.info["sfreq"]

        window_samples = int(window_duration * sfreq)
        step_samples = int(window_samples * (1 - overlap))

        windows = []
        start = 0
        while start + window_samples <= data.shape[1]:
            window = data[:, start : start + window_samples]
            windows.append(window)
            start += step_samples

        return windows
