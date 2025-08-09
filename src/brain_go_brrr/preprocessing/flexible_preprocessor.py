"""Flexible EEG preprocessor that handles heterogeneous data formats."""

import logging
from typing import Literal

import mne
import numpy as np

from brain_go_brrr._typing import MNERaw

logger = logging.getLogger(__name__)

# Channel name mappings for common EEG datasets
CHANNEL_MAPPINGS = {
    # Sleep-EDF mappings
    "EEG Fpz-Cz": "Fpz",
    "EEG Pz-Oz": "Pz",
    "EEG C3-A2": "C3",
    "EEG C4-A1": "C4",
    "EOG horizontal": "EOG",
    "EOG vertical": "VEOG",
    "Resp oro-nasal": "RESP",
    "EMG submental": "EMG",
    "Temp rectal": "MISC",
    "Event marker": "STIM",
    # Common variations
    "EEG Fp1-A2": "Fp1",
    "EEG Fp2-A1": "Fp2",
    "EEG F3-A2": "F3",
    "EEG F4-A1": "F4",
    "EEG O1-A2": "O1",
    "EEG O2-A1": "O2",
}

# Task-specific channel preferences
TASK_CHANNELS = {
    "sleep": ["C3", "C4", "F3", "F4", "O1", "O2", "EOG", "EMG"],
    "abnormality": [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Fz",
        "Cz",
        "Pz",
    ],
    "event_detection": ["Fp1", "Fp2", "F7", "F8", "T3", "T4", "T5", "T6"],
}


class FlexibleEEGPreprocessor:
    """Flexible preprocessor that adapts to different EEG data formats."""

    target_sfreq: float | None
    lowpass_freq: float | None
    highpass_freq: float
    notch_freq: float | None

    def __init__(
        self,
        mode: Literal["auto", "sleep", "abnormality", "event_detection", "minimal"] = "auto",
        target_sfreq: float | None = None,
        lowpass_freq: float | None = None,
        highpass_freq: float = 0.5,
        notch_freq: float | None = None,
        use_autoreject: bool = True,
        require_positions: bool = False,
    ):
        """Initialize flexible preprocessor.

        Args:
            mode: Preprocessing mode that sets defaults
            target_sfreq: Target sampling frequency
            lowpass_freq: Low-pass filter frequency
            highpass_freq: High-pass filter frequency
            notch_freq: Notch filter frequency (auto-detect if None)
            use_autoreject: Whether to use Autoreject (falls back if no positions)
            require_positions: Whether to require channel positions
        """
        self.mode = mode
        self.require_positions = require_positions
        self.use_autoreject = use_autoreject

        # Set mode-specific defaults
        if mode == "sleep":
            self.target_sfreq = target_sfreq or 100
            self.lowpass_freq = lowpass_freq or 35
            self.highpass_freq = 0.3  # Lower for sleep
        elif mode == "abnormality":
            self.target_sfreq = target_sfreq or 256  # EEGPT requirement
            self.lowpass_freq = lowpass_freq or 45
            self.highpass_freq = highpass_freq
        elif mode == "event_detection":
            self.target_sfreq = target_sfreq or 256
            self.lowpass_freq = lowpass_freq or 50
            self.highpass_freq = highpass_freq
        elif mode == "minimal":
            self.target_sfreq = target_sfreq  # Keep original
            self.lowpass_freq = lowpass_freq or 50
            self.highpass_freq = highpass_freq
        else:  # auto
            self.target_sfreq = target_sfreq
            self.lowpass_freq = lowpass_freq or 45
            self.highpass_freq = highpass_freq

        self.notch_freq = notch_freq

    def preprocess(self, raw: MNERaw) -> mne.io.BaseRaw:
        """Preprocess EEG data with flexible handling.

        Args:
            raw: Input EEG data

        Returns:
            Preprocessed EEG data
        """
        # Make a copy
        raw = raw.copy()

        # Check positions if required
        if self.require_positions and raw.get_montage() is None:
            raise ValueError("Channel positions required but not found in data")

        # Step 1: Map channel names to standard
        raw = self._standardize_channel_names(raw)

        # Step 2: Select relevant channels
        raw = self._select_channels(raw)

        # Step 3: Add montage if possible and missing
        raw = self._add_montage_if_possible(raw)

        # Step 4: Apply filters
        raw = self._apply_filters(raw)

        # Step 5: Resample if needed
        raw = self._resample(raw)

        # Step 6: Apply artifact rejection
        raw = self._apply_artifact_rejection(raw)

        # Step 7: Apply reference
        raw = self._apply_reference(raw)

        return raw

    def _standardize_channel_names(self, raw: MNERaw) -> mne.io.BaseRaw:
        """Standardize channel names to 10-20 system."""
        mapping = self._map_channel_names(raw.ch_names)
        if mapping:
            raw.rename_channels(mapping)
            logger.info(f"Renamed {len(mapping)} channels to standard names")
        return raw

    def _map_channel_names(self, ch_names: list[str]) -> dict[str, str]:
        """Create mapping from current names to standard names."""
        mapping = {}
        for ch in ch_names:
            if ch in CHANNEL_MAPPINGS:
                mapping[ch] = CHANNEL_MAPPINGS[ch]
            # Also check case-insensitive
            elif ch.upper() in [k.upper() for k in CHANNEL_MAPPINGS]:
                for key, val in CHANNEL_MAPPINGS.items():
                    if ch.upper() == key.upper():
                        mapping[ch] = val
                        break
        return mapping

    def _select_channels(self, raw: MNERaw) -> mne.io.BaseRaw:
        """Select channels based on task."""
        # First pick only EEG channels (and EOG/EMG for sleep)
        if self.mode == "sleep":
            # For sleep, keep EEG, EOG, and EMG channels
            picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True, exclude="bads")
            ch_names = [raw.ch_names[i] for i in picks]

            # Keep all EEG channels for sleep analysis
            eeg_channels = [ch for ch in ch_names if raw.get_channel_types([ch])[0] == "eeg"]
            if eeg_channels:
                raw.pick_channels(ch_names, ordered=True)
                logger.info(f"Selected {len(ch_names)} channels for sleep (EEG+EOG+EMG)")
            else:
                # If no EEG channels, just keep what we have
                logger.warning("No EEG channels found, keeping all channels for sleep")
        else:
            # For other modes, pick only EEG
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            ch_names = [raw.ch_names[i] for i in picks]

            # Then select based on task
            if self.mode in TASK_CHANNELS and ch_names:
                selected = self._select_channels_for_task(ch_names, self.mode)
                if selected:
                    raw.pick_channels(selected, ordered=True)
                    logger.info(f"Selected {len(selected)} channels for {self.mode}")
            elif ch_names:
                # Just pick EEG channels
                raw.pick_types(eeg=True, exclude="bads")

        return raw

    def _select_channels_for_task(self, ch_names: list[str], task: str) -> list[str]:
        """Select best channels for a specific task."""
        preferred = TASK_CHANNELS.get(task, [])
        selected = []

        for ch in preferred:
            if ch in ch_names:
                selected.append(ch)
            # Also check with common variations
            elif f"{ch}-A1" in ch_names:
                selected.append(f"{ch}-A1")
            elif f"{ch}-A2" in ch_names:
                selected.append(f"{ch}-A2")

        # If too few channels found, add more
        if len(selected) < 4:
            for ch in ch_names:
                if ch not in selected and len(selected) < 10:
                    selected.append(ch)

        return selected

    def _add_montage_if_possible(self, raw: MNERaw) -> mne.io.BaseRaw:
        """Add standard montage if channels match standard names."""
        if raw.get_montage() is not None:
            return raw

        # Try to add standard montage
        try:
            # Check if we have standard channel names
            standard_names = [
                "Fp1",
                "Fp2",
                "F3",
                "F4",
                "C3",
                "C4",
                "P3",
                "P4",
                "O1",
                "O2",
                "F7",
                "F8",
                "T3",
                "T4",
                "T5",
                "T6",
                "Fz",
                "Cz",
                "Pz",
                "Fpz",
                "Oz",
            ]

            matches = [ch for ch in raw.ch_names if ch in standard_names]

            if len(matches) >= 3:  # Need at least 3 standard channels
                montage = mne.channels.make_standard_montage("standard_1020")
                raw.set_montage(montage, match_case=False, on_missing="ignore")
                logger.info("Added standard 10-20 montage")
        except Exception as e:
            logger.debug(f"Could not add montage: {e}")

        return raw

    def _apply_filters(self, raw: MNERaw) -> mne.io.BaseRaw:
        """Apply frequency filters."""
        # High-pass filter
        if self.highpass_freq > 0:
            raw.filter(
                l_freq=self.highpass_freq, h_freq=None, picks="eeg", method="fir", verbose=False
            )

        # Low-pass filter
        if self.lowpass_freq and self.lowpass_freq < raw.info["sfreq"] / 2:
            raw.filter(
                l_freq=None, h_freq=self.lowpass_freq, picks="eeg", method="fir", verbose=False
            )

        # Notch filter
        if self.notch_freq is None:
            # Auto-detect power line frequency
            self.notch_freq = 50 if raw.info.get("line_freq", 50) == 50 else 60

        if self.notch_freq < raw.info["sfreq"] / 2:
            raw.notch_filter(freqs=self.notch_freq, picks="eeg", verbose=False)

        return raw

    def _resample(self, raw: MNERaw) -> mne.io.BaseRaw:
        """Resample to target frequency if needed."""
        if self.target_sfreq and raw.info["sfreq"] != self.target_sfreq:
            raw.resample(sfreq=self.target_sfreq, verbose=False)
            logger.info(f"Resampled to {self.target_sfreq} Hz")
        return raw

    def _apply_artifact_rejection(self, raw: MNERaw) -> mne.io.BaseRaw:
        """Apply artifact rejection with fallback methods."""
        if not self.use_autoreject:
            return raw

        has_positions = raw.get_montage() is not None

        if has_positions:
            # Try to use Autoreject
            try:
                from autoreject import AutoReject

                # Create epochs for Autoreject
                epochs = mne.make_fixed_length_epochs(
                    raw, duration=2.0, preload=True, proj=False, verbose=False
                )

                ar = AutoReject(n_jobs=1, verbose=False)
                epochs_clean = ar.fit_transform(epochs)

                # Convert back to raw
                raw = mne.concatenate_raws(
                    [mne.io.RawArray(epoch, epochs.info, verbose=False) for epoch in epochs_clean]
                )

                logger.info("Applied Autoreject artifact rejection")

            except Exception as e:
                logger.warning(f"Autoreject failed: {e}, using fallback")
                raw = self._fallback_artifact_rejection(raw)
        else:
            # Use fallback method
            raw = self._fallback_artifact_rejection(raw)

        return raw

    def _fallback_artifact_rejection(self, raw: MNERaw) -> mne.io.BaseRaw:
        """Simple amplitude-based artifact rejection."""
        # Find and interpolate bad segments
        data = raw.get_data(picks="eeg")

        # Detect high amplitude artifacts
        threshold = 150e-6  # 150 μV
        bad_times = np.any(np.abs(data) > threshold, axis=0)

        if np.any(bad_times):
            # Simple clipping for now
            raw._data[:, bad_times] = np.clip(raw._data[:, bad_times], -threshold, threshold)
            logger.info(
                f"Clipped {np.sum(bad_times)} samples with amplitude > {threshold * 1e6:.0f} μV"
            )

        return raw

    def _apply_reference(self, raw: MNERaw) -> mne.io.BaseRaw:
        """Apply appropriate reference."""
        # Average reference if we have enough channels
        if len(raw.ch_names) >= 10:
            raw.set_eeg_reference("average", projection=False, verbose=False)
            logger.info("Applied average reference")
        else:
            logger.info("Kept original reference (too few channels for average)")

        return raw
