"""Minimal preprocessing for sleep EEG data.

This module provides a simplified preprocessing pipeline specifically designed
for sleep analysis with YASA, following the recommendations from the YASA
reference implementation.
"""

import logging

import mne

from brain_go_brrr._typing import MNERaw

logger = logging.getLogger(__name__)


class SleepPreprocessor:
    """Minimal preprocessor for sleep EEG data.

    This preprocessor follows YASA's recommendations:
    - Bandpass filter 0.3-35 Hz
    - Resample to 100 Hz
    - Average reference
    - No aggressive artifact rejection
    - No notch filtering

    Designed for PSG/sleep data with limited channels (e.g., Sleep-EDF with 2 EEG channels).
    """

    def __init__(
        self,
        l_freq: float = 0.3,
        h_freq: float = 35.0,
        target_sfreq: float = 100.0,
        reference: str = "average",
    ):
        """Initialize sleep preprocessor.

        Args:
            l_freq: Low-pass filter frequency (Hz)
            h_freq: High-pass filter frequency (Hz)
            target_sfreq: Target sampling frequency (Hz)
            reference: Reference type ('average' or None)
        """
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.reference = reference

    def preprocess(self, raw: MNERaw) -> MNERaw:
        """Apply minimal preprocessing for sleep analysis.

        Args:
            raw: Raw EEG data

        Returns:
            Preprocessed EEG data
        """
        # Make a copy to avoid modifying original
        raw = raw.copy()

        # Log initial state
        logger.info(f"Sleep preprocessing: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

        # 1. Bandpass filter (sleep-specific frequencies)
        # Only filter data channels (EEG, EOG, EMG) - not misc/stim channels
        logger.info(f"Applying bandpass filter: {self.l_freq}-{self.h_freq} Hz")
        raw.filter(
            self.l_freq,
            self.h_freq,
            picks=["eeg", "eog", "emg"],  # Only filter relevant channels
            fir_design="firwin",
            verbose=False,
        )

        # 2. Resample if needed (YASA standard is 100 Hz)
        if raw.info["sfreq"] != self.target_sfreq:
            logger.info(f"Resampling from {raw.info['sfreq']} Hz to {self.target_sfreq} Hz")
            raw.resample(self.target_sfreq, npad="auto")

        # 3. Set average reference (if requested)
        if self.reference == "average":
            logger.info("Setting average reference")
            raw.set_eeg_reference("average", projection=False)

        # Log final state
        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        logger.info(f"Preprocessing complete: {n_eeg} EEG channels at {raw.info['sfreq']} Hz")

        return raw

    def preprocess_for_yasa(
        self,
        raw: MNERaw,
        eeg_channels: list[str] | None = None,
        eog_channels: list[str] | None = None,
        emg_channels: list[str] | None = None,
    ) -> mne.io.BaseRaw:
        """Preprocess and prepare channels for YASA sleep staging.

        This method ensures channel types are properly set for YASA.

        Args:
            raw: Raw EEG data
            eeg_channels: List of EEG channel names
            eog_channels: List of EOG channel names
            emg_channels: List of EMG channel names

        Returns:
            Preprocessed data ready for YASA
        """
        # Make a copy to avoid modifying original
        raw = raw.copy()

        # Set channel types BEFORE preprocessing (needed for filtering)
        if eeg_channels:
            raw.set_channel_types({ch: "eeg" for ch in eeg_channels if ch in raw.ch_names})

        if eog_channels:
            raw.set_channel_types({ch: "eog" for ch in eog_channels if ch in raw.ch_names})

        if emg_channels:
            raw.set_channel_types({ch: "emg" for ch in emg_channels if ch in raw.ch_names})

        # Now apply standard preprocessing
        return self.preprocess(raw)
