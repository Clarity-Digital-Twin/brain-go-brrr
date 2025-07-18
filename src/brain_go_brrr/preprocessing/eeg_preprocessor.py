"""EEG preprocessing pipeline following BioSerenity-E1 specifications.

This module provides a standardized preprocessing pipeline for EEG data
that prepares recordings for abnormality detection using EEGPT.
"""

import mne
import numpy as np


class EEGPreprocessor:
    """EEG preprocessing pipeline following BioSerenity-E1 specifications."""

    def __init__(
        self,
        target_sfreq: int = 128,
        lowpass_freq: float = 45.0,
        highpass_freq: float = 0.5,
        notch_freq: float = 50.0,
        channel_subset_size: int = 16
    ):
        """Initialize preprocessor with filtering parameters.

        Args:
            target_sfreq: Target sampling frequency (128 Hz for BioSerenity-E1)
            lowpass_freq: Low-pass filter cutoff (45 Hz)
            highpass_freq: High-pass filter cutoff (0.5 Hz)
            notch_freq: Notch filter frequency (50/60 Hz)
            channel_subset_size: Number of channels to select (16)
        """
        self.target_sfreq = target_sfreq
        self.lowpass_freq = lowpass_freq
        self.highpass_freq = highpass_freq
        self.notch_freq = notch_freq
        self.channel_subset_size = channel_subset_size

    def preprocess(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply full preprocessing pipeline.

        Args:
            raw: Input EEG data

        Returns:
            Preprocessed EEG data
        """
        # Make a copy to avoid modifying original
        raw = raw.copy()

        # Apply filters in order
        raw = self._apply_highpass_filter(raw)
        raw = self._apply_lowpass_filter(raw)
        raw = self._apply_notch_filter(raw)
        raw = self._resample_to_target(raw)
        raw = self._apply_average_reference(raw)
        raw = self._select_channel_subset(raw)

        return raw

    def _apply_highpass_filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply 0.5 Hz high-pass filter to remove DC and drift."""
        raw.filter(
            l_freq=self.highpass_freq,
            h_freq=None,
            picks='eeg',
            method='iir',
            iir_params={'order': 5, 'ftype': 'butter'},
            verbose=False
        )
        return raw

    def _apply_lowpass_filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply 45 Hz low-pass filter to remove high-frequency noise."""
        raw.filter(
            l_freq=None,
            h_freq=self.lowpass_freq,
            picks='eeg',
            method='iir',
            iir_params={'order': 5, 'ftype': 'butter'},
            verbose=False
        )
        return raw

    def _apply_notch_filter(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply notch filter to remove powerline interference."""
        raw.notch_filter(
            freqs=self.notch_freq,
            picks='eeg',
            method='iir',
            verbose=False
        )
        return raw

    def _resample_to_target(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Resample to target frequency (128 Hz for BioSerenity-E1)."""
        if raw.info['sfreq'] != self.target_sfreq:
            raw.resample(sfreq=self.target_sfreq, verbose=False)
        return raw

    def _apply_average_reference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply average re-referencing."""
        raw.set_eeg_reference('average', projection=False, verbose=False)
        return raw

    def _select_channel_subset(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Select 16-channel subset as per BioSerenity-E1."""
        # If we already have 16 or fewer channels, return as is
        if len(raw.ch_names) <= self.channel_subset_size:
            return raw

        # Define priority channels for 16-channel montage
        # Based on standard 10-20 system, excluding T5, T6, Pz
        priority_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'Fz', 'Cz'
        ]

        # Find channels that exist in the data
        available_channels = []
        for ch in priority_channels:
            if ch in raw.ch_names:
                available_channels.append(ch)

        # If we have enough priority channels, use them
        if len(available_channels) >= self.channel_subset_size:
            channels_to_keep = available_channels[:self.channel_subset_size]
        else:
            # Otherwise, keep priority channels and fill with others
            channels_to_keep = available_channels.copy()
            for ch in raw.ch_names:
                if ch not in channels_to_keep:
                    channels_to_keep.append(ch)
                if len(channels_to_keep) >= self.channel_subset_size:
                    break

        # Pick the selected channels
        raw.pick_channels(channels_to_keep, ordered=True)
        return raw

    def extract_windows(
        self,
        raw: mne.io.BaseRaw,
        window_duration: float = 16.0,
        overlap: float = 0.0
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
        sfreq = raw.info['sfreq']

        window_samples = int(window_duration * sfreq)
        step_samples = int(window_samples * (1 - overlap))

        windows = []
        start = 0
        while start + window_samples <= data.shape[1]:
            window = data[:, start:start + window_samples]
            windows.append(window)
            start += step_samples

        return windows
