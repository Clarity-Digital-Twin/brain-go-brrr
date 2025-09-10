"""TUEV Event Extractor for paper parity - extracts 5s segments at 200Hz around events."""

from pathlib import Path

import mne
import numpy as np
import numpy.typing as npt


class TUEVEventExtractor:
    """Extract 5-second segments around annotated events per EEGPT paper.

    This implements the exact preprocessing from reference_repos/EEGPT/downstream_tueg/dataset_maker/make_TUEV.py:
    - 5-second segments at 200Hz (not 4s at 256Hz)
    - Bandpass filter 0.1-75Hz, notch filter at 50Hz
    - Uses referential "-REF" channels (NO bipolar conversion)
    - Outputs (23, 1000) segments in Volts
    """

    # EXACT channel order from EEGPT reference (23 channels)
    TUEV_CHANNELS_REF = [
        'EEG FP1-REF',
        'EEG FP2-REF',
        'EEG F3-REF',
        'EEG F4-REF',
        'EEG C3-REF',
        'EEG C4-REF',
        'EEG P3-REF',
        'EEG P4-REF',
        'EEG O1-REF',
        'EEG O2-REF',
        'EEG F7-REF',
        'EEG F8-REF',
        'EEG T3-REF',
        'EEG T4-REF',
        'EEG T5-REF',
        'EEG T6-REF',
        'EEG A1-REF',
        'EEG A2-REF',
        'EEG FZ-REF',
        'EEG CZ-REF',
        'EEG PZ-REF',
        'EEG T1-REF',
        'EEG T2-REF',
    ]

    def __init__(
        self,
        target_fs: int = 200,  # EEGPT uses 200Hz not 256Hz!
        segment_duration: float = 5.0,  # 5 seconds
        tmin: float = -2.0,  # 2 seconds before event
        tmax: float = 3.0,
    ):  # 3 seconds after event
        """Initialize TUEV event extractor with paper parameters.

        Args:
            target_fs: Target sampling rate (200Hz per EEGPT)
            segment_duration: Duration of each segment (5 seconds)
            tmin: Time before event center (-2 seconds)
            tmax: Time after event center (+3 seconds)
        """
        self.target_fs = target_fs
        self.segment_duration = segment_duration
        self.tmin = tmin
        self.tmax = tmax

        # Validate parameters match paper
        assert self.segment_duration == self.tmax - self.tmin, "Duration must equal tmax - tmin"
        assert self.target_fs == 200, "EEGPT TUEV uses 200Hz, not 256Hz"

    def extract_segments(
        self, edf_path: Path, annotations: list[dict[str, float | int]]
    ) -> list[tuple[npt.NDArray[np.float32], int]]:
        """Extract event-centered segments from EDF file.

        Args:
            edf_path: Path to EDF file
            annotations: List of annotation dicts with 'start', 'end', 'label' keys

        Returns:
            List of (segment, label) tuples where:
                segment shape: (23, 1000) - 23 channels, 5 seconds @ 200Hz
                label: integer class label
        """
        # Load EDF with MNE
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

        # Apply EEGPT preprocessing: bandpass 0.1-75Hz, notch 50Hz
        raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
        raw.notch_filter(freqs=50.0, verbose=False)

        # Resample to 200Hz (EEGPT standard for TUEV)
        if raw.info['sfreq'] != self.target_fs:
            raw.resample(self.target_fs, verbose=False)

        # Select and reorder channels to match reference
        available_channels = [ch for ch in self.TUEV_CHANNELS_REF if ch in raw.ch_names]
        # Use inst.pick to avoid legacy pick_channels and preserve order explicitly
        raw.pick(picks=available_channels)
        if raw.ch_names != available_channels:
            raw.reorder_channels(available_channels)

        # Get data and handle missing channels
        data = raw.get_data()  # Shape: (n_available_channels, n_samples)

        # Pad with zeros if channels missing (maintain 23 channel format)
        if len(available_channels) < 23:
            # Create full 23-channel array with zeros
            full_data = np.zeros((23, data.shape[1]), dtype=np.float32)
            for i, ch in enumerate(available_channels):
                idx = self.TUEV_CHANNELS_REF.index(ch)
                full_data[idx] = data[i].astype(np.float32)
            data = full_data
        else:
            data = data.astype(np.float32)

        # Extract segments around events using authors' offset-concatenation trick
        segments = []
        fs = int(self.target_fs)
        samples_per_segment = int(self.segment_duration * self.target_fs)  # 1000 samples
        times = np.arange(data.shape[1]) / float(self.target_fs)

        # Precompute extended signal to avoid boundary drops
        offset = data.shape[1]
        extended = np.concatenate([data, data, data], axis=1)

        for annot in annotations:
            # Use annotation start/end directly, as in reference
            start_sec = float(annot['start'])
            end_sec = float(annot['end'])

            # Map to sample indices similar to np.where(times >= t)[0][0]
            start_idx = int(np.searchsorted(times, start_sec, side='left'))
            end_idx = int(np.searchsorted(times, end_sec, side='left'))

            # Authors' code assumes end-start == fs (1s). Enforce 1s if needed.
            if (end_idx - start_idx) != fs:
                end_idx = start_idx + fs

            # Slice from extended buffer: [offset + start - 2s : offset + end + 2s]
            cut_start = offset + start_idx - 2 * fs
            cut_end = offset + end_idx + 2 * fs

            # Ensure exact length of 5 seconds
            if (cut_end - cut_start) != samples_per_segment:
                # Adjust end to maintain exact segment length
                cut_end = cut_start + samples_per_segment

            segment = extended[:, cut_start:cut_end]
            assert segment.shape[1] == samples_per_segment, (
                f"Segment length {segment.shape[1]} != {samples_per_segment}"
            )

            label = int(annot['label']) if not isinstance(annot['label'], int) else annot['label']
            segments.append((segment.astype(np.float32), label))

        return segments
