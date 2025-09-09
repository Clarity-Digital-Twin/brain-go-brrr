"""MNE+Autoreject preprocessing for TUEV dataset.

Extends TUABPreprocessor to handle TUEV's 23-channel format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from brain_go_brrr.utils import mask_path_for_log

if TYPE_CHECKING:
    from pathlib import Path

import mne
import numpy as np

from brain_go_brrr.infra.data.channels import CHANNELS_TUEV_20

from .channel_utils import canonicalize_channel_labels, canonicalize_channel_types
from .mne_preprocessor import TUABPreprocessor

logger = logging.getLogger(__name__)

# 23 channels for paper parity (using canonical/modern names after T3→T7 mapping)
# CRITICAL: Use mixed-case naming to match canonicalize_channel_labels output
CHANNELS_TUEV_23_CANONICAL = [
    'Fp1',  # Mixed-case: Fp not FP
    'Fp2',
    'F3',
    'F4',
    'C3',
    'C4',
    'P3',
    'P4',
    'O1',
    'O2',
    'F7',
    'F8',
    'T7',
    'T8',
    'P7',
    'P8',  # Modern names (T3→T7, etc.)
    'A1',
    'A2',
    'Fz',  # Mixed-case: lowercase z
    'Cz',
    'Pz',
    'T1',
    'T2',
]


class TUEVPreprocessor(TUABPreprocessor):
    """MNE+Autoreject preprocessing for TUEV dataset.

    Modes:
    - Paper parity (use_paper_parity=True): Keep all 23 raw channels (incl. A1/A2/T1/T2), no synthesis; mapper handles 23→20 before EEGPT.
    - Legacy (use_paper_parity=False): Map to a canonical 20‑channel interface (drops A1/A2; may synthesize Fpz). Not paper‑parity.

    Key differences from TUAB:
    1. 23 input channels (parity mode) vs 20 (legacy interface)
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

    # Map TUEV channels to standard 20 (dropping A1, A2 only)
    # Also handle old naming (T3→T7, T4→T8, T5→P7, T6→P8)
    # Override parent class type to allow None values for dropped channels
    CHANNEL_MAPPING: dict[str, str] = {
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8',
        # Note: A1, A2, FPZ are handled separately in _map_to_standard_channels
    }

    # Final 20 channels we want (canonical 10-20 WITH Fz and Fpz, WITHOUT Oz)
    # Single Source of Truth: import from infra.data.channels
    STANDARD_CHANNELS = CHANNELS_TUEV_20

    def __init__(self, config: dict[str, Any] | None = None, use_paper_parity: bool = False):
        """Initialize TUEV preprocessor.

        Args:
            config: Optional configuration dict
                - disable_ransac: Skip RANSAC bad channel detection (default: True for TUEV)
            use_paper_parity: If True, keep all 23 channels for learned mapper.
                            If False, use existing 20-ch preprocessing approach.
        """
        super().__init__(config)
        # Default to True for TUEV since RANSAC has internal bug with our channel config
        self.disable_ransac = (config or {}).get('disable_ransac', True)
        self.use_paper_parity = use_paper_parity

        if use_paper_parity:
            # Override parent settings for 23-channel mode
            self.STANDARD_CHANNELS = CHANNELS_TUEV_23_CANONICAL
            self.n_channels = 23
            logger.info(
                f"TUEVPreprocessor: Paper parity mode - keeping 23 channels "
                f"(RANSAC: {'disabled' if self.disable_ransac else 'enabled'})"
            )
        else:
            # Use existing 20-channel mapping (legacy, not paper parity)
            self.STANDARD_CHANNELS = CHANNELS_TUEV_20
            self.n_channels = 20
            logger.info(
                f"TUEVPreprocessor: Legacy mode - mapping to 20 channels (NOT paper parity) "
                f"(RANSAC: {'disabled' if self.disable_ransac else 'enabled'})"
            )

    def _apply_channel_mapping(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply TUEV channel mapping.

        For paper parity (23 channels):
        - Keep all channels including A1, A2, T1, T2
        - No synthesis, no dropping

        For standard mode (20 channels):
        - Old to modern naming (T3→T7, etc.)
        - Dropping reference channels (A1, A2)
        - Including Fpz (synthesized if missing), excluding Oz
        - Case normalization

        Args:
            raw: Raw MNE object

        Returns:
            Raw object with appropriate number of channels
        """
        if self.use_paper_parity:
            return self._apply_23_channel_mapping(raw)
        else:
            return self._apply_20_channel_mapping(raw)

    def _apply_20_channel_mapping(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply standard 20-channel mapping (existing behavior)."""
        # After canonicalization, channel names are already standardized
        # We just need to drop unwanted reference channels (A1, A2)
        channels_to_drop = []

        for ch_name in raw.ch_names:
            # Check if this should be dropped
            if ch_name in ['A1', 'A2']:  # Drop only reference channels
                channels_to_drop.append(ch_name)

        # Drop unwanted channels
        if channels_to_drop:
            logger.info(f"Dropping {len(channels_to_drop)} channels: {channels_to_drop}")
            raw.drop_channels(channels_to_drop)

        # Synthesize any missing canonical channels (e.g., Fpz) as zeros
        raw = self._synthesize_missing_channels(raw)

        # Set channel types to 'eeg' for canonical channels to ensure proper filtering
        # This also prevents the "misc channels" montage warning
        channel_types = {}
        for ch in raw.ch_names:
            if ch in self.STANDARD_CHANNELS:
                channel_types[ch] = 'eeg'
            # Keep existing types for non-standard channels (EOG, ECG, etc.)

        if channel_types:
            raw.set_channel_types(channel_types, verbose=False)
            logger.debug(f"Set {len(channel_types)} channels to 'eeg' type")

        # Set standard 1020 montage for all channels including synthesized ones
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='warn')  # warn first to see any issues
            logger.info(f"Set standard_1020 montage for {len(raw.ch_names)} channels")
        except Exception as e:
            logger.warning(f"Could not set montage: {e}")
            # Could retry with on_missing='ignore' if needed

        # Select and reorder to standard 20 channels
        available_standard = [ch for ch in self.STANDARD_CHANNELS if ch in raw.ch_names]
        missing_channels = [ch for ch in self.STANDARD_CHANNELS if ch not in raw.ch_names]

        # Log any remaining missing channels after synthesis
        if missing_channels:
            logger.warning(f"Missing standard channels after synthesis: {missing_channels}")

        # Pick and reorder channels - enforce exactly 20 for TUEV
        # CRITICAL: TUEV needs exactly 20 channels (including Fz and Fpz, excluding Oz)
        # This is different from TUAB which uses 19 channels (excluding Fz)
        if len(available_standard) != 20:
            logger.warning(f"Expected 20 channels for TUEV, found {len(available_standard)}")
            missing = [ch for ch in self.STANDARD_CHANNELS if ch not in available_standard]
            extra = [ch for ch in available_standard if ch not in self.STANDARD_CHANNELS]

            if missing:
                logger.error(f"Missing required channels: {missing}")
            if extra:
                logger.warning(f"Extra non-standard channels will be dropped: {extra}")
            # After synthesis, enforce exactly 20
            if len(available_standard) < 20:
                raise ValueError(
                    f"Too few channels ({len(available_standard)}). TUEV requires exactly 20. "
                    f"Missing: {missing}"
                )
            elif len(available_standard) > 20:
                # Drop any extra channels not in standard list
                available_standard = [
                    ch for ch in self.STANDARD_CHANNELS if ch in available_standard
                ]
                if len(available_standard) != 20:
                    raise ValueError(
                        f"Could not get exactly 20 standard channels. Got {len(available_standard)}"
                    )

        # Final sanity check - MUST be exactly 20 channels for TUEV
        assert len(available_standard) == 20, (
            f"TUEV must have exactly 20 channels, got {len(available_standard)}"
        )

        raw.pick(available_standard)
        logger.info(
            f"Selected {len(raw.ch_names)} standard channels (enforced to exactly 20 for TUEV)"
        )

        return raw

    def _apply_23_channel_mapping(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply 23-channel mapping for paper parity.

        Keep all 23 TUEV channels without synthesis or dropping.
        """
        # Check which channels are available after canonicalization
        available = [ch for ch in self.STANDARD_CHANNELS if ch in raw.ch_names]
        missing = [ch for ch in self.STANDARD_CHANNELS if ch not in raw.ch_names]

        if missing:
            logger.warning(f"Missing channels for 23-ch parity: {missing}")
            # For paper parity, we should have all 23 channels
            # If not, the dataset may be incomplete
            raise ValueError(f"Paper parity requires all 23 channels. Missing: {missing}")

        # Use raw.pick() to select channels
        raw.pick(available)

        # Ensure we have exactly 23 channels for mapper
        if len(raw.ch_names) != 23:
            raise ValueError(f"Paper parity requires exactly 23 channels, got {len(raw.ch_names)}")

        logger.info(f"Selected {len(raw.ch_names)} channels for paper parity")
        return raw

    def _apply_channel_mapping_with_tracking(self, raw: mne.io.Raw) -> tuple[mne.io.Raw, list[str]]:
        """Apply channel mapping and return missing channels list."""
        # Determine missing channels after canonicalization but before mapping
        standard_upper = {ch.upper() for ch in self.STANDARD_CHANNELS}
        raw_upper = {ch.upper() for ch in raw.ch_names if ch.upper() not in {'A1', 'A2'}}
        missing = sorted(standard_upper - raw_upper)

        # Apply channel mapping (drop refs, synthesize, reorder)
        raw = self._apply_channel_mapping(raw)

        return raw, missing

    def _synthesize_missing_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Synthesize missing canonical channels intelligently.

        Ensures final selection can reach exactly 20 channels by adding
        missing channels using interpolation when possible, zeros otherwise.
        """
        present_lower = {ch.lower() for ch in raw.ch_names}
        need = [ch for ch in self.STANDARD_CHANNELS if ch.lower() not in present_lower]

        if not need:
            return raw

        import numpy as np

        sfreq = raw.info['sfreq']
        n_times = len(raw.times)

        # Build mapping of channel names to indices for easy lookup
        ch_to_idx = {ch.lower(): i for i, ch in enumerate(raw.ch_names)}

        for ch in need:
            info = mne.create_info([ch], sfreq, ['eeg'])

            # Special case: Interpolate Fpz from Fp1 and Fp2 if available
            if ch.lower() == 'fpz' and 'fp1' in ch_to_idx and 'fp2' in ch_to_idx:
                # Use public API to get channel data
                fp1_data = raw.get_data(picks=['Fp1'])[0]
                fp2_data = raw.get_data(picks=['Fp2'])[0]
                # Average of Fp1 and Fp2 (biologically sensible for midline)
                fpz_data = (fp1_data + fp2_data) / 2.0
                fpz_raw = mne.io.RawArray(fpz_data[np.newaxis, :], info, verbose=False)
                raw.add_channels([fpz_raw], force_update_info=True)
                logger.info(f"Synthesized {ch} by interpolating from Fp1 and Fp2")
            else:
                # Default: synthesize as zeros
                zero_data = np.zeros((1, n_times))
                zero_raw = mne.io.RawArray(zero_data, info, verbose=False)
                raw.add_channels([zero_raw], force_update_info=True)
                logger.info(f"Synthesized missing channel as zeros: {ch}")

        return raw

    def _apply_mne_preprocessing(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply MNE preprocessing with optional RANSAC disable.

        Args:
            raw: Raw MNE object

        Returns:
            Preprocessed raw object
        """
        if self.disable_ransac:
            # Skip RANSAC, just do filtering and resampling
            logger.info("TUEV _apply_mne_preprocessing: RANSAC disabled - applying filters only")
            raw.filter(self.bandpass_low, self.bandpass_high, picks='eeg', verbose=False)
            if self.notch_freq:
                # Apply notch at fundamental and harmonics
                raw.notch_filter([self.notch_freq, self.notch_freq * 2], picks='eeg', verbose=False)
            return raw
        else:
            # Use parent's full preprocessing including RANSAC
            return super()._apply_mne_preprocessing(raw)

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
        logger.info(f"Processing {mask_path_for_log(edf_path)} with {len(annotations)} annotations")

        # Load and preprocess raw data
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        logger.info(f"Loaded {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

        # Canonicalize channel types (EDF loses types, everything becomes 'eeg')
        raw = canonicalize_channel_types(raw)

        # Canonicalize channel labels (strip prefixes, map legacy names, fix case)
        raw = canonicalize_channel_labels(raw)

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

        for _i, (win_start, _win_end) in enumerate(windows):
            start_sample = int(win_start * raw.info['sfreq'])
            events.append([start_sample, 0, window_event_code])  # All windows get code 1

        events_array = np.array(events, dtype=int)  # MNE requires int dtype for events
        event_id = {"window": window_event_code}  # Simple event_id dict

        # Create epochs
        # CRITICAL FIX: tmax must be exclusive to get exactly 1024 samples
        # Use actual sfreq from data (post-resample) for precision
        sfreq = float(raw.info["sfreq"])
        tmax = self.window_duration - (1.0 / sfreq)
        # At 256 Hz: tmax = 4.0 - (1/256) = 3.99609375
        # This gives samples from t=0 to t=3.99609375, inclusive of both endpoints
        # Sample indices: 0, 1, 2, ..., 1023 (exactly 1024 samples)
        epochs = mne.Epochs(
            raw,
            events_array,
            event_id=event_id,
            tmin=0,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )

        # Assert epoch length immediately (fail fast with context)
        n_required = round(self.window_duration * sfreq)  # 1024
        actual_samples = epochs.get_data().shape[-1]
        if actual_samples != n_required:
            raise RuntimeError(
                f"TUEV epoch length={actual_samples} != {n_required} "
                f"(sfreq={sfreq}, tmax={tmax}) for file={raw.filenames[0]}"
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
                self._warned_files: set[str] = set()

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

        windows: list[tuple[float, float]] = []
        start: float = 0.0
        while start + self.window_duration <= duration:
            windows.append((start, start + self.window_duration))
            start += step

        logger.info(f"Created {len(windows)} fixed-grid windows with {overlap * 100:.0f}% overlap")
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
        overlaps: dict[str, float] = {}

        for ann in annotations:
            # Calculate overlap in seconds
            overlap_start = max(win_start, ann['start'])
            overlap_end = min(win_end, ann['end'])
            overlap_duration = max(0.0, float(overlap_end) - float(overlap_start))

            if overlap_duration > 0:
                label = str(ann['label'])  # Ensure label is always a string
                if label not in overlaps:
                    overlaps[label] = 0.0
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

    def _apply_autoreject_tuev(self, epochs: mne.Epochs) -> tuple[mne.Epochs, dict[str, Any]]:
        """Apply Autoreject with gentle parameters for TUEV spike preservation.

        Args:
            epochs: Input epochs

        Returns:
            Tuple of (clean epochs, learned parameters dict)
        """
        import warnings

        from autoreject import AutoReject

        try:
            # Suppress expected NumPy warnings from empty CV folds
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', category=RuntimeWarning, message='.*Mean of empty slice.*'
                )
                warnings.filterwarnings(
                    'ignore', category=RuntimeWarning, message='.*invalid value encountered.*'
                )

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

        except Exception as e:
            logger.warning(f"Autoreject failed: {e}. Proceeding without artifact rejection.")
            # Return original epochs if AR fails
            return epochs, {}
