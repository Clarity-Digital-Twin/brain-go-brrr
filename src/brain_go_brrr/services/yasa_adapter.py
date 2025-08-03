#!/usr/bin/env python
"""YASA Sleep Staging Adapter - Real Implementation.

Integrates YASA sleep staging into our hierarchical pipeline.
YASA includes pre-trained models and requires no additional weights.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import numpy.typing as npt
import yasa

logger = logging.getLogger(__name__)


@dataclass
class YASAConfig:
    """Configuration for YASA sleep staging."""
    # Model selection
    use_consensus: bool = True  # Use consensus of multiple models
    eeg_backend: str = "lightgbm"  # "lightgbm" or "perceptron"
    eog_backend: str = "lightgbm"
    emg_backend: str = "lightgbm"

    # Feature extraction
    freq_broad: tuple[float, float] = (0.5, 35.0)  # Broad frequency range

    # Confidence thresholds
    min_confidence: float = 0.5

    # Performance
    n_jobs: int = 1  # Number of parallel jobs


class YASASleepStager:
    """YASA-based sleep staging with our pipeline integration."""

    def __init__(self, config: YASAConfig | None = None):
        """Initialize YASA sleep stager."""
        self.config = config or YASAConfig()
        self._validate_installation()

        # Track performance metrics
        self.stages_processed = 0
        self.avg_confidence = 0.0

    def _validate_installation(self):
        """Validate YASA is properly installed with models."""
        try:
            # Check if lightgbm is available
            import lightgbm  # noqa: F401
            logger.info("LightGBM available for YASA")
        except ImportError:
            logger.warning("LightGBM not available, falling back to perceptron")
            self.config.eeg_backend = "perceptron"

    def stage_sleep(
        self,
        eeg_data: npt.NDArray[np.float64],
        sfreq: float = 256,
        ch_names: list[str] | None = None,
        epoch_duration: int = 30
    ) -> tuple[list[str], list[float], dict[str, Any]]:
        """Perform sleep staging on EEG data.

        Args:
            eeg_data: EEG data array (n_channels, n_times)
            sfreq: Sampling frequency
            ch_names: Channel names
            epoch_duration: Duration of each epoch in seconds

        Returns:
            Tuple of:
                - List of sleep stages per epoch
                - List of confidence scores per epoch
                - Dictionary with additional metrics
        """
        # Validate input
        n_channels, n_times = eeg_data.shape
        duration_sec = n_times / sfreq

        if duration_sec < epoch_duration:
            raise ValueError(
                f"Data too short: {duration_sec:.1f}s, "
                f"need at least {epoch_duration}s"
            )

        # Create MNE Raw object if needed
        if ch_names is None:
            ch_names = [f"EEG{i}" for i in range(n_channels)]

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(eeg_data, info)

        # Select frontal channels for sleep staging (if available)
        eeg_name = self._select_eeg_channel(ch_names)

        # Run YASA sleep staging
        sls = yasa.SleepStaging(
            raw,
            eeg_name=eeg_name,
            eog_name=None,  # We typically don't have EOG
            emg_name=None,  # We typically don't have EMG
            metadata=None
        )

        # Get predictions
        hypnogram = sls.predict()  # Returns array of stages
        proba = sls.predict_proba()  # Returns probability matrix

        # Convert to our format
        stages = [self._yasa_to_standard_stage(s) for s in hypnogram]

        # Calculate confidence (max probability for each epoch)
        confidences = np.max(proba, axis=1).tolist()

        # Calculate additional metrics
        metrics = self._calculate_sleep_metrics(stages, confidences)

        # Update tracking
        self.stages_processed += len(stages)
        self.avg_confidence = (
            (self.avg_confidence * (self.stages_processed - len(stages)) +
             sum(confidences)) / self.stages_processed
        )

        return stages, confidences, metrics

    def _select_eeg_channel(self, ch_names: list[str]) -> str | None:
        """Select best EEG channel for sleep staging.

        YASA prefers frontal channels (e.g., C3, C4, Cz).
        """
        # Preference order for sleep staging
        preferred = ['C3', 'C4', 'Cz', 'F3', 'F4', 'Fz']

        for ch in preferred:
            if ch in ch_names:
                logger.info(f"Using {ch} for sleep staging")
                return ch

        # Fallback to first channel
        logger.warning(
            f"No preferred channel found, using {ch_names[0]}"
        )
        return ch_names[0]

    def _yasa_to_standard_stage(self, yasa_stage: int) -> str:
        """Convert YASA numeric stage to standard string.

        YASA uses: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        We use: W, N1, N2, N3, REM
        """
        mapping = {
            0: 'W',
            1: 'N1',
            2: 'N2',
            3: 'N3',
            4: 'REM'
        }
        return mapping.get(yasa_stage, 'W')

    def _calculate_sleep_metrics(
        self,
        stages: list[str],
        confidences: list[float]
    ) -> dict[str, Any]:
        """Calculate sleep quality metrics from staging results."""
        n_epochs = len(stages)

        if n_epochs == 0:
            return {}

        # Count stages
        stage_counts = {
            'W': stages.count('W'),
            'N1': stages.count('N1'),
            'N2': stages.count('N2'),
            'N3': stages.count('N3'),
            'REM': stages.count('REM')
        }

        # Calculate percentages
        stage_percentages = {
            stage: (count / n_epochs) * 100
            for stage, count in stage_counts.items()
        }

        # Sleep efficiency
        sleep_epochs = n_epochs - stage_counts['W']
        sleep_efficiency = (sleep_epochs / n_epochs) * 100

        # Find sleep onset and offset
        sleep_onset = None
        sleep_offset = None

        for i, stage in enumerate(stages):
            if stage != 'W' and sleep_onset is None:
                sleep_onset = i
            if stage != 'W':
                sleep_offset = i

        # Calculate WASO (Wake After Sleep Onset)
        waso_epochs = 0
        if sleep_onset is not None and sleep_offset is not None:
            for i in range(sleep_onset, sleep_offset + 1):
                if stages[i] == 'W':
                    waso_epochs += 1

        # Mean confidence
        mean_confidence = np.mean(confidences)

        return {
            'stage_counts': stage_counts,
            'stage_percentages': stage_percentages,
            'sleep_efficiency': sleep_efficiency,
            'sleep_onset_epoch': sleep_onset,
            'sleep_offset_epoch': sleep_offset,
            'waso_epochs': waso_epochs,
            'mean_confidence': mean_confidence,
            'n_epochs': n_epochs
        }

    def process_full_night(
        self,
        eeg_path: Path,
        output_hypnogram: bool = True
    ) -> dict[str, Any]:
        """Process a full night recording.

        Args:
            eeg_path: Path to EEG file
            output_hypnogram: Whether to include full hypnogram

        Returns:
            Dictionary with sleep analysis results
        """
        # Load EEG data
        raw = mne.io.read_raw(str(eeg_path), preload=True)

        # Get data
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names

        # Run staging
        stages, confidences, metrics = self.stage_sleep(
            data, sfreq, ch_names
        )

        # Prepare results
        results = {
            'file': str(eeg_path),
            'duration_hours': len(stages) * 30 / 3600,  # 30s epochs
            'metrics': metrics,
            'quality_check': {
                'mean_confidence': metrics['mean_confidence'],
                'low_confidence_epochs': sum(1 for c in confidences if c < self.config.min_confidence),
                'confidence_warning': bool(metrics['mean_confidence'] < 0.7)
            }
        }

        if output_hypnogram:
            results['hypnogram'] = stages
            results['confidences'] = confidences

        return results


class HierarchicalPipelineYASAAdapter:
    """Adapter to integrate YASA with our hierarchical pipeline."""

    def __init__(self, yasa_config: YASAConfig | None = None):
        """Initialize the adapter."""
        self.stager = YASASleepStager(yasa_config)

    def stage(self, eeg: npt.NDArray[np.float64]) -> tuple[str, float]:
        """Simple interface matching our mock SleepStager.

        Args:
            eeg: EEG data (n_channels, n_times)

        Returns:
            Tuple of (stage, confidence) for the dominant stage
        """
        try:
            # Run full staging
            stages, confidences, metrics = self.stager.stage_sleep(eeg)

            # Return most common stage and mean confidence
            if stages:
                # Find most common stage
                from collections import Counter
                stage_counter = Counter(stages)
                dominant_stage = stage_counter.most_common(1)[0][0]

                # Use mean confidence
                confidence = np.mean(confidences)

                return dominant_stage, confidence
            else:
                # Fallback
                return 'W', 0.5

        except Exception as e:
            logger.error(f"YASA staging failed: {e}")
            # Return wake with low confidence
            return 'W', 0.0
