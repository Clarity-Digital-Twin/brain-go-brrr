"""Pure domain settings for abnormality detection.

These are DOMAIN settings - no infrastructure or application concerns.
Following Clean Architecture: domain entities and value objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AbnormalitySettings:
    """Pure domain settings for abnormality detection.

    This is a value object - immutable and contains only domain logic.
    """

    # Detection thresholds
    abnormal_threshold: float = 0.5
    confidence_threshold: float = 0.7
    min_confidence: float = 0.3

    # Clinical triage thresholds
    urgent_threshold: float = 0.95
    expedite_threshold: float = 0.85
    routine_threshold: float = 0.70

    # Processing parameters
    window_duration: float = 4.0
    window_overlap: float = 0.5
    min_windows: int = 3

    # Quality thresholds
    min_quality_score: float = 0.5
    artifact_threshold: float = 0.3

    def validate(self) -> None:
        """Validate settings consistency - pure domain logic."""
        if not 0 <= self.abnormal_threshold <= 1:
            raise ValueError(f"abnormal_threshold must be in [0,1], got {self.abnormal_threshold}")

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be in [0,1], got {self.confidence_threshold}"
            )

        if self.urgent_threshold <= self.expedite_threshold:
            raise ValueError("urgent_threshold must be > expedite_threshold")

        if self.expedite_threshold <= self.routine_threshold:
            raise ValueError("expedite_threshold must be > routine_threshold")

        if self.window_duration <= 0:
            raise ValueError(f"window_duration must be positive, got {self.window_duration}")

        if not 0 <= self.window_overlap < 1:
            raise ValueError(f"window_overlap must be in [0,1), got {self.window_overlap}")


@dataclass(frozen=True)
class QualitySettings:
    """Pure domain settings for quality control."""

    # Bad channel detection
    flat_channel_threshold: float = 1e-6
    noise_multiplier: float = 5.0
    min_unique_values: int = 100

    # Artifact detection
    high_amplitude_threshold: float = 100e-6  # 100 µV
    jump_threshold: float = 50e-6  # 50 µV

    # Quality scoring weights
    bad_channel_weight: float = 0.6
    artifact_weight: float = 0.4

    # Minimum requirements
    min_channels: int = 4
    min_duration_seconds: float = 10.0
    min_sampling_rate: float = 50.0


@dataclass(frozen=True)
class FeatureSettings:
    """Pure domain settings for feature extraction."""

    # Window parameters
    window_size: float = 4.0
    window_overlap: float = 0.5

    # Frequency bands (Hz)
    delta_band: tuple[float, float] = (0.5, 4.0)
    theta_band: tuple[float, float] = (4.0, 8.0)
    alpha_band: tuple[float, float] = (8.0, 13.0)
    beta_band: tuple[float, float] = (13.0, 30.0)
    gamma_band: tuple[float, float] = (30.0, 45.0)

    # Feature extraction parameters
    n_fft_bins: int = 256
    entropy_bins: int = 50

    # Preprocessing
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    notch_freq: float = 50.0
