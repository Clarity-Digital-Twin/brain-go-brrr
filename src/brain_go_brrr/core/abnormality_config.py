"""Configuration for EEG abnormality detection.

This module defines all configuration parameters and thresholds
used in the abnormality detection pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class QualityConfig:
    """Configuration for EEG quality assessment."""
    
    # Channel quality thresholds
    flat_channel_std_threshold: float = 1e-10  # Channels with std below this are considered flat
    flat_channel_nanovolt_threshold: float = 1e-9  # Less than 1 nanovolt
    saturated_amplitude_millivolt: float = 1e-3  # Saturation threshold (1 millivolt)
    saturated_sample_ratio: float = 0.1  # If >10% samples exceed saturation
    
    # Window quality thresholds (for normalized data)
    artifact_amplitude_threshold: float = 5.0  # Z-score threshold for artifacts
    excessive_noise_threshold: float = 10.0  # Z-score for excessive noise
    
    # Quality score penalties
    flat_channel_penalty: float = 0.5
    artifact_channel_penalty: float = 0.3
    excessive_noise_penalty: float = 0.5
    
    # Recording quality grades
    excellent_avg_quality: float = 0.8
    excellent_bad_channel_ratio: float = 0.1
    good_avg_quality: float = 0.6
    good_bad_channel_ratio: float = 0.2
    fair_avg_quality: float = 0.4
    fair_bad_channel_ratio: float = 0.3


@dataclass
class ProcessingConfig:
    """Configuration for EEG processing parameters."""
    
    # Duration constraints
    min_recording_duration_seconds: float = 60.0  # 1 minute minimum
    min_windows_for_prediction: int = 10
    
    # Channel constraints
    min_required_channels: int = 19
    max_bad_channel_ratio: float = 0.3  # Maximum 30% bad channels
    
    # Normalization parameters
    channel_std_epsilon: float = 1e-8  # Epsilon for numerical stability
    
    # Window extraction
    window_duration_seconds: float = 4.0
    window_overlap_ratio: float = 0.5
    target_sampling_rate: int = 256


@dataclass
class ClassificationConfig:
    """Configuration for classification and triage."""
    
    # Classification thresholds
    abnormal_threshold: float = 0.5
    
    # Triage thresholds
    urgent_score_threshold: float = 0.8
    expedite_score_threshold: float = 0.6
    routine_score_threshold: float = 0.4
    
    # Confidence calculation
    confidence_std_weight: float = 0.7
    confidence_extremity_weight: float = 0.3
    confidence_std_multiplier: float = 2.0
    
    # Aggregation parameters
    voting_quality_threshold: float = 0.5  # Minimum quality for voting


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Classifier architecture
    classifier_hidden_1: int = 256
    classifier_hidden_2: int = 128
    classifier_dropout: float = 0.3
    feature_dim: int = 512
    num_classes: int = 2
    
    # Model versioning
    default_model_version: str = "eegpt-v1.0"


@dataclass
class AbnormalityConfig:
    """Complete configuration for abnormality detection."""
    
    quality: QualityConfig = field(default_factory=QualityConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    @classmethod
    def from_spec(cls) -> "AbnormalityConfig":
        """Create configuration following clinical specifications."""
        return cls()  # Default values follow the spec