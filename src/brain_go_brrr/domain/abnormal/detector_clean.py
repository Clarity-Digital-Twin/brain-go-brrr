"""Clean Architecture EEG Abnormality Detection.

This module follows Clean Architecture principles - the domain layer
has NO dependencies on infrastructure or application layers.
All dependencies are inverted through ports/interfaces.
"""

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
import torch

from brain_go_brrr._typing import MNERaw
from brain_go_brrr.domain.ports import (
    AbnormalityConfigPort,
    EEGModelPort,
    LoggerPort,
    PreprocessorPort,
)


class TriageLevel(str, Enum):
    """Clinical triage levels for EEG prioritization."""

    NORMAL = "NORMAL"  # Low priority
    ROUTINE = "ROUTINE"  # Standard workflow (< 48 hours)
    EXPEDITE = "EXPEDITE"  # Priority review (< 4 hours)
    URGENT = "URGENT"  # Immediate review needed


@dataclass
class AbnormalityResult:
    """Result of abnormality detection analysis."""

    is_abnormal: bool
    confidence: float
    triage_level: TriageLevel
    processing_time_ms: float
    features_shape: tuple[int, ...]
    metadata: dict


class CleanAbnormalityDetector:
    """Clean Architecture Abnormality Detector using dependency injection.

    This class follows Clean Architecture principles:
    - Domain logic is pure (no infrastructure dependencies)
    - All dependencies are injected through ports/interfaces
    - Business rules are isolated from implementation details
    """

    def __init__(
        self,
        model: EEGModelPort,
        preprocessor: PreprocessorPort,
        config: AbnormalityConfigPort,
        logger: LoggerPort | None = None,
        linear_probe: torch.nn.Module | None = None,
    ):
        """Initialize detector with injected dependencies.

        Args:
            model: EEG model for feature extraction (port)
            preprocessor: EEG preprocessor (port)
            config: Configuration (port)
            logger: Logger (port, optional)
            linear_probe: Linear probe head for classification (optional)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.logger = logger
        self.linear_probe = linear_probe
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize linear probe if not provided
        if self.linear_probe is None:
            self._initialize_linear_probe()

    def _initialize_linear_probe(self) -> None:
        """Initialize linear probe head for binary classification."""
        feature_dim = self.model.get_feature_dim()
        self.linear_probe = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 2),  # Binary classification
        ).to(self.device)

        if self.logger:
            self.logger.info(f"Initialized linear probe with {feature_dim} input features")

    def detect(self, raw: MNERaw) -> AbnormalityResult:
        """Detect abnormalities in EEG recording.

        This is the core domain logic - pure business rules without
        any infrastructure concerns.

        Args:
            raw: Raw EEG data

        Returns:
            AbnormalityResult with detection outcome
        """
        start_time = time.perf_counter()

        # Step 1: Preprocess the EEG data
        preprocessed = self.preprocessor.preprocess(
            raw,
            bandpass=(self.config.bandpass_low, self.config.bandpass_high),
            notch=50.0,  # Standard power line frequency
        )

        # Step 2: Convert to array for model input
        eeg_array = self.preprocessor.transform_to_array(preprocessed)

        # Step 3: Extract features using the model
        features = self.model.extract_features(
            eeg_array,
            sampling_rate=int(preprocessed.info["sfreq"]),
        )

        # Step 4: Run inference with linear probe
        confidence = self._run_inference(features)

        # Step 5: Apply business rules for classification
        is_abnormal = confidence > self.config.confidence_threshold
        triage_level = self._determine_triage_level(confidence, is_abnormal)

        # Step 6: Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Step 7: Log if logger is available
        if self.logger:
            self.logger.info(
                f"Detection complete: abnormal={is_abnormal}, "
                f"confidence={confidence:.3f}, triage={triage_level.value}, "
                f"time={processing_time_ms:.1f}ms"
            )

        return AbnormalityResult(
            is_abnormal=is_abnormal,
            confidence=float(confidence),
            triage_level=triage_level,
            processing_time_ms=processing_time_ms,
            features_shape=features.shape,
            metadata={
                "n_channels": eeg_array.shape[0],
                "n_samples": eeg_array.shape[1],
                "sampling_rate": preprocessed.info["sfreq"],
                "threshold": self.config.confidence_threshold,
            },
        )

    def _run_inference(self, features: npt.NDArray[np.float32]) -> float:
        """Run inference using the linear probe.

        Args:
            features: Extracted features from EEG model

        Returns:
            Confidence score for abnormality (0-1)
        """
        with torch.inference_mode():
            # Convert to tensor and move to device
            features_tensor = torch.from_numpy(features).float().to(self.device)

            # Add batch dimension if needed
            if features_tensor.dim() == 1:
                features_tensor = features_tensor.unsqueeze(0)

            # Run through linear probe
            logits = self.linear_probe(features_tensor)

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Return abnormality probability (class 1)
            return probs[0, 1].cpu().item()

    def _determine_triage_level(self, confidence: float, is_abnormal: bool) -> TriageLevel:
        """Determine clinical triage level based on confidence and abnormality.

        This encapsulates the business rules for clinical prioritization.

        Args:
            confidence: Confidence score (0-1)
            is_abnormal: Whether EEG is classified as abnormal

        Returns:
            Appropriate triage level
        """
        if not is_abnormal:
            return TriageLevel.NORMAL

        # Business rules for triage based on confidence
        if confidence >= 0.95:
            return TriageLevel.URGENT  # Very high confidence abnormality
        elif confidence >= 0.85:
            return TriageLevel.EXPEDITE  # High confidence
        elif confidence >= 0.70:
            return TriageLevel.ROUTINE  # Moderate confidence
        else:
            return TriageLevel.NORMAL  # Low confidence, needs review

    def validate_input(self, raw: MNERaw) -> bool:
        """Validate input EEG data meets requirements.

        Pure domain validation logic.

        Args:
            raw: Raw EEG data

        Returns:
            True if valid, raises exception otherwise
        """
        # Check sampling rate
        if raw.info["sfreq"] < 100:
            raise ValueError(f"Sampling rate too low: {raw.info['sfreq']}Hz (minimum 100Hz)")

        # Check duration
        duration = raw.n_times / raw.info["sfreq"]
        if duration < 10:
            raise ValueError(f"Recording too short: {duration:.1f}s (minimum 10s)")

        # Check channels
        available_channels = set(raw.ch_names)
        required_channels = set(self.config.channels)
        missing = required_channels - available_channels

        if missing:
            raise ValueError(f"Missing required channels: {missing}")

        return True
