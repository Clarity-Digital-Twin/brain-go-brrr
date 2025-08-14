"""PURE CLEAN ARCHITECTURE Abnormality Detector.

This is the PURE domain service - ZERO dependencies on outer layers.
Following Uncle Bob's Clean Architecture to the letter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from .ports import (  # noqa: TC001
    AbnormalityHeadPort,
    EEGPreprocessorPort,
    FeatureExtractorPort,
    LoggerPort,
    MneRaw,
)
from .settings import AbnormalitySettings  # noqa: TC001


class TriageLevel(str, Enum):
    """Clinical triage levels - pure domain enum."""

    NORMAL = "NORMAL"
    ROUTINE = "ROUTINE"
    EXPEDITE = "EXPEDITE"
    URGENT = "URGENT"


@dataclass(frozen=True)
class AbnormalityResult:
    """Abnormality detection result - pure domain value object."""

    probability: float
    confidence: float
    is_abnormal: bool
    triage_level: TriageLevel
    n_windows: int
    quality_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "probability": self.probability,
            "confidence": self.confidence,
            "is_abnormal": self.is_abnormal,
            "triage_level": self.triage_level.value,
            "n_windows": self.n_windows,
            "quality_score": self.quality_score,
        }


class PureAbnormalityDetector:
    """PURE domain service for abnormality detection.

    This class is 100% CLEAN:
    - Depends ONLY on abstractions (ports)
    - Contains ONLY business logic
    - NO infrastructure concerns
    - NO application concerns
    - PURE DOMAIN LOGIC
    """

    def __init__(
        self,
        preprocessor: EEGPreprocessorPort,
        feature_extractor: FeatureExtractorPort,
        classifier: AbnormalityHeadPort,
        settings: AbnormalitySettings,
        logger: LoggerPort | None = None,
    ) -> None:
        """Initialize with injected dependencies.

        ALL dependencies are abstractions (ports), not concretions.
        This is Dependency Inversion Principle in action.
        """
        self._preprocessor = preprocessor
        self._feature_extractor = feature_extractor
        self._classifier = classifier
        self._settings = settings
        self._logger = logger

        # Validate settings on initialization
        settings.validate()

    def detect(self, raw: MneRaw) -> AbnormalityResult:
        """Detect abnormalities in EEG - PURE BUSINESS LOGIC.

        This method contains ONLY domain logic:
        1. Validate input
        2. Preprocess
        3. Extract features
        4. Classify
        5. Apply business rules for triage

        Args:
            raw: Raw EEG data (protocol, not concrete type)

        Returns:
            Detection result with triage level
        """
        # Step 1: Validate input (domain rule)
        self._validate_input(raw)

        # Step 2: Preprocess using injected port
        if self._logger:
            self._logger.debug("Starting EEG preprocessing")

        preprocessed = self._preprocessor.transform(raw)

        # Step 3: Extract windows (domain logic)
        windows = self._extract_windows(preprocessed, raw.info["sfreq"])

        if self._logger:
            self._logger.info(f"Extracted {len(windows)} windows for analysis")

        # Step 4: Process each window
        window_scores = []
        for window in windows:
            # Extract features
            features = self._feature_extractor.extract(window)

            # Classify
            prob = self._classifier.predict_proba(features)
            window_scores.append(prob)

        # Step 5: Aggregate results (domain logic)
        final_prob = self._aggregate_probabilities(window_scores)
        confidence = self._calculate_confidence(window_scores)
        quality = self._calculate_quality(preprocessed)

        # Step 6: Apply business rules
        is_abnormal = final_prob >= self._settings.abnormal_threshold
        triage = self._determine_triage(final_prob, confidence, quality)

        if self._logger:
            self._logger.info(
                f"Detection complete: prob={final_prob:.3f}, "
                f"confidence={confidence:.3f}, triage={triage.value}"
            )

        return AbnormalityResult(
            probability=final_prob,
            confidence=confidence,
            is_abnormal=is_abnormal,
            triage_level=triage,
            n_windows=len(windows),
            quality_score=quality,
        )

    def _validate_input(self, raw: MneRaw) -> None:
        """Validate input - PURE DOMAIN RULES."""
        # Check duration
        duration = raw.n_times / raw.info["sfreq"]
        min_duration = self._settings.window_duration * self._settings.min_windows

        if duration < min_duration:
            raise ValueError(
                f"Recording too short: {duration:.1f}s "
                f"(need {min_duration:.1f}s for {self._settings.min_windows} windows)"
            )

        # Check channels
        n_channels = len(raw.ch_names)
        if n_channels < 4:  # Domain rule: minimum 4 channels
            raise ValueError(f"Too few channels: {n_channels} (minimum 4)")

    def _extract_windows(
        self, data: npt.NDArray[np.float32], sfreq: float
    ) -> list[npt.NDArray[np.float32]]:
        """Extract sliding windows - PURE DOMAIN LOGIC."""
        window_samples = int(self._settings.window_duration * sfreq)
        step_samples = int(window_samples * (1 - self._settings.window_overlap))

        windows = []
        start = 0

        while start + window_samples <= data.shape[1]:
            window = data[:, start : start + window_samples]
            windows.append(window)
            start += step_samples

        return windows

    def _aggregate_probabilities(self, scores: list[float]) -> float:
        """Aggregate window scores - PURE BUSINESS LOGIC."""
        if not scores:
            return 0.5  # Uncertain if no windows

        # Domain rule: use weighted average with quality weighting
        # Higher scores get more weight (attention mechanism)
        scores_array = np.array(scores)
        weights = np.exp(scores_array * 2)  # Exponential weighting
        weights = weights / weights.sum()

        return float(np.sum(scores_array * weights))

    def _calculate_confidence(self, scores: list[float]) -> float:
        """Calculate confidence - PURE DOMAIN LOGIC."""
        if len(scores) < 2:
            return 0.5  # Low confidence with few windows

        scores_array = np.array(scores)

        # Confidence based on:
        # 1. Consistency (low std = high confidence)
        std_score = 1.0 - min(np.std(scores_array) * 2, 1.0)

        # 2. Extremity (far from 0.5 = high confidence)
        mean_score = np.mean(scores_array)
        extremity_score = abs(mean_score - 0.5) * 2

        # 3. Number of windows (more = higher confidence)
        n_score = min(len(scores) / 10, 1.0)

        # Weighted combination
        confidence = 0.4 * std_score + 0.4 * extremity_score + 0.2 * n_score

        return float(np.clip(confidence, 0, 1))

    def _calculate_quality(self, data: npt.NDArray[np.float32]) -> float:
        """Calculate data quality - PURE DOMAIN LOGIC."""
        quality_factors = []

        # Factor 1: Signal variance (not flat)
        channel_stds = np.std(data, axis=1)
        non_flat_ratio = np.mean(channel_stds > 1e-6)
        quality_factors.append(non_flat_ratio)

        # Factor 2: No saturation
        max_vals = np.max(np.abs(data), axis=1)
        non_saturated_ratio = np.mean(max_vals < 1000)  # Assuming normalized
        quality_factors.append(non_saturated_ratio)

        # Factor 3: Reasonable range
        in_range_ratio = np.mean((data > -100) & (data < 100))
        quality_factors.append(in_range_ratio)

        # Combine factors
        quality = float(np.mean(quality_factors))
        return float(np.clip(quality, 0, 1))

    def _determine_triage(
        self, probability: float, confidence: float, quality: float
    ) -> TriageLevel:
        """Determine triage level - PURE BUSINESS RULES."""
        # Low quality always needs review
        if quality < self._settings.min_quality_score:
            return TriageLevel.URGENT

        # Low confidence needs review
        if confidence < self._settings.min_confidence:
            return TriageLevel.EXPEDITE

        # High confidence abnormality
        if probability >= self._settings.urgent_threshold:
            return TriageLevel.URGENT
        elif probability >= self._settings.expedite_threshold:
            return TriageLevel.EXPEDITE
        elif probability >= self._settings.routine_threshold:
            return TriageLevel.ROUTINE
        else:
            return TriageLevel.NORMAL
