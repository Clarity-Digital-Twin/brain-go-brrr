"""Protocol types for Clean Architecture factories.

This module defines the Protocol types that factories return.
Following Uncle Bob's Clean Architecture:
- Factories return abstractions (Protocols), not concrete implementations
- This maintains loose coupling between layers
- Consumers depend on abstractions, not concretions

The application layer (factories) is the ONLY place that knows about
concrete implementations. Everything else depends on these abstractions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    # Import only for type checking to avoid runtime dependencies
    from brain_go_brrr._typing import MNERaw
    from brain_go_brrr.domain.abnormal.detector_pure import (
        AbnormalityResult,
    )
    from brain_go_brrr.domain.preprocessing.features.extractor_clean import (
        ExtractedFeatures,
    )
    from brain_go_brrr.domain.quality.controller_clean import (
        QualityMetrics,
    )


@runtime_checkable
class AbnormalityDetectorPort(Protocol):
    """Protocol for abnormality detection service.

    This is what consumers of the factory get - they don't know
    if it's PureAbnormalityDetector or any other implementation.
    """

    def detect(self, raw: MNERaw) -> AbnormalityResult:
        """Detect abnormalities in EEG data.

        Args:
            raw: Raw EEG data

        Returns:
            Detection result with triage level
        """
        ...


@runtime_checkable
class QualityControllerPort(Protocol):
    """Protocol for quality control service.

    Consumers depend on this abstraction, not the concrete
    CleanQualityController implementation.
    """

    def run_quality_check(self, raw: MNERaw) -> QualityMetrics:
        """Run comprehensive quality check on EEG data.

        Args:
            raw: Raw EEG data

        Returns:
            Quality metrics with QC results
        """
        ...

    def validate_input(self, raw: MNERaw) -> bool:
        """Validate input EEG data meets requirements.

        Args:
            raw: Raw EEG data

        Returns:
            True if valid

        Raises:
            QualityCheckError: If validation fails
        """
        ...


@runtime_checkable
class FeatureExtractorPort(Protocol):
    """Protocol for feature extraction service.

    Abstracts away the concrete feature extraction implementation.
    """

    def extract_features(self, raw: MNERaw) -> ExtractedFeatures:
        """Extract features from EEG recording.

        Args:
            raw: Raw EEG data

        Returns:
            Extracted features with embeddings and metadata
        """
        ...

    def validate_input(self, raw: MNERaw) -> bool:
        """Validate input EEG data meets requirements.

        Args:
            raw: Raw EEG data

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        ...


@runtime_checkable
class EEGProcessorPort(Protocol):
    """Protocol for general EEG processing.

    This is a composite protocol that can handle multiple
    types of EEG analysis.
    """

    def process(self, raw: "MNERaw", task: str = "detect") -> dict[str, Any]:
        """Process EEG data for specified task.

        Args:
            raw: Raw EEG data
            task: Processing task (detect/qc/features)

        Returns:
            Processing results as dictionary
        """
        ...


@runtime_checkable
class ModelPort(Protocol):
    """Protocol for ML model operations.

    Abstracts the underlying model implementation
    (PyTorch, TensorFlow, etc).
    """

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:  # noqa: N803
        """Run model prediction.

        Args:
            X: Input features

        Returns:
            Model predictions
        """
        ...

    def extract_embeddings(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:  # noqa: N803
        """Extract embeddings/features from model.

        Args:
            X: Input data

        Returns:
            Extracted embeddings
        """
        ...
