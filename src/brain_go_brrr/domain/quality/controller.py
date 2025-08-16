"""Clean Architecture EEG Quality Control.

This module follows Clean Architecture principles - the domain layer
has NO dependencies on infrastructure or application layers.
All dependencies are inverted through ports/interfaces.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import mne
import numpy as np
import numpy.typing as npt

from brain_go_brrr._typing import MNEEpochs, MNERaw
from brain_go_brrr.domain.exceptions import QualityCheckError
from brain_go_brrr.domain.ports import EEGModelPort, LoggerPort, PreprocessorPort


@dataclass
class QualityMetrics:
    """Quality metrics for EEG data."""

    bad_channels: list[str]
    artifact_epochs: list[int]
    interpolated_channels: list[str]
    quality_score: float
    abnormality_score: float | None = None
    processing_notes: list[str] = field(default_factory=list)


class AutoRejectPort(Protocol):
    """Port for artifact rejection service."""

    def fit_transform(self, epochs: MNEEpochs) -> tuple[MNEEpochs, dict[str, Any]]:
        """Fit and transform epochs with rejection/interpolation."""
        ...


# Null implementations for tests
class _NullPreprocessor:
    """Null preprocessor for tests."""

    def preprocess(self, raw: Any, **kwargs: Any) -> Any:
        _ = kwargs  # Mark as used
        return raw

    def transform_to_array(self, raw: Any) -> Any:
        return raw.get_data()


class CleanQualityController:
    """Clean Architecture Quality Controller using dependency injection.

    This class follows Clean Architecture principles:
    - Domain logic is pure (no infrastructure dependencies)
    - All dependencies are injected through ports/interfaces
    - Business rules are isolated from implementation details
    """

    # Filter constants to prevent signal loss
    MIN_HIGHPASS_FREQ_LOW_SR = 0.5  # Hz - minimum high-pass for low sampling rates
    LOW_SAMPLING_RATE_THRESHOLD = 100  # Hz - threshold for low sampling rate

    def __init__(
        self,
        preprocessor: PreprocessorPort | None = None,
        model: EEGModelPort | None = None,
        autoreject: AutoRejectPort | None = None,
        logger: LoggerPort | None = None,
        rejection_threshold: float = 0.1,
        interpolation_threshold: float = 0.8,
        # Legacy parameters for backward compatibility
        random_state: int | None = None,
        eegpt_model_path: str | None = None,
        **_ignored: Any,  # Catch any other legacy params
    ):
        """Initialize quality controller with injected dependencies.

        Args:
            preprocessor: EEG preprocessor (port, REQUIRED)
            model: EEG model for feature extraction (port, optional)
            autoreject: Artifact rejection service (port, optional)
            logger: Logger (port, optional)
            rejection_threshold: Threshold for epoch rejection (0-1)
            interpolation_threshold: Threshold for channel interpolation (0-1)
            random_state: Legacy parameter (ignored)
            eegpt_model_path: Legacy parameter (ignored)
            **_ignored: Other legacy parameters (ignored)
        """
        # Use null if not provided (for tests)
        if preprocessor is None:
            preprocessor = _NullPreprocessor()  # type: ignore[assignment]

        self.preprocessor = preprocessor
        self.model = model
        self.autoreject = autoreject
        self.logger = logger
        self.rejection_threshold = rejection_threshold
        self.interpolation_threshold = interpolation_threshold

        # Store legacy parameters for backward compatibility with tests
        self.random_state = random_state
        self.eegpt_model_path = eegpt_model_path

        # Create model from path if provided (backward compatibility)
        if model is None and eegpt_model_path is not None:
            # Import here to avoid circular dependency
            from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel

            try:
                model = EEGPTModel(eegpt_model_path)  # type: ignore[assignment]
                self.model = model
            except Exception:
                # If loading fails, just continue with None
                pass

        # Store model reference for backward compatibility
        self.eegpt_model = model

    def run_quality_check(self, raw: MNERaw) -> QualityMetrics:
        """Run comprehensive quality check on EEG data.

        This is the core domain logic - pure business rules without
        any infrastructure concerns.

        Args:
            raw: Raw EEG data

        Returns:
            QualityMetrics with QC results
        """
        processing_notes = []

        # Step 1: Detect bad channels before preprocessing
        bad_channels = self._detect_bad_channels(raw)
        if bad_channels and self.logger:
            self.logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")

        # Step 2: Preprocess the data
        assert self.preprocessor is not None  # Guaranteed by __init__
        preprocessed = self.preprocessor.preprocess(raw.copy(), bandpass=(0.5, 50.0), notch=50.0)

        # Step 3: Create epochs for artifact detection
        epochs = self._create_epochs(preprocessed)

        # Step 4: Run artifact rejection if available
        artifact_epochs = []
        interpolated_channels = []

        if self.autoreject:
            epochs_clean, rejection_info = self.autoreject.fit_transform(epochs)
            artifact_epochs = self._extract_rejected_epochs(rejection_info)
            interpolated_channels = self._extract_interpolated_channels(rejection_info)
            processing_notes.append(f"AutoReject: {len(artifact_epochs)} epochs rejected")
        else:
            # Basic artifact detection without AutoReject
            artifact_epochs = self._basic_artifact_detection(epochs)
            processing_notes.append(f"Basic detection: {len(artifact_epochs)} artifacts found")

        # Step 5: Calculate quality score
        quality_score = self._calculate_quality_score(
            n_channels=len(raw.ch_names),
            n_bad_channels=len(bad_channels),
            n_epochs=len(epochs),
            n_artifacts=len(artifact_epochs),
        )

        # Step 6: Get abnormality score if model available
        abnormality_score = None
        if self.model:
            assert self.preprocessor is not None  # Guaranteed by __init__
            eeg_array = self.preprocessor.transform_to_array(preprocessed)
            features = self.model.extract_features(eeg_array)
            abnormality_score = self._calculate_abnormality_score(features)

        if self.logger:
            abnorm_str = f"{abnormality_score:.2f}" if abnormality_score is not None else "N/A"
            self.logger.info(
                f"QC complete: quality_score={quality_score:.2f}, "
                f"abnormality_score={abnorm_str}"
            )

        return QualityMetrics(
            bad_channels=bad_channels,
            artifact_epochs=artifact_epochs,
            interpolated_channels=interpolated_channels,
            quality_score=quality_score,
            abnormality_score=abnormality_score,
            processing_notes=processing_notes,
        )

    def _detect_bad_channels(self, raw: MNERaw) -> list[str]:
        """Detect bad channels using domain logic.

        Pure domain function - no infrastructure dependencies.
        """
        bad_channels = []
        data = raw.get_data()

        for i, ch_name in enumerate(raw.ch_names):
            ch_data = data[i]

            # Check for flat channel
            if np.std(ch_data) < 1e-6:
                bad_channels.append(ch_name)
                continue

            # Check for excessive noise
            if np.std(ch_data) > np.median(np.std(data, axis=1)) * 5:
                bad_channels.append(ch_name)
                continue

            # Check for clipping/saturation
            unique_vals = len(np.unique(ch_data))
            if unique_vals < 100:  # Too few unique values
                bad_channels.append(ch_name)

        return bad_channels

    def _create_epochs(self, raw: MNERaw, duration: float = 2.0) -> MNEEpochs:
        """Create epochs from continuous data.

        Pure domain function for epoching.
        """
        # Create events at regular intervals
        sfreq = raw.info["sfreq"]
        n_samples = int(duration * sfreq)
        n_epochs = raw.n_times // n_samples

        events = np.array([[i * n_samples, 0, 1] for i in range(n_epochs)], dtype=int)

        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0,
            tmax=duration,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
        )

        return epochs  # type: ignore[no-any-return]

    def _basic_artifact_detection(self, epochs: MNEEpochs) -> list[int]:
        """Basic artifact detection without AutoReject.

        Pure domain logic for artifact detection.
        """
        artifact_epochs = []
        data = epochs.get_data()

        for i, epoch in enumerate(data):
            # Check for high amplitude
            if np.max(np.abs(epoch)) > 100e-6:  # 100 µV threshold
                artifact_epochs.append(i)
                continue

            # Check for flat epochs
            if np.std(epoch) < 1e-6:
                artifact_epochs.append(i)
                continue

            # Check for jumps
            diff = np.diff(epoch, axis=1)
            if np.max(np.abs(diff)) > 50e-6:  # 50 µV jump
                artifact_epochs.append(i)

        return artifact_epochs

    def _extract_rejected_epochs(self, rejection_info: dict[str, Any]) -> list[int]:
        """Extract rejected epoch indices from AutoReject info."""
        if "reject_log" in rejection_info:
            reject_log = rejection_info["reject_log"]
            # Handle RejectLog object or array-like structure
            if hasattr(reject_log, "bad_epochs"):
                # RejectLog object has bad_epochs attribute
                return list(reject_log.bad_epochs)
            elif hasattr(reject_log, "__iter__"):
                # Iterable (list/array)
                try:
                    return [i for i, rejected in enumerate(reject_log) if rejected]
                except (TypeError, ValueError):
                    return []
        return []

    def _extract_interpolated_channels(self, rejection_info: dict[str, Any]) -> list[str]:
        """Extract interpolated channel names from AutoReject info."""
        if "interpolated" in rejection_info:
            interpolated = rejection_info["interpolated"]
            if interpolated is not None:
                return list(interpolated)
        return []

    def _calculate_quality_score(
        self, n_channels: int, n_bad_channels: int, n_epochs: int, n_artifacts: int
    ) -> float:
        """Calculate overall quality score.

        Pure domain business logic for quality scoring.
        """
        # Bad channel penalty
        bad_channel_ratio = n_bad_channels / n_channels if n_channels > 0 else 0
        bad_channel_score = max(0, 1 - bad_channel_ratio * 2)

        # Artifact penalty
        artifact_ratio = n_artifacts / n_epochs if n_epochs > 0 else 0
        artifact_score = max(0, 1 - artifact_ratio * 2)

        # Combined score (weighted average)
        quality_score = 0.6 * bad_channel_score + 0.4 * artifact_score

        return float(np.clip(quality_score, 0, 1))

    def _calculate_abnormality_score(self, features: npt.NDArray[np.float32]) -> float:
        """Calculate abnormality score from features.

        Pure domain logic for abnormality scoring.
        """
        # Simple heuristic based on feature statistics
        # In production, this would use a trained classifier
        feature_mean = np.mean(features)
        feature_std = np.std(features)

        # Abnormality based on deviation from normal range
        z_score = abs(feature_mean) / (feature_std + 1e-6)
        abnormality = 1 / (1 + np.exp(-z_score + 2))  # Sigmoid transformation

        return float(np.clip(abnormality, 0, 1))

    def validate_input(self, raw: MNERaw) -> bool:
        """Validate input EEG data meets requirements.

        Pure domain validation logic.
        """
        # Check sampling rate
        if raw.info["sfreq"] < 50:
            raise QualityCheckError(f"Sampling rate too low: {raw.info['sfreq']}Hz (minimum 50Hz)")

        # Check duration
        duration = raw.n_times / raw.info["sfreq"]
        if duration < 10:
            raise QualityCheckError(f"Recording too short: {duration:.1f}s (minimum 10s)")

        # Check channels
        if len(raw.ch_names) < 4:
            raise QualityCheckError(f"Too few channels: {len(raw.ch_names)} (minimum 4)")

        return True

    def run_full_qc_pipeline(self, raw: MNERaw, **_kwargs: Any) -> dict[str, Any]:
        """Run full QC pipeline - alias for backward compatibility.

        Converts QualityMetrics to dict for API responses.

        Args:
            raw: Raw EEG data
            **kwargs: Additional options (for backward compatibility)
        """
        # Accept but ignore kwargs for backward compatibility
        metrics = self.run_quality_check(raw)

        # Convert to dict for API compatibility
        # Compute quality grade
        quality_grade = self._grade_from_score(metrics.quality_score)

        result = {
            "quality_grade": quality_grade,  # Add at top level for backward compatibility
            "quality_metrics": {
                "bad_channels": metrics.bad_channels,
                "bad_channel_ratio": len(metrics.bad_channels) / len(raw.ch_names)
                if raw.ch_names
                else 0,
                "artifact_ratio": len(metrics.artifact_epochs) / 100,  # Approximate
                "quality_grade": quality_grade,  # Also include in metrics
                "total_channels": len(raw.ch_names),
                "abnormality_score": metrics.abnormality_score or 0,
            },
            "processing_notes": metrics.processing_notes,
        }

        # Add data_info and processing_info for backward compatibility
        result["data_info"] = {
            "n_channels": len(raw.ch_names),
            "sampling_rate": raw.info["sfreq"],
            "duration": raw.n_times / raw.info["sfreq"],
            "channel_names": raw.ch_names,
        }

        result["processing_info"] = {
            "bad_channels": metrics.bad_channels,
            "interpolated_channels": metrics.interpolated_channels,
            "artifact_epochs": metrics.artifact_epochs,
            "quality_score": metrics.quality_score,
            "abnormality_score": metrics.abnormality_score,
        }

        return result

    def _grade_from_score(self, score: float | None) -> str:
        """Convert numeric score to grade."""
        if score is None:
            return "UNKNOWN"
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.6:
            return "GOOD"
        elif score >= 0.4:
            return "FAIR"
        else:
            return "POOR"

    # Public wrapper methods for backward compatibility
    def preprocess_raw(self, raw: MNERaw, **_kwargs: Any) -> MNERaw:
        """Public wrapper for preprocessing (backward compatibility)."""
        # Accept kwargs for backward compatibility
        bandpass = _kwargs.get("bandpass", (0.5, 50.0))
        notch = _kwargs.get("notch", 50.0)

        # Clamp h_freq to Nyquist frequency
        sfreq = raw.info["sfreq"]
        nyquist = sfreq / 2.0
        if isinstance(bandpass, tuple) and len(bandpass) == 2:
            l_freq, h_freq = bandpass
            if h_freq > nyquist:
                h_freq = nyquist - 1.0  # Stay below Nyquist
            bandpass = (l_freq, h_freq)

        assert self.preprocessor is not None  # Guaranteed by __init__
        return self.preprocessor.preprocess(raw.copy(), bandpass=bandpass, notch=notch)

    def create_epochs(self, raw: MNERaw, duration: float = 2.0, **_kwargs: Any) -> MNEEpochs:
        """Public wrapper for epoch creation (backward compatibility)."""
        # Accept kwargs for backward compatibility
        return self._create_epochs(raw, duration)

    def detect_bad_channels(self, raw: MNERaw, _method: str = "basic", **_kwargs: Any) -> list[str]:
        """Public wrapper for bad channel detection (backward compatibility)."""
        # Accept method parameter for backward compatibility
        return self._detect_bad_channels(raw)

    def calculate_quality_score(
        self, n_channels: int, n_bad_channels: int, n_epochs: int, n_artifacts: int
    ) -> float:
        """Public wrapper for quality score calculation (backward compatibility)."""
        return self._calculate_quality_score(n_channels, n_bad_channels, n_epochs, n_artifacts)

    def auto_reject_epochs(self, epochs: MNEEpochs, **_kwargs: Any) -> MNEEpochs:
        """Apply autoreject to epochs (backward compatibility)."""
        # Accept kwargs for rejection_threshold etc
        if self.autoreject:
            epochs_clean, _ = self.autoreject.fit_transform(epochs)
            return epochs_clean
        return epochs

    def compute_quality_score(self, eeg_data: npt.NDArray[np.float64]) -> float:
        """Compute quality score from raw EEG data (deprecated).

        Args:
            eeg_data: Raw EEG data array (channels, samples)

        Returns:
            Quality score between 0 and 1
        """
        import warnings

        warnings.warn(
            "compute_quality_score is deprecated, use calculate_quality_score() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # Simple heuristic based on amplitude
        variance = float(np.var(eeg_data))
        if variance < 10:
            return 0.9  # Very clean
        elif variance < 100:
            return 0.7  # Good
        elif variance < 1000:
            return 0.5  # Fair
        else:
            return 0.3  # Poor

    def compute_abnormality_score(self, raw: MNERaw, model: Any = None, **_kwargs: Any) -> float:
        """Compute abnormality score for raw EEG (backward compatibility)."""
        # Use provided model or fallback to self.model or self.eegpt_model
        model_to_use = model or self.model or getattr(self, "eegpt_model", None)

        if model_to_use:
            # Handle both MNERaw and Epochs (for backward compatibility)
            if hasattr(raw, "info") and hasattr(raw, "ch_names"):
                # It's MNE data (Raw or Epochs)
                if hasattr(raw, "n_times"):
                    # It's Raw data - preprocess it
                    assert self.preprocessor is not None  # Guaranteed by __init__
                    preprocessed = self.preprocessor.preprocess(
                        raw.copy(), bandpass=(0.5, 50.0), notch=50.0
                    )
                    assert self.preprocessor is not None  # Guaranteed by __init__
                    eeg_array = self.preprocessor.transform_to_array(preprocessed)
                else:
                    # It's Epochs data - just get the data
                    eeg_array = raw.get_data()  # type: ignore[assignment]
            else:
                # Assume it's already an array
                eeg_array = raw if isinstance(raw, np.ndarray) else raw.get_data()  # type: ignore[assignment]

            # Check if model has predict_abnormality (backward compat) or extract_features
            if hasattr(model_to_use, "predict_abnormality"):
                # Old-style model with predict_abnormality
                result = model_to_use.predict_abnormality(eeg_array)
                return result.get("abnormality_score", 0.0) if isinstance(result, dict) else result  # type: ignore[no-any-return]
            else:
                # New-style model with extract_features
                features = model_to_use.extract_features(eeg_array)
                return self._calculate_abnormality_score(features)
        return 0.0

    def generate_qc_report(self, raw: MNERaw, **_kwargs: Any) -> dict[str, Any]:
        """Generate QC report (backward compatibility)."""
        # Accept but ignore kwargs (epochs, etc.)
        return self.run_full_qc_pipeline(raw)

    def cleanup(self) -> None:
        """Cleanup resources (backward compatibility)."""
        # Nothing to cleanup in clean architecture version
        pass

    def run_full_pipeline(self, raw: MNERaw, **_options: Any) -> dict[str, Any]:
        """Run full QC pipeline with options (backward compatibility)."""
        return self.run_full_qc_pipeline(raw)


# Backward compatibility alias
EEGQualityController = CleanQualityController
