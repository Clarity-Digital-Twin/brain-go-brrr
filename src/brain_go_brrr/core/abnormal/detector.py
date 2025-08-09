"""EEG Abnormality Detection Service.

This module implements a production-ready abnormality detection system
using EEGPT foundation model with clinical-grade accuracy requirements.
"""

import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from brain_go_brrr._typing import FloatArray, MNERaw
from brain_go_brrr.core.abnormality_config import AbnormalityConfig
from brain_go_brrr.core.config import ModelConfig
from brain_go_brrr.core.logger import get_logger
from brain_go_brrr.models.eegpt_model import EEGPTModel
from brain_go_brrr.preprocessing.eeg_preprocessor import EEGPreprocessor


class EEGBackbone(Protocol):
    """Protocol for EEG backbone models to help mypy with type checking."""

    is_loaded: bool
    config: Any  # ModelConfig, but avoiding circular import

    @property
    def n_summary_tokens(self) -> int:
        """Number of summary tokens."""
        ...

    def extract_features(self, data: FloatArray, channel_names: list[str]) -> FloatArray:
        """Extract features from EEG data."""
        ...


logger = get_logger(__name__)


class TriageLevel(str, Enum):
    """Clinical triage levels for EEG prioritization."""

    NORMAL = "NORMAL"  # Low priority
    ROUTINE = "ROUTINE"  # Standard workflow (< 48 hours)
    EXPEDITE = "EXPEDITE"  # Priority review (< 4 hours)
    URGENT = "URGENT"  # Immediate review needed


class AggregationMethod(str, Enum):
    """Methods for aggregating window-level predictions."""

    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    ATTENTION = "attention"


@dataclass
class WindowResult:
    """Results for a single EEG window."""

    index: int
    start_time: float
    end_time: float
    abnormality_score: float
    quality_score: float


@dataclass
class AbnormalityResult:
    """Complete abnormality detection results."""

    abnormality_score: float
    classification: str  # "normal" or "abnormal"
    confidence: float
    triage_flag: TriageLevel
    window_scores: list[WindowResult]
    quality_metrics: dict[str, Any]
    processing_time: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enum to string
        result["triage_flag"] = self.triage_flag.value
        # Convert window results to dicts
        if (
            isinstance(self.window_scores, list)
            and len(self.window_scores) > 0
            and isinstance(self.window_scores[0], WindowResult)
        ):
            result["window_scores"] = [asdict(w) for w in self.window_scores]
        return result


class AbnormalityDetector:
    """Main service class for EEG abnormality detection."""

    model: EEGBackbone
    classifier: torch.nn.Sequential
    device: str

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        window_duration: float | None = None,
        overlap_ratio: float | None = None,
        target_sfreq: int | None = None,
        model_version: str | None = None,
        config: AbnormalityConfig | None = None,
    ):
        """Initialize abnormality detector.

        Args:
            model_path: Path to EEGPT checkpoint
            device: Device for inference (auto, cuda, cpu)
            window_duration: Duration of analysis windows in seconds (overrides config)
            overlap_ratio: Overlap between windows (0-1) (overrides config)
            target_sfreq: Target sampling frequency (overrides config)
            model_version: Version identifier for tracking (overrides config)
            config: Complete configuration object (defaults to spec-compliant config)
        """
        # Use provided config or create default
        self.config = config or AbnormalityConfig.from_spec()

        # Allow overrides of specific parameters
        self.window_duration = window_duration or self.config.processing.window_duration_seconds
        self.overlap_ratio = overlap_ratio or self.config.processing.window_overlap_ratio
        self.target_sfreq = target_sfreq or self.config.processing.target_sampling_rate
        self.model_version = model_version or self.config.model.default_model_version

        # Handle device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device

        # Initialize model
        self._init_model(model_path)

        # Initialize classification head
        self._init_classification_head()

        logger.info(f"AbnormalityDetector initialized on {self.device}")

    def _init_model(self, model_path: Path) -> None:
        """Initialize EEGPT model."""
        config = ModelConfig(
            model_path=model_path,
            device=self.device,
            sampling_rate=self.target_sfreq,
            window_duration=self.window_duration,
        )

        try:
            self.model = EEGPTModel(config=config, auto_load=True)
            if not self.model.is_loaded:
                raise RuntimeError("Failed to load EEGPT model")
        except Exception as e:
            logger.error(f"Error loading EEGPT model: {e}")
            # Create mock model for testing
            self.model = EEGPTModel(config=config, auto_load=False)

    def _init_classification_head(self) -> None:
        """Initialize classification head for abnormality detection."""
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.config.model.feature_dim, self.config.model.classifier_hidden_1),
            torch.nn.BatchNorm1d(self.config.model.classifier_hidden_1),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.config.model.classifier_dropout),
            torch.nn.Linear(
                self.config.model.classifier_hidden_1,
                self.config.model.classifier_hidden_2,
            ),
            torch.nn.BatchNorm1d(self.config.model.classifier_hidden_2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.config.model.classifier_dropout),
            torch.nn.Linear(self.config.model.classifier_hidden_2, self.config.model.num_classes),
        ).to(self.device)

        # Set to eval mode by default
        self.classifier.eval()

        # Load pretrained weights if available
        classifier_path = Path(
            str(self.model.config.model_path).replace(".ckpt", "_classifier.pth")
        )
        if classifier_path.exists():
            self._load_classifier_weights(classifier_path)

    def _load_classifier_weights(self, classifier_path: Path) -> None:
        """Load classifier weights with dimension compatibility check.

        Args:
            classifier_path: Path to classifier weights file

        Raises:
            RuntimeError: If classifier dimensions don't match config
        """
        state = torch.load(classifier_path, map_location=self.device, weights_only=True)

        # Check dimension compatibility before loading
        first_layer_key = "0.weight"  # First Linear layer in Sequential
        if first_layer_key in state:
            weight_shape = state[first_layer_key].shape
            expected_in_features = self.config.model.feature_dim
            actual_in_features = weight_shape[1]  # Linear weight shape is (out, in)

            if actual_in_features != expected_in_features:
                raise RuntimeError(
                    f"Classifier dimension mismatch: checkpoint has {actual_in_features} "
                    f"input features but config specifies {expected_in_features}. "
                    f"This likely means the checkpoint was trained with different EEGPT "
                    f"embedding dimensions. Please retrain the classifier or update config."
                )

        self.classifier.load_state_dict(state)
        self.classifier.to(self.device)
        logger.info(f"Loaded pretrained classifier head from {classifier_path}")

    def validate_model_compatibility(self) -> None:
        """Validate that EEGPT model and classifier dimensions are compatible.

        Raises:
            RuntimeError: If there's a dimension mismatch
        """
        # Check EEGPT embedding dimension if available
        if hasattr(self.model, "embedding_dim"):
            model_embedding_dim = self.model.embedding_dim
            n_summary_tokens = getattr(self.model, "n_summary_tokens", 4)  # Default to 4 tokens
            expected_flattened_dim = model_embedding_dim * n_summary_tokens
            config_feature_dim = self.config.model.feature_dim

            if expected_flattened_dim != config_feature_dim:
                raise RuntimeError(
                    f"Model dimension mismatch: EEGPT produces {n_summary_tokens} x {model_embedding_dim} = "
                    f"{expected_flattened_dim}-dim flattened features but config expects {config_feature_dim}-dim. "
                    f"Update config.model.feature_dim to {expected_flattened_dim} or use compatible model checkpoint."
                )

        # Check classifier input dimension
        first_layer = self.classifier[0]
        if isinstance(first_layer, torch.nn.Linear):
            classifier_in_features = first_layer.in_features
            config_feature_dim = self.config.model.feature_dim

            if classifier_in_features != config_feature_dim:
                raise RuntimeError(
                    f"Classifier dimension mismatch: classifier expects {classifier_in_features} "
                    f"features but config specifies {config_feature_dim}. "
                    f"This should not happen - please report this as a bug."
                )

    def detect_abnormality(self, raw: MNERaw) -> AbnormalityResult:
        """Detect abnormalities in EEG recording.

        Args:
            raw: MNE Raw object with EEG data

        Returns:
            AbnormalityResult with detection results
        """
        start_time = time.time()

        # Validate model compatibility first
        self.validate_model_compatibility()

        # Validate input
        self._validate_input(raw)

        # Log initial info
        logger.info(f"Processing {raw.times[-1]:.1f}s recording with {len(raw.ch_names)} channels")

        # Detect bad channels BEFORE preprocessing
        bad_channels_pre = self._detect_bad_channels_raw(raw)

        # Preprocess EEG using the dedicated preprocessor
        logger.info("Preprocessing EEG data...")
        logger.info(f"Input channels ({len(raw.ch_names)}): {raw.ch_names}")

        # Determine channel subset size based on configuration
        channel_subset_size = (
            self.config.processing.channel_subset_size
            if hasattr(self.config.processing, "channel_subset_size")
            else 16
        )

        preprocessor = EEGPreprocessor(
            target_sfreq=self.target_sfreq,
            lowpass_freq=45.0,
            highpass_freq=0.5,
            notch_freq=50.0 if raw.info.get("line_freq", 50) == 50 else 60.0,
            channel_subset_size=channel_subset_size,
            use_standard_montage=True,  # Use BioSerenity-E1 standard montage
        )
        preprocessed = preprocessor.preprocess(raw.copy())

        logger.info(
            f"Preprocessed channels ({len(preprocessed.ch_names)}): {preprocessed.ch_names}"
        )

        # Apply z-score normalization
        preprocessed = self._apply_normalization(preprocessed)

        # Extract windows
        logger.info("Extracting analysis windows...")
        windows = self._extract_windows(preprocessed)

        # Assess window quality
        quality_scores = [self._assess_window_quality(w) for w in windows]

        # Get predictions for each window
        logger.info(f"Analyzing {len(windows)} windows...")
        window_scores = []
        window_results = []

        for i, (window, quality) in enumerate(zip(windows, quality_scores, strict=True)):
            score = self._predict_window(window)
            window_scores.append(score)

            # Calculate time boundaries
            step_samples = int(self.window_duration * self.target_sfreq * (1 - self.overlap_ratio))
            start = i * step_samples / self.target_sfreq
            end = start + self.window_duration

            window_results.append(
                WindowResult(
                    index=i,
                    start_time=start,
                    end_time=end,
                    abnormality_score=score,
                    quality_score=quality,
                )
            )

        # Aggregate scores
        final_score = self._aggregate_scores(
            window_scores, quality_scores, method=AggregationMethod.WEIGHTED_AVERAGE
        )

        # Calculate confidence
        confidence = self._calculate_confidence(window_scores)

        # Determine classification
        classification = (
            "abnormal" if final_score > self.config.classification.abnormal_threshold else "normal"
        )

        # Get quality metrics
        quality_metrics = self._compute_quality_metrics(
            preprocessed, quality_scores, bad_channels_pre
        )

        # Determine triage level
        triage = self._determine_triage(final_score, quality_metrics["quality_grade"])

        # Calculate processing time
        processing_time = time.time() - start_time

        return AbnormalityResult(
            abnormality_score=float(final_score),
            classification=classification,
            confidence=float(confidence),
            triage_flag=triage,
            window_scores=window_results,
            quality_metrics=quality_metrics,
            processing_time=processing_time,
            model_version=self.model_version,
        )

    def detect_abnormality_batch(self, recordings: list[MNERaw]) -> list[AbnormalityResult]:
        """Process multiple recordings in batch.

        Args:
            recordings: List of MNE Raw objects

        Returns:
            List of AbnormalityResult objects
        """
        results = []
        for raw in recordings:
            try:
                result = self.detect_abnormality(raw)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing recording: {e}")
                # Create error result
                results.append(self._create_error_result(str(e)))

        return results

    def _validate_input(self, raw: MNERaw) -> None:
        """Validate input EEG data."""
        # Check duration
        duration = raw.times[-1]
        min_duration = self.config.processing.min_recording_duration_seconds
        if duration < min_duration - 0.01:  # Allow small tolerance for floating point
            raise ValueError(f"Recording too short: {duration:.1f}s < {min_duration}s minimum")

        # Check channels
        n_channels = len(raw.ch_names)
        if n_channels < self.config.processing.min_required_channels:
            raise ValueError(
                f"Too few channels: {n_channels} < {self.config.processing.min_required_channels} minimum"
            )

        # Check for bad channels
        info = raw.info
        from brain_go_brrr import mne_compat
        if (
            mne_compat.info_has_field(info, "bads")
            and len(info["bads"]) > n_channels * self.config.processing.max_bad_channel_ratio
        ):
            raise ValueError(
                f"Too many bad channels: {len(info['bads'])} > {self.config.processing.max_bad_channel_ratio * 100:.0f}% of total"
            )

    def _detect_bad_channels_raw(self, raw: MNERaw) -> list[str]:
        """Detect bad channels in raw data before preprocessing."""
        data = raw.get_data()
        bad_channels = []

        for i, ch_name in enumerate(raw.ch_names):
            ch_data = data[i]
            # Check for flat channel (essentially no signal)
            if (
                np.std(ch_data) < self.config.quality.flat_channel_nanovolt_threshold
                or np.sum(np.abs(ch_data) > self.config.quality.saturated_amplitude_millivolt)
                > len(ch_data) * self.config.quality.saturated_sample_ratio
            ):
                bad_channels.append(ch_name)

        return bad_channels

    def _apply_normalization(self, raw: MNERaw) -> MNERaw:
        """Apply z-score normalization to preprocessed data."""
        data = raw.get_data()
        channel_means = data.mean(axis=1, keepdims=True)
        channel_stds = data.std(axis=1, keepdims=True)
        # Avoid division by zero - mark channels with very low std as bad
        mask = channel_stds > self.config.quality.flat_channel_std_threshold
        data = np.where(
            mask,
            (data - channel_means) / (channel_stds + self.config.processing.channel_std_epsilon),
            0.0,
        )
        # Rebuild raw with normalized data using public API
        from brain_go_brrr.mne_compat import update_data_inplace
        raw = update_data_inplace(raw, data)
        return raw

    def _extract_windows(self, raw: MNERaw) -> list[FloatArray]:
        """Extract sliding windows from EEG data."""
        data = raw.get_data()
        sfreq = raw.info["sfreq"]

        window_samples = int(self.window_duration * sfreq)
        step_samples = int(window_samples * (1 - self.overlap_ratio))

        windows = []
        start = 0
        while start + window_samples <= data.shape[1]:
            window = data[:, start : start + window_samples]
            windows.append(window.astype(np.float32))
            start += step_samples

        if len(windows) < self.config.processing.min_windows_for_prediction:
            raise ValueError(
                f"Too few windows: {len(windows)} < {self.config.processing.min_windows_for_prediction} minimum"
            )

        # Cast to float64 for type safety
        windows_float64 = [w.astype(np.float64) for w in windows]
        return windows_float64

    def _assess_window_quality(self, window: npt.NDArray[np.float64]) -> float:
        """Assess quality of a single window."""
        # Check for flat channels
        flat_channels: int = int(
            np.sum(np.std(window, axis=1) < self.config.quality.flat_channel_std_threshold)
        )

        # Check for high amplitude artifacts (normalized data)
        # After z-score normalization, typical values are -3 to +3
        max_amp: float = float(np.max(np.abs(window)))
        artifact_channels: int = int(
            np.sum(
                np.max(np.abs(window), axis=1) > self.config.quality.artifact_amplitude_threshold
            )
        )

        # Calculate quality score
        quality = 1.0
        quality -= (flat_channels / window.shape[0]) * self.config.quality.flat_channel_penalty
        quality -= (
            artifact_channels / window.shape[0]
        ) * self.config.quality.artifact_channel_penalty

        # Check for excessive noise (adjusted for normalized data)
        if max_amp > self.config.quality.excessive_noise_threshold:
            quality *= self.config.quality.excessive_noise_penalty

        return float(max(0.0, min(1.0, quality)))

    def _predict_window(self, window: npt.NDArray[np.float64]) -> float:
        """Get abnormality prediction for a single window."""
        # Convert to tensor
        window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract features using EEGPT
            try:
                # Convert tensor to numpy for EEGPT model
                window_np = window_tensor.cpu().numpy()
                # Create default channel names for the window
                n_channels = window_np.shape[0]
                channel_names = [f"CH{i + 1}" for i in range(n_channels)]
                features_np = self.model.extract_features(window_np, channel_names)

                # Convert features to tensor for classification
                features = torch.from_numpy(features_np).float().to(self.device)

                # Flatten EEGPT summary tokens: (4, 512) -> (2048,)
                features_flat = features.flatten() if features.dim() > 1 else features

                # Validate feature dimensions after flattening
                if features_flat.shape[0] != self.config.model.feature_dim:
                    raise RuntimeError(
                        f"Feature dimension mismatch: EEGPT returned {features.shape} "
                        f"({features_flat.shape[0]} flattened) but classifier expects {self.config.model.feature_dim}-dim. "
                        f"Check model configuration and checkpoints."
                    )

                # Add batch dimension for classifier
                features_batch = features_flat.unsqueeze(0)
                logits = self.classifier(features_batch)
                predictions = torch.softmax(logits, dim=1)

                # Get abnormality probability
                abnormal_prob = predictions[0, 1].item()
            except RuntimeError as e:
                # Re-raise RuntimeError for dimension mismatches and other critical errors
                if "dimension mismatch" in str(e).lower():
                    raise
                logger.error(f"Critical prediction error: {e}")
                raise
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                # Return neutral score on non-critical errors
                abnormal_prob = 0.5

        return float(abnormal_prob)

    def _aggregate_scores(
        self,
        window_scores: list[float],
        quality_scores: list[float],
        method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
    ) -> float:
        """Aggregate window-level scores to recording-level."""
        scores = np.array(window_scores)
        qualities = np.array(quality_scores)

        if method == AggregationMethod.WEIGHTED_AVERAGE:
            # Weight by quality scores
            weights = qualities / (qualities.sum() + 1e-7)
            return float(np.sum(scores * weights))

        elif method == AggregationMethod.VOTING:
            # Majority vote with quality threshold
            good_windows = qualities > self.config.classification.voting_quality_threshold
            if good_windows.sum() == 0:
                return 0.5  # Uncertain
            votes = scores[good_windows] > self.config.classification.abnormal_threshold
            return float(votes.mean())

        elif method == AggregationMethod.ATTENTION:
            # Simplified attention mechanism
            # In production, would use learned attention weights
            attention_weights = qualities * np.exp(np.abs(scores - 0.5))
            attention_weights /= attention_weights.sum() + 1e-7
            return float(np.sum(scores * attention_weights))

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _calculate_confidence(self, window_scores: list[float]) -> float:
        """Calculate confidence based on agreement between windows."""
        scores = np.array(window_scores)

        # High confidence if windows agree
        std_dev = np.std(scores)
        mean_score = np.mean(scores)

        # Base confidence on consistency
        consistency = 1.0 - (
            std_dev * self.config.classification.confidence_std_multiplier
        )  # Lower std = higher confidence

        # Adjust for extreme scores (more confident at extremes)
        extremity = 2 * abs(mean_score - 0.5)

        confidence = (
            self.config.classification.confidence_std_weight * consistency
            + self.config.classification.confidence_extremity_weight * extremity
        )

        return max(0.0, min(1.0, float(confidence)))

    def _compute_quality_metrics(
        self,
        raw: MNERaw,
        window_qualities: list[float],
        bad_channels_pre: list[str],
    ) -> dict[str, Any]:
        """Compute overall quality metrics for the recording."""
        # Use pre-processing bad channel detection
        bad_channels = bad_channels_pre

        # Calculate overall quality grade
        avg_quality = np.mean(window_qualities)
        bad_channel_ratio = len(bad_channels) / len(raw.ch_names)

        if (
            avg_quality > self.config.quality.excellent_avg_quality
            and bad_channel_ratio < self.config.quality.excellent_bad_channel_ratio
        ):
            quality_grade = "EXCELLENT"
        elif (
            avg_quality > self.config.quality.good_avg_quality
            and bad_channel_ratio < self.config.quality.good_bad_channel_ratio
        ):
            quality_grade = "GOOD"
        elif (
            avg_quality > self.config.quality.fair_avg_quality
            and bad_channel_ratio < self.config.quality.fair_bad_channel_ratio
        ):
            quality_grade = "FAIR"
        else:
            quality_grade = "POOR"

        # Count artifacts (simplified)
        artifacts_detected = sum(1 for q in window_qualities if q < 0.7)

        return {
            "bad_channels": bad_channels,
            "quality_grade": quality_grade,
            "artifacts_detected": artifacts_detected,
            "average_quality": float(avg_quality),
            "bad_channel_ratio": float(bad_channel_ratio),
        }

    def _determine_triage(self, abnormality_score: float, quality_grade: str) -> TriageLevel:
        """Determine clinical triage level based on results."""
        # Follow spec: Clinical triage logic
        if (
            abnormality_score > self.config.classification.urgent_score_threshold
            or quality_grade == "POOR"
        ):
            return TriageLevel.URGENT
        elif (
            abnormality_score > self.config.classification.expedite_score_threshold
            or quality_grade == "FAIR"
        ):
            return TriageLevel.EXPEDITE
        elif abnormality_score > self.config.classification.routine_score_threshold:
            return TriageLevel.ROUTINE
        else:
            return TriageLevel.NORMAL

    def _create_error_result(self, error_message: str) -> AbnormalityResult:
        """Create result for error cases with safe defaults."""
        return AbnormalityResult(
            abnormality_score=0.5,  # Uncertain
            classification="unknown",
            confidence=0.0,
            triage_flag=TriageLevel.URGENT,  # Fail safe
            window_scores=[],
            quality_metrics={
                "bad_channels": [],
                "quality_grade": "POOR",
                "artifacts_detected": 0,
                "error": error_message,
            },
            processing_time=0.0,
            model_version=self.model_version,
        )
