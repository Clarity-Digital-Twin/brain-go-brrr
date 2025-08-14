"""PURE CLEAN ARCHITECTURE Application Factories - THE COMPOSITION ROOT.

This is where we wire everything together following Uncle Bob's Clean Architecture.
The application layer is the ONLY place that knows about concrete implementations.

Following Dependency Injection and Composition Root patterns:
- Domain depends on abstractions (ports)
- Infrastructure implements abstractions (adapters)
- Application wires them together (this file)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from brain_go_brrr.application.config import AbnormalityConfig

if TYPE_CHECKING:
    from brain_go_brrr.application.factories_types import (
        AbnormalityDetectorPort,
        FeatureExtractorPort,
        QualityControllerPort,
    )
from brain_go_brrr.domain.abnormal.detector_pure import PureAbnormalityDetector
from brain_go_brrr.domain.abnormal.settings import (
    AbnormalitySettings,
)
from brain_go_brrr.infra.adapters.eegpt_classifier import EEGPTClassifierAdapter
from brain_go_brrr.infra.adapters.eegpt_feature_extractor import (
    EEGPTFeatureExtractorAdapter,
)
from brain_go_brrr.infra.adapters.logger_adapter import PythonLoggerAdapter
from brain_go_brrr.infra.adapters.preprocessor_flexible import (
    FlexiblePreprocessorAdapter,
)
from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel


def create_pure_abnormality_detector(
    config: AbnormalityConfig | None = None,
    model_path: str | None = None,
    device: str = "auto",
    logger: logging.Logger | None = None,
) -> AbnormalityDetectorPort:
    """Create a PURE abnormality detector with all dependencies wired.

    This is THE composition root - the ONLY place that knows about
    concrete implementations. Everything else depends on abstractions.

    Args:
        config: Application configuration (or use defaults)
        model_path: Path to EEGPT model checkpoint
        device: Device for inference (auto/cuda/cpu)
        logger: Python logger instance

    Returns:
        AbnormalityDetectorPort implementation ready for use
    """
    # Step 1: Configuration
    if config is None:
        config = AbnormalityConfig.from_spec()

    if model_path is None:
        # Try to get checkpoint path from config, default to None
        checkpoint = getattr(config.model, "checkpoint_path", None)
        model_path = str(checkpoint) if checkpoint else None

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 2: Create domain settings (pure value objects)
    settings = AbnormalitySettings(
        abnormal_threshold=config.classification.abnormal_threshold,
        confidence_threshold=getattr(config.classification, "confidence_threshold", 0.7),
        min_confidence=getattr(config.classification, "min_confidence", 0.3),
        urgent_threshold=config.classification.urgent_score_threshold,
        expedite_threshold=config.classification.expedite_score_threshold,
        routine_threshold=config.classification.routine_score_threshold,
        window_duration=config.processing.window_duration_seconds,
        window_overlap=config.processing.window_overlap_ratio,
        min_windows=config.processing.min_windows_for_prediction,
        min_quality_score=config.quality.fair_avg_quality,
        artifact_threshold=config.quality.artifact_amplitude_threshold,
    )

    # Step 3: Create infrastructure components

    # Load EEGPT model (pass primitives, not config object)
    eegpt_model = EEGPTModel(
        checkpoint_path=model_path,
        device=device,
        sampling_rate=config.processing.target_sampling_rate,
        window_duration=config.processing.window_duration_seconds,
        auto_load=True,
    )

    # Load classifier head
    if model_path:
        classifier_path = Path(model_path).parent / "abnormal_classifier.pth"
        classifier_exists = classifier_path.exists()
    else:
        classifier_exists = False

    if classifier_exists and model_path:
        classifier_path = Path(model_path).parent / "abnormal_classifier.pth"
        classifier = torch.load(classifier_path, map_location=device)
    else:
        # Create default classifier architecture
        feature_dim = config.model.feature_dim
        classifier = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 2),
        )

    # Step 4: Create adapters (infrastructure implementing domain ports)
    preprocessor = FlexiblePreprocessorAdapter(
        target_sfreq=config.processing.target_sampling_rate,
        lowpass_freq=45.0,
        highpass_freq=0.5,
        notch_freq=50.0,
        channel_subset_size=config.processing.channel_subset_size,
    )

    feature_extractor = EEGPTFeatureExtractorAdapter(eegpt_model)

    classifier_adapter = EEGPTClassifierAdapter(classifier, device=device)

    logger_adapter = PythonLoggerAdapter(
        logger or logging.getLogger("brain_go_brrr.domain.abnormal")
    )

    # Step 5: Wire everything together
    detector = PureAbnormalityDetector(
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        classifier=classifier_adapter,
        settings=settings,
        logger=logger_adapter,
    )

    return detector  # type: ignore[return-value]


def create_quality_controller_pure(
    model_path: str | None = None,
    device: str = "auto",
    enable_autoreject: bool = True,
    logger: logging.Logger | None = None,
) -> QualityControllerPort:
    """Create a PURE quality controller with all dependencies wired.

    Args:
        model_path: Path to EEGPT model checkpoint
        device: Device for inference
        enable_autoreject: Whether to use AutoReject
        logger: Python logger instance

    Returns:
        QualityControllerPort implementation ready for use
    """
    # Import the clean quality controller
    from brain_go_brrr.domain.quality.controller_clean import CleanQualityController
    from brain_go_brrr.infra.adapters.autoreject_adapter import AutoRejectAdapter
    from brain_go_brrr.infra.adapters.model_adapter import (
        EEGPreprocessorAdapter,
        EEGPTModelAdapter,
    )

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use default model path if not provided
    if model_path is None:
        model_path = "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"

    # Create adapters
    model_adapter = EEGPTModelAdapter(model_path=model_path, device=device)
    preprocessor_adapter = EEGPreprocessorAdapter()

    autoreject_adapter = None
    if enable_autoreject:
        autoreject_adapter = AutoRejectAdapter()

    logger_adapter = None
    if logger:
        from brain_go_brrr.infra.adapters.model_adapter import LoggerAdapter

        logger_adapter = LoggerAdapter("brain_go_brrr.domain.quality")

    # Wire together
    controller = CleanQualityController(
        preprocessor=preprocessor_adapter,
        model=model_adapter,
        autoreject=autoreject_adapter,
        logger=logger_adapter,
    )

    return controller


def create_feature_extractor_pure(
    model_path: str | None = None,
    device: str = "auto",
    window_size: float = 4.0,
    overlap: float = 0.5,
    logger: logging.Logger | None = None,
) -> FeatureExtractorPort:
    """Create a PURE feature extractor with all dependencies wired.

    Args:
        model_path: Path to EEGPT model checkpoint
        device: Device for inference
        window_size: Window size in seconds
        overlap: Window overlap ratio
        logger: Python logger instance

    Returns:
        FeatureExtractorPort implementation ready for use
    """
    from brain_go_brrr.domain.preprocessing.features.extractor_clean import (
        CleanFeatureExtractor,
    )
    from brain_go_brrr.infra.adapters.model_adapter import (
        EEGPreprocessorAdapter,
        EEGPTModelAdapter,
        LoggerAdapter,
    )

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use default model path if not provided
    if model_path is None:
        model_path = "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"

    # Create adapters
    model_adapter = EEGPTModelAdapter(model_path=model_path, device=device)
    preprocessor_adapter = EEGPreprocessorAdapter()

    logger_adapter = None
    if logger:
        logger_adapter = LoggerAdapter("brain_go_brrr.domain.features")

    # Wire together
    extractor = CleanFeatureExtractor(
        model=model_adapter,
        preprocessor=preprocessor_adapter,
        logger=logger_adapter,
        window_size=window_size,
        overlap=overlap,
    )

    return extractor
