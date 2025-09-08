"""Application factories for dependency injection.

This is the composition root where we wire together domain services
with their infrastructure implementations, following Clean Architecture.
"""

from brain_go_brrr.application.config import AbnormalityConfig
from brain_go_brrr.domain.abnormal.detector import CleanAbnormalityDetector
from brain_go_brrr.domain.ports import AbnormalityConfigPort
from brain_go_brrr.infra.adapters.model_adapter import (
    EEGPreprocessorAdapter,
    EEGPTModelAdapter,
    LoggerAdapter,
)
from brain_go_brrr.services.yasa_adapter import YASASleepStager


class ConfigAdapter(AbnormalityConfigPort):
    """Adapter to make AbnormalityConfig conform to port interface."""

    def __init__(self, config: AbnormalityConfig):
        """Initialize config adapter.

        Args:
            config: Application configuration
        """
        self._config = config

    @property
    def confidence_threshold(self) -> float:
        """Confidence threshold for abnormality detection."""
        return getattr(self._config.classification, "confidence_threshold", 0.7)

    @property
    def min_confidence(self) -> float:
        """Minimum confidence for valid prediction."""
        return getattr(self._config.classification, "min_confidence", 0.3)

    @property
    def channels(self) -> list[str]:
        """Required EEG channels."""
        return getattr(self._config.quality, "required_channels", [])

    @property
    def bandpass_low(self) -> float:
        """Low frequency for bandpass filter."""
        return getattr(self._config.processing, "bandpass_low", 0.5)

    @property
    def bandpass_high(self) -> float:
        """High frequency for bandpass filter."""
        return getattr(self._config.processing, "bandpass_high", 50.0)


def create_abnormality_detector(
    model_path: str | None = None,
    config: AbnormalityConfig | None = None,
    device: str = "cpu",
    enable_logging: bool = True,
) -> CleanAbnormalityDetector:
    """Factory to create abnormality detector with all dependencies.

    This is the composition root where we wire together:
    - Domain logic (CleanAbnormalityDetector)
    - Infrastructure adapters (EEGPTModelAdapter, etc.)
    - Configuration

    Args:
        model_path: Path to EEGPT model checkpoint (optional, uses default if None)
        config: Abnormality detection configuration
        device: Device to run model on (cpu/cuda)
        enable_logging: Whether to enable logging

    Returns:
        Fully configured abnormality detector
    """
    # Use default config if not provided
    if config is None:
        config = AbnormalityConfig()

    # Use default model path if not provided
    if model_path is None:
        import os

        model_path = os.environ.get(
            "BGB_MODEL_PATH", "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
        )

    # Create infrastructure adapters
    model_adapter = EEGPTModelAdapter(model_path=model_path, device=device)
    preprocessor_adapter = EEGPreprocessorAdapter()
    config_adapter = ConfigAdapter(config)

    # Create logger adapter if enabled
    logger_adapter = None
    if enable_logging:
        logger_adapter = LoggerAdapter("brain_go_brrr.domain.abnormal")

    # Wire everything together in the domain service
    detector = CleanAbnormalityDetector(
        model=model_adapter,
        preprocessor=preprocessor_adapter,
        config=config_adapter,
        logger=logger_adapter,
    )

    return detector


def create_quality_controller(
    model_path: str | None = None,
    device: str = "cpu",
    enable_logging: bool = True,
    enable_autoreject: bool = True,
) -> object:
    """Factory to create quality controller with clean dependencies.

    Args:
        model_path: Path to EEGPT model checkpoint (optional, uses default if None)
        device: Device to run model on
        enable_logging: Whether to enable logging
        enable_autoreject: Whether to use AutoReject

    Returns:
        Configured quality controller
    """
    from brain_go_brrr.domain.quality.controller import CleanQualityController
    from brain_go_brrr.infra.adapters.autoreject_adapter import AutoRejectAdapter

    # Use default model path if not provided
    if model_path is None:
        import os

        model_path = os.environ.get(
            "BGB_MODEL_PATH", "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
        )

    # Create infrastructure adapters
    model_adapter = EEGPTModelAdapter(model_path=model_path, device=device)
    preprocessor_adapter = EEGPreprocessorAdapter()

    # Create AutoReject adapter if enabled
    autoreject_adapter = None
    if enable_autoreject:
        autoreject_adapter = AutoRejectAdapter()

    # Create logger adapter if enabled
    logger_adapter = None
    if enable_logging:
        logger_adapter = LoggerAdapter("brain_go_brrr.domain.quality")

    # Wire everything together
    controller = CleanQualityController(
        preprocessor=preprocessor_adapter,
        model=model_adapter,
        autoreject=autoreject_adapter,
        logger=logger_adapter,
    )

    return controller


def create_feature_extractor(
    model_path: str,
    device: str = "cpu",
    window_size: float = 4.0,
    overlap: float = 0.5,
    enable_logging: bool = True,
) -> object:
    """Factory to create feature extractor with clean dependencies.

    Args:
        model_path: Path to EEGPT model checkpoint
        device: Device to run model on
        window_size: Window size in seconds
        overlap: Window overlap ratio
        enable_logging: Whether to enable logging

    Returns:
        Configured feature extractor
    """
    from brain_go_brrr.domain.preprocessing.features.extractor import CleanFeatureExtractor

    # Create infrastructure adapters
    model_adapter = EEGPTModelAdapter(model_path=model_path, device=device)
    preprocessor_adapter = EEGPreprocessorAdapter()

    # Create logger adapter if enabled
    logger_adapter = None
    if enable_logging:
        logger_adapter = LoggerAdapter("brain_go_brrr.domain.features")

    # Wire everything together
    extractor = CleanFeatureExtractor(
        model=model_adapter,
        preprocessor=preprocessor_adapter,
        logger=logger_adapter,
        window_size=window_size,
        overlap=overlap,
    )

    return extractor


def create_sleep_analyzer(
    consensus: bool = True,
    min_confidence: float = 0.5,
    enable_logging: bool = True,  # noqa: ARG001
) -> YASASleepStager:
    """Factory to create sleep analyzer (YASA adapter).

    This maintains backward compatibility with the documented API
    while providing sensible defaults.

    Args:
        consensus: Whether to use consensus model (default: True)
        min_confidence: Minimum confidence threshold (default: 0.5)
        enable_logging: Whether to enable logging (default: True)

    Returns:
        Configured YASA sleep stager
    """
    # Import the config class
    from brain_go_brrr.infra.external.yasa_adapter import (
        YASAAdapterConfig,
    )  # P1 FIX: Use renamed class

    # Create config with parameters
    config = YASAAdapterConfig()  # P1 FIX: Use renamed class
    config.use_consensus = consensus
    config.min_confidence = min_confidence

    # Create YASA adapter with configuration
    stager = YASASleepStager(config=config)

    return stager
