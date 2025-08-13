"""Domain layer - pure business logic with no external dependencies."""

# Channels
try:
    from .channels import ChannelMapper, ChannelProcessor, ChannelValidator
except ImportError:
    ChannelMapper = None
    ChannelProcessor = None
    ChannelValidator = None

# Exceptions - import all available
from .exceptions import (
    AbnormalityDetectionError,
    BrainGoBrrrError,
    ConfigurationError,
    EdfLoadError,
    FeatureExtractionError,
    FileFormatError,
    GPUNotAvailableError,
    InsufficientDataError,
    InsufficientMemoryError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotInitializedError,
    ProcessingError,
    QualityCheckError,
    ResourceError,
    SleepAnalysisError,
    UnsupportedMontageError,
)

# Aliases for compatibility
DataLoadingError = EdfLoadError  # Alias
PreprocessingError = ProcessingError  # Alias
ValidationError = ConfigurationError  # Alias

__all__ = [
    # Channels
    "ChannelMapper",
    "ChannelProcessor",
    "ChannelValidator",
    # Exceptions
    "AbnormalityDetectionError",
    "BrainGoBrrrError",
    "ConfigurationError",
    "DataLoadingError",  # Alias for EdfLoadError
    "EdfLoadError",
    "FeatureExtractionError",
    "FileFormatError",
    "GPUNotAvailableError",
    "InsufficientDataError",
    "InsufficientMemoryError",
    "ModelError",
    "ModelInferenceError",
    "ModelLoadError",
    "ModelNotInitializedError",
    "PreprocessingError",  # Alias for ProcessingError
    "ProcessingError",
    "QualityCheckError",
    "ResourceError",
    "SleepAnalysisError",
    "UnsupportedMontageError",
    "ValidationError",  # Alias for ConfigurationError
]
