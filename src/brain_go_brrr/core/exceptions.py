"""Custom exceptions for Brain-Go-Brrr core functionality."""


class BrainGoBrrrError(Exception):
    """Base exception for all Brain-Go-Brrr errors."""

    pass


# File I/O Errors
class EdfLoadError(BrainGoBrrrError):
    """Raised when EDF file loading fails.

    This exception wraps various MNE errors that can occur during
    EDF file loading, providing a consistent interface for error handling.
    """

    pass


class FileFormatError(EdfLoadError):
    """Raised when file format is invalid or corrupted."""

    pass


class InsufficientDataError(EdfLoadError):
    """Raised when EDF file doesn't contain enough data for analysis."""

    pass


# Processing Errors
class ProcessingError(BrainGoBrrrError):
    """Base class for all processing-related errors."""

    pass


class QualityCheckError(ProcessingError):
    """Raised when quality check processing fails."""

    pass


class SleepAnalysisError(ProcessingError):
    """Raised when sleep analysis fails."""

    pass


class FeatureExtractionError(ProcessingError):
    """Raised when feature extraction fails."""

    pass


class AbnormalityDetectionError(ProcessingError):
    """Raised when abnormality detection fails."""

    pass


# Model Errors
class ModelError(BrainGoBrrrError):
    """Base class for model-related errors."""

    pass


class ModelNotInitializedError(ModelError):
    """Raised when attempting to use a model that hasn't been initialized."""

    pass


class ModelLoadError(ModelError):
    """Raised when model checkpoint cannot be loaded."""

    pass


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""

    pass


# Configuration Errors
class ConfigurationError(BrainGoBrrrError):
    """Raised when configuration is invalid or missing."""

    pass


# Resource Errors
class ResourceError(BrainGoBrrrError):
    """Base class for resource-related errors."""

    pass


class InsufficientMemoryError(ResourceError):
    """Raised when operation runs out of memory."""

    pass


class GPUNotAvailableError(ResourceError):
    """Raised when GPU is required but not available."""

    pass
