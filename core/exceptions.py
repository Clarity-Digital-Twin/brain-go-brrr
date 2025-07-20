"""Custom exceptions for Brain-Go-Brrr core functionality."""


class BrainGoBrrrError(Exception):
    """Base exception for all Brain-Go-Brrr errors."""

    pass


class EdfLoadError(BrainGoBrrrError):
    """Raised when EDF file loading fails.

    This exception wraps various MNE errors that can occur during
    EDF file loading, providing a consistent interface for error handling.
    """

    pass


class QualityCheckError(BrainGoBrrrError):
    """Raised when quality check processing fails."""

    pass


class ModelNotInitializedError(BrainGoBrrrError):
    """Raised when attempting to use a model that hasn't been initialized."""

    pass
