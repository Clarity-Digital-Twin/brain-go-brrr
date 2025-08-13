"""Domain layer - pure business logic with no external dependencies."""

from .channels import ChannelMapper, ChannelProcessor, ChannelValidator
from .exceptions import (
    AbnormalityDetectionError,
    BrainGoBrrrError,
    ConfigurationError,
    DataLoadingError,
    ModelError,
    PreprocessingError,
    ValidationError,
)

__all__ = [
    # Channels
    "ChannelMapper",
    "ChannelProcessor",
    "ChannelValidator",
    # Exceptions
    "BrainGoBrrrError",
    "ConfigurationError",
    "DataLoadingError",
    "PreprocessingError",
    "ModelError",
    "ValidationError",
    "AbnormalityDetectionError",
]
