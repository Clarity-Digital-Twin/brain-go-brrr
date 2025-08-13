"""Domain layer - pure business logic with no external dependencies."""

from .channels import CHANNEL_GROUPS, ChannelConfig, StandardChannels
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
    "StandardChannels",
    "ChannelConfig",
    "CHANNEL_GROUPS",
    # Exceptions
    "BrainGoBrrrError",
    "ConfigurationError",
    "DataLoadingError",
    "PreprocessingError",
    "ModelError",
    "ValidationError",
    "AbnormalityDetectionError",
]