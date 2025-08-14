"""Back-compat shim to the clean detector."""

from typing import Any

# Import everything from detector_clean
from .detector_clean import (
    AbnormalityResult,
    CleanAbnormalityDetector,
    TriageLevel,
)

# Back-compat aliases for legacy code
AbnormalityDetector = CleanAbnormalityDetector
PureAbnormalityDetector = CleanAbnormalityDetector


# Placeholder for test mocking - tests expect these classes to exist
class EEGPTModel:
    """Placeholder for backward compatibility with tests that mock EEGPTModel."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ModelConfig:
    """Placeholder for backward compatibility with tests that mock ModelConfig."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


__all__ = [
    "AbnormalityDetector",  # Legacy alias
    "AbnormalityResult",
    "CleanAbnormalityDetector",
    "EEGPTModel",  # Placeholder for tests
    "ModelConfig",  # Placeholder for tests
    "PureAbnormalityDetector",  # Alias
    "TriageLevel",
]
