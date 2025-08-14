"""Back-compat shim: silent re-export of the clean extractor.

No import-time warnings; pytest treats DeprecationWarning as error in this repo.
"""

from typing import Any

from .extractor_clean import (
    CleanFeatureExtractor,
    ExtractedFeatures,
)

# Back-compat names and aliases
EEGPTFeatureExtractor = CleanFeatureExtractor
FeatureExtractor = CleanFeatureExtractor  # Another common alias


# Placeholder for test mocking - tests expect this class to exist
class EEGPTModel:
    """Placeholder for backward compatibility with tests that mock EEGPTModel."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize placeholder."""
        pass


__all__ = [
    "CleanFeatureExtractor",
    "EEGPTFeatureExtractor",
    "EEGPTModel",  # Placeholder for tests
    "ExtractedFeatures",
    "FeatureExtractor",
]
