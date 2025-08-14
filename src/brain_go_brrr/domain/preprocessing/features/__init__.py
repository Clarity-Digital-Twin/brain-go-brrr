"""Feature extraction domain module.

This module provides feature extraction functionality for EEG data.
The clean architecture version is available through the application factories.
"""

# Import from the clean module directly (extractor.py is now a silent shim)
from .extractor_clean import (
    CleanFeatureExtractor,
    EEGPTFeatureExtractor,  # This is re-exported from extractor_clean
    ExtractedFeatures,
)

__all__ = [
    "CleanFeatureExtractor",  # Clean Architecture
    "EEGPTFeatureExtractor",  # Legacy
    "ExtractedFeatures",
]
