"""Feature extraction domain module.

This module provides feature extraction functionality for EEG data.
The clean architecture version is available through the application factories.
"""

# Import the legacy version for backward compatibility
from .extractor import EEGPTFeatureExtractor

# Import the clean version
from .extractor_clean import CleanFeatureExtractor, ExtractedFeatures

__all__ = [
    "EEGPTFeatureExtractor",  # Legacy
    "CleanFeatureExtractor",  # Clean Architecture
    "ExtractedFeatures",
]
