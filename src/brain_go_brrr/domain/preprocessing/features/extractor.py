"""Back-compat shim: silent re-export of the clean extractor.

No import-time warnings; pytest treats DeprecationWarning as error in this repo.
"""
from .extractor_clean import (
    CleanFeatureExtractor,
    ExtractedFeatures,
)

# Back-compat name
EEGPTFeatureExtractor = CleanFeatureExtractor

__all__ = ["CleanFeatureExtractor", "EEGPTFeatureExtractor", "ExtractedFeatures"]