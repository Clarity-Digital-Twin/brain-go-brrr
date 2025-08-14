"""Feature extraction package.

Clean implementation lives in extractor_clean. Legacy name is kept for BC.
"""
from .extractor_clean import CleanFeatureExtractor, ExtractedFeatures

# BC alias so old imports keep working
EEGPTFeatureExtractor = CleanFeatureExtractor

__all__ = ["CleanFeatureExtractor", "EEGPTFeatureExtractor", "ExtractedFeatures"]