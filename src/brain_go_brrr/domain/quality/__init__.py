"""EEG Quality Control Domain.

This module provides quality control functionality for EEG data.
The clean architecture version is available through the application factories.
"""

# Import from the single controller file
from .controller import CleanQualityController, QualityMetrics

# Alias for backward compatibility
EEGQualityController = CleanQualityController

__all__ = [
    "CleanQualityController",  # Clean Architecture
    "EEGQualityController",  # Legacy
    "QualityMetrics",
]
