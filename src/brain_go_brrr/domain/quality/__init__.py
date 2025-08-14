"""EEG Quality Control Domain.

This module provides quality control functionality for EEG data.
The clean architecture version is available through the application factories.
"""

# Import the legacy version for backward compatibility
from .controller import EEGQualityController

# Import the clean version
from .controller_clean import CleanQualityController, QualityMetrics

__all__ = [
    "CleanQualityController",  # Clean Architecture
    "EEGQualityController",  # Legacy
    "QualityMetrics",
]
