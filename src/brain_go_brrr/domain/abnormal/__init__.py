"""Abnormality detection domain module.

This module provides abnormality detection functionality for EEG data.
The clean architecture version is available through the application factories.
"""

# Import the legacy version for backward compatibility
from .detector import AbnormalityDetector, AbnormalityResult, TriageLevel

# Import the clean version
from .detector_clean import CleanAbnormalityDetector

__all__ = [
    "AbnormalityDetector",  # Legacy
    "CleanAbnormalityDetector",  # Clean Architecture
    "AbnormalityResult",
    "TriageLevel",
]
