"""Abnormality detection domain module.

This module provides abnormality detection functionality for EEG data.
The clean architecture version is available through the application factories.
"""

# Import from the single detector file
from .detector import AbnormalityResult, CleanAbnormalityDetector, TriageLevel

# Alias for backward compatibility
AbnormalityDetector = CleanAbnormalityDetector

__all__ = [
    "AbnormalityDetector",  # Legacy
    "AbnormalityResult",
    "CleanAbnormalityDetector",  # Clean Architecture
    "TriageLevel",
]
