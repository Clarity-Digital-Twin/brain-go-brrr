"""Back-compat shim to the clean detector."""

# Import everything from detector_clean
from .detector_clean import (
    AbnormalityResult,
    CleanAbnormalityDetector,
    TriageLevel,
)

# Back-compat aliases for legacy code
AbnormalityDetector = CleanAbnormalityDetector
PureAbnormalityDetector = CleanAbnormalityDetector

__all__ = [
    "AbnormalityDetector",  # Legacy alias
    "AbnormalityResult",
    "CleanAbnormalityDetector",
    "PureAbnormalityDetector",  # Alias
    "TriageLevel",
]
