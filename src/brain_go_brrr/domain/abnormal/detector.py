"""Back-compat shim to the pure detector (silent re-export)."""

from .detector_pure import (
    AbnormalityResult,
    PureAbnormalityDetector,
    TriageLevel,
)

# Backward compatibility alias
AbnormalityDetector = PureAbnormalityDetector

__all__ = [
    "AbnormalityDetector",
    "AbnormalityResult", 
    "PureAbnormalityDetector",
    "TriageLevel",
]