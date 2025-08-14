"""Back-compat shim to the clean controller (silent re-export)."""

from .controller_clean import (
    CleanQualityController,
    QualityMetrics,
)

# Backward compatibility alias
EEGQualityController = CleanQualityController

__all__ = [
    "CleanQualityController",
    "EEGQualityController",
    "QualityMetrics",
]