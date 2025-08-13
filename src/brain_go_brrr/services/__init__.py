"""Services module for brain_go_brrr - compatibility redirects."""

# Re-export from new locations
try:
    from brain_go_brrr.application.pipeline.hierarchical_pipeline import HierarchicalEEGAnalyzer
except ImportError:
    HierarchicalEEGAnalyzer = None

try:
    from brain_go_brrr.infra.external.yasa_adapter import (
        HierarchicalPipelineYASAAdapter,
        YASAConfig,
        YASASleepStager,
    )
except ImportError:
    HierarchicalPipelineYASAAdapter = None
    YASAConfig = None
    YASASleepStager = None

__all__ = [
    "HierarchicalEEGAnalyzer",
    "HierarchicalPipelineYASAAdapter",
    "YASAConfig",
    "YASASleepStager",
]
