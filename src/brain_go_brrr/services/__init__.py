"""Services module for brain_go_brrr - compatibility layer."""

# Re-export for backward compatibility
try:
    from brain_go_brrr.application.pipeline.hierarchical_pipeline import (
        HierarchicalEEGAnalyzer,
        HierarchicalPipeline,
        PipelineConfig,
    )
except ImportError:
    HierarchicalEEGAnalyzer = None
    HierarchicalPipeline = None
    PipelineConfig = None

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
    "HierarchicalPipeline",
    "HierarchicalPipelineYASAAdapter",
    "PipelineConfig",
    "YASAConfig",
    "YASASleepStager",
]
