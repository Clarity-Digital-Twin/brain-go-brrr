"""Services module for brain_go_brrr - compatibility layer."""

# Re-export for backward compatibility
from brain_go_brrr.application.pipeline.hierarchical_pipeline import (
    HierarchicalEEGAnalyzer,
    PipelineConfig,
)

# Alias for backward compatibility
HierarchicalPipeline = HierarchicalEEGAnalyzer

from brain_go_brrr.infra.external.yasa_adapter import (
    HierarchicalPipelineYASAAdapter,
    YASAConfig,
    YASASleepStager,
)

__all__ = [
    "HierarchicalEEGAnalyzer",
    "HierarchicalPipeline",
    "HierarchicalPipelineYASAAdapter",
    "PipelineConfig",
    "YASAConfig",
    "YASASleepStager",
]
