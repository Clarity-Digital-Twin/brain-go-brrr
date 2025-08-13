"""Services compatibility exports."""

from brain_go_brrr.application.pipeline.hierarchical_pipeline import (
    HierarchicalEEGAnalyzer,
    PipelineConfig,
)
# Optional legacy alias (some code/tests expect this name)
HierarchicalPipeline = HierarchicalEEGAnalyzer

from brain_go_brrr.infra.external.yasa_adapter import (
    HierarchicalPipelineYASAAdapter,
    YASAConfig,
    YASASleepStager,
)

__all__ = [
    "HierarchicalEEGAnalyzer",
    "PipelineConfig",
    "HierarchicalPipeline",  # legacy alias
    "HierarchicalPipelineYASAAdapter",
    "YASAConfig",
    "YASASleepStager",
]
