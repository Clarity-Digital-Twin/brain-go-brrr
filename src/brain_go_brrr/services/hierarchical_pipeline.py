# shim for backward compatibility
from brain_go_brrr.application.pipeline.hierarchical_pipeline import *  # noqa: F403
from brain_go_brrr.application.pipeline.hierarchical_pipeline import (
    AnalysisResult,
    HierarchicalEEGAnalyzer,
    PipelineConfig,
)

# Provide backward compatible alias
HierarchicalPipeline = HierarchicalEEGAnalyzer

__all__ = ["HierarchicalPipeline", "HierarchicalEEGAnalyzer", "PipelineConfig", "AnalysisResult"]
