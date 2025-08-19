# shim for backward compatibility - lazy import to avoid side effects

__all__ = ["HierarchicalPipeline", "HierarchicalEEGAnalyzer", "PipelineConfig", "AnalysisResult"]

def __getattr__(name):
    """Lazy import from new location."""
    from brain_go_brrr.application.pipeline.hierarchical_pipeline import (
        AnalysisResult,
        HierarchicalEEGAnalyzer,
        PipelineConfig,
    )
    
    if name == "HierarchicalPipeline":
        return HierarchicalEEGAnalyzer  # Backward compatible alias
    elif name == "HierarchicalEEGAnalyzer":
        return HierarchicalEEGAnalyzer
    elif name == "PipelineConfig":
        return PipelineConfig
    elif name == "AnalysisResult":
        return AnalysisResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
