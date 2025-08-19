"""Services compatibility exports."""

# Use lazy imports to avoid circular dependencies and hanging
__all__ = [
    "HierarchicalEEGAnalyzer",
    "HierarchicalPipeline",  # legacy alias
    "HierarchicalPipelineYASAAdapter", 
    "PipelineConfig",
    "YASAConfig",
    "YASASleepStager",
]

def __getattr__(name):
    """Lazy import services to avoid circular dependencies."""
    if name in ("HierarchicalEEGAnalyzer", "PipelineConfig"):
        from brain_go_brrr.application.pipeline.hierarchical_pipeline import (
            HierarchicalEEGAnalyzer,
            PipelineConfig,
        )
        if name == "HierarchicalEEGAnalyzer":
            return HierarchicalEEGAnalyzer
        return PipelineConfig
    elif name == "HierarchicalPipeline":
        from brain_go_brrr.application.pipeline.hierarchical_pipeline import HierarchicalEEGAnalyzer
        return HierarchicalEEGAnalyzer  # Legacy alias
    elif name in ("HierarchicalPipelineYASAAdapter", "YASAConfig", "YASASleepStager"):
        from brain_go_brrr.infra.external.yasa_adapter import (
            HierarchicalPipelineYASAAdapter,
            YASAConfig,
            YASASleepStager,
        )
        if name == "HierarchicalPipelineYASAAdapter":
            return HierarchicalPipelineYASAAdapter
        elif name == "YASAConfig":
            return YASAConfig
        return YASASleepStager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
