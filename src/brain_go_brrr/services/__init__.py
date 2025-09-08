"""Services compatibility exports."""

from typing import Any

# Use lazy imports to avoid circular dependencies and hanging
__all__ = [
    "HierarchicalEEGAnalyzer",
    "HierarchicalPipeline",  # legacy alias
    "HierarchicalPipelineYASAAdapter",
    "PipelineConfig",
    "YASAConfig",
    "YASASleepStager",
]


def __getattr__(name: str) -> Any:
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
    elif name in ("HierarchicalPipelineYASAAdapter", "YASASleepStager"):
        from brain_go_brrr.infra.external.yasa_adapter import (
            HierarchicalPipelineYASAAdapter,
            YASASleepStager,
        )

        if name == "HierarchicalPipelineYASAAdapter":
            return HierarchicalPipelineYASAAdapter
        return YASASleepStager
    elif name == "YASAConfig":
        # P1 FIX: Direct import from infra where the alias is properly defined
        from brain_go_brrr.infra.external.yasa_adapter import YASAConfig

        return YASAConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
