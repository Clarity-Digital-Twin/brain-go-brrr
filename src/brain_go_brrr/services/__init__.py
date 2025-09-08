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
    elif name in ("HierarchicalPipelineYASAAdapter", "YASAConfig", "YASASleepStager"):
        from brain_go_brrr.infra.external.yasa_adapter import (
            HierarchicalPipelineYASAAdapter,
            YASAAdapterConfig,  # P1 FIX: Use renamed class
            YASASleepStager,
        )

        # P1 FIX: Create compatibility alias
        yasa_config = YASAAdapterConfig  # Use lowercase to avoid N806

        if name == "HierarchicalPipelineYASAAdapter":
            return HierarchicalPipelineYASAAdapter
        elif name == "YASAConfig":
            return yasa_config
        return YASASleepStager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
