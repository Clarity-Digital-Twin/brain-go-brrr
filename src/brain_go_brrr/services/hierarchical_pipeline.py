# shim for backward compatibility - lazy import to avoid side effects

from typing import Any

# Don't use __all__ with lazy imports to avoid F822 errors
# __all__ is handled dynamically via __getattr__


def __getattr__(name: str) -> Any:
    """Lazy import from new location."""
    from brain_go_brrr.application.pipeline.hierarchical_pipeline import (
        AnalysisResult,
        HierarchicalEEGAnalyzer,
        PipelineConfig,
    )

    mapping = {
        "HierarchicalPipeline": HierarchicalEEGAnalyzer,  # Backward compatible alias
        "HierarchicalEEGAnalyzer": HierarchicalEEGAnalyzer,
        "PipelineConfig": PipelineConfig,
        "AnalysisResult": AnalysisResult,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
