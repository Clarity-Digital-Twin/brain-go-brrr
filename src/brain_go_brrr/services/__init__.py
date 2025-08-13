"""Services module for brain_go_brrr - compatibility redirects."""

# Import shims for moved modules
class _ServiceShim:
    """Lazy import shim for services."""

    def __getattr__(self, name):
        # Try to import from new locations
        if name == "hierarchical_pipeline":
            from brain_go_brrr.application.pipeline import hierarchical_pipeline
            return hierarchical_pipeline
        elif name == "yasa_adapter":
            from brain_go_brrr.infra.external import yasa_adapter
            return yasa_adapter
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

import sys

sys.modules[__name__] = _ServiceShim()

# Re-export from new locations for direct imports
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
    "HierarchicalPipelineYASAAdapter",
    "YASAConfig",
    "YASASleepStager",
]
