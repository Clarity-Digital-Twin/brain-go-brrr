"""Services compatibility exports."""

from brain_go_brrr.application.pipeline.hierarchical_pipeline import (
    HierarchicalEEGAnalyzer,
    PipelineConfig,
)

# Optional legacy alias (some code/tests expect this name)
HierarchicalPipeline = HierarchicalEEGAnalyzer

from brain_go_brrr.utils.deprecated_redirect import redirect

# Keep the module available at the legacy path so tests can patch it
redirect(
    old="brain_go_brrr.services.yasa_adapter",
    new="brain_go_brrr.infra.external.yasa_adapter",
    globals_dict=globals(),
    warn_on_import=False,
)

from brain_go_brrr.infra.external.yasa_adapter import (
    HierarchicalPipelineYASAAdapter,
    YASAConfig,
    YASASleepStager,
)

__all__ = [
    "HierarchicalEEGAnalyzer",
    "HierarchicalPipeline",  # legacy alias
    "HierarchicalPipelineYASAAdapter",
    "PipelineConfig",
    "YASAConfig",
    "YASASleepStager",
]
