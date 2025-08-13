"""Services compatibility exports."""

from brain_go_brrr.application.pipeline.hierarchical_pipeline import (
    HierarchicalEEGAnalyzer,
    PipelineConfig,
)
from brain_go_brrr.infra.external.yasa_adapter import (
    HierarchicalPipelineYASAAdapter,
    YASAConfig,
    YASASleepStager,
)
from brain_go_brrr.utils.deprecated_redirect import redirect

# Optional legacy alias (some code/tests expect this name)
HierarchicalPipeline = HierarchicalEEGAnalyzer

# Keep the module available at the legacy path so tests can patch it
redirect(
    old="brain_go_brrr.services.yasa_adapter",
    new="brain_go_brrr.infra.external.yasa_adapter",
    globals_dict=globals(),
    warn_on_import=False,
)

__all__ = [
    "HierarchicalEEGAnalyzer",
    "HierarchicalPipeline",  # legacy alias
    "HierarchicalPipelineYASAAdapter",
    "PipelineConfig",
    "YASAConfig",
    "YASASleepStager",
]
