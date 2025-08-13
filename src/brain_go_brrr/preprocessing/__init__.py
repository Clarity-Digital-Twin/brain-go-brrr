# DEPRECATED: preprocessing has been split. Removed in v2.0.0.
# Domain preprocessing: brain_go_brrr.domain.preprocessing
# Infrastructure preprocessing: brain_go_brrr.infra.preprocessing

import sys
from types import ModuleType

# For backward compatibility, export common items from both
from brain_go_brrr.domain.preprocessing import BasicEEGPreprocessor
from brain_go_brrr.domain.preprocessing.basic import (
    BandpassFilter,
    PreprocessingConfig,
    PreprocessingPipeline,
)
from brain_go_brrr.domain.preprocessing.window_extractor import WindowExtractor
from brain_go_brrr.infra.preprocessing.autoreject_adapter import AutorejectAdapter
from brain_go_brrr.infra.preprocessing.flexible_preprocessor import FlexibleEEGPreprocessor

# Create features submodule shim
class FeaturesModule(ModuleType):
    """Shim for preprocessing.features - redirects to domain.preprocessing.features."""
    def __getattr__(self, name):
        from brain_go_brrr.domain.preprocessing import features
        return getattr(features, name)

# Install the features submodule
sys.modules['brain_go_brrr.preprocessing.features'] = FeaturesModule('brain_go_brrr.preprocessing.features')

__all__ = [
    "AutorejectAdapter",
    "BandpassFilter",
    "BasicEEGPreprocessor",
    "FlexibleEEGPreprocessor",
    "PreprocessingConfig",
    "PreprocessingPipeline",
    "WindowExtractor",
]
