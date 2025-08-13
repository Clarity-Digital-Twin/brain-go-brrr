# DEPRECATED: preprocessing has been split. Removed in v2.0.0.
# Domain preprocessing: brain_go_brrr.domain.preprocessing
# Infrastructure preprocessing: brain_go_brrr.infra.preprocessing

# For backward compatibility, export common items from both
from brain_go_brrr.domain.preprocessing.basic import BasicEEGPreprocessor
from brain_go_brrr.domain.preprocessing.window_extractor import WindowExtractor
from brain_go_brrr.infra.preprocessing.autoreject_adapter import AutorejectAdapter
from brain_go_brrr.infra.preprocessing.flexible_preprocessor import FlexibleEEGPreprocessor

__all__ = [
    "AutorejectAdapter",
    "BasicEEGPreprocessor",
    "FlexibleEEGPreprocessor",
    "WindowExtractor",
]
