"""Domain preprocessing - pure business logic for EEG preprocessing."""

from brain_go_brrr.domain.preprocessing.basic import PreprocessingPipeline
from brain_go_brrr.domain.preprocessing.basic import (
    PreprocessingPipeline as BasicEEGPreprocessor,  # Alias for compatibility
)
from brain_go_brrr.domain.preprocessing.sleep_preprocessor import SleepPreprocessor
from brain_go_brrr.domain.preprocessing.window_extractor import WindowExtractor

__all__ = [
    "BasicEEGPreprocessor",
    "PreprocessingPipeline",
    "SleepPreprocessor",
    "WindowExtractor",
]
