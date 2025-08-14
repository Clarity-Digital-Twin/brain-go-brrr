"""Domain preprocessing - pure business logic for EEG preprocessing."""

from typing import Any


# Lazy loading to avoid circular imports
def __getattr__(name: str) -> Any:
    if name == "PreprocessingPipeline" or name == "BasicEEGPreprocessor":
        from brain_go_brrr.domain.preprocessing.basic import PreprocessingPipeline
        return PreprocessingPipeline
    elif name == "SleepPreprocessor":
        from brain_go_brrr.domain.preprocessing.sleep_preprocessor import SleepPreprocessor
        return SleepPreprocessor
    elif name == "WindowExtractor":
        from brain_go_brrr.domain.preprocessing.window_extractor import WindowExtractor
        return WindowExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BasicEEGPreprocessor",
    "PreprocessingPipeline",
    "SleepPreprocessor",
    "WindowExtractor",
]
