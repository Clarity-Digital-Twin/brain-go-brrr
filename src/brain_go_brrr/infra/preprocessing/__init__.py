"""Infrastructure preprocessing - external dependencies and I/O.

This module provides preprocessing utilities. Imports are lazy to avoid
forcing heavy dependencies like MNE when not needed.
"""

# Lazy imports to avoid forcing MNE installation for modules that don't need it
__all__ = [
    "AutorejectAdapter",
    "ChunkedAutorejectPipeline",
    "EEGPreprocessor",
    "FlexibleEEGPreprocessor",
    "TUEVEventExtractor",
]


def __getattr__(name: str) -> type:
    """Lazy import to avoid forcing MNE when importing preprocessing package."""
    if name == "AutorejectAdapter":
        from brain_go_brrr.infra.preprocessing.autoreject_adapter import AutorejectAdapter

        return AutorejectAdapter
    elif name == "ChunkedAutorejectPipeline":
        from brain_go_brrr.infra.preprocessing.chunked_autoreject import ChunkedAutorejectPipeline

        return ChunkedAutorejectPipeline
    elif name == "EEGPreprocessor":
        from brain_go_brrr.infra.preprocessing.eeg_preprocessor import EEGPreprocessor

        return EEGPreprocessor
    elif name == "FlexibleEEGPreprocessor":
        from brain_go_brrr.infra.preprocessing.flexible_preprocessor import FlexibleEEGPreprocessor

        return FlexibleEEGPreprocessor
    elif name == "TUEVEventExtractor":
        from brain_go_brrr.infra.preprocessing.tuev_event_extractor import TUEVEventExtractor

        return TUEVEventExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
