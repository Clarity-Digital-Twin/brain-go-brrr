"""Factory for creating feature extractor with dependencies."""

from pathlib import Path
from typing import Any

from brain_go_brrr.domain.preprocessing.features.extractor import CleanFeatureExtractor
from brain_go_brrr.infra.adapters.model_adapter import (
    EEGPreprocessorAdapter,
    EEGPTModelAdapter,
)


def create_feature_extractor(
    model_path: str | Path | None = None,
    device: str = "cpu",
    window_size: float = 4.0,
    overlap: float = 0.0,
    **kwargs: Any,
) -> CleanFeatureExtractor:
    """Create feature extractor with all dependencies injected.

    Args:
        model_path: Path to EEGPT model checkpoint
        device: Device for inference
        window_size: Window size in seconds
        overlap: Window overlap
        **kwargs: Additional configuration

    Returns:
        Configured feature extractor
    """
    # Create infrastructure adapters
    model = EEGPTModelAdapter(
        model_path=str(model_path)
        if model_path
        else "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt",
        device=device,
    )
    preprocessor = EEGPreprocessorAdapter()

    # Create domain service with injected dependencies
    return CleanFeatureExtractor(
        model=model,
        preprocessor=preprocessor,
        window_size=window_size,
        overlap=overlap,
        **kwargs,
    )
