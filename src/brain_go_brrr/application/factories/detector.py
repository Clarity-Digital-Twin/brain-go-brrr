"""Factory for creating abnormality detector with dependencies."""

from pathlib import Path
from typing import Any

from brain_go_brrr.domain.abnormal.detector import CleanAbnormalityDetector
from brain_go_brrr.infra.adapters.model_adapter import (
    EEGPreprocessorAdapter,
    EEGPTModelAdapter,
)


def create_abnormality_detector(
    model_path: str | Path | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> CleanAbnormalityDetector:
    """Create abnormality detector with all dependencies injected.
    
    Args:
        model_path: Path to EEGPT model checkpoint
        device: Device for inference (cpu/cuda)
        **kwargs: Additional configuration
        
    Returns:
        Configured abnormality detector
    """
    # Create infrastructure adapters
    model = EEGPTModelAdapter(
        model_path=str(model_path) if model_path else "data/models/pretrained/eegpt.ckpt",
        device=device,
    )
    preprocessor = EEGPreprocessorAdapter()

    # Create domain service with injected dependencies
    return CleanAbnormalityDetector(
        model=model,
        preprocessor=preprocessor,
        device=device,
        **kwargs,
    )
