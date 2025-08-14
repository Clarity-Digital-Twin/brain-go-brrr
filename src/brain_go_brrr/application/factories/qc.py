"""Factory for creating quality controller with dependencies."""

from typing import Any

from brain_go_brrr.domain.quality.controller import CleanQualityController
from brain_go_brrr.infra.adapters.model_adapter import (
    EEGPreprocessorAdapter,
    EEGPTModelAdapter,
)


def create_quality_controller(
    model_path: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> CleanQualityController:
    """Create quality controller with all dependencies injected.
    
    Args:
        model_path: Optional path to EEGPT model
        device: Device for inference
        **kwargs: Additional configuration
        
    Returns:
        Configured quality controller
    """
    # Create infrastructure adapters
    preprocessor = EEGPreprocessorAdapter()

    # Model is optional for QC
    model = None
    if model_path:
        model = EEGPTModelAdapter(model_path=model_path, device=device)

    # Create domain service with injected dependencies
    return CleanQualityController(
        preprocessor=preprocessor,
        model=model,
        **kwargs,
    )
