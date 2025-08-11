"""Safe torch.load wrapper with backward compatibility."""

import inspect
from pathlib import Path
from typing import Any

import torch


def safe_load(
    path: Path | str,
    *,
    weights_only: bool = True,
    map_location: str | torch.device | None = None,
    **kwargs: Any
) -> Any:
    """Safe wrapper for torch.load with backward compatibility.
    
    Args:
        path: Path to checkpoint file
        weights_only: Whether to restrict unpickling to tensors/primitives (safer)
        map_location: Device to map tensors to
        **kwargs: Additional arguments for torch.load
    
    Returns:
        Loaded checkpoint data
        
    Note:
        For EEGPT checkpoints which require weights_only=False, use:
        `safe_load(path, weights_only=False)  # nosec:weights_only - EEGPT format`
    """
    # Build kwargs dict
    load_kwargs: dict[str, Any] = {}

    # Add map_location if specified
    if map_location is not None:
        load_kwargs["map_location"] = map_location

    # Add weights_only if supported by this PyTorch version
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = weights_only

    # Add any additional kwargs
    load_kwargs.update(kwargs)

    # Load checkpoint
    return torch.load(path, **load_kwargs)
