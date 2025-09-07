"""P0 CRITICAL FIX: Adapter for preparing EEGPT features for probes.

This module provides a centralized adapter that ensures EEGPT features are
always in the correct shape (B, 2048) before being passed to probes.

This is the SINGLE SOURCE OF TRUTH for feature preparation per P0_CRITICAL_FIXES.md.
"""

import numpy as np
import numpy.typing as npt
import torch


def prepare_probe_features(
    features: npt.NDArray[np.float32] | torch.Tensor,
) -> torch.Tensor:
    """Prepare EEGPT features for probe consumption.
    
    This is the SSOT adapter that ensures features are always in the correct
    shape (B, 2048) for probes, regardless of input shape.
    
    Args:
        features: EEGPT features in one of these shapes:
            - (512,): Single summary vector (will error - invalid)
            - (4, 512): Single sample, 4 tokens
            - (2048,): Single sample, already flattened
            - (B, 512): Batch of summaries (will error - invalid)
            - (B, 4, 512): Batch of 4 tokens each
            - (B, 2048): Batch, already flattened
    
    Returns:
        Torch tensor of shape (B, 2048) ready for probe consumption.
    
    Raises:
        ValueError: If features have invalid shape (e.g., 512-d summaries).
    """
    # Convert to tensor if needed
    if isinstance(features, np.ndarray):
        features_tensor = torch.as_tensor(features, dtype=torch.float32)
    else:
        features_tensor = features
    
    # Ensure we have at least 2D
    if features_tensor.dim() == 1:
        # Single vector - check dimensions
        if features_tensor.shape[0] == 512:
            raise ValueError(
                "P0 ERROR: Received 512-d summary features. "
                "Call extract_features with summary=False to get 2048-d features!"
            )
        elif features_tensor.shape[0] == 2048:
            # Already flattened, just add batch dimension
            features_tensor = features_tensor.unsqueeze(0)
        else:
            raise ValueError(
                f"Invalid feature dimension: {features_tensor.shape[0]}. "
                f"Expected 2048 (or unflattenned 4x512)."
            )
    
    elif features_tensor.dim() == 2:
        # Check if it's (4, 512) or (B, D)
        if features_tensor.shape[0] == 4 and features_tensor.shape[1] == 512:
            # Single sample with 4 tokens - flatten and add batch dim
            features_tensor = features_tensor.flatten().unsqueeze(0)
        elif features_tensor.shape[1] == 512:
            # Batch of summaries - ERROR!
            raise ValueError(
                f"P0 ERROR: Received batch of 512-d summaries (shape {features_tensor.shape}). "
                "Call extract_features with summary=False to get (B, 4, 512) features!"
            )
        elif features_tensor.shape[1] == 2048:
            # Already in correct shape (B, 2048)
            pass
        else:
            raise ValueError(
                f"Invalid feature shape: {features_tensor.shape}. "
                f"Expected (B, 2048) or (B, 4, 512)."
            )
    
    elif features_tensor.dim() == 3:
        # Should be (B, 4, 512)
        if features_tensor.shape[1] == 4 and features_tensor.shape[2] == 512:
            # Flatten the last two dimensions
            features_tensor = features_tensor.flatten(1)  # (B, 2048)
        else:
            raise ValueError(
                f"Invalid 3D feature shape: {features_tensor.shape}. "
                f"Expected (B, 4, 512)."
            )
    
    else:
        raise ValueError(
            f"Invalid feature dimensions: {features_tensor.dim()}D. "
            f"Expected 1D, 2D, or 3D tensor."
        )
    
    # Final validation
    if features_tensor.shape[-1] != 2048:
        raise ValueError(
            f"P0 CRITICAL: Final shape {features_tensor.shape} doesn't have 2048 features! "
            f"This is a bug in the adapter."
        )
    
    return features_tensor