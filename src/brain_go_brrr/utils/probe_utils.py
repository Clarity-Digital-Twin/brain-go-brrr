"""P0 CRITICAL FIX: Adapter for preparing EEGPT features for probes.

This module provides a centralized adapter that ensures EEGPT features are
always in the correct shape (B, EEGPT_PROBE_INPUT_DIM) before being passed to probes.

This is the SINGLE SOURCE OF TRUTH for feature preparation per P0_CRITICAL_FIXES.md.
"""

import numpy as np
import numpy.typing as npt
import torch

from brain_go_brrr.domain.constants import (
    EEGPT_PROBE_INPUT_DIM,
    EEGPT_SUMMARY_TOKENS,
    EEGPT_TOKEN_DIM,
)


def prepare_probe_features(
    features: npt.NDArray[np.float32] | npt.NDArray[np.float64] | torch.Tensor,
) -> torch.Tensor:
    """Prepare EEGPT features for probe consumption.

    This is the SSOT adapter that ensures features are always in the correct
    shape (B, EEGPT_PROBE_INPUT_DIM) for probes, regardless of input shape.

    Args:
        features: EEGPT features in one of these shapes:
            - (EEGPT_TOKEN_DIM,): Single summary vector (will error - invalid)
            - (EEGPT_SUMMARY_TOKENS, EEGPT_TOKEN_DIM): Single sample, 4 tokens
            - (EEGPT_PROBE_INPUT_DIM,): Single sample, already flattened
            - (B, EEGPT_TOKEN_DIM): Batch of summaries (will error - invalid)
            - (B, EEGPT_SUMMARY_TOKENS, EEGPT_TOKEN_DIM): Batch of 4 tokens each
            - (B, EEGPT_PROBE_INPUT_DIM): Batch, already flattened

    Returns:
        Torch tensor of shape (B, EEGPT_PROBE_INPUT_DIM) ready for probe consumption.

    Raises:
        ValueError: If features have invalid shape (e.g., EEGPT_TOKEN_DIM-d summaries).
    """
    # Convert to tensor if needed
    if isinstance(features, np.ndarray):
        features_tensor = torch.as_tensor(features, dtype=torch.float32)
    else:
        features_tensor = features

    # Ensure we have at least 2D
    if features_tensor.dim() == 1:
        # Single vector - check dimensions
        if features_tensor.shape[0] == EEGPT_TOKEN_DIM:
            raise ValueError(
                f"P0 ERROR: Received {EEGPT_TOKEN_DIM}-d summary features. "
                f"Call extract_features with summary=False to get {EEGPT_PROBE_INPUT_DIM}-d features!"
            )
        elif features_tensor.shape[0] == EEGPT_PROBE_INPUT_DIM:
            # Already flattened, just add batch dimension
            features_tensor = features_tensor.unsqueeze(0)
        else:
            raise ValueError(
                f"Invalid feature dimension: {features_tensor.shape[0]}. "
                f"Expected {EEGPT_PROBE_INPUT_DIM} (or unflattened {EEGPT_SUMMARY_TOKENS}x{EEGPT_TOKEN_DIM})."
            )

    elif features_tensor.dim() == 2:
        # Check if it's (EEGPT_SUMMARY_TOKENS, EEGPT_TOKEN_DIM) or (B, D)
        if (
            features_tensor.shape[0] == EEGPT_SUMMARY_TOKENS
            and features_tensor.shape[1] == EEGPT_TOKEN_DIM
        ):
            # Single sample with 4 tokens - flatten and add batch dim
            features_tensor = features_tensor.flatten().unsqueeze(0)
        elif features_tensor.shape[1] == EEGPT_TOKEN_DIM:
            # Batch of summaries - ERROR!
            raise ValueError(
                f"P0 ERROR: Received batch of {EEGPT_TOKEN_DIM}-d summaries (shape {features_tensor.shape}). "
                f"Call extract_features with summary=False to get (B, {EEGPT_SUMMARY_TOKENS}, {EEGPT_TOKEN_DIM}) features!"
            )
        elif features_tensor.shape[1] == EEGPT_PROBE_INPUT_DIM:
            # Already in correct shape (B, EEGPT_PROBE_INPUT_DIM)
            pass
        else:
            raise ValueError(
                f"Invalid feature shape: {features_tensor.shape}. "
                f"Expected (B, {EEGPT_PROBE_INPUT_DIM}) or (B, {EEGPT_SUMMARY_TOKENS}, {EEGPT_TOKEN_DIM})."
            )

    elif features_tensor.dim() == 3:
        # Should be (B, EEGPT_SUMMARY_TOKENS, EEGPT_TOKEN_DIM)
        if (
            features_tensor.shape[1] == EEGPT_SUMMARY_TOKENS
            and features_tensor.shape[2] == EEGPT_TOKEN_DIM
        ):
            # Flatten the last two dimensions
            features_tensor = features_tensor.flatten(1)  # (B, EEGPT_PROBE_INPUT_DIM)
        else:
            raise ValueError(
                f"Invalid 3D feature shape: {features_tensor.shape}. Expected (B, {EEGPT_SUMMARY_TOKENS}, {EEGPT_TOKEN_DIM})."
            )

    else:
        raise ValueError(
            f"Invalid feature dimensions: {features_tensor.dim()}D. Expected 1D, 2D, or 3D tensor."
        )

    # Final validation
    if features_tensor.shape[-1] != EEGPT_PROBE_INPUT_DIM:
        raise ValueError(
            f"P0 CRITICAL: Final shape {features_tensor.shape} doesn't have {EEGPT_PROBE_INPUT_DIM} features! "
            f"This is a bug in the adapter."
        )

    return features_tensor
