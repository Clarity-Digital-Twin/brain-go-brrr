"""TUAB-specific collate function - strictly enforces 19 channels."""

import logging

import torch

logger = logging.getLogger(__name__)


def collate_tuab_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor | int | float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for TUAB dataset - expects exactly 19 channels.

    TUAB specifications:
    - Expects exactly 19 channels (no Fz)
    - BCEWithLogitsLoss expects float labels (0.0, 1.0)

    Cache audit results (Jan 2025):
    - 100/100 sampled cache files have exactly 19 channels
    - 0% have 20 channels (no Fz contamination found)
    - Legacy workaround removed per technical debt cleanup

    Args:
        batch: List of (eeg_data, label) tuples from DataLoader

    Returns:
        Tuple of (batch_data, batch_labels)
        - batch_data: (B, 19, 1024) tensor
        - batch_labels: (B,) tensor with float32 dtype

    Raises:
        RuntimeError: If any sample doesn't have exactly 19 channels
    """
    # Validate and collect samples - strict 19-channel enforcement
    processed_samples = []
    for idx, (x, label) in enumerate(batch):
        if x.shape[0] == 19:
            # Correct shape - proceed
            processed_samples.append(x)
        elif x.shape[0] == 20:
            # Log warning but don't fail - helps identify if contamination reappears
            # In production, this would be an error after cache is verified clean
            logger.warning(
                f"TUAB batch item {idx}: Found 20 channels (expected 19). "
                f"This suggests cache contamination. Label={label}, Shape={x.shape}"
            )
            # For now, strictly reject as per audit results
            raise RuntimeError(
                f"TUAB batch item {idx}: Expected exactly 19 channels, got {x.shape[0]}. "
                f"Cache audit showed 0% contamination - this shouldn't happen. Shape: {x.shape}"
            )
        else:
            raise RuntimeError(
                f"TUAB batch item {idx}: Unexpected channel count {x.shape[0]}. "
                f"Expected exactly 19 channels. Shape: {x.shape}"
            )

    data = torch.stack(processed_samples)

    # Handle labels - TUAB uses float labels for BCEWithLogitsLoss
    first_label = batch[0][1]

    if isinstance(first_label, torch.Tensor):
        # Labels are already tensors - stack and preserve dtype
        labels = torch.stack(
            [
                sample[1] if isinstance(sample[1], torch.Tensor) else torch.tensor(sample[1])
                for sample in batch
            ]
        )
        # Ensure shape is (B,) not (B, 1) for compatibility
        labels = labels.view(-1).float()  # Ensure float for BCE loss
    else:
        # Labels are Python scalars - convert to float32 for TUAB
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)

    return data, labels
