"""Simple collate function for cached TUAB dataset with consistent 20 channels."""

import torch


def collate_eeg_batch_fixed(
    batch: list[tuple[torch.Tensor, torch.Tensor | int | float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple collate function that preserves label dtype from dataset.

    Works for both:
    - TUAB: BCEWithLogitsLoss expects float labels (0.0, 1.0)
    - TUEV: CrossEntropyLoss expects long labels (0, 1, 2, 3, 4, 5)
    
    The key insight: preserve the dtype from the cached data!
    """
    # Stack data
    data = torch.stack([sample[0] for sample in batch])
    
    # Handle labels - preserve dtype from dataset
    first_label = batch[0][1]
    
    if isinstance(first_label, torch.Tensor):
        # Labels are already tensors - stack and preserve dtype
        labels = torch.stack([sample[1] for sample in batch])
        # Ensure shape is (B,) not (B, 1) for compatibility
        labels = labels.view(-1)
    else:
        # Labels are Python scalars (int or float)
        # Infer dtype from the first label
        if isinstance(first_label, float):
            labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
        else:
            labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)

    return data, labels
