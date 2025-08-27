"""Simple collate function for cached TUAB dataset with consistent 19 channels."""

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
    # Stack data - handle 19 vs 20 channel mismatch by truncating to 19
    # This is a temporary fix until cache is rebuilt
    processed_samples = []
    for sample in batch:
        x = sample[0]
        if x.shape[0] == 20:
            # Drop Fz channel (typically channel 4) to get 19 channels
            # Standard 10-20 order: Fp1, Fp2, F7, F3, Fz, F4, F8, ...
            # We want to exclude Fz (index 4)
            x = torch.cat([x[:4], x[5:]], dim=0)  # Skip channel 4 (Fz)
        elif x.shape[0] != 19:
            raise RuntimeError(f"Unexpected channel count: {x.shape[0]}. Expected 19 or 20.")
        processed_samples.append(x)

    data = torch.stack(processed_samples)

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
