"""Simple collate function for cached TUAB dataset with consistent 20 channels."""

import torch


def collate_eeg_batch_fixed(
    batch: list[tuple[torch.Tensor, torch.Tensor | int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple collate function for cached data with consistent channels.

    Handles both tensor and integer labels for compatibility.
    MNE cache stores labels as tensors, so we need to handle both cases.
    """
    # Stack data
    data = torch.stack([sample[0] for sample in batch])
    
    # Handle labels - they might be tensors or ints
    labels_list = []
    for sample in batch:
        label = sample[1]
        if isinstance(label, torch.Tensor):
            # Convert tensor to int (handles both scalar tensors and 0-d tensors)
            labels_list.append(label.item())
        else:
            # Already an int
            labels_list.append(label)
    
    labels = torch.tensor(labels_list, dtype=torch.long)

    return data, labels
