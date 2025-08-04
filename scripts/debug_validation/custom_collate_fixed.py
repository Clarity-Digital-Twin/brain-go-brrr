"""Fixed custom collate function for handling variable channel counts."""

import torch


def collate_eeg_batch_fixed(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function that handles variable channel counts by padding.

    Some cached samples have 19 channels, others have 20. This function
    ensures all samples have the same number of channels by padding with zeros.
    """
    # Find the maximum number of channels
    max_channels = max(sample[0].shape[0] for sample in batch)

    # Pad all samples to have the same number of channels
    padded_data = []
    labels = []

    for data, label in batch:
        n_channels, n_samples = data.shape

        if n_channels < max_channels:
            # Pad with zeros for missing channels
            padding = torch.zeros(max_channels - n_channels, n_samples, dtype=data.dtype)
            data = torch.cat([data, padding], dim=0)

        padded_data.append(data)
        labels.append(label)

    # Stack into batch tensors
    batch_data = torch.stack(padded_data)
    batch_labels = torch.tensor(labels, dtype=torch.long)

    return batch_data, batch_labels
