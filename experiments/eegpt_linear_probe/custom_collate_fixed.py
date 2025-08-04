"""Fixed custom collate function for handling variable channel counts."""

import torch
from typing import List, Tuple


def collate_eeg_batch_fixed(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function that handles variable channel counts by padding.
    
    Some cached samples have 19 channels, others have 20. This function
    ensures all samples have the same number of channels by padding with zeros.
    """
    # First check for NaN/Inf in input data
    for i, (data, label) in enumerate(batch):
        if not torch.isfinite(data).all():
            raise RuntimeError(f"NaN/Inf detected in batch sample {i} before collation")
        if data.abs().max() > 1000:
            raise RuntimeError(f"Extreme values in batch sample {i}: max={data.abs().max():.2f}")
    
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