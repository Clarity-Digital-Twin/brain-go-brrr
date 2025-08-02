"""Custom collate function for handling channel and window size mismatches."""

import torch
from typing import List, Tuple


def collate_eeg_batch(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function that handles channel and window size mismatches.
    
    Forces all samples to have the same dimensions by padding/trimming.
    """
    # Get target dimensions from first sample
    target_channels = 19  # Standard EEGPT channels (without OZ sometimes)
    target_samples = 1024  # 5.12s @ 200Hz
    
    # Process each sample
    processed_batch = []
    labels = []
    
    for eeg_data, label in batch:
        # Handle channel dimension
        n_channels, n_samples = eeg_data.shape
        
        if n_channels != target_channels:
            # Create padded/trimmed tensor
            processed_eeg = torch.zeros(target_channels, n_samples, dtype=eeg_data.dtype)
            if n_channels < target_channels:
                # Pad with zeros
                processed_eeg[:n_channels] = eeg_data
            else:
                # Trim excess channels
                processed_eeg = eeg_data[:target_channels]
        else:
            processed_eeg = eeg_data
        
        # Handle time dimension
        if n_samples != target_samples:
            if n_samples < target_samples:
                # Pad with zeros
                padded = torch.zeros(target_channels, target_samples, dtype=eeg_data.dtype)
                padded[:, :n_samples] = processed_eeg
                processed_eeg = padded
            else:
                # Trim to size
                processed_eeg = processed_eeg[:, :target_samples]
        
        processed_batch.append(processed_eeg)
        labels.append(label)
    
    # Stack into batch tensors
    batch_eeg = torch.stack(processed_batch, dim=0)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_eeg, batch_labels