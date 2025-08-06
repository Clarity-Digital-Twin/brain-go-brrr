"""Simple collate function for cached TUAB dataset with consistent 20 channels."""

import torch
from typing import List, Tuple


def collate_eeg_batch_fixed(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple collate function for cached data with consistent 20 channels.
    
    All cached samples have exactly 20 channels after preprocessing,
    so we just need to stack them into a batch.
    """
    # Stack data and labels
    data = torch.stack([sample[0] for sample in batch])
    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    
    return data, labels