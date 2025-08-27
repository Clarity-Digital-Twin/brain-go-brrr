"""TUEV Cached Dataset with automatic padding to 1024 samples."""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TUEVCachedDatasetPadded(Dataset):
    """TUEV cached dataset that pads from 1000 to 1024 samples for EEGPT."""

    def __init__(
        self,
        cache_dir: Path,
        split: str = 'train',
        padding: str = 'edge',  # 'edge' or 'zero'
    ):
        """Initialize cached dataset with padding.

        Args:
            cache_dir: Path to cache directory
            split: 'train' or 'eval'
            padding: Type of padding ('edge' repeats last values, 'zero' adds zeros)
        """
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.split_cache = self.cache_dir / f"tuev_{split}_cache"
        self.padding = padding

        if not self.split_cache.exists():
            raise ValueError(f"Cache not found at {self.split_cache}")

        # Load index
        index_file = self.split_cache / "index.json"
        if not index_file.exists():
            raise ValueError(f"Index file not found at {index_file}")

        with open(index_file, 'r') as f:
            self.index = json.load(f)

        self.samples = self.index['samples']
        self.class_counts = self.index['class_counts']

        logger.info(f"Loaded cached dataset with {len(self.samples)} samples")
        logger.info(f"Will pad from 1000 to 1024 samples using '{padding}' padding")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get padded sample.

        Returns:
            Tuple of (data, label) where data is shape (23, 1024)
        """
        sample_info = self.samples[idx]
        cache_file = self.split_cache / sample_info['cache_file']

        # Load cached data
        data = torch.load(cache_file, weights_only=True)
        x = data['x']  # Shape: (23, 1000)
        y = data['y']

        # Pad from 1000 to 1024
        if self.padding == 'edge':
            # Repeat last 24 samples
            padding = x[:, -24:]
            x_padded = torch.cat([x, padding], dim=1)
        elif self.padding == 'zero':
            # Add 24 zeros
            padding = torch.zeros(23, 24, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            raise ValueError(f"Unknown padding type: {self.padding}")

        assert x_padded.shape == (23, 1024), f"Wrong shape after padding: {x_padded.shape}"

        return x_padded, y

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced loss."""
        # Convert dict to list if needed
        if isinstance(self.class_counts, dict):
            counts_list = [self.class_counts[str(i)] for i in range(6)]
        else:
            counts_list = self.class_counts

        counts = torch.tensor(counts_list, dtype=torch.float32)
        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-8)
        # Normalize so mean weight is 1
        weights = weights / weights.mean()
        return weights
