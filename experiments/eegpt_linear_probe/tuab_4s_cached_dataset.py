"""Simple dataset for loading cached 4s TUAB windows."""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TUAB4sCachedDataset(Dataset):
    """Load pre-cached 4s TUAB windows directly from .pt files."""
    
    def __init__(self, cache_dir: Path, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        # Load index
        index_path = self.cache_dir / "index.json"
        with open(index_path) as f:
            self.index = json.load(f)
        
        # Filter files by split
        self.samples = []
        for file_path, info in self.index["files"].items():
            if split in file_path:  # Simple split detection
                cache_file = self.cache_dir / info["cache_file"]
                n_windows = info["n_windows"]
                # FIXED: Check the actual label from the cached tensor
                label = info.get("label", 0 if "normal" in file_path else 1)
                
                # Add all windows from this file
                for window_idx in range(n_windows):
                    self.samples.append((cache_file, window_idx, label))
        
        logger.info(f"Loaded {len(self.samples)} windows for {split} split")
        
        # Count classes
        n_normal = sum(1 for _, _, label in self.samples if label == 0)
        n_abnormal = len(self.samples) - n_normal
        logger.info(f"Class distribution: normal={n_normal}, abnormal={n_abnormal}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        cache_file, window_idx, _ = self.samples[idx]
        
        # Load cached tensor
        data = torch.load(cache_file, weights_only=True)
        
        # Get specific window and its label
        x = data["x"][window_idx]  # Shape: (20, 1024)
        y = data["y"][window_idx].item()  # Get actual label from cache
        
        return x, y