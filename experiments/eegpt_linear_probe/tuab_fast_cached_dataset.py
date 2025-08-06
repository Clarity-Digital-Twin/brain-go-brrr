"""FAST cached dataset that pre-loads all windows into memory."""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TUABFastCachedDataset(Dataset):
    """Pre-load ALL windows into memory for FAST training."""
    
    def __init__(self, cache_dir: Path, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        # Load index
        index_path = self.cache_dir / "index.json"
        with open(index_path) as f:
            self.index = json.load(f)
        
        # Pre-load all windows
        logger.info(f"Pre-loading {split} windows into memory...")
        self.X = []
        self.y = []
        
        for file_path, info in tqdm(self.index["files"].items(), desc=f"Loading {split}"):
            if split in file_path:
                cache_file = self.cache_dir / info["cache_file"]
                
                # Load entire file
                data = torch.load(cache_file, weights_only=True)
                
                # Add all windows
                self.X.append(data["x"])  # Shape: (n_windows, 20, 1024)
                self.y.append(data["y"])  # Shape: (n_windows,)
        
        # Concatenate all
        self.X = torch.cat(self.X, dim=0)
        self.y = torch.cat(self.y, dim=0)
        
        logger.info(f"Loaded {len(self.X)} windows for {split} split")
        
        # Count classes
        n_normal = (self.y == 0).sum().item()
        n_abnormal = (self.y == 1).sum().item()
        logger.info(f"Class distribution: normal={n_normal}, abnormal={n_abnormal}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].item()