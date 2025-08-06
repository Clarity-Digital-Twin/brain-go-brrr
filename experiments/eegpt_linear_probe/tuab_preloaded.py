"""TRULY pre-loaded dataset for FAST training."""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TUABPreloadedDataset(Dataset):
    """Pre-loads ALL windows into memory for 400+ it/s training."""
    
    def __init__(self, cache_dir: Path, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        # Load index
        index_path = self.cache_dir / "index.json"
        with open(index_path) as f:
            self.index = json.load(f)
        
        # Pre-load ALL data
        logger.info(f"Pre-loading ALL {split} windows into memory...")
        logger.info("This will take 2-3 minutes but then training will be FAST")
        
        X_list = []
        y_list = []
        
        # Load all files for this split
        for file_path, info in tqdm(self.index["files"].items(), desc=f"Loading {split}"):
            if split in file_path:
                cache_file = self.cache_dir / info["cache_file"]
                data = torch.load(cache_file, weights_only=True)
                X_list.append(data["x"])
                y_list.append(data["y"])
        
        # Concatenate everything
        self.X = torch.cat(X_list, dim=0)
        self.y = torch.cat(y_list, dim=0)
        
        logger.info(f"Pre-loaded {len(self.X)} windows into memory")
        logger.info(f"Memory usage: ~{self.X.nbytes / 1e9:.1f} GB")
        
        # Count classes
        n_normal = (self.y == 0).sum().item()
        n_abnormal = (self.y == 1).sum().item()
        logger.info(f"Class distribution: normal={n_normal}, abnormal={n_abnormal}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Direct memory access - FAST!
        return self.X[idx], self.y[idx].item()