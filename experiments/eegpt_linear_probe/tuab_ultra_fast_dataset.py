"""Ultra-fast dataset using memory-mapped numpy arrays."""

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import logging
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class TUABUltraFastDataset(Dataset):
    """Ultra-fast dataset using pre-built memory-mapped arrays."""
    
    def __init__(self, cache_dir: Path, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        # Check for pre-built arrays
        array_file = self.cache_dir / f"{split}_data.npy"
        label_file = self.cache_dir / f"{split}_labels.npy"
        
        if array_file.exists() and label_file.exists():
            # Load memory-mapped arrays
            logger.info(f"Loading memory-mapped arrays for {split}...")
            self.X = np.load(array_file, mmap_mode='r')
            self.y = np.load(label_file, mmap_mode='r')
            logger.info(f"Loaded {len(self.X)} windows for {split} split")
        else:
            # Build arrays if they don't exist
            logger.info(f"Building arrays for {split}...")
            self._build_arrays()
    
    def _build_arrays(self):
        """Build memory-mapped arrays from cached .pt files."""
        # Load index
        index_path = self.cache_dir / "index.json"
        with open(index_path) as f:
            index = json.load(f)
        
        # Collect all tensors
        X_list = []
        y_list = []
        
        for file_path, info in tqdm(index["files"].items(), desc=f"Building {self.split} arrays"):
            if self.split in file_path:
                cache_file = self.cache_dir / info["cache_file"]
                data = torch.load(cache_file, weights_only=True)
                X_list.append(data["x"].numpy())
                y_list.append(data["y"].numpy())
        
        # Concatenate and save
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # Save as memory-mapped arrays
        array_file = self.cache_dir / f"{self.split}_data.npy"
        label_file = self.cache_dir / f"{self.split}_labels.npy"
        
        np.save(array_file, X)
        np.save(label_file, y)
        
        logger.info(f"Saved {len(X)} windows to {array_file}")
        
        # Load as memory-mapped
        self.X = np.load(array_file, mmap_mode='r')
        self.y = np.load(label_file, mmap_mode='r')
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Convert to tensor on access
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = int(self.y[idx])
        return x, y