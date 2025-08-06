"""Simple and FAST cached dataset that actually works."""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TUABSimpleCachedDataset(Dataset):
    """Dead simple dataset that loads cached .pt files directly."""
    
    def __init__(self, cache_dir: Path, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        # Load index
        index_path = self.cache_dir / "index.json"
        with open(index_path) as f:
            self.index = json.load(f)
        
        # Build file list for this split
        self.files = []
        for file_path, info in self.index["files"].items():
            if split in file_path:
                cache_file = self.cache_dir / info["cache_file"]
                n_windows = info["n_windows"]
                
                # Each file has multiple windows
                for window_idx in range(n_windows):
                    self.files.append((cache_file, window_idx))
        
        logger.info(f"Loaded {len(self.files)} windows for {split} split")
        
        # Keep cache of loaded files to avoid repeated I/O
        self._cache = {}
        self._cache_size = 100  # Cache last 100 files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        cache_file, window_idx = self.files[idx]
        
        # Use cached data if available
        if cache_file not in self._cache:
            # Load file
            data = torch.load(cache_file, weights_only=True)
            
            # Maintain cache size
            if len(self._cache) >= self._cache_size:
                # Remove oldest
                self._cache.pop(next(iter(self._cache)))
            
            self._cache[cache_file] = data
        else:
            data = self._cache[cache_file]
        
        # Get window
        x = data["x"][window_idx]  # Shape: (20, 1024)
        y = data["y"][window_idx].item()
        
        return x, y