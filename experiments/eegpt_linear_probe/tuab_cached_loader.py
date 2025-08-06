"""SIMPLE cached dataset that ACTUALLY loads our cached tensors fast."""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TUABCachedLoader(Dataset):
    """Loads cached .pt files with smart caching for speed."""
    
    def __init__(self, cache_dir: Path, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        # Load index
        index_path = self.cache_dir / "index.json"
        with open(index_path) as f:
            self.index = json.load(f)
        
        # Build list of (file, window_idx) for this split
        self.samples = []
        for file_path, info in self.index["files"].items():
            if split in file_path:
                cache_file = self.cache_dir / info["cache_file"]
                n_windows = info["n_windows"]
                for idx in range(n_windows):
                    self.samples.append((cache_file, idx))
        
        logger.info(f"Loaded {len(self.samples)} windows for {split}")
        
        # Smart cache - keep last N loaded files in memory
        self._cache = {}
        self._cache_order = []
        self._max_cache = 50  # Keep 50 files in memory
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        cache_file, window_idx = self.samples[idx]
        
        # Load from cache or disk
        if cache_file not in self._cache:
            # Load file
            data = torch.load(cache_file, weights_only=True)
            
            # Update cache
            if len(self._cache) >= self._max_cache:
                # Remove oldest
                old_file = self._cache_order.pop(0)
                del self._cache[old_file]
            
            self._cache[cache_file] = data
            self._cache_order.append(cache_file)
        
        # Get data from cache
        data = self._cache[cache_file]
        x = data["x"][window_idx]
        y = data["y"][window_idx].item()
        
        return x, y