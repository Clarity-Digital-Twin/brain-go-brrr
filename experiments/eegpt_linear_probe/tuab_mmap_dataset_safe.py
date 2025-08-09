"""Memory-mapped dataset for FAST training - WSL SAFE VERSION."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


class TUABMemoryMappedDatasetSafe(Dataset):
    """Ultra-fast dataset using memory-mapped numpy arrays - WSL SAFE.
    
    This version is optimized for WSL and avoids multiprocessing issues.
    """
    
    def __init__(self, cache_dir: Path, split: str = "train"):
        self.cache_dir = Path(cache_dir)
        self.split = split
        
        # Load memory-mapped arrays
        X_path = self.cache_dir / f"{split}_data.npy"
        y_path = self.cache_dir / f"{split}_labels.npy"
        
        if not X_path.exists():
            raise FileNotFoundError(
                f"Memory-mapped arrays not found at {X_path}\n"
                f"Run: python build_mmap_cache.py"
            )
        
        # Store paths instead of opening memmaps in __init__
        # This avoids issues with forking in multiprocessing
        self.X_path = X_path
        self.y_path = y_path
        
        # Get dataset size without opening memmap
        # Load just the labels to get length (small file)
        temp_y = np.memmap(y_path, dtype='int64', mode='r')
        self.length = len(temp_y)
        del temp_y  # Close immediately
        
        # Store shape info
        self.data_shape = (20, 1024)  # channels, samples
        
        # Lazy load the memmaps (will be done in worker processes)
        self._X = None
        self._y = None
        
        logger.info(f"Dataset {split}: {self.length} windows")
        logger.info(f"Using WSL-safe lazy loading strategy")
        
        # Detect if we're in WSL
        if 'microsoft' in os.uname().release.lower():
            logger.warning("WSL detected - use num_workers=0 for stability!")
    
    def _ensure_memmaps(self):
        """Lazily open memory maps (once per process)."""
        if self._X is None:
            self._X = np.memmap(self.X_path, dtype='float32', mode='r')
            self._X = self._X.reshape(self.length, *self.data_shape)
            self._y = np.memmap(self.y_path, dtype='int64', mode='r')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Ensure memmaps are open in this process
        self._ensure_memmaps()
        
        # Direct memory-mapped access without .copy() for speed
        # PyTorch will handle the memory copy when creating tensor
        x = torch.from_numpy(self._X[idx]).float()
        y = int(self._y[idx])
        return x, y
    
    def get_class_weights(self, max_samples=10000):
        """Compute class weights for balanced loss."""
        self._ensure_memmaps()
        
        # Sample subset for speed
        n_samples = min(max_samples, len(self))
        indices = np.random.choice(len(self), n_samples, replace=False)
        
        labels = self._y[indices]
        unique, counts = np.unique(labels, return_counts=True)
        
        # Compute weights (inverse frequency)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique)
        
        class_weights = {int(cls): float(w) for cls, w in zip(unique, weights)}
        logger.info(f"Class weights: {class_weights}")
        
        return class_weights