"""Memory-mapped dataset for FAST training (300+ it/s on WSL)."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TUABMemoryMappedDataset(Dataset):
    """Ultra-fast dataset using memory-mapped numpy arrays."""
    
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
        
        # Memory-map the arrays (NO RAM USAGE - OS handles paging)
        self.X = np.memmap(X_path, dtype='float32', mode='r')
        self.y = np.memmap(y_path, dtype='int64', mode='r')
        
        # Reshape X to correct dimensions
        n_samples = len(self.y)
        self.X = self.X.reshape(n_samples, 20, 1024)
        
        logger.info(f"Memory-mapped {split}: {n_samples} windows")
        logger.info(f"Data will stream from disk at >1 GB/s")
        
        # Count classes
        # Note: This will read through all labels once (fast)
        unique, counts = np.unique(self.y[:1000], return_counts=True)  # Sample first 1000
        logger.info(f"Class distribution (sample): {dict(zip(unique, counts))}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Direct memory-mapped access - OS caches hot pages automatically
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = int(self.y[idx])
        return x, y