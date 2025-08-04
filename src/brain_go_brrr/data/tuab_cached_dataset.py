"""TUAB dataset with cached metadata for FAST loading."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import mne
from collections import defaultdict

from .tuab_dataset import TUABDataset
from .tuab_enhanced_dataset import TUABEnhancedDataset

logger = logging.getLogger(__name__)


class TUABCachedDataset(TUABDataset):
    """TUAB dataset that uses cached metadata to avoid scanning files.
    
    This dataset loads instantly by using pre-computed file metadata.
    """
    
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        sampling_rate: int = 256,
        window_duration: float = 30.0,
        window_stride: float = 30.0,
        preload: bool = False,
        normalize: bool = True,
        cache_dir: Path | None = None,
        cache_index_path: Path | None = None,
        max_files: Optional[int] = None,  # For testing!
    ) -> None:
        """Initialize cached TUAB dataset.
        
        Args:
            cache_index_path: Path to tuab_index.json file
            max_files: Limit number of files (for testing)
            ... (other args same as parent)
        """
        # Store params before parent init
        self.root_dir = Path(root_dir)
        self.split = split
        self.sampling_rate = sampling_rate
        self.window_duration = window_duration
        self.window_stride = window_stride
        self.window_samples = int(window_duration * sampling_rate)
        self.stride_samples = int(window_stride * sampling_rate)
        self.preload = preload
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.max_files = max_files
        
        # Set MNE log level
        mne.set_log_level("ERROR")
        
        # Load from cache instead of scanning
        if cache_index_path is None:
            cache_index_path = Path("tuab_index.json")
        
        if not cache_index_path.exists():
            raise ValueError(
                f"Cache index not found: {cache_index_path}\n"
                f"Run: python scripts/build_tuab_index.py"
            )
        
        logger.info(f"Loading cached metadata from {cache_index_path}")
        with open(cache_index_path, "r") as f:
            index = json.load(f)
        
        # Initialize collections
        self.samples = []
        self.file_list = []
        self.class_counts = defaultdict(int)
        
        # Process cached metadata
        split_data = index["files"].get(split, {})
        file_count = 0
        
        for class_name in ["normal", "abnormal"]:
            files = split_data.get(class_name, [])
            label = self.LABEL_MAP[class_name]
            
            # Apply file limit for testing
            if self.max_files is not None:
                files = files[:self.max_files // 2]
                logger.info(f"LIMITED to {len(files)} {class_name} files")
            
            for file_info in files:
                # Calculate windows from cached duration
                duration = file_info["duration"]
                n_windows = int((duration - self.window_duration) / self.window_stride) + 1
                
                if n_windows > 0:
                    # Store file info
                    self.file_list.append({
                        "path": self.root_dir / file_info["path"],
                        "label": label,
                        "class_name": class_name,
                        "n_windows": n_windows,
                        "duration": duration,
                        "sfreq": file_info["sfreq"],
                    })
                    
                    # Add window indices
                    for window_idx in range(n_windows):
                        self.samples.append({
                            "file_idx": file_count,
                            "window_idx": window_idx,
                            "label": label,
                            "class_name": class_name,
                        })
                    
                    self.class_counts[class_name] += n_windows
                    file_count += 1
        
        logger.info(
            f"Loaded TUAB {split} split from cache: {len(self.samples)} windows from "
            f"{len(self.file_list)} files ({dict(self.class_counts)})"
        )
        
        # Initialize file cache
        self._file_cache = {}
        self._cache_size = 100
        
        # Preload if requested
        if self.preload:
            logger.info("Preloading all data into memory...")
            self._preload_data()
    
    def _collect_samples(self) -> None:
        """Override parent method - we already collected in __init__."""
        pass  # Already done in __init__ from cache!
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a sample with CONSISTENT window size.
        
        Override parent to ensure all windows are the same size!
        """
        # Get base window from parent
        eeg_tensor, label = super().__getitem__(idx)
        
        # FORCE to expected size
        target_samples = self.window_samples  # Should be 1024 for 5.12s @ 200Hz
        current_samples = eeg_tensor.shape[1]
        
        if current_samples != target_samples:
            # Pad or trim to target size
            if current_samples < target_samples:
                # Pad with zeros
                pad_size = target_samples - current_samples
                eeg_tensor = torch.nn.functional.pad(eeg_tensor, (0, pad_size), mode='constant', value=0)
            else:
                # Trim to size
                eeg_tensor = eeg_tensor[:, :target_samples]
        
        # Add safety check for NaN/Inf
        if not torch.isfinite(eeg_tensor).all():
            raise RuntimeError(f"NaN/Inf detected in sample {idx} after loading")
        
        # Check for extreme values that indicate bad normalization
        max_val = eeg_tensor.abs().max()
        if max_val > 1000:
            raise RuntimeError(f"Extreme values detected in sample {idx}: max={max_val:.2f}. Check normalization.")
        
        return eeg_tensor, label