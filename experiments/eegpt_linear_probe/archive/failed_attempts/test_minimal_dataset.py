#!/usr/bin/env python3
"""Test EEGPT training with MINIMAL dataset to bypass WSL2 bottleneck."""

import os
import sys
from pathlib import Path
import time
import torch
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.brain_go_brrr.data.tuab_dataset import TUABDataset
from torch.utils.data import DataLoader, Subset
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_minimal_loading():
    """Test with only 10 files to see if DataLoader works at all."""
    
    logger.info("=" * 80)
    logger.info("TESTING MINIMAL DATASET LOADING")
    logger.info("=" * 80)
    
    # Configuration
    data_root = Path(os.environ.get("BGB_DATA_ROOT", "data"))
    tuab_dir = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    
    # Check if directory exists
    if not tuab_dir.exists():
        logger.error(f"TUAB directory not found: {tuab_dir}")
        return
    
    # Create dataset with AGGRESSIVE limits
    logger.info("Creating dataset with file limit...")
    
    # Monkey patch to limit files
    original_collect = TUABDataset._collect_samples
    
    def limited_collect(self):
        """Collect only first 10 files."""
        self.file_list = []
        self.class_counts = {}
        self.samples = []
        
        file_count = 0
        for class_name in ["normal", "abnormal"]:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
                
            # LIMIT TO 5 FILES PER CLASS
            edf_files = list(class_dir.glob("**/*.edf"))[:5]
            logger.info(f"Loading {len(edf_files)} {class_name} files (LIMITED)")
            
            for edf_file in edf_files:
                self.file_list.append({
                    "path": edf_file,
                    "label": 0 if class_name == "normal" else 1,
                    "class_name": class_name,
                    "n_windows": 10  # Fake it
                })
                
                # Add only 10 windows per file
                for i in range(10):
                    self.samples.append({
                        "file_idx": file_count,
                        "window_idx": i,
                        "label": 0 if class_name == "normal" else 1,
                        "class_name": class_name,
                    })
                file_count += 1
                
        self.class_counts = {"normal": 50, "abnormal": 50}
        logger.info(f"LIMITED dataset: {len(self.samples)} windows from {len(self.file_list)} files")
    
    # Apply monkey patch
    TUABDataset._collect_samples = limited_collect
    
    try:
        # Create dataset
        start = time.time()
        dataset = TUABDataset(
            root_dir=tuab_dir,
            split="train",
            sampling_rate=256,
            window_duration=8.0,
            window_stride=8.0,
            preload=False,
            normalize=True,
        )
        logger.info(f"Dataset created in {time.time() - start:.2f}s")
        logger.info(f"Dataset size: {len(dataset)} windows")
        
        # Create DataLoader
        logger.info("\nCreating DataLoader...")
        start = time.time()
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # CRITICAL for WSL2
            pin_memory=False,
            prefetch_factor=None,  # Disable prefetching
        )
        logger.info(f"DataLoader created in {time.time() - start:.2f}s")
        
        # Test loading one batch
        logger.info("\nLoading first batch...")
        start = time.time()
        for i, (data, labels) in enumerate(loader):
            logger.info(f"Batch {i}: data shape={data.shape}, labels shape={labels.shape}")
            logger.info(f"First batch loaded in {time.time() - start:.2f}s")
            
            # Check data
            logger.info(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
            logger.info(f"Labels: {labels.tolist()}")
            
            if i >= 2:  # Load 3 batches
                break
                
        logger.info("\n✅ SUCCESS! DataLoader works with minimal dataset!")
        
    except Exception as e:
        logger.error(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original method
        TUABDataset._collect_samples = original_collect

def test_lazy_wrapper():
    """Test a lazy-loading wrapper that doesn't scan all files upfront."""
    
    logger.info("\n" + "=" * 80)
    logger.info("TESTING LAZY LOADING APPROACH")
    logger.info("=" * 80)
    
    class LazyTUABDataset:
        """Lazy dataset that only loads file paths, not metadata."""
        
        def __init__(self, root_dir, split="train", max_files=None):
            self.root_dir = Path(root_dir)
            self.split_dir = self.root_dir / split
            self.max_files = max_files
            
            # Just collect file paths, don't open them!
            self.files = []
            for class_name in ["normal", "abnormal"]:
                class_dir = self.split_dir / class_name
                if class_dir.exists():
                    files = list(class_dir.glob("**/*.edf"))
                    if self.max_files:
                        files = files[:self.max_files//2]
                    self.files.extend([(f, class_name) for f in files])
            
            logger.info(f"Found {len(self.files)} files (lazy mode)")
            
            # Assume 100 windows per file (we'll handle errors later)
            self.windows_per_file = 100
            
        def __len__(self):
            return len(self.files) * self.windows_per_file
            
        def __getitem__(self, idx):
            # Figure out which file and window
            file_idx = idx // self.windows_per_file
            window_idx = idx % self.windows_per_file
            
            if file_idx >= len(self.files):
                raise IndexError(f"Index {idx} out of range")
                
            file_path, class_name = self.files[file_idx]
            
            # For testing, just return random data
            data = np.random.randn(20, 2048).astype(np.float32)
            label = 0 if class_name == "normal" else 1
            
            return torch.from_numpy(data), torch.tensor(label)
    
    # Test it
    data_root = Path(os.environ.get("BGB_DATA_ROOT", "data"))
    tuab_dir = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    
    start = time.time()
    dataset = LazyTUABDataset(tuab_dir, split="train", max_files=20)
    logger.info(f"Lazy dataset created in {time.time() - start:.2f}s")
    
    # Create loader
    loader = DataLoader(dataset, batch_size=16, num_workers=0)
    
    # Test loading
    start = time.time()
    for i, (data, labels) in enumerate(loader):
        if i == 0:
            logger.info(f"First batch loaded in {time.time() - start:.2f}s")
            logger.info(f"Batch shape: {data.shape}")
        if i >= 5:
            break
            
    logger.info("\n✅ Lazy loading works instantly!")

if __name__ == "__main__":
    # Test both approaches
    test_minimal_loading()
    test_lazy_wrapper()
    
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS:")
    logger.info("1. Use lazy loading for large datasets on WSL2")
    logger.info("2. Cache file metadata to avoid re-scanning")
    logger.info("3. Or just use a real Linux machine FFS")
    logger.info("=" * 80)