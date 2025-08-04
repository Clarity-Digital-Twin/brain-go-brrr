#!/usr/bin/env python3
"""Test dataloader to find bad samples."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from torch.utils.data import DataLoader
from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from experiments.eegpt_linear_probe.custom_collate_fixed import collate_eeg_batch_fixed

print("Testing dataloader...")
ds = TUABCachedDataset(
    split="train",
    cache_dir=Path("data/cache/tuab_enhanced"),
    root_dir=Path("data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf"),
    window_duration=8.0, 
    window_stride=4.0
)

print(f"Dataset has {len(ds)} samples")

loader = DataLoader(ds, batch_size=32, num_workers=0, collate_fn=collate_eeg_batch_fixed)

print("Loading batches...")
for i, (x, y) in enumerate(loader):
    print(f"batch {i} OK  shape={x.shape}")
    if i == 2: 
        print("SUCCESS! DataLoader works fine!")
        break