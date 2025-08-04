#!/usr/bin/env python3
"""Check complete validation set class balance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import torch
from custom_collate_fixed import collate_eeg_batch_fixed
from torch.utils.data import DataLoader

from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset

print("Loading validation dataset...")
val_ds = TUABCachedDataset(
    split="eval",
    cache_dir=Path("data/cache/tuab_enhanced"),
    root_dir=Path("data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf"),
    window_duration=8.0,
    window_stride=4.0,
)

print(f"Total validation samples: {len(val_ds)}")

# Check ALL labels
labels = []
loader = DataLoader(val_ds, batch_size=256, num_workers=0, collate_fn=collate_eeg_batch_fixed)

print("Checking all labels...")
for i, (_data, label) in enumerate(loader):
    labels.append(label)
    if i % 10 == 0:
        print(f"Processed {i * 256} samples...")

labels = torch.cat(labels)
print("\nFULL VALIDATION SET:")
print(f"Class 0 (normal): {(labels == 0).sum().item()}")
print(f"Class 1 (abnormal): {(labels == 1).sum().item()}")
print(f"Total samples: {len(labels)}")

# Check file-level distribution
print("\nChecking file-level class distribution...")
normal_files = set()
abnormal_files = set()

for sample in val_ds.samples:
    file_idx = sample["file_idx"]
    file_info = val_ds.file_list[file_idx]
    if sample["label"] == 0:
        normal_files.add(file_info["path"])
    else:
        abnormal_files.add(file_info["path"])

print(f"Normal files: {len(normal_files)}")
print(f"Abnormal files: {len(abnormal_files)}")
