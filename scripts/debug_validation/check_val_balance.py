#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import torch
from custom_collate_fixed import collate_eeg_batch_fixed
from torch.utils.data import DataLoader

from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset

val_ds = TUABCachedDataset(
    split="eval",
    cache_dir=Path("data/cache/tuab_enhanced"),
    root_dir=Path("data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf"),
    window_duration=8.0,
    window_stride=4.0,
)

labels = []
loader = DataLoader(val_ds, batch_size=128, num_workers=0, collate_fn=collate_eeg_batch_fixed)
for i, (data, label) in enumerate(loader):
    labels.append(label)
    if i >= 10:  # First 10 batches = 1280 samples
        break

labels = torch.cat(labels)
print("Val set counts  -> 0:", (labels == 0).sum().item(), " 1:", (labels == 1).sum().item())
print("Total samples checked:", len(labels))
