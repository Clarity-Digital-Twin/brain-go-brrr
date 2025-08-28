#!/usr/bin/env python3
"""Diagnose cache channel inconsistency."""

import json
from pathlib import Path

import torch
from tqdm import tqdm

cache_dir = Path("../../data/cache/tuab_mne_preprocessed")

# Load indices
for split in ["train", "eval"]:
    print(f"\nChecking {split} split...")
    index_file = cache_dir / f"index_{split}_mne-ar-v2.json"

    with open(index_file) as f:
        index = json.load(f)

    print(f"Total windows: {index['total_windows']}")

    # Check a sample of cache files
    bad_files = []
    channel_counts = {}

    windows = index['windows']
    for window_id in tqdm(list(windows.keys())[:1000], desc=f"Checking {split}"):
        cache_file = cache_dir / windows[window_id]['cache_file']
        if cache_file.exists():
            data = torch.load(cache_file, map_location='cpu', weights_only=True)
            shape = data['x'].shape
            channels = shape[0]
            channel_counts[channels] = channel_counts.get(channels, 0) + 1

            if channels != 19:
                bad_files.append((cache_file.name, shape))

    print("Channel distribution in first 1000 files:")
    for channels, count in sorted(channel_counts.items()):
        print(f"  {channels} channels: {count} files")

    if bad_files:
        print("\nFiles with wrong channel count:")
        for name, shape in bad_files[:10]:  # Show first 10
            print(f"  {name}: {shape}")
