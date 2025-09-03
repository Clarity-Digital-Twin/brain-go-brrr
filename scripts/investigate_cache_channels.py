#!/usr/bin/env python3
"""Deep investigation of TUAB cache channel counts - EVIDENCE BASED."""

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def investigate_cache():
    """Scan cache for actual channel counts - NO GUESSING."""
    cache_dir = Path("data/cache/tuab_mne_v2")

    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        return

    print(f"📂 Investigating cache: {cache_dir}")

    # Count .pkl files (cache uses pickle format)
    pkl_files = list(cache_dir.glob("*.pkl"))
    print(f"📊 Total .pkl files: {len(pkl_files)}")

    if len(pkl_files) == 0:
        print("❌ No .pkl files found!")
        return

    # Sample some files to check channel counts
    channel_counts = Counter()
    problematic_files = []

    # Check first 100 files and any that match the problematic pattern
    files_to_check = pkl_files[:100]

    # Also specifically check for the known problematic files
    for pattern in ["*aaaaakfo_s004*", "*aaaaakfo_s005*"]:
        files_to_check.extend(cache_dir.glob(pattern))

    files_to_check = list(set(files_to_check))  # Remove duplicates

    print(f"🔍 Checking {len(files_to_check)} files for channel counts...")

    for i, pt_file in enumerate(files_to_check):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(files_to_check)}")

        try:
            # Load pickle file
            with pt_file.open('rb') as f:
                data = pickle.load(f)

            # Handle both dict format and direct tensor
            if isinstance(data, dict) and 'x' in data:
                n_channels = data['x'].shape[0]
            elif isinstance(data, np.ndarray | torch.Tensor):
                n_channels = data.shape[0] if len(data.shape) >= 2 else None
            else:
                continue

            if n_channels:
                channel_counts[n_channels] += 1

                if n_channels == 20:
                    problematic_files.append((pt_file.name, n_channels))
                    print(f"  ⚠️ Found 20-channel file: {pt_file.name}")
                elif n_channels != 19:
                    problematic_files.append((pt_file.name, n_channels))
                    print(f"  ⚠️ Unexpected channel count {n_channels}: {pt_file.name}")

        except Exception as e:
            print(f"  ❌ Error loading {pt_file.name}: {e}")

    print("\n📊 CHANNEL COUNT DISTRIBUTION:")
    for n_channels, count in sorted(channel_counts.items()):
        marker = "✅" if n_channels == 19 else "⚠️"
        print(f"  {marker} {n_channels} channels: {count} files")

    if problematic_files:
        print(f"\n⚠️ PROBLEMATIC FILES ({len(problematic_files)}):")
        for fname, n_ch in problematic_files[:10]:  # Show first 10
            print(f"  - {fname}: {n_ch} channels")
        if len(problematic_files) > 10:
            print(f"  ... and {len(problematic_files) - 10} more")
    else:
        print("\n✅ NO PROBLEMATIC FILES FOUND!")

    # Check specifically for the files mentioned in tech debt
    print("\n🔍 CHECKING SPECIFIC FILES FROM TECH DEBT:")
    for pattern in ["aaaaakfo_s004", "aaaaakfo_s005"]:
        matching = [f for f in pkl_files if pattern in f.name]
        if matching:
            print(f"  Found {len(matching)} files matching {pattern}")
            for f in matching[:5]:
                try:
                    with f.open('rb') as file:
                        data = pickle.load(file)
                    if isinstance(data, dict) and 'x' in data:
                        print(f"    {f.name}: {data['x'].shape[0]} channels")
                    elif isinstance(data, np.ndarray | torch.Tensor):
                        print(f"    {f.name}: {data.shape[0]} channels")
                except Exception as e:
                    print(f"    Error loading {f.name}: {e}")
        else:
            print(f"  ❌ No files matching {pattern}")

    return channel_counts, problematic_files


if __name__ == "__main__":
    print("=" * 60)
    print("TUAB CACHE CHANNEL INVESTIGATION - EVIDENCE ONLY")
    print("=" * 60)

    channel_counts, problematic = investigate_cache()

    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT:")
    if 20 in channel_counts:
        print(f"⚠️ CONTAMINATION CONFIRMED: {channel_counts[20]} files have 20 channels")
        print("⚠️ COLLATE WORKAROUND IS NECESSARY!")
    else:
        print("✅ NO 20-CHANNEL CONTAMINATION FOUND")
        print("✅ COLLATE WORKAROUND MAY BE OBSOLETE")
    print("=" * 60)
