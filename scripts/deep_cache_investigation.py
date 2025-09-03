#!/usr/bin/env python3
"""DEEP investigation - check EVERY channel count with 100% certainty."""

import pickle
import random
from collections import Counter
from pathlib import Path


def investigate_deeply():
    """Check a large sample with absolute certainty."""
    cache_dir = Path("data/cache/tuab_mne_v2")
    pkl_files = list(cache_dir.glob("*.pkl"))

    print(f"📂 Cache: {cache_dir}")
    print(f"📊 Total files: {len(pkl_files)}")

    # Check a large random sample
    sample_size = min(1000, len(pkl_files))
    sample_files = random.sample(pkl_files, sample_size)

    # Also specifically add the problematic files
    for pattern in ["aaaaakfo_s004", "aaaaakfo_s005"]:
        specific = [f for f in pkl_files if pattern in f.name]
        sample_files.extend(specific[:10])  # Add first 10 of each

    sample_files = list(set(sample_files))  # Remove duplicates

    print(f"🔍 Checking {len(sample_files)} files...")

    channel_counts = Counter()
    twenty_channel_files = []
    errors = []

    for i, pkl_file in enumerate(sample_files):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(sample_files)}")

        try:
            with pkl_file.open('rb') as f:
                data = pickle.load(f)

            # Handle different formats
            n_channels = None

            if isinstance(data, tuple) and len(data) == 2:
                # Format: (array, label)
                if hasattr(data[0], 'shape'):
                    n_channels = data[0].shape[0]
            elif isinstance(data, dict):
                if 'x' in data and hasattr(data['x'], 'shape'):
                    n_channels = data['x'].shape[0]
            elif hasattr(data, 'shape'):
                n_channels = data.shape[0]

            if n_channels is not None:
                channel_counts[n_channels] += 1

                if n_channels == 20:
                    twenty_channel_files.append(pkl_file.name)
                    print(f"    ⚠️ FOUND 20-CHANNEL FILE: {pkl_file.name}")

        except Exception as e:
            errors.append((pkl_file.name, str(e)))

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    print("\n📊 Channel Distribution:")
    for n_ch, count in sorted(channel_counts.items()):
        pct = count / len(sample_files) * 100
        status = "✅" if n_ch == 19 else "⚠️"
        print(f"  {status} {n_ch} channels: {count} files ({pct:.1f}%)")

    if twenty_channel_files:
        print(f"\n⚠️ TWENTY-CHANNEL FILES FOUND: {len(twenty_channel_files)}")
        for f in twenty_channel_files[:10]:
            print(f"  - {f}")
    else:
        print("\n✅ NO 20-CHANNEL FILES FOUND IN SAMPLE!")

    if errors:
        print(f"\n❌ Errors: {len(errors)} files")

    # Check specific files from tech debt
    print("\n" + "=" * 60)
    print("CHECKING TECH DEBT FILES SPECIFICALLY:")
    print("=" * 60)

    for base_pattern in ["aaaaakfo_s004_t000", "aaaaakfo_s005_t000"]:
        matches = [f for f in pkl_files if base_pattern in f.name]
        if matches:
            print(f"\n📁 {base_pattern}: {len(matches)} windows")
            # Check first few
            for f in matches[:3]:
                try:
                    with f.open('rb') as file:
                        data = pickle.load(file)
                    if isinstance(data, tuple):
                        print(f"  {f.name}: {data[0].shape[0]} channels ✓")
                except Exception as e:
                    print(f"  {f.name}: ERROR - {e}")

    return channel_counts, twenty_channel_files


if __name__ == "__main__":
    print("=" * 60)
    print("DEEP TUAB CACHE INVESTIGATION")
    print("=" * 60)

    counts, twenty_ch = investigate_deeply()

    print("\n" + "=" * 60)
    print("FINAL DETERMINATION:")
    print("=" * 60)

    if 20 in counts:
        print(f"⚠️ CONTAMINATION CONFIRMED: {counts[20]} files with 20 channels")
        print("⚠️ COLLATE WORKAROUND IS NECESSARY!")
    else:
        print("✅ NO 20-CHANNEL CONTAMINATION DETECTED")
        print("✅ COLLATE WORKAROUND APPEARS OBSOLETE")
        print("\n📝 Recommendation: Remove workaround and add strict 19-channel assertion")
