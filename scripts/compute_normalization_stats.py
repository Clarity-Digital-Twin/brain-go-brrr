#!/usr/bin/env python3
"""Compute global normalization statistics for EEGPT from a dataset.

This ensures reproducible inference by using fixed statistics rather than
estimating per-batch.
"""

import json
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm


def compute_eeg_statistics(data_dir: Path, n_samples: int = 1000) -> dict:
    """Compute mean and std from EEG dataset.

    Args:
        data_dir: Directory containing EDF files
        n_samples: Number of random windows to sample

    Returns:
        Dictionary with statistics
    """
    edf_files = list(data_dir.glob("**/*.edf"))
    if not edf_files:
        raise ValueError(f"No EDF files found in {data_dir}")

    print(f"Found {len(edf_files)} EDF files")

    # Collect samples
    all_values = []
    files_used = 0

    for edf_file in tqdm(edf_files[:50], desc="Processing files"):  # Limit to 50 files
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)

            # Get EEG channels only
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            if len(picks) == 0:
                continue

            # Sample random 4-second windows
            sfreq = raw.info["sfreq"]
            window_samples = int(4.0 * sfreq)
            max_start = raw.n_times - window_samples

            if max_start <= 0:
                continue

            # Sample up to 10 windows per file
            for _ in range(min(10, n_samples // len(edf_files))):
                start = np.random.randint(0, max_start)
                data, _ = raw[picks, start : start + window_samples]
                all_values.extend(data.flatten())

            files_used += 1

            if len(all_values) > n_samples * 1000:  # Enough samples
                break

        except Exception as e:
            print(f"Error processing {edf_file}: {e}")
            continue

    if not all_values:
        raise ValueError("No valid EEG data found")

    # Compute statistics
    all_values = np.array(all_values)

    # Remove outliers (> 5 std)
    mean = np.mean(all_values)
    std = np.std(all_values)
    mask = np.abs(all_values - mean) < 5 * std
    clean_values = all_values[mask]

    # Final statistics
    final_mean = float(np.mean(clean_values))
    final_std = float(np.std(clean_values))

    stats = {
        "mean": final_mean,
        "std": final_std,
        "n_samples": len(clean_values),
        "n_files": files_used,
        "unit": "V",  # Volts (MNE loads in V, not µV)
        "percentiles": {
            "1": float(np.percentile(clean_values, 1)),
            "5": float(np.percentile(clean_values, 5)),
            "25": float(np.percentile(clean_values, 25)),
            "50": float(np.percentile(clean_values, 50)),
            "75": float(np.percentile(clean_values, 75)),
            "95": float(np.percentile(clean_values, 95)),
            "99": float(np.percentile(clean_values, 99)),
        },
    }

    return stats


def main():
    """Compute and save normalization statistics."""
    # Use Sleep-EDF dataset
    data_dir = Path("data/datasets/external/sleep-edf/sleep-cassette")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please download Sleep-EDF dataset first")
        return

    print("Computing EEG statistics from Sleep-EDF dataset...")
    stats = compute_eeg_statistics(data_dir)

    print("\nStatistics computed:")
    print(f"  Mean: {stats['mean']:.6f} V ({stats['mean'] * 1e6:.1f} µV)")
    print(f"  Std:  {stats['std']:.6f} V ({stats['std'] * 1e6:.1f} µV)")
    print(f"  Based on {stats['n_samples']:,} samples from {stats['n_files']} files")

    # Save statistics
    output_dir = Path("data/models/eegpt/pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "eegpt_normalization_stats.json"
    with output_file.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to: {output_file}")

    # Also create a simplified version for the wrapper
    simple_stats = {
        "mean": stats["mean"],
        "std": stats["std"],
        "computed_from": "sleep-edf",
        "n_samples": stats["n_samples"],
    }

    simple_file = output_dir / "normalization.json"
    with simple_file.open("w") as f:
        json.dump(simple_stats, f, indent=2)

    print(f"Saved simplified to: {simple_file}")


if __name__ == "__main__":
    main()
