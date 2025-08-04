#!/usr/bin/env python3
"""Build TUAB metadata cache to avoid scanning 2700+ files every time."""

import json
import os
import sys
import time
from pathlib import Path

import mne
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def build_tuab_index():
    """Scan all TUAB files ONCE and cache metadata."""
    print("=" * 80)
    print("BUILDING TUAB METADATA CACHE")
    print("=" * 80)

    # Get data root
    data_root = Path(os.environ.get("BGB_DATA_ROOT", "data"))
    tuab_dir = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"

    if not tuab_dir.exists():
        print(f"ERROR: TUAB directory not found: {tuab_dir}")
        return

    # Suppress MNE logging
    mne.set_log_level("ERROR")

    index = {"version": "1.0", "created": time.time(), "files": {}}

    # Scan each split
    for split in ["train", "eval"]:
        split_dir = tuab_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found")
            continue

        index["files"][split] = {"normal": [], "abnormal": []}

        for class_name in ["normal", "abnormal"]:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            # Find all EDF files
            edf_files = list(class_dir.glob("**/*.edf"))
            print(f"\nProcessing {len(edf_files)} {split}/{class_name} files...")

            for edf_file in tqdm(edf_files, desc=f"{split}/{class_name}"):
                try:
                    # Read header only
                    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)

                    file_info = {
                        "path": str(edf_file.relative_to(tuab_dir)),
                        "duration": float(raw.n_times / raw.info["sfreq"]),
                        "sfreq": float(raw.info["sfreq"]),
                        "n_channels": len(raw.ch_names),
                        "channels": raw.ch_names,
                        "n_times": int(raw.n_times),
                    }

                    index["files"][split][class_name].append(file_info)

                except Exception as e:
                    print(f"\nError reading {edf_file}: {e}")

    # Save index
    output_path = Path("data/cache/tuab_index.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    # Print summary
    total_files = sum(
        len(files) for split_data in index["files"].values() for files in split_data.values()
    )

    print(f"\n✅ Index created: {output_path}")
    print(f"   Total files: {total_files}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build_tuab_index()
