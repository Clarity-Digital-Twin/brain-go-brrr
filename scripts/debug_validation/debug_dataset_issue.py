#!/usr/bin/env python3
"""Debug why validation dataset only has normal samples."""

from pathlib import Path

from brain_go_brrr  # noqa: E402.data.tuab_cached_dataset import TUABCachedDataset

# Create dataset
print("Creating validation dataset...")
val_ds = TUABCachedDataset(
    split="eval",
    cache_dir=Path("data/cache/tuab_enhanced"),
    root_dir=Path("data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf"),
    window_duration=8.0,
    window_stride=4.0,
    cache_index_path=Path("data/cache/tuab_index.json"),
)

print("\nDataset info:")
print(f"Total samples: {len(val_ds)}")
print(f"Class counts: {val_ds.class_counts}")
print(f"Total files: {len(val_ds.file_list)}")

# Check first few files
print("\nFirst 10 files:")
for i, file_info in enumerate(val_ds.file_list[:10]):
    print(f"  {i}: {file_info['class_name']} - {Path(file_info['path']).name}")

# Check label distribution in samples
normal_count = sum(1 for s in val_ds.samples if s["label"] == 0)
abnormal_count = sum(1 for s in val_ds.samples if s["label"] == 1)
print("\nSample label distribution:")
print(f"  Normal (0): {normal_count}")
print(f"  Abnormal (1): {abnormal_count}")

# Check if any abnormal files exist
abnormal_files = [f for f in val_ds.file_list if f["label"] == 1]
print(f"\nNumber of abnormal files in file_list: {len(abnormal_files)}")
if abnormal_files:
    print(f"First abnormal file: {abnormal_files[0]['path']}")
