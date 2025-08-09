#!/usr/bin/env python
"""BUILD THE FUCKING CACHE PROPERLY - 4 SECOND WINDOWS."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ['BGB_DATA_ROOT'] = str(Path(__file__).parent.parent.parent / "data")

import json
import torch
import numpy as np
from tqdm import tqdm
import mne
mne.set_log_level('ERROR')

def build_4s_cache():
    """Build a WORKING 4s cache for TUAB dataset."""
    
    # PATHS
    DATA_ROOT = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
    EDF_ROOT = DATA_ROOT / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    CACHE_DIR = DATA_ROOT / "cache/tuab_4s_final"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL PARAMETERS - MUST BE 4 SECONDS
    WINDOW_SIZE = 4.0  # seconds - EEGPT PRETRAINED ON THIS
    STRIDE = 2.0  # 50% overlap
    SAMPLING_RATE = 256  # Hz
    WINDOW_SAMPLES = int(WINDOW_SIZE * SAMPLING_RATE)  # 1024
    STRIDE_SAMPLES = int(STRIDE * SAMPLING_RATE)  # 512
    
    # EEGPT expects these channels
    TARGET_CHANNELS = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'O2', 'OZ'
    ]
    
    # Channel mapping for old naming
    CHANNEL_MAP = {
        'T3': 'T7', 'T4': 'T8',
        'T5': 'P7', 'T6': 'P8'
    }
    
    print("=" * 80)
    print("BUILDING 4-SECOND CACHE FOR TUAB DATASET")
    print("=" * 80)
    print(f"EDF Root: {EDF_ROOT}")
    print(f"Cache Dir: {CACHE_DIR}")
    print(f"Window: {WINDOW_SIZE}s, Stride: {STRIDE}s")
    print(f"Target channels: {len(TARGET_CHANNELS)}")
    print()
    
    # Collect all EDF files
    all_files = []
    for split in ['train', 'eval']:
        split_dir = EDF_ROOT / split
        if not split_dir.exists():
            continue
        for label in ['normal', 'abnormal']:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            files = list(label_dir.rglob("*.edf"))
            for f in files:
                all_files.append({
                    'path': f,
                    'split': split,
                    'label': 0 if label == 'normal' else 1,
                    'label_str': label
                })
    
    print(f"Found {len(all_files)} EDF files")
    print()
    
    # Process files
    cache_index = {
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'sampling_rate': SAMPLING_RATE,
        'channels': TARGET_CHANNELS,
        'n_channels': len(TARGET_CHANNELS),
        'cache_dir': str(CACHE_DIR),
        'files': {}
    }
    
    total_windows = 0
    failed_files = []
    
    for file_info in tqdm(all_files, desc="Processing files"):
        try:
            # Load EDF
            raw = mne.io.read_raw_edf(str(file_info['path']), preload=False, verbose=False)
            
            # Clean channel names - FIXED ORDER
            ch_mapping = {}
            for ch in raw.ch_names:
                clean = ch.upper()
                # Remove prefixes FIRST
                for prefix in ['EEG ', 'EEG_']:
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):]
                # Remove suffixes SECOND
                for suffix in ['-REF', '-LE', '-AR', '_REF']:
                    if clean.endswith(suffix):
                        clean = clean[:-len(suffix)]
                # Apply mapping LAST
                clean = CHANNEL_MAP.get(clean, clean)
                
                if clean in TARGET_CHANNELS:
                    ch_mapping[ch] = clean
            
            # Need at least 10 channels
            if len(ch_mapping) < 10:
                failed_files.append((str(file_info['path']), f"Only {len(ch_mapping)} channels"))
                continue
            
            # Rename and pick channels - KEEP ORDER
            raw.rename_channels(ch_mapping)
            available = [ch for ch in TARGET_CHANNELS if ch in raw.ch_names]
            raw.pick_channels(available)  # ordered=True by default
            
            # Load and resample ONLY IF NEEDED
            raw.load_data()
            if abs(raw.info['sfreq'] - SAMPLING_RATE) > 1e-3:
                raw.resample(SAMPLING_RATE)
            
            # NO FILTERING - EEGPT was trained on unfiltered data!
            
            # Get data
            data = raw.get_data()
            n_samples = data.shape[1]
            
            # Skip if too short
            if n_samples < WINDOW_SAMPLES:
                failed_files.append((str(file_info['path']), "Too short"))
                continue
            
            # Create padded array with all channels
            full_data = np.zeros((len(TARGET_CHANNELS), n_samples), dtype=np.float32)
            for i, ch in enumerate(available):
                ch_idx = TARGET_CHANNELS.index(ch)
                full_data[ch_idx] = data[i]
            
            # Extract windows - OPTIMIZED STORAGE
            window_list = []
            for start in range(0, n_samples - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
                window = full_data[:, start:start + WINDOW_SAMPLES]
                
                # Normalize
                mean = window.mean(axis=1, keepdims=True)
                std = window.std(axis=1, keepdims=True) + 1e-8
                window = (window - mean) / std
                
                window_list.append(window)
            
            if window_list:
                # Save as tensors for fast loading
                windows_tensor = torch.from_numpy(np.stack(window_list).astype(np.float32))
                labels_tensor = torch.full((len(window_list),), file_info['label'], dtype=torch.int8)
                
                # Save cache file - FIX NAMING COLLISIONS
                rel_path = file_info['path'].relative_to(EDF_ROOT)
                safe_name = rel_path.with_suffix('').as_posix().replace('/', '__')
                cache_file = CACHE_DIR / f"{safe_name}.pt"
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({'x': windows_tensor, 'y': labels_tensor}, cache_file)
                
                # Update index
                cache_index['files'][str(rel_path)] = {
                    'cache_file': str(cache_file.relative_to(CACHE_DIR)),
                    'n_windows': len(window_list),
                    'split': file_info['split'],
                    'label': file_info['label'],
                    'channels': len(available)
                }
                
                total_windows += len(window_list)
                
        except Exception as e:
            failed_files.append((str(file_info['path']), str(e)))
    
    # Save index
    cache_index['total_windows'] = total_windows
    cache_index['n_files'] = len(cache_index['files'])
    cache_index['failed_files'] = len(failed_files)
    
    index_path = CACHE_DIR / "index.json"
    with open(index_path, 'w') as f:
        json.dump(cache_index, f, indent=2)
    
    # Save failed files log
    if failed_files:
        with open(CACHE_DIR / "failed_files.txt", 'w') as f:
            for path, reason in failed_files:
                f.write(f"{path}: {reason}\n")
    
    print()
    print("=" * 80)
    print("CACHE BUILD COMPLETE")
    print("=" * 80)
    print(f"✅ Cached {total_windows:,} windows from {len(cache_index['files'])} files")
    print(f"❌ Failed {len(failed_files)} files")
    print(f"📁 Cache directory: {CACHE_DIR}")
    print(f"📋 Index file: {index_path}")
    print()
    
    # Summary by split
    train_windows = sum(
        info['n_windows'] for info in cache_index['files'].values()
        if info['split'] == 'train'
    )
    eval_windows = sum(
        info['n_windows'] for info in cache_index['files'].values()
        if info['split'] == 'eval'
    )
    
    print(f"Train windows: {train_windows:,}")
    print(f"Eval windows: {eval_windows:,}")
    print()
    
    if total_windows < 1000000:
        print("⚠️  WARNING: Less than 1M windows cached. Training may be slow.")
    else:
        print("✅ Sufficient data cached for fast training!")
    
    return index_path

if __name__ == "__main__":
    index = build_4s_cache()
    print(f"\nUSE THIS INDEX: {index}")