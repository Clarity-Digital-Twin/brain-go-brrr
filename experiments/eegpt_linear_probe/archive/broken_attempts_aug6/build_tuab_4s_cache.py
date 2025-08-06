#!/usr/bin/env python
"""Build PROPER fucking cache for TUAB with 4-second windows that ACTUALLY WORKS."""

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import mne
import pickle
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def build_4s_cache():
    """Build cache that ACTUALLY FUCKING WORKS."""
    
    # CRITICAL PATHS - MUST MATCH EXACTLY
    BGB_DATA_ROOT = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
    EDF_ROOT = BGB_DATA_ROOT / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    CACHE_DIR = BGB_DATA_ROOT / "cache_4s"
    
    print(f"🚨 BUILDING PROPER 4S CACHE")
    print(f"📁 EDF Root: {EDF_ROOT}")
    print(f"💾 Cache Dir: {CACHE_DIR}")
    print(f"✅ BGB_DATA_ROOT: {BGB_DATA_ROOT}")
    
    # Verify paths exist
    if not EDF_ROOT.exists():
        raise FileNotFoundError(f"EDF ROOT NOT FOUND: {EDF_ROOT}")
    
    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parameters for 4s windows (EEGPT pretrained)
    WINDOW_SIZE = 4.0  # seconds
    WINDOW_STRIDE = 2.0  # 50% overlap for training  
    SAMPLING_RATE = 256  # Hz
    WINDOW_SAMPLES = int(WINDOW_SIZE * SAMPLING_RATE)  # 1024 samples
    STRIDE_SAMPLES = int(WINDOW_STRIDE * SAMPLING_RATE)  # 512 samples
    
    # Standard 20 channels (modern naming)
    REQUIRED_CHANNELS = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',  # Modern naming
        'P7', 'P3', 'PZ', 'P4', 'P8',   # Modern naming
        'O1', 'O2', 'OZ'
    ]
    
    # Old to modern channel mapping
    CHANNEL_MAPPING = {
        'T3': 'T7', 'T4': 'T8',
        'T5': 'P7', 'T6': 'P8'
    }
    
    def process_edf_file(edf_path: Path, label: int):
        """Process single EDF file and extract windows."""
        windows = []
        
        try:
            # Load EDF
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            
            # Map old channel names to modern
            if raw.ch_names:
                mapping = {}
                for old_name, new_name in CHANNEL_MAPPING.items():
                    if old_name in raw.ch_names and new_name not in raw.ch_names:
                        mapping[old_name] = new_name
                if mapping:
                    raw.rename_channels(mapping)
            
            # Check we have required channels
            missing = set(REQUIRED_CHANNELS) - set(raw.ch_names)
            if missing:
                print(f"⚠️  Missing channels in {edf_path.name}: {missing}")
                return []  # Skip files with missing channels
            
            # Pick and reorder channels
            raw.pick_channels(REQUIRED_CHANNELS, ordered=True)
            
            # Resample to 256 Hz if needed
            if raw.info['sfreq'] != SAMPLING_RATE:
                raw.resample(SAMPLING_RATE)
            
            # Filter
            raw.filter(0.5, 50.0, fir_design='firwin', verbose=False)
            
            # Get data
            data = raw.get_data()  # (20, n_samples)
            n_samples = data.shape[1]
            
            # Extract windows
            for start_idx in range(0, n_samples - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
                window = data[:, start_idx:start_idx + WINDOW_SAMPLES]
                
                # Normalize window
                window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)
                
                windows.append({
                    'data': window.astype(np.float32),
                    'label': label,
                    'file': str(edf_path.relative_to(EDF_ROOT)),
                    'start_sample': start_idx,
                })
                
        except Exception as e:
            print(f"❌ Error processing {edf_path.name}: {e}")
            
        return windows
    
    # Process train and eval splits
    all_cache_data = {}
    
    for split in ['train', 'eval']:
        print(f"\n📂 Processing {split.upper()} split...")
        
        split_windows = []
        file_info = []
        
        # Find all EDF files
        for class_name, label in [('normal', 0), ('abnormal', 1)]:
            class_dir = EDF_ROOT / split / class_name
            if not class_dir.exists():
                print(f"⚠️  {class_dir} not found, skipping")
                continue
                
            edf_files = list(class_dir.rglob("*.edf"))
            print(f"  Found {len(edf_files)} {class_name} files")
            
            # Process each file
            for edf_path in tqdm(edf_files[:100] if split == 'train' else edf_files[:20], 
                               desc=f"  {class_name}", leave=False):
                windows = process_edf_file(edf_path, label)
                
                if windows:
                    # Save windows to cache file
                    cache_file = CACHE_DIR / f"{edf_path.stem}_{split}_{class_name}.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(windows, f)
                    
                    # Track metadata
                    for i, w in enumerate(windows):
                        split_windows.append({
                            'cache_file': str(cache_file),
                            'window_idx': i,
                            'label': label,
                            'file': w['file'],
                        })
                    
                    file_info.append({
                        'file': str(edf_path.relative_to(EDF_ROOT)),
                        'n_windows': len(windows),
                        'label': label,
                        'cache_file': str(cache_file),
                    })
        
        # Save split metadata
        all_cache_data[split] = {
            'windows': split_windows,
            'files': file_info,
            'n_windows': len(split_windows),
            'n_normal': sum(1 for w in split_windows if w['label'] == 0),
            'n_abnormal': sum(1 for w in split_windows if w['label'] == 1),
        }
        
        print(f"✅ {split.upper()}: {len(split_windows)} windows from {len(file_info)} files")
        print(f"   Normal: {all_cache_data[split]['n_normal']}, Abnormal: {all_cache_data[split]['n_abnormal']}")
    
    # Save master index
    index_file = CACHE_DIR / "tuab_index_4s.json"
    with open(index_file, 'w') as f:
        json.dump({
            'cache_dir': str(CACHE_DIR),
            'window_size': WINDOW_SIZE,
            'window_stride': WINDOW_STRIDE,
            'sampling_rate': SAMPLING_RATE,
            'channels': REQUIRED_CHANNELS,
            'train': all_cache_data.get('train', {}),
            'eval': all_cache_data.get('eval', {}),
        }, f, indent=2)
    
    print(f"\n🎉 CACHE BUILD COMPLETE!")
    print(f"📍 Index: {index_file}")
    print(f"💾 Cache size: {sum(f.stat().st_size for f in CACHE_DIR.glob('*.pkl')) / 1e9:.2f} GB")
    
    # Create symlink for compatibility
    tuab_link = BGB_DATA_ROOT / "datasets/external/tuab"
    if not tuab_link.exists():
        tuab_link.symlink_to(EDF_ROOT)
        print(f"🔗 Created symlink: tuab -> {EDF_ROOT}")
    
    return index_file

if __name__ == "__main__":
    index_path = build_4s_cache()
    print(f"\n✅ USE THIS IN YOUR CONFIG:")
    print(f"   cache_index: {index_path}")