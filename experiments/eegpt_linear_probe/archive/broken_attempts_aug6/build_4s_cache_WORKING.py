#!/usr/bin/env python
"""Build WORKING 4s cache for TUAB that handles the actual channel format."""

import json
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import mne
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def build_working_4s_cache():
    """Build a cache that ACTUALLY FUCKING WORKS."""
    
    # Paths
    BGB_DATA_ROOT = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
    EDF_ROOT = BGB_DATA_ROOT / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    CACHE_DIR = BGB_DATA_ROOT / "cache_4s_working"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parameters - MUST BE 4 SECONDS!
    WINDOW_SIZE = 4.0  # seconds - EEGPT WAS PRETRAINED ON 4S!
    WINDOW_STRIDE = 2.0  # 50% overlap
    SAMPLING_RATE = 256  # Hz
    WINDOW_SAMPLES = int(WINDOW_SIZE * SAMPLING_RATE)  # 1024
    STRIDE_SAMPLES = int(WINDOW_STRIDE * SAMPLING_RATE)  # 512
    
    # Required channels (what EEGPT expects)
    REQUIRED_CHANNELS = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'O2', 'OZ'
    ]
    
    # Mapping for old naming
    OLD_TO_NEW = {
        'T3': 'T7', 'T4': 'T8',
        'T5': 'P7', 'T6': 'P8'
    }
    
    def clean_channel_name(ch_name):
        """Clean TUAB channel names: 'EEG FP1-REF' -> 'FP1'."""
        # Remove 'EEG ' prefix
        if ch_name.startswith('EEG '):
            ch_name = ch_name[4:]
        # Remove '-REF' or '-LE' suffix
        for suffix in ['-REF', '-LE', '-AR']:
            if ch_name.endswith(suffix):
                ch_name = ch_name[:-len(suffix)]
        # Apply old->new mapping
        ch_name = OLD_TO_NEW.get(ch_name, ch_name)
        return ch_name.upper()
    
    print("🚀 Building WORKING 4s cache for TUAB")
    print(f"📁 EDF Root: {EDF_ROOT}")
    print(f"💾 Cache Dir: {CACHE_DIR}")
    print(f"⏱️  Window: {WINDOW_SIZE}s (stride: {WINDOW_STRIDE}s)")
    
    # Process files
    all_windows = []
    file_index = {}
    
    for split in ['train', 'eval']:
        print(f"\n📂 Processing {split} split...")
        split_dir = EDF_ROOT / split
        
        if not split_dir.exists():
            print(f"⚠️  {split_dir} not found")
            continue
        
        for class_name in ['normal', 'abnormal']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            label = 0 if class_name == 'normal' else 1
            edf_files = list(class_dir.rglob("*.edf"))
            
            print(f"  Found {len(edf_files)} {class_name} files")
            
            # Process subset for quick testing
            for edf_path in tqdm(edf_files[:500], desc=f"  {class_name}"):
                try:
                    # Load file
                    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
                    
                    # Clean channel names
                    cleaned_mapping = {}
                    for ch in raw.ch_names:
                        cleaned = clean_channel_name(ch)
                        if cleaned in REQUIRED_CHANNELS:
                            cleaned_mapping[ch] = cleaned
                    
                    # Check if we have enough channels
                    found_channels = set(cleaned_mapping.values())
                    missing = set(REQUIRED_CHANNELS) - found_channels
                    
                    if len(missing) > 10:  # Skip if too many missing
                        continue
                    
                    # Rename channels
                    raw.rename_channels(cleaned_mapping)
                    
                    # Pick available channels (pad missing ones later)
                    available = [ch for ch in REQUIRED_CHANNELS if ch in raw.ch_names]
                    if len(available) < 10:  # Need at least 10 channels
                        continue
                        
                    raw.pick_channels(available, ordered=False)
                    
                    # Load and resample
                    raw.load_data()
                    if raw.info['sfreq'] != SAMPLING_RATE:
                        raw.resample(SAMPLING_RATE)
                    
                    # Basic filtering
                    raw.filter(0.5, 50.0, fir_design='firwin', verbose=False)
                    
                    # Get data
                    data = raw.get_data()
                    n_samples = data.shape[1]
                    
                    # Create padded array with all channels
                    full_data = np.zeros((len(REQUIRED_CHANNELS), n_samples), dtype=np.float32)
                    for i, ch in enumerate(available):
                        ch_idx = REQUIRED_CHANNELS.index(ch)
                        full_data[ch_idx] = data[i]
                    
                    # Extract windows
                    file_windows = []
                    for start_idx in range(0, n_samples - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
                        window = full_data[:, start_idx:start_idx + WINDOW_SAMPLES]
                        
                        # Normalize
                        window_mean = window.mean(axis=1, keepdims=True)
                        window_std = window.std(axis=1, keepdims=True) + 1e-8
                        window = (window - window_mean) / window_std
                        
                        window_info = {
                            'data': window,
                            'label': label,
                            'file': str(edf_path.relative_to(EDF_ROOT)),
                            'start_sample': start_idx,
                            'split': split,
                            'class': class_name,
                        }
                        file_windows.append(window_info)
                        all_windows.append(window_info)
                    
                    if file_windows:
                        # Save to cache file
                        cache_file = CACHE_DIR / f"{edf_path.stem}_{split}_{class_name}.pt"
                        torch.save(file_windows, cache_file)
                        
                        file_index[str(edf_path.relative_to(EDF_ROOT))] = {
                            'cache_file': str(cache_file),
                            'n_windows': len(file_windows),
                            'label': label,
                            'split': split,
                        }
                        
                except Exception as e:
                    continue
    
    print(f"\n✅ Cached {len(all_windows)} windows from {len(file_index)} files")
    
    # Save index
    index_data = {
        'cache_dir': str(CACHE_DIR),
        'window_size': WINDOW_SIZE,
        'window_stride': WINDOW_STRIDE,
        'sampling_rate': SAMPLING_RATE,
        'n_channels': len(REQUIRED_CHANNELS),
        'channels': REQUIRED_CHANNELS,
        'n_windows': len(all_windows),
        'n_files': len(file_index),
        'files': file_index,
        'splits': {
            'train': sum(1 for w in all_windows if w['split'] == 'train'),
            'eval': sum(1 for w in all_windows if w['split'] == 'eval'),
        },
        'labels': {
            'normal': sum(1 for w in all_windows if w['class'] == 'normal'),
            'abnormal': sum(1 for w in all_windows if w['class'] == 'abnormal'),
        }
    }
    
    index_file = CACHE_DIR / "tuab_index_4s.json"
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"📍 Index saved: {index_file}")
    print(f"   Train windows: {index_data['splits']['train']}")
    print(f"   Eval windows: {index_data['splits']['eval']}")
    print(f"   Normal: {index_data['labels']['normal']}")
    print(f"   Abnormal: {index_data['labels']['abnormal']}")
    
    return index_file

if __name__ == "__main__":
    index_path = build_working_4s_cache()
    print(f"\n🎉 SUCCESS! Use this cache:")
    print(f"   cache_dir: {index_path.parent}")
    print(f"   index_file: {index_path}")