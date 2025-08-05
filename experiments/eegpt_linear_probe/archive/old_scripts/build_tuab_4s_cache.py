#!/usr/bin/env python
"""Build TUAB dataset cache with 4-second windows to match EEGPT paper."""

import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.brain_go_brrr.data.tuab_cached_dataset import build_cache


def main():
    """Build TUAB cache with 4-second windows."""
    
    # Configuration matching EEGPT paper
    config = {
        'root_dir': os.environ.get('BGB_DATA_ROOT', 'data') + '/datasets/external/tuh_eeg_abnormal/v3.0.1/edf',
        'cache_dir': os.environ.get('BGB_DATA_ROOT', 'data') + '/cache/tuab_4s_windows',
        'window_duration': 4.0,  # 4 seconds (EEGPT paper)
        'window_stride': 2.0,    # 50% overlap
        'sampling_rate': 256,    # 256 Hz
        'bandpass_low': 0.5,     # Standard EEG filter
        'bandpass_high': 38.0,   # Match paper for MI tasks
        'channels': None,        # Use all available channels
        'n_jobs': 8,            # Parallel processing
        'batch_size': 100,      # Files per batch
    }
    
    print("Building TUAB cache with 4-second windows")
    print("=" * 50)
    print(f"Root dir: {config['root_dir']}")
    print(f"Cache dir: {config['cache_dir']}")
    print(f"Window: {config['window_duration']}s, stride: {config['window_stride']}s")
    print(f"Sampling rate: {config['sampling_rate']} Hz")
    print(f"Bandpass: {config['bandpass_low']}-{config['bandpass_high']} Hz")
    print("=" * 50)
    
    # Build cache for train split
    print("\nBuilding train split...")
    train_cache_info = build_cache(
        root_dir=config['root_dir'],
        cache_dir=config['cache_dir'],
        split='train',
        window_duration=config['window_duration'],
        window_stride=config['window_stride'],
        sampling_rate=config['sampling_rate'],
        bandpass_low=config['bandpass_low'],
        bandpass_high=config['bandpass_high'],
        channels=config['channels'],
        n_jobs=config['n_jobs'],
        batch_size=config['batch_size']
    )
    
    print(f"\nTrain split complete:")
    print(f"  Total windows: {train_cache_info['total_windows']}")
    print(f"  Normal: {train_cache_info['normal_windows']}")
    print(f"  Abnormal: {train_cache_info['abnormal_windows']}")
    
    # Build cache for eval split
    print("\nBuilding eval split...")
    eval_cache_info = build_cache(
        root_dir=config['root_dir'],
        cache_dir=config['cache_dir'],
        split='eval',
        window_duration=config['window_duration'],
        window_stride=config['window_stride'],
        sampling_rate=config['sampling_rate'],
        bandpass_low=config['bandpass_low'],
        bandpass_high=config['bandpass_high'],
        channels=config['channels'],
        n_jobs=config['n_jobs'],
        batch_size=config['batch_size']
    )
    
    print(f"\nEval split complete:")
    print(f"  Total windows: {eval_cache_info['total_windows']}")
    print(f"  Normal: {eval_cache_info['normal_windows']}")
    print(f"  Abnormal: {eval_cache_info['abnormal_windows']}")
    
    print("\n✅ Cache building complete!")
    print(f"Cache location: {config['cache_dir']}")


if __name__ == "__main__":
    main()