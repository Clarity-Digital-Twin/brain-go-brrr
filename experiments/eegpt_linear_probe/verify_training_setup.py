#!/usr/bin/env python
"""Comprehensive verification before training - MUST PASS ALL CHECKS"""

import os
import sys
import time
from pathlib import Path

# Setup
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

def main():
    print("=" * 80)
    print("COMPREHENSIVE TRAINING VERIFICATION")
    print("=" * 80)
    
    all_checks_passed = True
    
    # 1. Environment check
    print("\n1. ENVIRONMENT CHECK")
    print("-" * 40)
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    print(f"BGB_DATA_ROOT: {data_root}")
    print(f"Exists: {data_root.exists()}")
    
    # 2. Critical paths
    print("\n2. CRITICAL PATHS CHECK")
    print("-" * 40)
    paths = {
        "EEGPT checkpoint": data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt",
        "TUAB dataset": data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        "Cache directory": data_root / "cache/tuab_enhanced",
        "Cache index": data_root / "cache/tuab_index.json",
    }
    
    for name, path in paths.items():
        exists = path.exists()
        print(f"{name}: {exists}")
        if not exists:
            print(f"  ERROR: {path} not found!")
            all_checks_passed = False
    
    # 3. Cache verification
    print("\n3. CACHE VERIFICATION")
    print("-" * 40)
    cache_dir = paths["Cache directory"]
    cache_files = list(cache_dir.glob("*.pkl"))
    print(f"Cache files: {len(cache_files)}")
    if len(cache_files) < 900000:
        print("  WARNING: Expected ~976,698 cache files")
        all_checks_passed = False
    
    # 4. Import test
    print("\n4. IMPORT TEST")
    print("-" * 40)
    try:
        from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
        print("TUABCachedDataset: OK")
    except Exception as e:
        print(f"TUABCachedDataset: FAILED - {e}")
        all_checks_passed = False
    
    try:
        from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
        print("EEGPTTwoLayerProbe: OK")
    except Exception as e:
        print(f"EEGPTTwoLayerProbe: FAILED - {e}")
        all_checks_passed = False
    
    try:
        from brain_go_brrr.tasks.enhanced_abnormality_detection import EnhancedAbnormalityDetectionProbe
        print("EnhancedAbnormalityDetectionProbe: OK")
    except Exception as e:
        print(f"EnhancedAbnormalityDetectionProbe: FAILED - {e}")
        all_checks_passed = False
    
    # 5. Dataset loading test
    print("\n5. DATASET LOADING TEST")
    print("-" * 40)
    
    start_time = time.time()
    train_dataset = TUABCachedDataset(
        root_dir=paths["TUAB dataset"],
        split="train",
        sampling_rate=256,
        window_duration=8.0,
        window_stride=4.0,
        preload=False,
        normalize=True,
        cache_dir=cache_dir,
        cache_index_path=paths["Cache index"],
    )
    load_time = time.time() - start_time
    
    print(f"Train dataset loaded in {load_time:.2f}s")
    print(f"Train windows: {len(train_dataset)}")
    
    if load_time > 10:
        print("  WARNING: Dataset loading too slow!")
        all_checks_passed = False
    
    # 6. Sample test
    print("\n6. SAMPLE LOADING TEST")
    print("-" * 40)
    
    sample_start = time.time()
    x, y = train_dataset[0]
    sample_time = time.time() - sample_start
    
    print(f"Sample loaded in {sample_time:.3f}s")
    print(f"Sample shape: {x.shape}")
    print(f"Expected shape: torch.Size([19, 2048])")
    
    if x.shape != (19, 2048):
        print("  ERROR: Wrong sample shape!")
        all_checks_passed = False
    
    # 7. GPU check
    print("\n7. GPU CHECK")
    print("-" * 40)
    
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: No GPU available!")
    
    # 8. Config check
    print("\n8. CONFIG CHECK")
    print("-" * 40)
    
    config_path = project_root / "experiments/eegpt_linear_probe/configs/tuab_cached.yaml"
    print(f"Config exists: {config_path.exists()}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - READY FOR TRAINING!")
        print("\nRun training with:")
        print("  bash experiments/eegpt_linear_probe/launch_cached_training_WORKING.sh")
    else:
        print("❌ SOME CHECKS FAILED - FIX BEFORE TRAINING!")
    print("=" * 80)
    
    return all_checks_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)