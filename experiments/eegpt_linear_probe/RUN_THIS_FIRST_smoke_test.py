#!/usr/bin/env python
"""SMOKE TEST - Run this FIRST to verify everything works before training!"""

import sys
from pathlib import Path

# Colors for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def test_imports():
    """Test all required imports."""
    print(f"\n{BLUE}=== Testing Imports ==={RESET}")
    try:
        import torch
        print(f"{GREEN}✓ PyTorch: {torch.__version__}{RESET}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"{RED}✗ PyTorch import failed: {e}{RESET}")
        return False
    
    try:
        import pytorch_lightning as pl
        print(f"{GREEN}✓ PyTorch Lightning: {pl.__version__}{RESET}")
    except Exception as e:
        print(f"{RED}✗ PyTorch Lightning import failed: {e}{RESET}")
        return False
    
    try:
        import mne
        print(f"{GREEN}✓ MNE: {mne.__version__}{RESET}")
    except Exception as e:
        print(f"{RED}✗ MNE import failed: {e}{RESET}")
        return False
    
    return True

def test_paths():
    """Test all required paths exist."""
    print(f"\n{BLUE}=== Testing Paths ==={RESET}")
    
    project_root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr")
    data_root = project_root / "data"
    
    paths_to_check = {
        "Project root": project_root,
        "Data root": data_root,
        "EEGPT checkpoint": data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt",
        "TUAB root": data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        "TUAB train": data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf/train",
        "TUAB eval": data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf/eval",
        "Config file": project_root / "experiments/eegpt_linear_probe/configs/tuab_config.yaml",
    }
    
    all_good = True
    for name, path in paths_to_check.items():
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"{GREEN}✓ {name}: {path} ({size_mb:.1f} MB){RESET}")
            else:
                print(f"{GREEN}✓ {name}: {path}{RESET}")
        else:
            print(f"{RED}✗ {name}: {path} NOT FOUND!{RESET}")
            all_good = False
    
    return all_good

def test_dataset_structure():
    """Test TUAB dataset structure."""
    print(f"\n{BLUE}=== Testing Dataset Structure ==={RESET}")
    
    data_root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
    tuab_root = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    
    if not tuab_root.exists():
        print(f"{RED}✗ TUAB root not found: {tuab_root}{RESET}")
        return False
    
    # Count files
    splits = ["train", "eval"]
    classes = ["normal", "abnormal"]
    
    total_files = 0
    for split in splits:
        print(f"\n{split.upper()}:")
        for cls in classes:
            path = tuab_root / split / cls
            if path.exists():
                edf_files = list(path.rglob("*.edf"))
                count = len(edf_files)
                total_files += count
                print(f"  {cls}: {count} files")
                # Show sample file
                if edf_files:
                    print(f"    Sample: {edf_files[0].relative_to(tuab_root)}")
            else:
                print(f"  {cls}: {RED}NOT FOUND{RESET}")
    
    print(f"\n{YELLOW}Total EDF files: {total_files}{RESET}")
    return total_files > 0

def test_tuab_dataset():
    """Test TUABDataset can be instantiated."""
    print(f"\n{BLUE}=== Testing TUABDataset ==={RESET}")
    
    # Add project to path
    sys.path.insert(0, "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr")
    
    try:
        from brain_go_brrr.data.tuab_dataset import TUABDataset
        print(f"{GREEN}✓ TUABDataset imported{RESET}")
    except Exception as e:
        print(f"{RED}✗ Failed to import TUABDataset: {e}{RESET}")
        return False
    
    # Try to create dataset
    try:
        data_root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
        dataset = TUABDataset(
            root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",  # CORRECT PATH!
            split="train",
            window_duration=8.0,
            window_stride=8.0,
            sampling_rate=256,
            preload=False,
            normalize=True,
        )
        print(f"{GREEN}✓ Created train dataset with {len(dataset)} windows{RESET}")
        
        # Try to load one sample
        if len(dataset) > 0:
            x, y = dataset[0]
            print(f"{GREEN}✓ Loaded sample: shape={x.shape}, label={y}{RESET}")
    except Exception as e:
        print(f"{RED}✗ Failed to create dataset: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_model():
    """Test model can be loaded."""
    print(f"\n{BLUE}=== Testing Model ==={RESET}")
    
    sys.path.insert(0, "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr")
    
    try:
        from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe
        print(f"{GREEN}✓ AbnormalityDetectionProbe imported{RESET}")
    except Exception as e:
        print(f"{RED}✗ Failed to import model: {e}{RESET}")
        return False
    
    try:
        data_root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
        checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
        
        model = AbnormalityDetectionProbe(checkpoint_path, n_input_channels=20)
        print(f"{GREEN}✓ Model loaded successfully{RESET}")
        
        # Test forward pass
        import torch
        dummy_input = torch.randn(1, 20, 2048)  # [batch, channels, time]
        with torch.no_grad():
            output = model(dummy_input)
        print(f"{GREEN}✓ Forward pass successful: output shape={output.shape}{RESET}")
        
    except Exception as e:
        print(f"{RED}✗ Failed to load model: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all smoke tests."""
    print(f"{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}EEGPT LINEAR PROBE SMOKE TEST{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")
    
    tests = [
        ("Imports", test_imports),
        ("Paths", test_paths),
        ("Dataset Structure", test_dataset_structure),
        ("TUABDataset", test_tuab_dataset),
        ("Model", test_model),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"{RED}✗ {name} test crashed: {e}{RESET}")
            results.append((name, False))
    
    # Summary
    print(f"\n{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}SUMMARY:{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")
    
    all_passed = True
    for name, passed in results:
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n{GREEN}✓ ALL TESTS PASSED! Ready to train.{RESET}")
        print(f"\n{BLUE}CORRECT DATASET PATH:{RESET}")
        print(f"root_dir = data_root / 'datasets/external/tuh_eeg_abnormal/v3.0.1/edf'")
    else:
        print(f"\n{RED}✗ SOME TESTS FAILED! Fix issues before training.{RESET}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)