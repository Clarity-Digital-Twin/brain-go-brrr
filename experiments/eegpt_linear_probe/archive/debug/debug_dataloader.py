"""Debug DataLoader issues - find out why workers are crashing."""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import os
import sys
import psutil
import traceback

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def check_system():
    """Check system resources."""
    print("=== SYSTEM CHECK ===")
    print(f"CPU count: {os.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"RAM available: {psutil.virtual_memory().available / 1e9:.1f} GB")
    
    # Check if WSL
    if 'microsoft' in os.uname().release.lower():
        print("⚠️  WSL DETECTED - Known issues with multiprocessing!")
        
        # Check shared memory
        try:
            shm_size = os.statvfs('/dev/shm')
            shm_gb = (shm_size.f_blocks * shm_size.f_frsize) / 1e9
            print(f"/dev/shm size: {shm_gb:.1f} GB")
            if shm_gb < 2:
                print("❌ /dev/shm too small! DataLoader workers will crash!")
        except:
            print("❌ Cannot check /dev/shm")
    
    print()

def test_dataloader(num_workers=0):
    """Test DataLoader with different worker configs."""
    print(f"=== TESTING num_workers={num_workers} ===")
    
    try:
        # Load config
        config_path = Path("configs/tuab_4s_paper_aligned.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Set environment
        os.environ['BGB_DATA_ROOT'] = '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data'
        cache_dir = Path(os.path.expandvars(config['data']['cache_dir']))
        
        print(f"Cache dir: {cache_dir}")
        
        # Try original dataset
        try:
            from tuab_mmap_dataset import TUABMemoryMappedDataset
            dataset = TUABMemoryMappedDataset(cache_dir, split='train')
            print(f"✅ Original dataset loaded: {len(dataset)} samples")
        except Exception as e:
            print(f"❌ Original dataset failed: {e}")
            # Try safe version
            from tuab_mmap_dataset_safe import TUABMemoryMappedDatasetSafe
            dataset = TUABMemoryMappedDatasetSafe(cache_dir, split='train')
            print(f"✅ Safe dataset loaded: {len(dataset)} samples")
        
        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(num_workers == 0),  # Only pin if single process
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        print(f"DataLoader created with {num_workers} workers")
        
        # Try to load batches
        print("Loading batches...")
        for i, (data, labels) in enumerate(loader):
            print(f"  Batch {i}: data={data.shape}, labels={labels.shape}")
            if i >= 2:  # Just test a few
                break
        
        print(f"✅ SUCCESS with num_workers={num_workers}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED with num_workers={num_workers}")
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        print()

def main():
    """Run diagnostic tests."""
    print("=" * 60)
    print("DATALOADER DIAGNOSTIC")
    print("=" * 60)
    print()
    
    # Check system
    check_system()
    
    # Test different worker configurations
    results = {}
    
    # Test num_workers=0 (should always work)
    print("Testing single process (safest)...")
    results[0] = test_dataloader(0)
    
    # Test num_workers=1
    print("Testing 1 worker...")
    results[1] = test_dataloader(1)
    
    # Test num_workers=2
    print("Testing 2 workers...")
    results[2] = test_dataloader(2)
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for num_workers, success in results.items():
        status = "✅ WORKS" if success else "❌ FAILS"
        print(f"num_workers={num_workers}: {status}")
    
    print()
    print("RECOMMENDATION:")
    if results[0] and not results[1]:
        print("⚠️  USE num_workers=0 for WSL!")
        print("This is a known WSL issue with multiprocessing + memory-mapped files")
    elif all(results.values()):
        print("✅ All configurations work! You can use multiple workers.")
    else:
        working = [k for k, v in results.items() if v]
        if working:
            print(f"Use num_workers={max(working)} for best performance")

if __name__ == "__main__":
    main()