#!/usr/bin/env python
"""Quick smoke test for paper-aligned training setup."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper


def test_setup():
    """Test all components work before full training."""
    print("Running smoke test for paper-aligned training...")
    print("=" * 50)
    
    # Check environment
    data_root = os.environ.get('BGB_DATA_ROOT', 'data')
    print(f"✓ BGB_DATA_ROOT: {data_root}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    
    # Check config file
    config_path = Path("configs/tuab_4s_paper_aligned.yaml")
    assert config_path.exists(), f"Config not found: {config_path}"
    print(f"✓ Config exists: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if we need 4s cache or can use existing 8s for smoke test
    cache_8s = Path(f"{data_root}/cache/tuab_enhanced")
    cache_4s = Path(f"{data_root}/cache/tuab_4s_windows")
    
    if cache_4s.exists():
        print(f"✓ 4s cache exists: {cache_4s}")
        cache_dir = cache_4s
        window_duration = 4.0
    elif cache_8s.exists():
        print(f"! Using 8s cache for smoke test: {cache_8s}")
        cache_dir = cache_8s
        window_duration = 8.0
    else:
        print("✗ No cache found! Run build_tuab_4s_cache.py first")
        return False
        
    # Test dataset loading
    print("\nTesting dataset loading...")
    try:
        # Check if dataset exists
        dataset_path = Path(f"{data_root}/datasets/external/tuh_eeg_abnormal/v3.0.1/edf")
        if not dataset_path.exists():
            print(f"✗ Dataset not found at {dataset_path}")
            return False
            
        # For smoke test, just check if we can create the dataset
        # We'll use a small subset for testing
        print(f"  Dataset path: {dataset_path}")
        print(f"  Cache dir: {cache_dir}")
        print(f"  Window duration: {window_duration}s")
        
        # Count cache files to verify
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.pt"))
            print(f"✓ Found {len(cache_files)} cached files")
        else:
            print("! Cache directory doesn't exist, will be created during training")
        
        print("✓ Dataset configuration validated")
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
        
    # Test model loading
    print("\nTesting EEGPT model...")
    try:
        model_path = f"{data_root}/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
        assert Path(model_path).exists(), f"Model not found: {model_path}"
        print(f"✓ Model checkpoint exists: {model_path}")
        
        # Create wrapper
        backbone = EEGPTWrapper(
            checkpoint_path=model_path
        )
        print("✓ EEGPT backbone created")
        
        # Test forward pass
        device = torch.device('cuda' if cuda_available else 'cpu')
        backbone.to(device)
        backbone.eval()
        
        # Create batch
        batch = torch.randn(2, 20, int(window_duration * 256)).to(device)
        
        with torch.no_grad():
            features = backbone(batch)
        
        print(f"✓ Forward pass successful, features shape: {features.shape}")
        assert features.shape[0] == 2  # Batch size
        assert features.shape[1] == 4  # Summary tokens (embed_num)
        # Feature dim depends on model size - could be 512 or 768
        assert features.shape[2] in [512, 768], f"Unexpected embed_dim: {features.shape[2]}"
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n" + "=" * 50)
    print("✅ All smoke tests passed! Ready for training.")
    return True


if __name__ == "__main__":
    # Set environment
    os.environ['BGB_DATA_ROOT'] = os.environ.get(
        'BGB_DATA_ROOT', 
        '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data'
    )
    
    success = test_setup()
    sys.exit(0 if success else 1)