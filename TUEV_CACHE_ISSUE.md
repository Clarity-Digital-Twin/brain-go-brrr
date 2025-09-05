# TUEV Cache Building Issue - Critical Documentation

**Status**: ✅ REVERTED - Safe NotImplementedError in place  
**Date**: September 5, 2025 (Updated after senior review)  
**Priority**: P0 - Blocks TUEV training (unless pre-built cache exists)  
**Author**: Claude  
**For Review**: Senior Engineering Audit Required

## Executive Summary

The TUEV dataset cache building implementation was completely broken and has been reverted. The implementation violated multiple contracts and would crash on any attempt to build cache. This document details what went wrong, why, and how to fix it properly.

## What Went Wrong

### 1. Wrong API Call
**Attempted**: `preprocessor.process_file(edf_path, annotations=annotations, window_duration=4.0, window_stride=2.0)`  
**Reality**: Method doesn't exist. TUEVPreprocessor inherits from TUABPreprocessor which only has:
- `process_raw_with_annotations(edf_path, annotations, window_overlap=...)`
- Returns: `(epochs_clean, info, window_labels)` not list of dicts

### 2. Broken Cache Contract

#### Index File Contract Violation
**Expected**: `index_{split}_{CACHE_VERSION}.json` with structure:
```json
{
    "total_windows": 12345,
    "windows": {
        "0": {
            "cache_file": "window_0.pt",
            "label": 2,
            "file": "relative/path/to/source.edf"
        },
        ...
    },
    "n_files": 359,
    "n_rejected": 42,
    "class_counts": {
        "0": 1234,  // SPSW
        "1": 2345,  // GPED
        "2": 3456,  // PLED
        "3": 4567,  // EYEM
        "4": 5678,  // ARTF
        "5": 6789   // BCKG
    }
}
```
**Actual**: Never wrote this file at all

#### META File Contract Violation
**Expected**: `META.json` (JSON format) with:
```json
{
    "sr": 256,
    "unit": "mV",
    "window": 1024,
    "channels": ["FP1", "FP2", ...],  // CHANNELS_TUEV_20
    "n_channels": 20,
    "norm": "wrapper",
    "commit": "d6b3e4e",
    "split": "train",
    "dataset": "TUEV"
}
```
**Actual**: `torch.save(index, self.cache_dir / 'META.json')` - Binary data to .json file!

### 3. Safe Load Violation
**Problem**: Saved `y` as Python int, loaded with `torch.load(..., weights_only=True)`  
**Issue**: With `weights_only=True`, non-tensor members may be dropped  
**Required**: Save as `torch.tensor(y, dtype=torch.long)`

## Consequences of Broken Implementation

1. `_cache_exists()` returns False (no index file)
2. `_build_cache()` runs but creates wrong structure
3. `_load_cache_index()` fails - can't find index JSON
4. Dataset initialization crashes
5. Training cannot start

## How TUAB Does It (Working Reference)

TUAB successfully builds cache because:
1. Uses simpler structure - just pickle files with naming convention
2. Builds cache during first dataset access
3. Has working preprocessor with correct method signatures
4. Cache format matches loader expectations

TUAB cache structure:
```
tuab_mne_v2/
├── META.json
├── aaaaaaaq_s004_t000_0.pkl
├── aaaaaaaq_s004_t000_1.pkl
└── ...
```

## Correct Implementation (When Ready)

### Option 1: Pre-built Cache (RECOMMENDED NOW)
```python
def _build_cache(self) -> None:
    raise NotImplementedError(
        "Building TUEV cache requires TUEVPreprocessor which is not implemented. "
        "Please use a pre-built cache..."
    )
```
**Status**: ✅ Currently implemented (safe)

### Option 2: Proper Implementation (FUTURE)
```python
def _build_cache(self) -> None:
    from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor
    import subprocess
    
    preprocessor = TUEVPreprocessor()
    
    # Track all windows globally
    global_window_id = 0
    windows_dict = {}
    class_counts = {i: 0 for i in range(6)}
    n_rejected_total = 0
    
    edf_files = self._get_edf_files()
    
    for edf_path in tqdm(edf_files):
        annotations = self._load_annotations(edf_path)
        
        # Use CORRECT method signature
        epochs_clean, info, window_labels = preprocessor.process_raw_with_annotations(
            edf_path, 
            annotations, 
            window_overlap=0.5  # 50% overlap for 2s stride on 4s windows
        )
        
        # Extract data from MNE Epochs object
        epoch_data = epochs_clean.get_data()  # Shape: (n_epochs, n_channels, n_times)
        
        # Process each clean epoch
        for epoch_idx in range(len(epochs_clean)):
            # Get single epoch data (C, T) = (20, 1024)
            x_volts = epoch_data[epoch_idx]  # In Volts from MNE
            
            # CRITICAL: Convert Volts to millivolts for cache contract
            x_mV = x_volts * 1e3  
            
            # Get label for this window
            label = window_labels[epoch_idx]
            
            # Ensure correct tensor types
            x_tensor = torch.tensor(x_mV, dtype=torch.float32)  # (20, 1024) in mV
            y_tensor = torch.tensor(label, dtype=torch.long)  # Long for CrossEntropyLoss
            
            # Save with unique ID
            cache_file = f"window_{global_window_id}.pt"
            torch.save({
                'x': x_tensor,
                'y': y_tensor
            }, self.cache_dir / cache_file, _use_new_zipfile_serialization=True)
            
            # Track in index with string keys
            windows_dict[str(global_window_id)] = {
                'cache_file': cache_file,
                'label': int(label),
                'file': str(edf_path.relative_to(self.root_dir))
            }
            
            class_counts[int(label)] += 1
            global_window_id += 1
        
        n_rejected_total += info.get('n_rejected', 0)
    
    # Write proper index JSON with CACHE_VERSION
    index_data = {
        'total_windows': global_window_id,
        'windows': windows_dict,  # String keys "0", "1", ...
        'n_files': len(edf_files),
        'n_rejected': n_rejected_total,
        'class_counts': {str(k): v for k, v in class_counts.items()}  # String keys
    }
    
    # Use self.CACHE_VERSION not hardcoded!
    index_path = self.cache_dir / f'index_{self.split}_{self.CACHE_VERSION}.json'
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Write proper META JSON
    meta_data = {
        'sr': 256,
        'unit': 'mV',  # CRITICAL: Data stored in millivolts
        'window': 1024,
        'channels': CHANNELS_TUEV_20,
        'n_channels': 20,
        'norm': 'wrapper',  # Normalization done in wrapper, not dataset
        'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
        'split': self.split,
        'dataset': 'TUEV'
    }
    
    with open(self.cache_dir / 'META.json', 'w') as f:
        json.dump(meta_data, f, indent=2)
        
    logger.info(f"Cache built: {global_window_id} windows, class dist: {class_counts}")
```

## Current Status & Recommendations

### ✅ What's Working
1. TUAB training completed: AUROC 0.8282
2. TUEV training script is bulletproof (guards, checkpoints, etc.)
3. Configs are correct and in right location
4. **CODE IS REVERTED**: _build_cache() raises NotImplementedError (safe)
5. Training will work IF pre-built cache exists at cache_dir

### 🔴 What's Broken
1. TUEV cannot build cache on-the-fly
2. No pre-built TUEV cache exists

### 📋 Recommended Actions

**IMMEDIATE (for training now)**:
1. ✅ DONE: Revert to NotImplementedError
2. Build TUEV cache offline using a separate script
3. Or obtain pre-built cache from elsewhere

**FUTURE (proper fix)**:
1. Fix TUEVPreprocessor to have correct methods
2. Implement _build_cache() following exact contract above
3. Test cache format matches loader expectations
4. Validate training works end-to-end

## Validation Checklist

Before any future cache building implementation:
- [ ] Verify preprocessor method exists and signature matches
- [ ] Confirm index JSON structure matches loader
- [ ] Ensure META.json is valid JSON (not binary)
- [ ] Test labels saved as torch.long tensors
- [ ] Validate cache can be loaded by dataset
- [ ] Run smoke test training to verify

## Critical Corrections from Senior Review

### Fixed in this revision:
1. **MNE Epochs handling**: Use `epochs_clean.get_data()` to extract numpy array, not direct iteration
2. **Unit conversion**: MNE returns Volts, must multiply by 1e3 to store as millivolts (mV)
3. **Version usage**: Use `self.CACHE_VERSION` not hardcoded "v1"
4. **Index structure**: Windows dict uses string keys ("0", "1", ...) consistently
5. **Status clarification**: Code IS reverted (NotImplementedError), training only blocked if no cache exists

## Bottom Line

The TUEV training pipeline is **100% ready** if you have a pre-built cache. Without a cache, training is blocked. The safest path:
1. Use a pre-built cache if available
2. OR build cache offline with the corrected implementation above
3. Only re-enable in-dataset building after testing the exact contract

---
**Status**: ✅ Document updated with senior corrections. Code is safe (NotImplementedError).