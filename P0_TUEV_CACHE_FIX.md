# P0 CRITICAL: TUEV Cache Building Fix

**Status**: 🔴 BLOCKING - TUEV training cannot start without cache
**Priority**: P0 - Runtime crash on cache build attempt
**Date**: September 5, 2025
**Owner**: Engineering Team
**Time to Fix**: 2-3 hours
**Reviewer**: Senior Engineer

## Executive Summary

TUEV training is COMPLETELY BLOCKED because:
1. No pre-built cache exists
2. Cache building was incorrectly implemented and reverted
3. **GOOD NEWS**: All required components EXIST and are CORRECT

## Critical Discovery

### ✅ WHAT EXISTS AND WORKS:
- `TUEVPreprocessor` class EXISTS at `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py`
- `process_raw_with_annotations()` method EXISTS with CORRECT signature (line 231-335)
- Method returns CORRECT tuple: `(epochs_clean, info, window_labels)`
- All preprocessing logic is ALREADY IMPLEMENTED

### 🔴 WHAT'S BROKEN:
- `_build_cache()` in `tuev_dataset.py` raises NotImplementedError (line 140-144)
- Previous attempt had wrong imports and data formats

## The Fix (2 Files, ~100 Lines)

### File 1: `/src/brain_go_brrr/infra/data/tuev_dataset.py`

**Location**: Lines 134-144 (method `_build_cache`)

**Current Code**:
```python
def _build_cache(self) -> None:
    raise NotImplementedError(
        "Building TUEV cache requires TUEVPreprocessor which is not implemented. "
        "Please use a pre-built cache..."
    )
```

**Fixed Code**:
```python
def _build_cache(self) -> None:
    """Build preprocessed cache with MNE+Autoreject."""
    from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor
    from brain_go_brrr.infra.data.channels import CHANNELS_TUEV_20
    import subprocess
    import json
    from tqdm import tqdm
    
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = TUEVPreprocessor()
    
    # Track all windows globally
    global_window_id = 0
    windows_dict = {}
    class_counts = {i: 0 for i in range(6)}
    n_rejected_total = 0
    
    edf_files = self._get_edf_files()
    
    for edf_path in tqdm(edf_files, desc="Processing TUEV files"):
        annotations = self._load_annotations(edf_path)
        
        # Call EXISTING method with CORRECT signature
        epochs_clean, info, window_labels = preprocessor.process_raw_with_annotations(
            edf_path, 
            annotations, 
            window_overlap=0.5  # 50% overlap for 2s stride on 4s windows
        )
        
        # Extract data from MNE Epochs object
        epoch_data = epochs_clean.get_data()  # Shape: (n_epochs, 20, 1024)
        
        # Process each clean epoch
        for epoch_idx in range(len(epochs_clean)):
            # Get single epoch data (20, 1024)
            x_volts = epoch_data[epoch_idx]  # In Volts from MNE
            
            # CRITICAL: Convert Volts to millivolts
            x_mV = x_volts * 1e3
            
            # Get label for this window
            label_str = window_labels[epoch_idx]
            label_int = CLASS_MAPPING[label_str]
            
            # Ensure correct tensor types
            x_tensor = torch.tensor(x_mV, dtype=torch.float32)  # (20, 1024) in mV
            y_tensor = torch.tensor(label_int, dtype=torch.long)  # Long for CrossEntropyLoss
            
            # Save individual window
            cache_file = f"window_{global_window_id}.pt"
            torch.save({
                'x': x_tensor,
                'y': y_tensor
            }, self.cache_dir / cache_file, _use_new_zipfile_serialization=True)
            
            # Track in index
            windows_dict[str(global_window_id)] = {
                'cache_file': cache_file,
                'label': int(label_int),
                'file': str(edf_path.relative_to(self.root_dir))
            }
            
            class_counts[label_int] += 1
            global_window_id += 1
        
        n_rejected_total += info.get('n_rejected', 0)
    
    # Write index JSON
    index_data = {
        'total_windows': global_window_id,
        'windows': windows_dict,
        'n_files': len(edf_files),
        'n_rejected': n_rejected_total,
        'class_counts': {str(k): v for k, v in class_counts.items()}
    }
    
    index_path = self.cache_dir / f'index_{self.split}_{self.CACHE_VERSION}.json'
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Write META JSON
    meta_data = {
        'sr': 256,
        'unit': 'mV',
        'window': 1024,
        'channels': CHANNELS_TUEV_20,
        'n_channels': 20,
        'norm': 'wrapper',
        'commit': subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).parent
        ).decode().strip(),
        'split': self.split,
        'dataset': 'TUEV'
    }
    
    with open(self.cache_dir / 'META.json', 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    logger.info(f"Cache built: {global_window_id} windows, class dist: {class_counts}")
```

## Critical Contract Requirements

### 1. Index File Structure
- **Filename**: `index_{split}_{CACHE_VERSION}.json`
- **Keys**: Must use STRING keys ("0", "1", ...) not integers
- **Format**: Valid JSON (no comments, no binary data)

### 2. META.json Structure
- **unit**: MUST be "mV" (millivolts)
- **channels**: MUST be CHANNELS_TUEV_20 list
- **Format**: Valid JSON (not torch binary!)

### 3. Data Storage
- **x**: Shape (20, 1024) in millivolts as float32
- **y**: Label as torch.long (for CrossEntropyLoss)
- **Conversion**: MNE returns Volts, multiply by 1e3

## Validation Commands

```bash
# 1. Check preprocessor exists
grep -n "class TUEVPreprocessor" src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py

# 2. Check method exists
grep -n "def process_raw_with_annotations" src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py

# 3. After fix, verify cache builds
python -c "
from pathlib import Path
from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset
dataset = TUEVMNEDataset(
    root_dir=Path('data/datasets/tuev'),
    split='train',
    force_rebuild=True
)
print(f'Built {len(dataset)} windows')
"
```

## Definition of Done

- [ ] `_build_cache()` method replaced with working implementation
- [ ] Import statements include TUEVPreprocessor and CHANNELS_TUEV_20
- [ ] Correct method called: `process_raw_with_annotations()`
- [ ] Data converted from Volts to millivolts (×1e3)
- [ ] Labels saved as torch.long tensors
- [ ] Index file uses string keys
- [ ] META.json is valid JSON (not binary)
- [ ] Cache builds successfully for at least 1 file
- [ ] Training can start with built cache

## Risk Assessment

**Risk Level**: LOW
- All components exist and are tested
- Just need to wire them together correctly
- TUAB uses simpler approach but works

## Why This Will Work

1. **Preprocessor EXISTS**: Line 231-335 of tuev_preprocessor.py
2. **Correct return type**: Tuple of (epochs, info, labels)
3. **MNE integration working**: Used in TUAB successfully
4. **Cache loader ready**: Lines 80-133 of tuev_dataset.py

## Bottom Line

This is a SIMPLE fix - just calling an existing method correctly and saving the output in the right format. The entire preprocessing pipeline is already implemented and tested. We just need to connect the pieces.