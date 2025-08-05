# Issues Encountered and Fixes Applied

## Executive Summary

This document details all the issues we encountered during EEGPT linear probe training setup and how we fixed them. This serves as a reference for future debugging and helps avoid repeating the same mistakes.

---

## 1. PyTorch Lightning 2.5.2 Critical Bug

### Issue
PyTorch Lightning hangs indefinitely when loading large cached datasets (>100k samples).

### Symptoms
- Training stuck at "Loading `train_dataloader` to estimate number of stepping batches"
- No progress after hours of waiting
- CPU at 100% but no actual computation

### Root Cause
Lightning 2.5.2 has a bug in its dataloader validation for large datasets with caching.

### Fix Applied
Switched to pure PyTorch implementation:
- Removed all Lightning dependencies
- Implemented manual training loop
- Direct DataLoader usage without Lightning wrappers

### Lessons Learned
- Always test with small datasets first
- Have fallback to pure PyTorch ready
- Monitor initial loading phase carefully

---

## 2. Environment Variable Resolution in YAML

### Issue
`${BGB_DATA_ROOT}` in config files not automatically resolved.

### Symptoms
```
FileNotFoundError: [Errno 2] No such file or directory: '${BGB_DATA_ROOT}/models/...'
```

### Root Cause
PyYAML doesn't resolve environment variables by default.

### Fix Applied
Created custom resolution function:
```python
def resolve_env_vars(obj):
    if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        env_var = obj[2:-1]
        return os.environ.get(env_var, obj)
    # ... recursive handling for dicts and lists
```

Also added manual path resolution in critical sections:
```python
if '${BGB_DATA_ROOT}' in root_dir:
    root_dir = root_dir.replace('${BGB_DATA_ROOT}', data_root)
```

### Lessons Learned
- Don't assume automatic env var resolution
- Add explicit path resolution in dataloaders
- Test with both relative and absolute paths

---

## 3. Dataset API Inconsistencies

### Issue
Different dataset classes have incompatible interfaces.

### Symptoms
```
TypeError: TUABEnhancedDataset.__init__() got an unexpected keyword argument 'use_cached_dataset'
```

### Root Cause
- TUABCachedDataset vs TUABEnhancedDataset have different parameters
- No clear documentation on which to use when

### Fix Applied
- Identified correct dataset class: TUABCachedDataset for cached data
- Used proper initialization parameters
- Added cache_index_path requirement

### Lessons Learned
- Document dataset class differences
- Create unified interface or factory pattern
- Always check __init__ parameters

---

## 4. Variable Channel Count in EEG Data

### Issue
Some files have 19 channels, others have 20, causing batch collation errors.

### Symptoms
```
RuntimeError: stack expects each tensor to be equal size
```

### Root Cause
TUAB dataset has inconsistent channel counts across recordings.

### Fix Applied
Created custom collate function:
```python
def collate_eeg_batch_fixed(batch):
    max_channels = max(sample[0].shape[0] for sample in batch)
    # Pad all samples to max_channels
```

### Lessons Learned
- Always inspect data variability first
- Custom collate functions are powerful
- Add data validation in collate function

---

## 5. Model Architecture Confusion

### Issue
Import errors and parameter mismatches with EEGPT model classes.

### Symptoms
```
ImportError: cannot import name 'EEGPTBackbone'
TypeError: EEGPTWrapper.__init__() got an unexpected keyword argument 'freeze_backbone'
```

### Root Cause
- Multiple model wrapper classes with similar names
- Inconsistent APIs between wrappers

### Fix Applied
- Use EEGPTWrapper for inference
- Manual .eval() instead of freeze_backbone parameter
- Correct import paths

### Lessons Learned
- Maintain consistent naming conventions
- Document class hierarchies
- Use type hints for clarity

---

## 6. Config Structure Navigation

### Issue
Nested config causing KeyError when accessing probe config.

### Symptoms
```
KeyError: 'probe'
```

### Root Cause
Passing entire config instead of model subsection.

### Fix Applied
```python
# Wrong: probe = LinearProbe(config)
# Right: probe = LinearProbe(config['model'])
```

### Lessons Learned
- Be explicit about config structure
- Add config validation
- Use dataclasses for config typing

---

## 7. Scheduler Parameter Types

### Issue
OneCycleLR failing due to string max_lr from YAML.

### Symptoms
```
TypeError: unsupported operand type(s) for /: 'str' and 'int'
```

### Root Cause
YAML loads numeric values as strings in some cases.

### Fix Applied
Explicit type casting:
```python
max_lr=float(config['training']['scheduler']['max_lr'])
```

### Lessons Learned
- Always cast numeric config values
- Add config validation layer
- Use pydantic for config models

---

## 8. Model Output Dimension Assumptions

### Issue
Expected 768-dim output but model produces 512-dim.

### Symptoms
```
AssertionError: assert features.shape[2] == 768
```

### Root Cause
Different EEGPT model variants have different embedding dimensions.

### Fix Applied
- Checked actual model output
- Updated config to use 512 input_dim
- Made assertions flexible

### Lessons Learned
- Never hardcode model dimensions
- Always inspect model architecture
- Add dimension discovery code

---

## 9. Missing Cache Index File

### Issue
Dataset initialization fails without tuab_index.json.

### Symptoms
```
ValueError: Cache index not found: .../cache/tuab_index.json
Run: python scripts/build_tuab_index.py
```

### Root Cause
Cached dataset requires pre-built file index for fast loading.

### Fix Applied
- Located existing tuab_index.json
- Updated path resolution
- Added existence check

### Lessons Learned
- Document all required files
- Add helpful error messages
- Create setup validation script

---

## 10. CUDA Memory Management

### Issue
OOM errors with large batch sizes.

### Symptoms
```
RuntimeError: CUDA out of memory
```

### Root Cause
- Large model + large batch size
- No gradient accumulation

### Fix Applied
- Reduced batch size to 256
- Added gradient accumulation option
- Enabled mixed precision training

### Lessons Learned
- Profile memory usage early
- Have batch size fallback options
- Monitor GPU memory during training

---

## Summary Statistics

- **Total Issues Encountered**: 10 major issues
- **Time Lost to Debugging**: ~8 hours
- **Most Common Category**: Path/Config issues (40%)
- **Most Time-Consuming**: PyTorch Lightning bug (3 hours)

## Recommendations

1. **Create validation script** that checks all requirements before training
2. **Use type hints** and dataclasses for configs
3. **Build comprehensive test suite** for data loading
4. **Document all external dependencies** clearly
5. **Maintain working script templates** for common tasks

## Prevention Checklist

Before starting new experiments:
- [ ] Validate all file paths exist
- [ ] Check dataset sample for dimensions
- [ ] Run smoke test with 10 samples
- [ ] Verify GPU memory with target batch size
- [ ] Test config loading and resolution
- [ ] Ensure all imports work
- [ ] Check model output dimensions
- [ ] Validate data consistency