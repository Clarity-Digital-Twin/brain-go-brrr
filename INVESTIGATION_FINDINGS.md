# CRITICAL INVESTIGATION FINDINGS - EEGPT TRAINING

## 🚨 ROOT CAUSE ANALYSIS

### 1. CONFIG FILE MISMATCH (CRITICAL ISSUE)

**LINE 138 in train_enhanced.py:**
```python
config_file = os.environ.get("EEGPT_CONFIG", "configs/tuab_memsafe.yaml")
```

**PROBLEM**: The script defaults to `tuab_memsafe.yaml` instead of `tuab_cached.yaml`
- This is why logs show "Configuration: configs/tuab_memsafe.yaml"
- The memsafe config has DIFFERENT settings than cached config
- This could cause file scanning if memsafe config doesn't have proper cache settings

### 2. CACHED DATASET IMPLEMENTATION (VERIFIED CORRECT)

**LINES 163-186 in train_enhanced.py:**
```python
# ALWAYS USE CACHED DATASET FOR FAST LOADING
DatasetClass = TUABCachedDataset
cache_mode = 'readonly'  # FORCE readonly - never scan files
logger.info(f"Using TUABCachedDataset for FAST cached loading")

# Additional kwargs for cached dataset
extra_kwargs = {}
if hasattr(cfg.data, 'max_files') and cfg.data.max_files:
    extra_kwargs['max_files'] = cfg.data.max_files
# ALWAYS use the cache index
extra_kwargs['cache_index_path'] = data_root / "cache/tuab_index.json"

# Create train dataset - ONLY pass parameters TUABCachedDataset accepts
train_dataset = DatasetClass(
    root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    window_duration=cfg.data.window_duration,
    window_stride=cfg.data.window_stride,
    sampling_rate=cfg.data.sampling_rate,
    preload=False,
    normalize=True,
    cache_dir=data_root / "cache/tuab_enhanced",
    **extra_kwargs
)
```

**ANALYSIS**:
- ✅ Using TUABCachedDataset (correct)
- ✅ Forcing readonly mode (correct)
- ✅ Specifying cache_index_path (correct)
- ✅ Using proper cache_dir (correct)
- ❌ BUT config file is wrong!

### 3. PARAMETER MISMATCHES BETWEEN CONFIGS

**tuab_memsafe.yaml settings (from logs):**
- window_duration: 5.12s
- sampling_rate: 200Hz  
- batch_size: 16
- bandpass: 0.1-75.0Hz

**tuab_cached.yaml settings:**
- window_duration: 8.0s (CACHE WAS BUILT WITH THIS)
- sampling_rate: 256Hz (CACHE WAS BUILT WITH THIS)
- batch_size: 32
- bandpass: 0.5-50.0Hz (CACHE WAS BUILT WITH THIS)

**CRITICAL**: The memsafe config has DIFFERENT window/sampling parameters than what the cache was built with!

### 4. ENVIRONMENT VARIABLE ISSUE

**RUN_TRAINING_NOW.sh doesn't set EEGPT_CONFIG:**
```bash
# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# MISSING: export EEGPT_CONFIG="experiments/eegpt_linear_probe/configs/tuab_cached.yaml"
```

### 5. LOG ANALYSIS FINDINGS

**From training.log:**
- Line 5: "Configuration: configs/tuab_memsafe.yaml" (WRONG CONFIG)
- Line 7: "✓ 5.12s windows @ 200Hz" (WRONG - cache expects 8s @ 256Hz)
- Line 19: "Loaded TUAB train split from cache: 1455702 windows" (Different from expected 930,495)
- Line 157: UserWarning about y_pred contains classes not in y_true

## 🔥 CRITICAL ISSUES SUMMARY

1. **WRONG CONFIG FILE**: Using memsafe instead of cached
2. **PARAMETER MISMATCH**: 5.12s/200Hz vs cache built with 8s/256Hz
3. **MISSING ENV VAR**: EEGPT_CONFIG not set in launch script
4. **WINDOW COUNT MISMATCH**: 1,455,702 vs expected 930,495 (likely due to different window size)

## 🛠️ REQUIRED FIXES

1. **Fix launch script to set correct config:**
   ```bash
   export EEGPT_CONFIG="experiments/eegpt_linear_probe/configs/tuab_cached.yaml"
   ```

2. **Verify cache was built with correct parameters**

3. **Check memsafe config contents vs cached config**

4. **Ensure all parameters match between config and cache**

## 📊 CONFIG COMPARISON

### tuab_memsafe.yaml (WRONG CONFIG BEING USED):
- window_duration: 5.12s (1024 samples @ 200Hz)
- window_stride: 2.56s
- sampling_rate: 200Hz
- bandpass: 0.1-75.0Hz
- notch_filter: 50.0Hz
- batch_size: 16
- num_workers: 0

### tuab_cached.yaml (CORRECT CONFIG FOR CACHE):
- window_duration: 8.0s (2048 samples @ 256Hz)
- window_stride: 4.0s
- sampling_rate: 256Hz
- bandpass: 0.5-50.0Hz
- notch_filter: null
- batch_size: 32
- num_workers: 2

## ⚠️ INCOMPATIBILITY ANALYSIS

The cache was built with 8s windows @ 256Hz, but memsafe config uses 5.12s @ 200Hz. This is **COMPLETELY INCOMPATIBLE**:

1. **Different window sizes**: 8s vs 5.12s
2. **Different sampling rates**: 256Hz vs 200Hz
3. **Different filtering**: 0.5-50Hz vs 0.1-75Hz
4. **Additional notch filter**: None vs 50Hz

This explains why it might be scanning files - the cached data doesn't match the requested parameters!

## 🔍 ADDITIONAL INVESTIGATION

### Cache Index Location Verification:
```bash
ls -la /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/
ls -la /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/tuab_index.json
```

### Check What Config the Cache Was Built With:
Need to examine the cache index to see what parameters were used during cache creation.

### Symlink Status:
Previously created: `ln -sf data/cache/tuab_index.json tuab_index.json`
This might be backwards - need to verify correct symlink direction.

## ✅ FIXES IMPLEMENTED

1. **Updated RUN_TRAINING_NOW.sh:**
   - Added: `export EEGPT_CONFIG="experiments/eegpt_linear_probe/configs/tuab_cached.yaml"`
   - Modified launch command to include EEGPT_CONFIG explicitly

2. **Killed wrong training session**

## 🚀 NEXT STEPS

1. Verify cache index file location and parameters
2. Check if cache was actually built with 8s/256Hz or different parameters
3. Either:
   - Use config that matches cache parameters, OR
   - Rebuild cache with desired parameters
4. Run training with correct config

## 📋 COMPLETE PARAMETER AUDIT

### From CACHE_TRAINING_GUIDE.md:
**EXACT CACHE BUILD PARAMETERS:**
- window_duration = 8.0 (8 seconds)
- window_stride = 4.0 (50% overlap for train)
- sampling_rate = 256 (256 Hz)
- bandpass_low = 0.5
- bandpass_high = 50.0
- notch_freq = None (No notch filter)
- 19 channels with OLD naming (T3, T4, T5, T6)

### From smoke_test_cache.py (lines 49-58):
```python
dataset = TUABCachedDataset(
    root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    sampling_rate=256,      # ✅ CORRECT
    window_duration=8.0,    # ✅ CORRECT
    window_stride=4.0,      # ✅ CORRECT
    normalize=True,
    cache_dir=cache_dir,
    cache_index_path=cache_index,
)
```

### From verify_training_setup.py (lines 84-94):
```python
train_dataset = TUABCachedDataset(
    root_dir=paths["TUAB dataset"],
    split="train",
    sampling_rate=256,      # ✅ CORRECT
    window_duration=8.0,    # ✅ CORRECT  
    window_stride=4.0,      # ✅ CORRECT
    preload=False,
    normalize=True,
    cache_dir=cache_dir,
    cache_index_path=paths["Cache index"],
)
```

### From train_enhanced.py (lines 176-186):
```python
train_dataset = DatasetClass(
    root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    window_duration=cfg.data.window_duration,    # ❌ FROM CONFIG (could be wrong)
    window_stride=cfg.data.window_stride,        # ❌ FROM CONFIG (could be wrong)
    sampling_rate=cfg.data.sampling_rate,        # ❌ FROM CONFIG (could be wrong)
    preload=False,
    normalize=True,
    cache_dir=data_root / "cache/tuab_enhanced",
    **extra_kwargs
)
```

## 🔴 CRITICAL ISSUE FOUND

**train_enhanced.py uses CONFIG VALUES instead of hardcoded cache parameters!**

If the config has different values than the cache was built with, it will fail:
- memsafe.yaml: 5.12s @ 200Hz ❌
- cached.yaml: 8.0s @ 256Hz ✅
- Cache was built: 8.0s @ 256Hz ✅

## 🛑 THE REAL PROBLEM

1. The environment variable EEGPT_CONFIG defaults to "configs/tuab_memsafe.yaml"
2. tuab_memsafe.yaml has INCOMPATIBLE parameters (5.12s @ 200Hz)
3. Even though we use TUABCachedDataset, it's being passed WRONG parameters from the config
4. The cache expects 8.0s @ 256Hz but gets 5.12s @ 200Hz

## ✅ SOLUTION

1. MUST use tuab_cached.yaml config (already fixed in launch script)
2. MUST ensure config has correct parameters (8.0s @ 256Hz)
3. OR hardcode the parameters in train_enhanced.py to match cache