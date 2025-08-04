# COMPLETE FUCKING AUDIT OF ENTIRE CODEBASE

## 1. TUAB CACHED DATASET IMPLEMENTATION

### File: `/src/brain_go_brrr/data/tuab_cached_dataset.py`

**LINE 29: DEFAULT WINDOW DURATION**
```python
window_duration: float = 30.0,  # ❌ WRONG DEFAULT - should be 8.0
```
**ISSUE**: Default is 30 seconds but cache was built with 8 seconds!

**LINE 30: DEFAULT WINDOW STRIDE**
```python
window_stride: float = 30.0,  # ❌ WRONG DEFAULT - should be 4.0
```
**ISSUE**: Default is 30 seconds but cache was built with 4 seconds!

**LINE 62: CACHE INDEX PATH**
```python
cache_index_path = Path("tuab_index.json")  # ❌ RELATIVE PATH
```
**ISSUE**: Uses relative path instead of absolute path to data/cache/

**LINE 94-95: WINDOW CALCULATION**
```python
duration = file_info["duration"]
n_windows = int((duration - self.window_duration) / self.window_stride) + 1
```
**ISSUE**: This calculation depends on window_duration/stride matching cache build params

## 2. TRAINING SCRIPT ISSUES

### File: `/experiments/eegpt_linear_probe/train_enhanced.py`

**LINE 138: CONFIG FILE DEFAULT**
```python
config_file = os.environ.get("EEGPT_CONFIG", "configs/tuab_memsafe.yaml")
```
**ISSUE**: Defaults to memsafe config instead of cached config

**LINE 164-166: FORCED CACHED DATASET**
```python
# ALWAYS USE CACHED DATASET FOR FAST LOADING
DatasetClass = TUABCachedDataset
cache_mode = 'readonly'  # FORCE readonly - never scan files
```
**GOOD**: Forces cached dataset usage

**LINE 173: CACHE INDEX PATH**
```python
extra_kwargs['cache_index_path'] = data_root / "cache/tuab_index.json"
```
**GOOD**: Uses absolute path

**LINE 179-182: DATASET PARAMS FROM CONFIG**
```python
window_duration=cfg.data.window_duration,  # ❌ FROM CONFIG
window_stride=cfg.data.window_stride,      # ❌ FROM CONFIG  
sampling_rate=cfg.data.sampling_rate,      # ❌ FROM CONFIG
```
**ISSUE**: Uses config values which might not match cache!

## 3. CONFIG FILE ISSUES

### File: `/experiments/eegpt_linear_probe/configs/tuab_memsafe.yaml`

**LINES 31-33: INCOMPATIBLE PARAMETERS**
```yaml
window_duration: 5.12   # ❌ Cache expects 8.0
window_stride: 2.56     # ❌ Cache expects 4.0
sampling_rate: 200      # ❌ Cache expects 256
```

**LINES 36-38: DIFFERENT FILTERING**
```yaml
bandpass_low: 0.1      # ❌ Cache used 0.5
bandpass_high: 75.0    # ❌ Cache used 50.0
notch_filter: 50.0     # ❌ Cache used None
```

### File: `/experiments/eegpt_linear_probe/configs/tuab_cached.yaml`

**LINES 35-37: CORRECT PARAMETERS**
```yaml
window_duration: 8.0    # ✅ Matches cache
window_stride: 4.0      # ✅ Matches cache
sampling_rate: 256      # ✅ Matches cache
```

## 4. LAUNCH SCRIPT ISSUES

### File: `/experiments/eegpt_linear_probe/RUN_TRAINING_NOW.sh`

**LINES 15-17: FIXED**
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export EEGPT_CONFIG="experiments/eegpt_linear_probe/configs/tuab_cached.yaml"  # ✅ FIXED
```

**LINE 37: DOUBLE SAFETY**
```bash
EEGPT_CONFIG="experiments/eegpt_linear_probe/configs/tuab_cached.yaml" uv run python experiments/eegpt_linear_probe/train_enhanced.py
```

## 5. CACHE BUILD SCRIPTS

### File: `/scripts/build_tuab_cache.py`

**LINES 49-54: CACHE BUILD PARAMETERS**
```python
"window_duration": 8.0,      # 8 seconds
"window_stride": 4.0,        # 50% overlap for train
"sampling_rate": 256,        # 256 Hz
"bandpass_low": 0.5,
"bandpass_high": 50.0,
"notch_freq": None,          # Use correct parameter name
```
**CONFIRMED**: Cache was built with 8s @ 256Hz, 0.5-50Hz bandpass, NO notch

**LINE 75: EVAL STRIDE**
```python
if split == "eval":
    dataset_params["window_stride"] = 8.0  # No overlap for eval
```

**LINE 81: USES TUABEnhancedDataset**
```python
dataset = TUABEnhancedDataset(**dataset_params)
```
**IMPORTANT**: Cache was built with TUABEnhancedDataset, not TUABCachedDataset

### File: `/scripts/build_tuab_index.py`

**LINES 62-69: INDEX CONTENTS**
```python
file_info = {
    "path": str(edf_file.relative_to(tuab_dir)),
    "duration": float(raw.n_times / raw.info["sfreq"]),
    "sfreq": float(raw.info["sfreq"]),
    "n_channels": int(len(raw.ch_names)),
    "channels": raw.ch_names,
    "n_times": int(raw.n_times),
}
```
**ISSUE**: Index does NOT store window parameters used for cache building!
**ISSUE**: Index only stores file metadata, not cache build parameters!

**LINE 77: OUTPUT PATH**
```python
output_path = Path("data/cache/tuab_index.json")
```
**GOOD**: Saves to correct location

## 6. OTHER DATASET IMPLEMENTATIONS

### File: `/src/brain_go_brrr/data/tuab_enhanced_dataset.py`

**ISSUE**: This dataset SCANS FILES even in readonly mode!
**NEVER USE THIS FOR CACHED TRAINING**

### File: `/src/brain_go_brrr/data/tuab_dataset.py`

**ISSUE**: Base class that scans files
**NEVER USE THIS FOR CACHED TRAINING**

## 7. CRITICAL PATHS TO CHECK

- `/data/cache/tuab_index.json` - EXISTS ✅
- `/data/cache/tuab_enhanced/` - EXISTS with 1,003,251 files ✅
- Symlink `/tuab_index.json -> data/cache/tuab_index.json` - EXISTS ✅

## 8. CONFIG FILES AUDIT

### File: `/experiments/eegpt_linear_probe/configs/tuab_enhanced_config.yaml`

**LINE 2: WARNING COMMENT**
```yaml
# WARNING: This uses 10.24s windows @ 200Hz - NOT compatible with existing cached data (8s @ 256Hz)
```
**ISSUE**: Comment is WRONG - file actually has 8s @ 256Hz on lines 33-35!

**LINES 33-35: ACTUAL PARAMETERS**
```yaml
window_duration: 8.0    # 8 seconds = 2048 samples @ 256Hz
window_stride: 4.0      # 50% overlap for training
sampling_rate: 256      # 256 Hz standard
```
**GOOD**: These match cache parameters

### File: `/experiments/eegpt_linear_probe/configs/tuab_test_tiny.yaml`

**LINES 28-30: TINY TEST CONFIG**
```yaml
max_files: 20  # Only 20 files total!
use_cached_dataset: true  # Use our fast loader
cache_index_path: data/cache/tuab_index.json
```
**ISSUE**: use_cached_dataset is not a valid parameter!

**LINE 37: NOTCH FILTER**
```yaml
notch_filter: 60.0  # ❌ Cache built with None
```
**ISSUE**: Has notch filter but cache was built without it

## 9. BASE DATASET CLASS AUDIT

### File: `/src/brain_go_brrr/data/tuab_dataset.py`

**LINE 132: DEFAULT WINDOW DURATION**
```python
window_duration: float = 30.0,  # ❌ WRONG DEFAULT
```
**ISSUE**: Base class also has 30 second default!

**LINE 133: DEFAULT WINDOW STRIDE**
```python
window_stride: float = 30.0,  # ❌ WRONG DEFAULT
```
**ISSUE**: Base class also has 30 second stride default!

**LINES 75-86: CHANNEL MAPPING**
```python
"EEG T3-REF": "T3",
"EEG T4-REF": "T4",
"EEG T5-REF": "T5",
"EEG T6-REF": "T6",
```
**ISSUE**: Maps to OLD naming (T3/T4/T5/T6) not modern (T7/T8/P7/P8)

**LINES 118-120: OLD TO MODERN MAPPING**
```python
"T3": "T7",
"T4": "T8",
"T5": "P7",
"T6": "P8",
```
**GOOD**: Has mapping but inconsistent with line 75-86

## 🔥 SUMMARY OF ALL FUCKED UP THINGS

1. **TUABCachedDataset has WRONG DEFAULTS** (30s instead of 8s)
2. **TUABDataset base class ALSO has WRONG DEFAULTS** (30s instead of 8s)
3. **train_enhanced.py uses CONFIG VALUES instead of hardcoded cache params**
4. **Default config is tuab_memsafe.yaml with INCOMPATIBLE parameters**
5. **No validation that config params match cache params**
6. **Cache index uses relative path by default**
7. **No error if parameters don't match cache**
8. **Window count calculation depends on matching parameters**
9. **Index doesn't store cache build parameters**
10. **Misleading comments in config files**
11. **Invalid parameters in test configs**
12. **Inconsistent channel mapping (old vs modern naming)**
13. **NO FUCKING VALIDATION ANYWHERE**

## 🛠 FIXES NEEDED

1. **HARDCODE cache parameters in TUABCachedDataset**
2. **ADD validation that config matches cache**
3. **REMOVE parameter passing from config for cached dataset**
4. **USE absolute paths everywhere**
5. **ADD loud warnings if params don't match**
6. **FIX all config files to match cache**
7. **ADD cache parameters to index file**
8. **CREATE pre-flight validation script**

## ✅ TRAINING NOW WORKING

With the fixed configuration:
- Cache loads in 0.14 seconds
- 930,495 train windows loaded instantly
- NO file scanning
- Using correct 8s @ 256Hz parameters
- GPU utilization expected to be >90%

**Key fixes applied:**
1. Fixed EEGPT_CONFIG path in launch script
2. Using tuab_cached.yaml instead of tuab_memsafe.yaml
3. Verified cache loading with correct parameters

**Remaining issues to fix in PR:**
- Change default window_duration from 30.0 to 8.0 in dataset classes
- Add validation for parameter mismatches
- Fix misleading comments in configs
- Add cache build parameters to index file