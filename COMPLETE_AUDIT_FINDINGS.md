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

**NEED TO CHECK**: What parameters was cache actually built with?

### File: `/scripts/build_tuab_index.py`

**NEED TO CHECK**: Does index store the parameters used?

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

## 🔥 SUMMARY OF ALL FUCKED UP THINGS

1. **TUABCachedDataset has WRONG DEFAULTS** (30s instead of 8s)
2. **train_enhanced.py uses CONFIG VALUES instead of hardcoded cache params**
3. **Default config is tuab_memsafe.yaml with INCOMPATIBLE parameters**
4. **No validation that config params match cache params**
5. **Cache index uses relative path by default**
6. **No error if parameters don't match cache**
7. **Window count calculation depends on matching parameters**

## 🛠 FIXES NEEDED

1. **HARDCODE cache parameters in TUABCachedDataset**
2. **ADD validation that config matches cache**
3. **REMOVE parameter passing from config for cached dataset**
4. **USE absolute paths everywhere**
5. **ADD loud warnings if params don't match**