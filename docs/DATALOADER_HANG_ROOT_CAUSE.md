# DATALOADER HANG ROOT CAUSE ANALYSIS

## THE FUCKING PROBLEM
The training hangs FOREVER at DataLoader initialization. It's not slow - it's BROKEN.

## ROOT CAUSES IDENTIFIED

### 1. DATASET INITIALIZATION IS O(N²) STUPIDITY
**Location**: `src/brain_go_brrr/data/tuab_dataset.py:189-240`

The `_collect_samples()` method:
```python
for class_name in ["normal", "abnormal"]:
    edf_files = list(class_dir.glob("**/*.edf"))  # 2717 files!
    for edf_file in edf_files:
        info = mne.io.read_raw_edf(edf_file, preload=False)  # DISK ACCESS!
        for window_idx in range(n_windows):  # 500+ windows per file!
            self.samples.append(...)  # 1.4 MILLION appends!
```

**PROBLEMS**:
- 2717 × file header reads through WSL2
- 1,455,702 × list.append() operations
- All happens BEFORE training starts

### 2. WSL2 FILESYSTEM BOTTLENECK
- Each `mne.io.read_raw_edf()` call = Windows filesystem translation
- 2717 files × WSL2 overhead = DEATH
- No caching between runs

### 3. PYTORCH DATALOADER + WSL2 = DISASTER
**Evidence**: 
- CPU at 67% for 4.5 HOURS
- Memory usage growing to 6GB
- Zero GPU utilization
- Process stuck in data collection loop

## DIFFERENTIAL DIAGNOSES EXPLORED

### ❌ NOT Memory Leak
- Memory stable at 6GB, not growing infinitely

### ❌ NOT GPU Issues  
- GPU detected correctly
- Memory allocated (2.4GB)
- CUDA available

### ❌ NOT Package Conflicts
- All imports successful
- Model loads fine

### ❌ NOT Configuration Issues
- Config loads correctly
- All paths valid

### ✅ IT'S THE DATASET CLASS DESIGN
- Eager loading of ALL file metadata
- No lazy evaluation
- No metadata caching

## THE SMOKING GUN
Line 210 in `tuab_dataset.py`:
```python
info = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
```

This runs 2717 times during `__init__`!

## SOLUTIONS

### 1. QUICK FIX - Reduce Dataset Size
```python
# Only load first 100 files for testing
edf_files = list(class_dir.glob("**/*.edf"))[:100]
```

### 2. PROPER FIX - Lazy Loading
- Don't scan files in __init__
- Load metadata on-demand
- Cache file info to disk

### 3. NUCLEAR OPTION - Precompute Index
```bash
# Create index file once
python scripts/create_tuab_index.py
# Use index instead of scanning
```

### 4. ESCAPE WSL2 HELL
- Use native Linux
- Or move data to WSL2 filesystem
- Or use Docker with proper mounts

## LESSONS LEARNED
1. NEVER trust "it'll work eventually" with WSL2
2. Dataset __init__ should be O(1), not O(N)
3. File I/O through WSL2 = PAIN
4. Always profile before "waiting it out"

## THE REAL CRIMINAL
The TUAB dataset implementation assumes fast local filesystem access. On WSL2 + Windows mount, this assumption is FUCKED.