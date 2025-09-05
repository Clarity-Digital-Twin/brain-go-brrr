# P0 CRITICAL: TUEV Autoreject Montage Failure - Complete Fix Documentation

**Status**: 🔴 BLOCKING - 0 windows produced, training impossible
**Priority**: P0 - Training completely blocked
**Date**: September 5, 2025
**Document**: This is the SINGLE AUTHORITATIVE document for fixing TUEV Autoreject

## 🔴 THE CRITICAL FAILURE

**Cache building produces 0 windows from 159 TUEV eval files**

Error in logs:
```
"Valid channel positions are needed for autoreject to work"
```

**Root Cause**: After synthesizing missing channels (Fpz), we never set a montage, so Autoreject can't interpolate bad channels.

## Investigation Results

### 1. Missing Montage After Channel Synthesis

**The Problem**:
- Parent class `TUABPreprocessor` sets montage at line 154-155
- But `TUEVPreprocessor` overrides `process_raw_with_annotations()` 
- After synthesizing Fpz as zeros (line 117), we never set montage
- Autoreject's `fit_transform()` (line 406) requires channel positions for interpolation
- Without montage → error → file skipped → 0 windows

**Evidence**:
```python
# Parent TUABPreprocessor line 154-155:
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='warn')

# TUEVPreprocessor: MISSING after synthesis!
```

### 2. Suspicious 4Hz Lowpass Filter (FALSE ALARM)

**Investigation**: The log shows "Upper passband edge: 4.00 Hz" but this is actually from the muscle detection filter, NOT the main filter.

**Evidence** (mne_preprocessor.py line 280):
```python
filter_freq=(muscle_band_low, muscle_band_high),  # 110-140Hz for muscle
```

The 4Hz is just MNE's internal filter design message for muscle detection. Main filter is correct: 0.5-45Hz.

### 3. Why Every File Failed

**Sequence of failure**:
1. File loads → channels canonicalized
2. Fpz synthesized as zeros (correct)
3. **MISSING**: Set montage for new channel set
4. Create epochs → attempt Autoreject
5. Autoreject tries to interpolate → no positions → ERROR
6. Exception caught → file skipped → 0 windows

## The Solution

### Fix 1: Set Montage After Channel Synthesis

**File**: `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py`
**Location**: After `_synthesize_missing_channels()` call (line 117)

```python
# After line 117 (raw = self._synthesize_missing_channels(raw))
# ADD THESE LINES:

# Set standard 1020 montage for all channels including synthesized ones
try:
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')  # ignore=OK for non-standard
    logger.info(f"Set standard_1020 montage for {len(raw.ch_names)} channels")
except Exception as e:
    logger.warning(f"Could not set montage: {e}")
```

### Fix 2: Add Fallback for Autoreject Failures

**File**: `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py`
**Location**: Wrap `_apply_autoreject_tuev()` method (line 385-417)

```python
def _apply_autoreject_tuev(self, epochs: mne.Epochs) -> tuple[mne.Epochs, dict[str, Any]]:
    """Apply Autoreject with gentle parameters for TUEV spike preservation."""
    from autoreject import AutoReject
    
    try:
        # Existing AR code...
        ar = AutoReject(
            n_interpolate=[1, 2],  
            consensus=[0.5, 0.7, 0.9],
            cv=3,
            thresh_method='bayesian_optimization',
            random_state=42,
            verbose=False,
        )
        
        epochs_clean = ar.fit_transform(epochs)
        
        # Collect learned parameters
        ar_params = {}
        if hasattr(ar, 'n_interpolate_'):
            ar_params['n_interpolate'] = ar.n_interpolate_.get('eeg', None)
        if hasattr(ar, 'consensus_'):
            ar_params['consensus'] = ar.consensus_.get('eeg', None)
            
        return epochs_clean, ar_params
        
    except Exception as e:
        logger.warning(f"Autoreject failed: {e}. Proceeding without artifact rejection.")
        # Return original epochs if AR fails
        return epochs, {}
```

### Fix 3: Early Exit if Cache Empty

**File**: `src/brain_go_brrr/infra/data/tuev_dataset.py`
**Location**: After cache building (line 238)

```python
# After line 238 (logger.info(f"Cache built: {global_window_id} windows..."))
# ADD:
if global_window_id == 0:
    raise ValueError(
        "Cache building failed: 0 windows produced. "
        "Check preprocessing logs for 'Valid channel positions' errors. "
        "Likely cause: montage not set after channel synthesis."
    )
```

## Verification Steps

### Quick Test (Single File)
```python
# Test preprocessor on one file
from pathlib import Path
from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor

proc = TUEVPreprocessor()
edf_path = Path("data/datasets/tuev/edf/eval/000/bckg_000_a_.edf")

# Load annotations...
# Process with fixed preprocessor
epochs, info, labels = proc.process_raw_with_annotations(edf_path, annotations)

# Should see:
# - "Set standard_1020 montage for 20 channels"
# - No "Valid channel positions" error
# - epochs should have >0 windows
```

### Full Verification
```bash
# Delete bad cache
rm -rf data/cache/tuev_mne_v2/

# Rebuild with fixed code
uv run python experiments/eegpt_linear_probe/train_tuev_mne.py \
    --config configs/tuev.yaml \
    --rebuild-cache

# Should see:
# - "Set standard_1020 montage" messages
# - No position errors
# - "Cache built: XXX windows" where XXX > 0
```

## Why This Happened

1. **TUEVPreprocessor overrides parent method** but doesn't replicate all steps
2. **Channel synthesis added** but montage setting forgotten
3. **No unit test** for TUEV with actual Autoreject
4. **Error silently caught** in cache builder, producing empty cache

## Definition of Done

- [ ] Montage set after channel synthesis
- [ ] Autoreject has try/except fallback
- [ ] Cache builder fails fast on 0 windows
- [ ] Test passes: Single TUEV file produces >0 epochs
- [ ] Test passes: Cache rebuild produces >0 windows
- [ ] Training starts with actual data

## Risk Assessment

**Risk**: LOW
- Simple addition of missing montage setting
- Follows established pattern from parent class
- Fallback ensures processing continues even if AR fails

## Testing Commands

```bash
# 1. Test single file processing
uv run python -c "
from pathlib import Path
from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor
import mne

proc = TUEVPreprocessor()
edf = Path('data/datasets/tuev/edf/eval/000/bckg_000_a_.edf')
raw = mne.io.read_raw_edf(edf, preload=True)
print(f'Channels before: {len(raw.ch_names)}')
print(f'Has montage before: {raw.get_montage() is not None}')

# Process it (this calls the fixed methods internally)
# Should NOT error on channel positions
"

# 2. Rebuild small cache
BGB_DATA_ROOT=data BGB_CACHE_DIR=data/cache \
uv run python -c "
from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset
from pathlib import Path

dataset = TUEVMNEDataset(
    root_dir=Path('data/datasets/tuev'),
    split='eval',
    force_rebuild=True
)
print(f'Dataset size: {len(dataset)} windows')
assert len(dataset) > 0, 'Still producing 0 windows!'
"
```

## Notes for Implementation

1. **The 4Hz filter log is a red herring** - it's from muscle detection, not main filtering
2. **Must set montage AFTER synthesis** - synthesized Fpz needs a position too
3. **Use on_missing='ignore'** - some channels might not be in standard_1020
4. **Don't skip whole file on AR failure** - better to have unfiltered data than no data

---

**CRITICAL**: Do not implement until senior review confirms this analysis is correct!