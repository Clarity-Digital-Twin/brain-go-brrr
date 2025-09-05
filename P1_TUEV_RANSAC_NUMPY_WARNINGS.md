# P1 NON-BLOCKING: TUEV RANSAC and NumPy Warning Issues

**Status**: 🟡 NON-BLOCKING - Training continues but logs are noisy
**Priority**: P1 - Fix for clean logs and robust operation
**Date**: September 5, 2025
**Document**: SSOT for RANSAC dtype and NumPy empty slice warnings

## 🟡 THE ISSUES (Both Non-Fatal)

### Issue 1: RANSAC Integer Index Error
```
"RANSAC failed: arrays used as indices must be of integer (or boolean) type"
```
- **Impact**: RANSAC bad channel detection skipped, but processing continues
- **Frequency**: Every file (100% occurrence)
- **Root Cause**: Float dtype in channel picks array passed to RANSAC

### Issue 2: NumPy Empty Slice Warnings
```
RuntimeWarning: Mean of empty slice
RuntimeWarning: invalid value encountered in scalar divide
```
- **Impact**: Autoreject CV produces empty folds on some channels
- **Frequency**: ~50% of files
- **Root Cause**: Edge case in Autoreject cross-validation with limited data

## Investigation Results

### 1. RANSAC Integer Type Issue

**Location**: `src/brain_go_brrr/infra/preprocessing/mne_preprocessor.py` lines 308-309

**The Problem**:
```python
# Current code doesn't pass picks to RANSAC
ransac = Ransac(n_jobs=1, random_state=42, verbose=False)
ransac.fit(epochs_temp)  # No picks parameter - uses internal default
```
- RANSAC's internal default pick detection may create float-typed arrays
- These get used as indices downstream, causing the type error

**Evidence from NumPy source** (`/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/lib/python3.12/site-packages/numpy/core/_methods.py`):
- NumPy requires integer indices for array indexing
- RANSAC's internal pick logic leads to float dtype in indexing operations

### 2. NumPy Mean of Empty Slice

**Location**: Autoreject's internal cross-validation

**The Problem**:
- Autoreject splits data into CV folds
- With gentle parameters (n_interpolate=[1,2], consensus=[0.5,0.7,0.9])
- Some folds end up with no valid samples after rejection
- NumPy warns when computing mean on empty array

**Evidence**:
- Occurs in `numpy/core/_methods.py:129` during mean calculation
- Triggered by Autoreject's internal validation loop

## Solutions

### Fix 1: Pass Explicit Integer Picks to RANSAC

**File**: `src/brain_go_brrr/infra/preprocessing/mne_preprocessor.py`
**Location**: Between lines 306-308 (after creating epochs_temp, before RANSAC)

```python
# After line 306 (epochs_temp creation)
# Add explicit integer picks to avoid RANSAC's internal dtype issues:
import numpy as np
ch_picks = mne.pick_types(epochs_temp.info, meg=False, eeg=True, exclude=[])
ch_picks = np.asarray(ch_picks, dtype=int)  # Force integer dtype

# Debug logging (remove after confirming fix):
logger.debug(f"RANSAC picks dtype: {ch_picks.dtype}, shape: {ch_picks.shape}")

# Modified RANSAC call with explicit picks
ransac = Ransac(
    n_jobs=1,
    random_state=42,
    picks=ch_picks,  # Now explicitly passing integer picks
    verbose=False
)
ransac.fit(epochs_temp)
```

**Rationale**: RANSAC's default pick detection may produce float-typed arrays in its internal logic. By passing explicit integer picks, we avoid this issue entirely.

### Fix 2: Add Config Option to Disable RANSAC for TUEV

**File**: `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py`
**Location**: In `__init__` method

```python
def __init__(self, config: dict[str, Any] | None = None):
    """Initialize TUEV preprocessor.
    
    Args:
        config: Optional configuration dict
            - disable_ransac: Skip RANSAC bad channel detection (default: False)
    """
    super().__init__(config)
    self.disable_ransac = (config or {}).get('disable_ransac', False)
    logger.info(f"Initialized TUEVPreprocessor (RANSAC: {'disabled' if self.disable_ransac else 'enabled'})")
```

Then override `_apply_mne_preprocessing`:

```python
def _apply_mne_preprocessing(self, raw: mne.io.Raw) -> mne.io.Raw:
    """Apply MNE preprocessing with optional RANSAC disable."""
    if self.disable_ransac:
        # Skip RANSAC, just do filtering and resampling
        raw.filter(self.bandpass_low, self.bandpass_high, picks='eeg', verbose=False)
        if self.notch_freq:
            # Apply notch at fundamental and harmonics
            raw.notch_filter([self.notch_freq, self.notch_freq * 2], picks='eeg', verbose=False)
        return raw
    else:
        # Use parent's full preprocessing including RANSAC
        return super()._apply_mne_preprocessing(raw)
```

### Fix 3: Suppress NumPy Warnings in Autoreject

**File**: `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py`
**Location**: In `_apply_autoreject_tuev` method (line ~395)

```python
def _apply_autoreject_tuev(self, epochs: mne.Epochs) -> tuple[mne.Epochs, dict[str, Any]]:
    """Apply Autoreject with gentle parameters for TUEV spike preservation."""
    from autoreject import AutoReject
    import warnings
    
    try:
        # Suppress expected NumPy warnings from empty CV folds
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
            
            ar = AutoReject(
                n_interpolate=[1, 2],  
                consensus=[0.5, 0.7, 0.9],
                cv=3,
                thresh_method='bayesian_optimization',
                random_state=42,
                verbose=False,
            )
            
            epochs_clean = ar.fit_transform(epochs)
        
        # Rest of the method...
```

## Verification Steps

### Quick Test (RANSAC Fix)
```bash
# Test with single file
uv run python -c "
import numpy as np
import mne
from pathlib import Path

# Simulate the issue
ch_picks = np.array([0.0, 1.0, 2.0])  # Float array
print(f'Float dtype: {ch_picks.dtype}')

# Fix
ch_picks_int = np.asarray(ch_picks, dtype=int)
print(f'Int dtype: {ch_picks_int.dtype}')
"
```

### Full Verification
```bash
# Test with RANSAC disabled
uv run python -c "
from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor

# Test with RANSAC disabled
proc = TUEVPreprocessor({'disable_ransac': True})
# Process a file - should see no RANSAC errors
"
```

## Implementation Priority

1. **Fix 1 (Integer dtype)**: Implement immediately - simple, safe fix
2. **Fix 3 (Warning suppression)**: Implement immediately - cosmetic improvement
3. **Fix 2 (Config option)**: Optional - only if RANSAC continues to fail after Fix 1

## Why These Issues Occurred

1. **RANSAC dtype issue**: RANSAC's internal default picks can lead to non-integer indexing; passing explicit integer picks avoids it
2. **Empty slice warnings**: TUEV has shorter recordings and aggressive AR parameters
3. **Not caught in tests**: Synthetic test data doesn't trigger these edge cases

## Definition of Done

- [ ] RANSAC runs without "arrays used as indices" error
- [ ] NumPy warnings suppressed in Autoreject context
- [ ] Optional: Config flag to disable RANSAC if needed
- [ ] Training logs are clean (no warnings)
- [ ] Cache builds successfully with >0 windows
- [ ] No regression in existing tests

## Risk Assessment

**Risk**: VERY LOW
- All fixes are defensive (try/except already in place)
- Warning suppression is scoped to specific context
- Integer casting is standard NumPy practice
- Config option provides escape hatch if needed

## Testing Commands

```bash
# 1. Test RANSAC integer fix
uv run python -c "
from brain_go_brrr.infra.preprocessing.mne_preprocessor import TUABPreprocessor
import numpy as np

# This should not error after fix
proc = TUABPreprocessor()
# Create test epochs and run RANSAC
"

# 2. Monitor cleaned logs
tmux attach -t tuev_training
# Should see:
# - No "arrays used as indices" errors
# - No "Mean of empty slice" warnings
# - Clean processing messages only
```

## Notes

1. **RANSAC is optional**: Bad channel detection improves quality but isn't critical
2. **Autoreject handles failures**: Already has internal fallbacks for edge cases
3. **Training continues**: Both issues are logged but don't stop processing
4. **Consider for future**: More robust RANSAC implementation or alternative bad channel detection

---

**STATUS**: Ready for implementation after senior review. These are quality-of-life improvements that will make logs cleaner and operation more robust.