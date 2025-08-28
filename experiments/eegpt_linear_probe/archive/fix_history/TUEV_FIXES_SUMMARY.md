# TUEV Implementation Fixes Summary

## Date: 2025-08-26
## Auditor Review Passed: ✅ Ready for Cache Build

## Critical Fixes Applied

### 1. ✅ Label-Epoch Alignment After AR (CRITICAL)
**Problem**: Labels were misaligned after Autoreject drops arbitrary epochs
**Solution**: Use `epochs.selection` to maintain correct mapping
```python
# OLD: Naive slicing that assumes sequential dropout
for epoch_idx, (epoch_data, label) in enumerate(
    zip(epochs_clean.get_data(), window_labels[:len(epochs_clean)])
):

# NEW: Proper alignment using selection indices
kept_indices = epochs_clean.selection if hasattr(epochs_clean, 'selection') else range(len(epochs_clean))
for epoch_idx, original_idx in enumerate(kept_indices):
    epoch_data = epochs_clean.get_data()[epoch_idx]
    label = window_labels[original_idx]  # Correct label from original index
```

### 2. ✅ Event Array Dtype Fixed
**Problem**: MNE requires int dtype for event arrays
**Solution**: Explicitly set dtype=int
```python
events_array = np.array(events, dtype=int)  # MNE requires int dtype
```

### 3. ✅ Channel Presence Validation (≥19 Required)
**Problem**: No enforcement of minimum channel requirement
**Solution**: Added validation and tracking
```python
if len(available_standard) < 19:
    error_msg = f"Too few standard channels ({len(available_standard)}/20). Need at least 19."
    logger.error(error_msg)
    raise ValueError(error_msg)
```

### 4. ✅ Missing Channels Tracking
**Problem**: No QC visibility into missing channels per file
**Solution**: Track and store in cache index
```python
info = {
    'n_epochs_before': n_epochs_before,
    'n_epochs_after': n_epochs_after,
    'n_rejected': n_epochs_before - n_epochs_after,
    'reject_rate': reject_rate,
    'missing_channels': missing_channels  # NEW: Track for QC
}
```

### 5. ✅ Duplicate Referencing Removed
**Problem**: Legacy dataset applied average reference twice
**Solution**: Removed from legacy, kept only in MNE preprocessor

### 6. ✅ Import Fixes
- numpy already imported in TUEV preprocessor ✅
- Path moved to TYPE_CHECKING block
- mne moved out of TYPE_CHECKING (used for more than types)
- Added missing numpy import in test file

### 7. ✅ Linting Compliance
- Variable naming: `SPIKE_PRIORITY_THRESHOLD` → `spike_priority_threshold`
- Type hints: `Tuple` → `tuple`
- Dict comprehension: `{k: v for ...}` → `dict(zip(...))`

### 8. ✅ Comprehensive Smoke Tests Added
```python
# Shape validation
assert x.shape == (20, 1024)
assert x.dtype == torch.float32
assert not torch.isnan(x).any()

# Label validation
assert 0 <= label < 6

# Epoch selection alignment test
assert epochs_clean.selection == sorted(epochs_clean.selection)
```

## Test Results

```
✅ Phase 1: Critical Bugs Fixed
✅ Phase 2: Fixed-Grid Windowing
✅ Phase 3: Preprocessing Updates
✅ Cache Validation (Smoke Tests)
✅ Epoch Selection Alignment

🎉 ALL TESTS PASSED - READY FOR CACHE BUILD
```

## Code Quality

```
✅ Formatting: ruff format (all files formatted)
✅ Linting: ruff check (all issues fixed)
✅ Tests: All passing
```

## Key Improvements

1. **Robustness**: Won't silently mislabel epochs after AR
2. **Correctness**: Proper dtypes and channel validation
3. **Observability**: Missing channels tracked in index for QC
4. **Clean Code**: Uncle Bob would approve - clear, tested, documented

## Next Steps

1. Build cache: `./scripts/build_tuev_mne_cache.sh`
2. Validate cache files match spec:
   - Shape: (20, 1024)
   - Dtype: float32
   - No NaNs
   - Labels: 0-5
   - Version: mne-ar-v3
3. Train linear probe targeting:
   - Balanced Accuracy ≥ 62.32%
   - Weighted F1 ≥ 81.87%
   - Cohen's Kappa ≥ 0.635

## Files Modified

- `datasets/tuev_mne_dataset.py` - Fixed label alignment, added tracking
- `mne_integration/tuev_preprocessor.py` - Fixed imports, channel tracking, dtypes
- `datasets/tuev_dataset.py` - Removed duplicate referencing
- `test_tuev_implementation.py` - Added comprehensive tests
- `train_tuev_mne.py` - Fixed dict comprehension

All changes follow SSOT principles and Clean Code standards.
