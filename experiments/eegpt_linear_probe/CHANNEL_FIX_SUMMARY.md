# Channel Enforcement Fix Summary

## What Was Fixed

### The Critical Issue
- **TUEV was incorrectly enforced to 19 channels** (like TUAB)
- **TUEV should have 20 channels** per EEGPT paper Table 13
- This was a **breaking change** that would have hurt TUEV model performance

### Changes Made

1. **TUEV Preprocessor** (`mne_integration/tuev_preprocessor.py`)
   - ✅ Changed assertion from 19 to 20 channels
   - ✅ Updated enforcement logic to require exactly 20
   - ✅ Fixed comments to clarify TUEV uses 20 (not 19)

2. **TUEV Config** (`configs/tuev.yaml`)
   - ✅ Fixed `target_20` channel list (was missing channels)
   - ✅ Now correctly lists all 20 channels with Fz, without Fpz
   - ✅ Added comments explaining we use Oz instead of Fpz

3. **TUEV Dataset** (`datasets/tuev_mne_dataset.py`)
   - ✅ Added `expected_shape: [20, 1024]` to cache index
   - ✅ Added channel validation during cache building
   - ✅ Windows with wrong channel count will be skipped

4. **Documentation**
   - ✅ Created `CHANNEL_SPECIFICATIONS.md` - comprehensive guide
   - ✅ Updated `TECH_DEBT_CRITICAL.md` to clarify both specs
   - ✅ Created test script to verify enforcement

### What Was Already Correct

1. **TUAB** - No changes needed
   - Already enforces 19 channels correctly
   - Collate workaround handles current cache contamination
   - Future cache builds will enforce 19 from the start

## Verification

Run the test script to verify:
```bash
uv run python test_channel_enforcement.py
```

Expected output:
```
✅ TUAB Preprocessor: 19 channels (no Fz)
✅ TUEV Preprocessor: 20 channels (with Fz, no Fpz)
✅ TUEV Config: 20 channels correctly specified
```

## Why This Matters

1. **Model Performance**: Using wrong channel count degrades performance
2. **Reproducibility**: Results must match paper specifications
3. **Event Detection**: TUEV needs full montage (20ch) for accurate IED detection
4. **Future Training**: TUEV cache must be built correctly from the start

## Current Status

- **TUAB Training**: Running with 19 channels + collate workaround ✅
- **TUEV**: Ready for training with correct 20-channel enforcement ✅
- **Code Quality**: All linting checks pass ✅
- **Documentation**: Clear specs for both datasets ✅

## Key Takeaway

**NEVER ASSUME** TUAB and TUEV have the same requirements:
- **TUAB = 19 channels** (no Fz)
- **TUEV = 20 channels** (with Fz, no Fpz)

Both are correct for their respective datasets!
