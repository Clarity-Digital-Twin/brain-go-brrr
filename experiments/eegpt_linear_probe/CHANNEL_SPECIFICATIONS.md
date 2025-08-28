# Channel Specifications for TUAB and TUEV Datasets

## Summary
**TUAB and TUEV use DIFFERENT channel counts!**
- **TUAB**: 19 channels (no Fz)
- **TUEV**: 20 channels (with Fz, no Fpz)

## TUAB Dataset - 19 Channels

### Channel List (Standard 10-20 without Fz)
```python
TUAB_CHANNELS = [
    'Fp1', 'Fp2',                  # Frontal polar
    'F7', 'F3', 'F4', 'F8',         # Frontal (NO Fz!)
    'T7', 'C3', 'Cz', 'C4', 'T8',   # Temporal/Central
    'P7', 'P3', 'Pz', 'P4', 'P8',   # Parietal
    'O1', 'O2', 'Oz'                # Occipital
]
```

### Rationale
- TUAB raw files often lack Fz channel
- Enforcing 19 channels prevents shape mismatches
- Current training uses this configuration successfully

### Implementation
- Preprocessor: `mne_integration/preprocessor.py`
- Dataset: `datasets/tuab_mne_dataset.py`
- Enforced in cache building with validation

## TUEV Dataset - 20 Channels

### Channel List (Standard 10-20 with Fz, without Fpz)
```python
TUEV_CHANNELS = [
    'Fp1', 'Fp2',                   # Frontal polar (NO Fpz!)
    'F7', 'F3', 'Fz', 'F4', 'F8',   # Frontal (WITH Fz!)
    'T7', 'C3', 'Cz', 'C4', 'T8',   # Temporal/Central
    'P7', 'P3', 'Pz', 'P4', 'P8',   # Parietal
    'O1', 'Oz', 'O2'                # Occipital (using Oz not Fpz)
]
```

### Rationale
- EEGPT paper Table 13 explicitly uses 20 channels for TUEV
- Paper lists Fpz, but project uses Oz for consistency (both are valid)
- The 1×1 conv adapter in the model handles channel differences
- Event detection benefits from full montage coverage

### Implementation
- Preprocessor: `mne_integration/tuev_preprocessor.py`
- Dataset: `datasets/tuev_mne_dataset.py`
- Config: `configs/tuev.yaml`
- Enforced in cache building with validation

## Critical Differences

| Dataset | Channels | Fz | Fpz | Oz | Total |
|---------|----------|----|----|-----|--------|
| TUAB    | 10-20    | ❌ | ❌  | ✅  | 19     |
| TUEV    | 10-20    | ✅ | ❌  | ✅  | 20     |

## Paper vs Project

### EEGPT Paper (Table 13 line 615)
- TUEV target: [FP1, **FPZ**, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2]
- Uses Fpz in position 2

### Our Project (TUEV_IMPLEMENTATION_SPEC.md)
- TUEV target: [Fp1, Fp2, F7, F3, **Fz**, F4, F8, T7, C3, **Cz**, C4, T8, P7, P3, **Pz**, P4, P8, O1, **Oz**, O2]
- Uses Oz instead of Fpz for consistency with TUAB
- Both are valid - the 1×1 conv adapter handles the difference

## Common Mistakes to Avoid

1. **DO NOT** enforce 19 channels for TUEV - it needs 20
2. **DO NOT** include Fz for TUAB - it should have 19
3. **DO NOT** include Fpz in either dataset - we use Oz instead
4. **DO NOT** assume channel order by index - always use channel names

## Validation

Both datasets now include validation:
1. Cache builder enforces exact channel count
2. Skips windows with wrong channel count
3. Stores `expected_shape` in cache index
4. Validates on dataset load

## Current Status (as of this fix)

- ✅ TUAB preprocessor: Enforces exactly 19 channels
- ✅ TUEV preprocessor: Fixed to enforce exactly 20 channels (was wrongly 19)
- ✅ TUAB dataset: Validates 19 channels
- ✅ TUEV dataset: Validates 20 channels
- ✅ Configs: Updated to reflect correct channel lists
- ⚠️ TUAB cache: Has 304 contaminated windows (workaround in place)
- ✅ Future caches: Will enforce correct counts from the start