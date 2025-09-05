# P0 CRITICAL: TUEV Cache Building - Complete Fix Documentation

**Status**: 🔴 BLOCKING - Cannot train TUEV without resolution
**Priority**: P0 - Training completely blocked
**Date**: September 5, 2025
**Owner**: Engineering Team
**Document**: This is the SINGLE AUTHORITATIVE document for TUEV fixes

## THE ACTUAL TRUTH (After Complete Investigation)

### What EEGPT Paper Says (Line 615)
**20 channels**: [FP1, **FPZ**, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2]
- **Has FPZ** ✓
- **No OZ** ✓

### What TUEV Files Actually Have (Verified with MNE)
Checked multiple files including `data/datasets/tuev/edf/eval/000/bckg_000_a_.edf`:
- **No FPZ** in any files (0/5 checked)
- **No OZ** in any files (0/5 checked)
- **HAS FZ, PZ** (midline channels present)
- Has: FP1, FP2, F3, F4, F7, F8, **FZ**, T3, T4, T5, T6, C3, C4, **CZ**, P3, P4, **PZ**, O1, O2 (+ A1, A2, extras)
- **Uses OLD naming**: T3, T4, T5, T6 (not T7, T8, P7, P8)

### What Our Code Currently Expects
- `configs/tuev.yaml` line 102: Uses **OZ** not FPZ (wrong per paper)
- `CHANNELS_TUEV_20`: Has **Oz**, missing **Fpz** (wrong per paper)

## The Problem

**THREE-WAY MISMATCH**:
1. **EEGPT Paper**: Wants FPZ, no OZ
2. **TUEV Files**: Have neither FPZ nor OZ  
3. **Our Code**: Expects OZ, drops FPZ

**Result**: Training blocked because we expect channels that don't exist

## The Root Cause

### What TUEV Actually Has (Data Reality from MNE checks)
- **359 training files, 159 eval files**
- **Standard 10/20 configuration** with TCP montage
- **MNE verified channels**: FP1, FP2, F7, F3, F4, F8, **FZ**, T3, C3, **CZ**, C4, T4, T5, P3, P4, **PZ**, T6, O1, O2, A1, A2
- **NOTE**: Uses OLD naming (T3/T4/T5/T6 not T7/T8/P7/P8)
- **MISSING**: **FPZ**, **OZ** (confirmed absent in all sampled files)
- **Variation**: AAREADME suggests FZ/PZ missing, but MNE checks show they're present in sampled files

### What EEGPT Paper Says (Table 13, line 615)
```
"The 23-channel input is first to reduce the number of channels to 20...
[FP1, FPZ, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2]"
```
- **Has FPZ** (we need to keep it)
- **Has FZ, PZ** (midline channels)
- **NO OZ** (paper doesn't expect it)

### What Our Code Expects (WRONG)
```python
CHANNELS_TUEV_20 = [..., "O1", "O2", "Oz"]  # Expects Oz - WRONG!
```

## The Channel Mismatch Matrix (CORRECTED)

| Channel | TUEV Files | EEGPT Paper | Our Code | Problem |
|---------|------------|-------------|----------|---------|
| FPZ | ✗ NO | ✓ YES | ✗ NO | Paper wants it, files don't have it, we drop it |
| FZ | ✓ YES | ✓ YES | ✓ YES | Present in files |
| PZ | ✓ YES | ✓ YES | ✓ YES | Present in files |
| OZ | ✗ NO | ✗ NO | ✓ YES | Never exists anywhere, but we expect it (WRONG) |

Note: All 5 TUEV files checked had FZ and PZ but lacked FPZ and OZ

## Critical Code Locations

### 1. Channel Definition (WRONG)
**File**: `src/brain_go_brrr/infra/data/channels.py` (lines 35-56)
```python
# CURRENT - WRONG (expects Oz, missing Fpz)
CHANNELS_TUEV_20 = [
    "Fp1", "Fp2",  # Missing Fpz! (should be between Fp1 and Fp2)
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2", "Oz"  # Has Oz - wrong per paper!
]
```

### 2. Preprocessor Channel Dropping (WRONG)
**File**: `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py` (line 128)
```python
# CURRENT - Drops Fpz (wrong per paper)
if ch_name in ['A1', 'A2', 'Fpz']:  # Should NOT drop Fpz!
    channels_to_drop.append(ch_name)
```

### 3. Preprocessor Assertion (TOO STRICT)
**File**: `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py` (lines 161, 187)
```python
# Requires EXACTLY 20 channels
assert len(available_standard) == 20  # Crashes with 19!
```

### 4. Cache Building Implementation (BLOCKED)
**File**: `src/brain_go_brrr/infra/data/tuev_dataset.py` (line 140)
```python
# Currently raises NotImplementedError (safe)
raise NotImplementedError("Building TUEV cache requires...")
```

## The Solution: ONE CORRECT Approach from First Principles

### THE ONLY CORRECT OPTION: Follow EEGPT Paper (Use FPZ, No OZ)

**Why this is correct**:
1. **Paper is authoritative**: EEGPT Table 13 (line 615) explicitly specifies the canonical 20 channels
2. **Reproducibility**: To reproduce paper results, we must match their exact channel interface
3. **Model flexibility**: EEGPT encoder accepts variable channels; the 20-channel contract is in preprocessing

**The canonical 20 channels (per paper)**:
```python
CHANNELS_TUEV_20 = [
    "Fp1", "Fpz", "Fp2",  # Fpz included (will synthesize as zeros)
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2"  # No Oz per paper
]
```

**Handling missing channels**:
- **FPZ**: Not in TUEV files → synthesize as zeros
- **OZ**: Not in paper, not in files → DO NOT include
- **T3→T7, T4→T8, T5→P7, T6→P8**: Map old naming to modern

## Implementation: Follow Paper's Canonical 20

### Fix 1: UPDATE CHANNELS_TUEV_20 to match paper
**File**: `src/brain_go_brrr/infra/data/channels.py`
```python
# CORRECT - Match EEGPT paper Table 13 line 615
CHANNELS_TUEV_20 = [
    "Fp1", "Fpz", "Fp2",  # Include Fpz per paper
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2"  # No Oz per paper
]
```

### Fix 2: Stop dropping Fpz, synthesize if missing
**File**: `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py`

Update channel dropping (line 128):
```python
# CORRECT - Drop only A1, A2 (not Fpz!)
if ch_name in ['A1', 'A2']:  # Remove 'Fpz' from this list
    channels_to_drop.append(ch_name)
```

Add synthesis for missing canonical channels:
```python
def _synthesize_missing_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
    """Synthesize missing canonical channels as zeros."""
    canonical_20 = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                    "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
                    "O1", "O2"]
    
    for ch in canonical_20:
        if ch not in raw.ch_names and ch.lower() not in [c.lower() for c in raw.ch_names]:
            import numpy as np
            info = mne.create_info([ch], raw.info['sfreq'], ['eeg'])
            zero_data = np.zeros((1, len(raw.times)))
            zero_raw = mne.io.RawArray(zero_data, info)
            raw.add_channels([zero_raw])
            logger.info(f"Synthesized missing {ch} channel as zeros")
    return raw
```

### Fix 3: Accept 19-20 channels with padding
**File**: `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py`, lines 161-189
```python
# CORRECT - Accept files with missing canonical channels
if len(available_standard) < 19:
    raise ValueError(f"Too few channels ({len(available_standard)})")
    
# Will be exactly 20 after synthesis
if len(available_standard) != 20:
    logger.info(f"Have {len(available_standard)} channels, will pad to 20")
```

### Fix 4: Update Config to match canonical 20
**File**: `experiments/eegpt_linear_probe/configs/tuev.yaml` line 96-103
```yaml
channels:
  target_20: [
    'FP1', 'FPZ', 'FP2',                # Include FPZ per paper
    'F7', 'F3', 'FZ', 'F4', 'F8',       # Frontal line
    'T7', 'C3', 'CZ', 'C4', 'T8',       # Temporal/Central
    'P7', 'P3', 'PZ', 'P4', 'P8',       # Parietal
    'O1', 'O2'                          # Occipital (NO OZ!)
  ]
```

### Fix 5: Cache Building Requirements (when implemented)
**File**: `src/brain_go_brrr/infra/data/tuev_dataset.py`, line 140
```python
# Keep as NotImplementedError until preprocessing fixed
raise NotImplementedError("Building TUEV cache requires fixed preprocessing")
```

**Critical Cache Requirements**:
- **Data format**: Store as float32 tensors, shape (20, 1024)
- **Units**: Store in **millivolts (mV)** - loader validates META.unit == 'mV'
- **META.json must include**:
  ```json
  {
    "unit": "mV",
    "channels": ["Fp1", "Fpz", "Fp2", ...],  // canonical 20
    "channel_policy": {
      "canonical": ["Fp1", "Fpz", "Fp2", ...],
      "mapping": {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"},
      "fill_missing": "zeros"
    }
  }
  ```

## Why This Happened

1. **Documentation Confusion**: EEGPT paper Table 13 wasn't clear about Fpz/Oz
2. **Dataset Reality**: TUEV files have non-standard channel sets
3. **Copy-Paste Error**: Someone copied TUAB config and modified wrongly
4. **No Testing**: Code never ran against actual TUEV files

## Validation Testing

### Test 1: Channel Availability
```python
# Check what channels TUEV files actually have
import mne
raw = mne.io.read_raw_edf('data/datasets/tuev/edf/train/any_file.edf')
channels = [ch.upper().replace('EEG ', '').replace('-REF', '') for ch in raw.ch_names]
print("Has FPZ:", 'FPZ' in channels)  # False - files don't have it
print("Has OZ:", 'OZ' in channels)    # False - files don't have it
print("Has FZ:", 'FZ' in channels)    # True - files have this
print("Has PZ:", 'PZ' in channels)    # True - files have this
```

### Test 2: After Fix
```python
# Should successfully build cache
from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset
dataset = TUEVMNEDataset(
    root_dir=Path('data/datasets/tuev'),
    split='train',
    force_rebuild=True
)
print(f"Success: {len(dataset)} windows")
```

## Risk Assessment

**Risk**: MEDIUM
- Changing channel expectations could affect model compatibility
- Need to verify EEGPT model accepts Fpz instead of Oz

**Mitigation**:
1. Test with small subset first
2. Verify model embedding layer accepts new config
3. Keep old config as fallback option

## Definition of Done

- [ ] CHANNELS_TUEV_20 updated to match EEGPT paper (with Fpz, without Oz)
- [ ] Preprocessor stops dropping Fpz
- [ ] Preprocessor synthesizes missing canonical channels (including Fpz)
- [ ] Config updated to canonical 20 (no Oz)
- [ ] Old naming mapped (T3→T7, T4→T8, T5→P7, T6→P8)
- [ ] Training starts successfully
- [ ] No more "missing Oz" errors

## THE CORRECT APPROACH: Follow the Paper

**From first principles**:
1. **Goal**: Reproduce EEGPT paper results on TUEV
2. **Paper specifies**: Canonical 20 with FPZ, without OZ (Table 13 line 615)
3. **Data reality**: Files lack FPZ (and OZ), have old naming
4. **Solution**: Standardize to paper's canonical 20, synthesize missing as zeros

**Why NOT Option B (keeping Oz)**:
- Contradicts the paper (no Oz in Table 13)
- Oz doesn't exist in files either
- Creates unnecessary drift from published results
- "Consistency" argument invalid when starting point was wrong

**The bulletproof implementation**:
1. Set CHANNELS_TUEV_20 = paper's canonical 20 (Fpz in, Oz out)
2. Map old naming, drop A1/A2, synthesize missing canonical channels
3. Update config and all references to match
4. Document clearly that we follow Table 13 line 615

## Appendix: Investigation Evidence

### From AAREADME.txt
- 359 training files confirmed
- TCP montage with standard 10/20 (but missing some midline)
- Uses old naming (T3/T4/T5/T6)

### From File Investigation (MNE checks on 5 files)
- 0/5 files had Oz channel (confirmed absent)
- 0/5 files had Fpz channel (confirmed absent)  
- 5/5 files had Fz and Pz (midline channels present)
- Variable total channels (27-33) due to extra non-EEG

### From EEGPT Paper
- Table 13 explicitly lists 20 channels with Fpz, without Oz
- Conv1d reduces 23→20 (drops A1, A2, something else)

---

**DELETE THESE REDUNDANT FILES AFTER REVIEW**:
- TUEV_CACHE_ISSUE.md (early investigation)
- TUEV_CRITICAL_INVESTIGATION.md (channel discovery)
- TUEV_FINAL_ANALYSIS.md (paper analysis)
- P0_TUEV_CACHE_FIX.md (incomplete fix)

**KEEP THIS ONE**: P0_TUEV_COMPLETE_FIX.md (authoritative)