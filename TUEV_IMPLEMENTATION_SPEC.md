# TUEV Implementation Specification - Single Source of Truth

## Status: DESIGN PHASE - NO CODE CHANGES YET

This document defines the **exact** implementation requirements for TUEV dataset preprocessing and training, based on:
- EEGPT paper analysis (literature/markdown/EEGPT/EEGPT.md)
- Current implementation audit (experiments/eegpt_linear_probe/)
- External expert review feedback
- TUAB successful implementation patterns

---

## 1. Critical Design Decisions

### 1.1 Montage: AVERAGE REFERENCE (Not TCP Bipolar)

**Decision**: Use **average reference** montage, NOT TCP bipolar derivations.

**Evidence from EEGPT paper**:
- Page 5: "Each dataset underwent... re-referencing (average)..." 
- Page 6: "we strictly follow the same strategy as BIOT" (BIOT uses average reference)
- No mention of bipolar montage for any dataset

**Why not TCP bipolar**: TCP bipolar creates a distribution shift from EEGPT's training data, materially changing the learned representations. TCP is kept only as an optional ablation study.

**Current Implementation Issues**:
- `experiments/eegpt_linear_probe/datasets/tuev_dataset.py` lines 314-319: **INCORRECTLY** attempts TCP bipolar
- `experiments/eegpt_linear_probe/tests/test_tuev_bipolar.py`: Entire test file should be **DELETED**
- `TCP_BIPOLAR_PAIRS` definition (lines 64-71): Should be **REMOVED**

**Correct Implementation**:
```python
# Apply average reference (same as TUAB) - use functional form
import mne
raw, _ = mne.set_eeg_reference(raw, ref_channels='average', projection=False)
```

### 1.2 Window Size: 1024 SAMPLES (4s @ 256Hz)

**Decision**: Standardize on **1024 samples** (4.0 seconds @ 256 Hz).

**Evidence**:
- EEGPT paper page 6: "input signal time length is T = 1024"
- "Each patch has a time length of d = 64" → 1024/64 = 16 patches (integer required)
- Same as TUAB implementation (verified working)

**Current Implementation**: 
- Already using 1024 in `train_tuev.py` line 118
- Cache builder expects 1024 samples

**Note**: Paper Table 13 might show 1000 samples (5s @ 200Hz) but this is incompatible with EEGPT's patch requirement (1000/64 = 15.625 patches).

### 1.3 Channel Configuration: 23→20 Mapping

**Decision**: Map TUEV's 23 channels to standard 20 channels.

**Current Mapping** (experiments/eegpt_linear_probe/mne_integration/tuev_preprocessor.py):
```python
# Drop these 3 channels:
'A1': None,  # Reference channel
'A2': None,  # Reference channel  
'Fpz': None  # Extra midline channel (note: MNE canonical casing)

# Keep standard 20 (with old→modern naming using MNE casing):
'T3' → 'T7'
'T4' → 'T8'
'T5' → 'P7'
'T6' → 'P8'
```

**This is CORRECT** - matches EEGPT's expected input format.

### 1.4 Window Labeling: ARGMAX WITH PRIORITY FOR SPIKES

**Decision**: Use **argmax overlap with ≥100ms minimum AND spike priority**.

**Critical Issue with Current Implementation**:
- Lines 239-250 in `tuev_dataset.py`: Splits events into 4-second chunks
- **WRONG**: Epileptiform discharges are typically 70-200ms, not 4 seconds!

**Correct Algorithm**:
```python
def label_window(window_start, window_end, annotations):
    """
    Label a window based on event annotations with spike priority.
    
    Args:
        window_start: Window start time in seconds
        window_end: Window end time in seconds  
        annotations: List of {'start', 'end', 'label'} dicts
    
    Returns:
        Label string (one of: 'spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg')
    """
    overlaps = {}
    
    for ann in annotations:
        # Calculate overlap in seconds
        overlap_start = max(window_start, ann['start'])
        overlap_end = min(window_end, ann['end'])
        overlap_duration = max(0, overlap_end - overlap_start)
        
        if overlap_duration > 0:
            label = ann['label']
            if label not in overlaps:
                overlaps[label] = 0
            overlaps[label] += overlap_duration
    
    # Priority override: if spike has sufficient overlap (≥120ms), prioritize it
    if overlaps.get('spsw', 0) >= 0.12:  # 120ms threshold for clear spike
        return 'spsw'
    
    # Otherwise use argmax with minimum duration
    if overlaps:
        # Only consider if overlap ≥ 100ms (0.1s)
        valid_overlaps = {k: v for k, v in overlaps.items() if v >= 0.1}
        
        if valid_overlaps:
            # Return class with max overlap
            return max(valid_overlaps, key=valid_overlaps.get)
    
    # Default to background
    return 'bckg'
```

**Priority for ties** (if equal overlap):
1. 'spsw' (spike/sharp wave - most important)
2. 'gped' (generalized periodic epileptiform)
3. 'pled' (periodic lateralized epileptiform)
4. 'eyem' (eye movement)
5. 'artf' (artifact)
6. 'bckg' (background)

### 1.5 Autoreject Parameters: GENTLER FOR SPIKE PRESERVATION

**Decision**: Use less aggressive parameters to preserve epileptiform morphology.

**TUEV-Specific Parameters**:
```python
# Gentler than TUAB (which uses [1,2,3,4])
n_interpolate = [1, 2]  # Max 2 channels interpolated
consensus = [0.5, 0.7, 0.9]  # Higher thresholds
cv = 3  # Faster than 5, sufficient for cleaner data
thresh_method = 'bayesian_optimization'
```

**Monitoring Requirements**:
- Log reject rate per split
- Log learned `n_interpolate_['eeg']` and `consensus_['eeg']`
- **FALLBACK if reject_rate > 15%**: Log warning, reduce aggressiveness or skip AR, flag in cache index (don't abort)

---

## 2. Files Requiring Changes

### 2.1 ARCHIVE These Files
- `experiments/eegpt_linear_probe/tests/test_tuev_bipolar.py` - Move to `tests/ablation/` or mark `@pytest.mark.xfail` with comment "kept for optional TCP ablation study"

### 2.2 MODIFY These Files

#### `experiments/eegpt_linear_probe/datasets/tuev_dataset.py`
- **REMOVE** lines 64-71 (TCP_BIPOLAR_PAIRS definition)
- **REMOVE** lines 314-324 (bipolar derivation logic)
- **REPLACE** lines 236-252 with correct window labeling (argmax with ≥100ms)
- **ADD** average reference application

#### `experiments/eegpt_linear_probe/mne_integration/tuev_preprocessor.py`
- **KEEP** 23→20 channel mapping (already correct)
- **UPDATE** Autoreject parameters (lines in `_apply_autoreject` method)
- **ADD** window labeling logic per specification

#### `experiments/eegpt_linear_probe/datasets/tuev_mne_dataset.py`
- **UPDATE** window labeling in `_load_annotations` method
- **ENSURE** cache version is bumped (e.g., "tuev-mne-v3")

### 2.3 Configuration Files

#### `experiments/eegpt_linear_probe/configs/tuev.yaml`
```yaml
data:
  window_samples: 1024  # KEEP THIS
  sampling_rate: 256    # KEEP THIS
  n_channels: 20        # AFTER preprocessing (was 23)

preprocessing:
  reference: 'average'  # NOT 'tcp_bipolar'
  autoreject:
    n_interpolate: [1, 2]
    consensus: [0.5, 0.7, 0.9]
    cv: 3
    max_reject_rate: 0.15  # Abort if exceeded
```

---

## 3. Cache Management

### 3.1 New Cache Directory Structure
```
data/cache/
├── tuab_mne_preprocessed/     # Existing, working perfectly
│   └── *_mne-ar-v2.pt         # Shape: (19-20, 1024)
└── tuev_avg_ref_v3/           # NEW - average reference
    ├── index_train_v3.json
    ├── index_eval_v3.json
    └── window_*_v3.pt         # Shape: (20, 1024)
```

**NEVER** mix old cache versions. The old `tuev_mne_preprocessed` should be deleted.

### 3.2 Cache Validation Requirements
Each cached tensor MUST have:
- Shape: `(20, 1024)`
- Dtype: `float32`
- No NaN values
- Label: One of 6 classes (0-5)

---

## 4. Testing Requirements

### 4.1 Critical Tests to Add

#### `tests/unit/test_tuev_window_labeling.py`
```python
def test_spike_labeling():
    """Ensure short spikes (100-200ms) get labeled correctly."""
    # Window: 0-4 seconds
    # Spike: 0.5-0.65 seconds (150ms)
    # Should label as 'spsw' not 'bckg'
    
def test_argmax_overlap():
    """Test that longest overlap wins."""
    # Window with 200ms spike, 300ms artifact
    # Should label as 'artf'
    
def test_minimum_duration():
    """Test that <100ms events are ignored."""
    # Window with 50ms spike
    # Should label as 'bckg'
```

#### `tests/unit/test_tuev_shape_contract.py`
```python
def test_cache_shape():
    """Every cached window must be (20, 1024)."""
    
def test_no_nans():
    """No NaN values in cached data."""
```

---

## 5. Logging Requirements

Each file processed must log:
```
Processing: aaaaaaar_00000001.edf
- Channels: 23 found, 20 after mapping
- Dropped: ['A1', 'A2', 'FPZ']  
- Reference: average
- Windows: 150 created
- Labels: SPSW=12, GPED=3, PLED=2, EYEM=8, ARTF=15, BCKG=110
- Autoreject: 150→142 epochs (5.3% rejected)
- AR learned: n_interpolate=1, consensus=0.7
```

---

## 6. Performance Targets

From EEGPT paper Table 3 (page 7):
- **Balanced Accuracy**: 0.6232 ± 0.0114
- **Weighted F1**: 0.8187 ± 0.0063  
- **Cohen's Kappa**: 0.6351 ± 0.0134

---

## 7. Migration Path

1. **Document Review** ← WE ARE HERE
2. **Write micro-tests** for window labeling
3. **Implement preprocessor changes**
4. **Build new cache** with correct parameters
5. **Verify shapes and labels**
6. **Train and compare metrics**

---

## References

- EEGPT Paper: `/literature/markdown/EEGPT/EEGPT.md`
- TUAB Working Implementation: `/experiments/eegpt_linear_probe/mne_integration/preprocessor.py`
- Current TUEV Code: `/experiments/eegpt_linear_probe/datasets/tuev_dataset.py`
- External Audit Feedback: Inline comments throughout