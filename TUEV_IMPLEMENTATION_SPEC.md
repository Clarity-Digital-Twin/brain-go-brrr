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

**Why not TCP bipolar**: TCP bipolar creates a distribution shift from EEGPT's training data. EEGPT was trained on average-referenced data, and using TCP would harm feature extraction. We're focusing on high-ROI approaches only.

**Current Implementation Issues**:
- `experiments/eegpt_linear_probe/datasets/tuev_dataset.py`: Contains TCP bipolar code that must be **REMOVED**
- `experiments/eegpt_linear_probe/tests/test_tuev_bipolar.py`: **DELETE** this file entirely
- All TCP/bipolar constants and functions: **DELETE** completely

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

### 1.6 Filtering Guard: NYQUIST SAFETY

**Decision**: Clamp high-frequency filter below Nyquist limit.

```python
# Guard against Nyquist violations (same as TUAB implementation)
sfreq = raw.info['sfreq']
high_freq = min(high_freq, 0.49 * sfreq)  # Stay below Nyquist
if high_freq <= low_freq:
    logger.warning(f"Skipping filter: high_freq {high_freq} <= low_freq {low_freq}")
    # Skip filtering for this file
```

---

## 2. Files Requiring Changes

### 2.1 DELETE These Files
- `experiments/eegpt_linear_probe/tests/test_tuev_bipolar.py` - DELETE completely (no ablation needed)

### 2.2 MODIFY These Files

#### `experiments/eegpt_linear_probe/datasets/tuev_dataset.py`
- **REMOVE** lines 43-72: All TCP channel/bipolar pair definitions
- **REMOVE** lines 75-120: `compute_bipolar_derivation()` function entirely
- **REMOVE** lines 286-329: All bipolar derivation logic in `__getitem__()`
- **REPLACE** lines 236-252: Window labeling with correct argmax+priority algorithm
- **ADD** average reference using `mne.set_eeg_reference()` functional call

#### `experiments/eegpt_linear_probe/mne_integration/tuev_preprocessor.py`
- **KEEP** 23→20 channel mapping (already correct but fix casing: `Fpz` not `FPZ`)
- **UPDATE** Autoreject parameters in `_apply_autoreject()` method
- **ADD** window labeling logic with spike priority per specification

#### `experiments/eegpt_linear_probe/datasets/tuev_mne_dataset.py`
- **UPDATE** window labeling in `_load_annotations()` method to use priority algorithm
- **ENSURE** cache version is bumped to "tuev_mne-ar-v3" (consistent naming)

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
└── tuev_mne-ar-v3/            # NEW - consistent naming with TUAB
    ├── index_train.json
    ├── index_eval.json
    └── window_*_mne-ar-v3.pt  # Shape: (20, 1024)
```

**NEVER** mix old cache versions. The old `tuev_mne_preprocessed` should be deleted.

### 3.2 Cache Validation Requirements
Each cached tensor MUST have:
- Shape: `(20, 1024)`
- Dtype: `float32`
- No NaN values
- Label: One of 6 classes (0-5)

Cache index should include:
- QC flags (e.g., "high_reject_rate", "fallback_applied")
- Autoreject metrics (reject_rate, n_interpolate, consensus)
- Missing/mapped channels per file

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
    
def test_spike_priority():
    """Test that spikes get priority when sufficient."""
    # Window with 200ms spike, 300ms artifact
    # Should label as 'spsw' (spike priority)
    
def test_argmax_overlap():
    """Test that longest overlap wins (when no spike priority)."""
    # Window with 50ms spike, 300ms artifact
    # Should label as 'artf' (spike too short for priority)
    
def test_minimum_duration():
    """Test that <100ms events are ignored."""
    # Window with 50ms spike
    # Should label as 'bckg'
```

#### `tests/unit/test_tuev_shape_contract.py`
```python
def test_cache_shape():
    """Every cached window must be (20, 1024) float32."""
    
def test_no_nans():
    """No NaN values in cached data."""
    
def test_labels_valid():
    """Labels must be in {0: spsw, 1: gped, 2: pled, 3: eyem, 4: artf, 5: bckg}."""

def test_qc_flags_logged():
    """Cache index contains QC flags for high reject rate files."""
```

---

## 5. Logging Requirements

Each file processed must log:
```
Processing: aaaaaaar_00000001.edf
- Channels: 23 found, 20 after mapping
- Dropped: ['A1', 'A2', 'Fpz']  # Note: MNE canonical casing
- Reference: average (functional call)
- Windows: 150 created
- Labels: SPSW=12, GPED=3, PLED=2, EYEM=8, ARTF=15, BCKG=110
- Autoreject: 150→142 epochs (5.3% rejected)
- AR learned: n_interpolate=1, consensus=0.7
- QC flags: None (or "high_reject_rate" if >15%)
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

## 8. Final Summary: The TUEV Pipeline

**NO EXPERIMENTS, NO ABLATIONS** - Just one clean implementation:

1. **Input**: TUEV EDF files with 23 channels @ 250Hz
2. **Channel mapping**: 23→20 (drop A1, A2, Fpz; rename T3/T4/T5/T6)
3. **Reference**: Average reference ONLY (matches EEGPT training)
4. **Filtering**: 0.5-45Hz bandpass + 60Hz notch (with Nyquist guard)
5. **Resampling**: 250→256Hz (EEGPT requirement)
6. **Windowing**: 4 seconds (1024 samples) with 50% overlap
7. **Labeling**: Argmax overlap ≥100ms WITH spike priority ≥120ms
8. **Autoreject**: Gentle parameters (n_interpolate=[1,2], consensus=[0.5,0.7,0.9])
9. **Output**: (20, 1024) float32 tensors with labels 0-5
10. **Cache**: `tuev_mne-ar-v3/` with QC flags in index

**Success Metrics**:
- Balanced Accuracy: ≥62.32%
- Weighted F1: ≥81.87%
- Cohen's Kappa: ≥0.635

---

## References

- EEGPT Paper: `/literature/markdown/EEGPT/EEGPT.md`
- TUAB Working Implementation: `/experiments/eegpt_linear_probe/mne_integration/preprocessor.py`
- Current TUEV Code: `/experiments/eegpt_linear_probe/datasets/tuev_dataset.py`