# Reference Repositories Analysis for EEGPT Linear Probe

## Critical for EEGPT Linear Probe Implementation

### 1. **EEGPT** ✅ (Already cloned)
- **Why Critical**: Contains the actual linear probe implementations we're following
- **Key Files**:
  - `downstream/linear_probe_EEGPT_*.py` - Linear probe examples
  - `downstream/Modules/models/EEGPT_mcae.py` - Model architecture
  - `downstream/Data_process/` - Data preprocessing pipelines

### 2. **mne-python** ✅ (Cloned)
- **Why Critical**: Core EEG data loading and preprocessing
- **Usage**:
  - Loading EDF files
  - Channel montage handling
  - Filtering and resampling
  - Already used extensively in our codebase

### 3. **yasa** ✅ (Cloned)
- **Why Critical**: For sleep staging evaluation and comparison
- **Usage**:
  - Sleep stage metrics
  - Hypnogram generation
  - Already integrated in `services/sleep_metrics.py`

### 4. **autoreject** ✅ (Cloned)
- **Why Critical**: Quality control before EEGPT
- **Usage**:
  - Artifact rejection
  - Bad channel detection
  - Already integrated in `services/qc_flagger.py`

## Nice to Have (Not Critical for Linear Probe)

### 5. **braindecode** ❌ (Skip for now)
- **Why Not Critical**: We're using EEGPT, not their models
- **Could be useful for**: Dataset utilities, but we already have our own

### 6. **mne-bids** ❌ (Skip for now)
- **Why Not Critical**: TUAB dataset isn't in BIDS format
- **Could be useful for**: Future BIDS dataset support

### 7. **pyEDFlib** ❌ (Skip for now)
- **Why Not Critical**: We already use `edfio` successfully
- **Could be useful for**: Performance optimization later

### 8. **tsfresh** ❌ (Skip for now)
- **Why Not Critical**: Not needed for EEGPT linear probe
- **Could be useful for**: Feature engineering in `snippet_maker.py`

## Implementation Priority

1. **First**: Use EEGPT repo to implement LinearWithConstraint and basic probe structure
2. **Second**: Use mne-python for data loading and preprocessing
3. **Third**: Use autoreject for quality control (optional but recommended)
4. **Fourth**: Use yasa for sleep staging evaluation metrics

## Key Code to Reference

### From EEGPT
```python
# Channel adaptation
self.chan_conv = nn.Conv1d(
    in_channels=self.chans_num,
    out_channels=58,
    kernel_size=1,
    bias=True
)

# Linear probe with constraint
self.linear_probe1 = LinearWithConstraint(embed_dim*4, embed_dim*4, max_norm=0.25)
self.linear_probe2 = LinearWithConstraint(embed_dim*4, n_classes, max_norm=0.25)
```

### From mne-python (already in our code)
```python
import mne
raw = mne.io.read_raw_edf(edf_path, preload=True)
raw.filter(0.5, 50.0)  # Bandpass filter
raw.resample(256)  # Resample to 256 Hz
```

### From autoreject (already in our code)
```python
from autoreject import AutoReject
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs)
```

### From yasa (already in our code)
```python
import yasa
hypno = yasa.predict(raw, eeg_name="C3-M2")
```

## Recommendation

Focus on:
1. **EEGPT** - For the exact implementation pattern
2. **mne-python** - For robust data handling
3. **autoreject** - For quality control (optional)
4. **yasa** - For sleep staging metrics only

Skip the others for now to avoid over-complexity. We can always add them later if needed.
