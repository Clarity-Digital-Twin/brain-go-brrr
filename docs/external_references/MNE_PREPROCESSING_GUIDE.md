# MNE-Python Preprocessing Documentation

## Overview

MNE-Python provides comprehensive tools for preprocessing MEG/EEG data. This guide covers essential preprocessing techniques based on official MNE documentation.

## Table of Contents
1. [Filtering](#filtering)
2. [Artifact Detection](#artifact-detection)
3. [Bad Channel Handling](#bad-channel-handling)
4. [Re-referencing](#re-referencing)
5. [ICA for Artifact Removal](#ica-for-artifact-removal)
6. [Epoching and Baseline Correction](#epoching-and-baseline-correction)

## Filtering

### Bandpass Filtering
```python
import mne

# High-pass filter to remove slow drifts
raw.filter(l_freq=0.5, h_freq=None)

# Low-pass filter to remove high-frequency noise
raw.filter(l_freq=None, h_freq=50.0)

# Bandpass filter (commonly used for EEG)
raw.filter(l_freq=0.5, h_freq=50.0)
```

### Notch Filtering (Remove Line Noise)
```python
# Remove 60 Hz line noise (US) or 50 Hz (Europe)
raw.notch_filter(freqs=60)  # or 50 for Europe

# Remove harmonics as well
raw.notch_filter(freqs=[60, 120, 180])
```

### Filter Parameters
- `method`: 'fir' (default) or 'iir'
- `fir_design`: 'firwin' or 'firwin2'
- `phase`: 'zero' (non-causal) or 'minimum' (causal)
- `filter_length`: 'auto' or specific length

## Artifact Detection

### Automated Detection Methods

1. **Amplitude-based Detection**
```python
# Find epochs with extreme values
from mne.preprocessing import annotate_amplitude

annotations, bad_epochs = annotate_amplitude(
    raw, 
    peak=dict(eeg=150e-6),  # 150 µV threshold
    flat=dict(eeg=5e-6)     # 5 µV flat threshold
)
raw.set_annotations(raw.annotations + annotations)
```

2. **Muscle Artifact Detection**
```python
from mne.preprocessing import annotate_muscle_zscore

# Detect muscle artifacts using z-score
annotations_muscle, scores_muscle = annotate_muscle_zscore(
    raw, 
    threshold=5,  # z-score threshold
    ch_type='eeg'
)
```

3. **Movement Artifact Detection (MEG only)**
```python
# NOTE: annotate_movement requires head position data, typically only available in MEG recordings
from mne.preprocessing import annotate_movement

# Detect movement artifacts (MEG with head position tracking)
annotations_movement, displacement = annotate_movement(
    raw, 
    pos=head_pos,  # head position data (MEG-specific)
    threshold=0.01
)
```

## Bad Channel Handling

### Manual Identification
```python
# Mark bad channels manually
raw.info['bads'] = ['EEG 003', 'EEG 007']

# Or interactively
raw.plot()  # Click on channels to mark as bad
```

### Automated Detection
```python
# Using RANSAC (from Autoreject)
from autoreject import Ransac
ransac = Ransac(n_jobs=1, random_state=42)
epochs_clean = ransac.fit_transform(epochs)

# Get bad channels
print(ransac.bad_chs_)
```

### Channel Interpolation
```python
# Interpolate bad channels
raw.interpolate_bads(reset_bads=True)

# For epochs
epochs.interpolate_bads(reset_bads=True)
```

## Re-referencing

### Common Average Reference (CAR)
```python
# Set average reference
raw.set_eeg_reference('average', projection=True)

# Apply the projection
raw.apply_proj()
```

### Linked Mastoids Reference
```python
# Reference to mastoid channels
raw.set_eeg_reference(['M1', 'M2'])
```

### REST Reference
```python
# REST reference requires a forward model
# fwd = mne.read_forward_solution('subject-fwd.fif')
# raw, _ = mne.set_eeg_reference(raw, ref_channels='REST', forward=fwd)
# For demonstration, using average reference:
raw, _ = mne.set_eeg_reference(raw, 'average')
```

### Bipolar Reference
```python
# Use the function form (not method)
import mne
# Create bipolar montage (result = anode - cathode)
raw = mne.set_bipolar_reference(
    raw,
    anode=['Fp1', 'F3'],
    cathode=['F3', 'C3']
)
```

## ICA for Artifact Removal

### Basic ICA Workflow
```python
from mne.preprocessing import ICA

# 1. Create ICA object
ica = ICA(
    n_components=20,  # Number of components
    method='fastica',  # or 'infomax', 'picard'
    random_state=42,
    max_iter=800
)

# 2. Fit ICA (use filtered data)
filtered_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
ica.fit(filtered_raw)

# 3. Find EOG components automatically
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices

# 4. Find ECG components
ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
ica.exclude.extend(ecg_indices)

# 5. Apply ICA to remove artifacts
raw_clean = ica.apply(raw.copy())
```

### Advanced ICA Parameters
```python
ica = ICA(
    n_components=0.95,  # Explain 95% of variance
    method='picard',    # Faster than fastica
    fit_params=dict(ortho=False),  # For picard
    max_iter='auto',
    random_state=42
)
```

### Component Selection
```python
# Manual inspection
ica.plot_components()  # Topographies
ica.plot_sources(raw)  # Time series

# Automated selection based on correlation
from mne.preprocessing import corrmap

# Use template from one subject
template_eog_component = ica_template.get_components()[:, 0]
corrmap([ica], template=template_eog_component, threshold=0.9)
```

## Epoching and Baseline Correction

### Creating Epochs
```python
# Define events
events = mne.find_events(raw)

# Create epochs
epochs = mne.Epochs(
    raw, 
    events,
    event_id={'stimulus': 1},
    tmin=-0.2,  # Start 200ms before event
    tmax=0.8,   # End 800ms after event
    baseline=(-0.2, 0),  # Baseline correction
    preload=True
)
```

### Baseline Correction Options
```python
# Different baseline periods
baseline = (-0.2, 0)    # Pre-stimulus baseline
baseline = (None, 0)    # From beginning to stimulus
baseline = (0.1, 0.2)   # Post-stimulus baseline
baseline = None         # No baseline correction

epochs = mne.Epochs(raw, events, baseline=baseline)
```

### Epoch Rejection
```python
# Rejection based on peak-to-peak amplitude
reject = dict(
    eeg=150e-6,     # 150 µV
    eog=250e-6      # 250 µV
)

# Flat channel detection
flat = dict(
    eeg=5e-6        # 5 µV
)

epochs = mne.Epochs(
    raw, events,
    reject=reject,
    flat=flat,
    reject_by_annotation=True
)
```

## Complete Preprocessing Pipeline Example

```python
import mne
from mne.preprocessing import ICA
from autoreject import AutoReject

def preprocess_eeg(raw_file_path):
    """Complete EEG preprocessing pipeline."""
    
    # 1. Load data
    raw = mne.io.read_raw_edf(raw_file_path, preload=True)
    
    # 2. Set channel types and montage
    raw.set_channel_types({'ECG': 'ecg', 'EOG': 'eog'})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # 3. Filter data
    raw.filter(l_freq=0.5, h_freq=50.0, fir_design='firwin')
    raw.notch_filter(freqs=60)  # Remove line noise
    
    # 4. Detect bad channels (manual or automated)
    # Manual: raw.info['bads'] = ['EEG 003']
    # Or use RANSAC from autoreject
    
    # 5. Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    
    # 6. Re-reference to average
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    
    # 7. ICA for artifact removal
    ica = ICA(n_components=20, random_state=42)
    ica.fit(raw.copy().filter(l_freq=1.0, h_freq=None))
    
    # Find and exclude EOG/ECG components
    eog_indices, _ = ica.find_bads_eog(raw)
    ecg_indices, _ = ica.find_bads_ecg(raw)
    ica.exclude = eog_indices + ecg_indices
    
    # Apply ICA
    raw = ica.apply(raw)
    
    # 8. Epoch data
    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=2.0,
        baseline=(0, 0.1),
        preload=True
    )
    
    # 9. Apply Autoreject
    ar = AutoReject(random_state=42, n_jobs=1)
    epochs_clean = ar.fit_transform(epochs)
    
    return epochs_clean
```

## Best Practices

### Order of Operations
1. **Channel setup** (types, locations, bad channels)
2. **Filtering** (high-pass, then low-pass/notch)
3. **Bad channel interpolation**
4. **Re-referencing**
5. **ICA** (optional but recommended)
6. **Epoching**
7. **Artifact rejection** (Autoreject or threshold-based)
8. **Baseline correction**

### Memory Management
```python
# Use preload=False for large files
raw = mne.io.read_raw_edf(file_path, preload=False)

# Process in chunks
raw.filter(l_freq=0.5, h_freq=50.0, n_jobs=4)

# Delete unnecessary objects
del raw  # Free memory
```

### Parallel Processing
```python
# Use multiple cores
from mne.parallel import parallel_func

# For ICA
ica = ICA(n_components=20)  # Note: ICA doesn't support n_jobs parameter

# For Autoreject
ar = AutoReject(n_jobs=4)  # Autoreject does support n_jobs
```

## Common Pitfalls and Solutions

### Issue: ICA removes too much signal
**Solution**: Reduce number of components or be more selective in component rejection

### Issue: Filtering introduces artifacts
**Solution**: Use appropriate filter length and transition bandwidth

### Issue: Reference choice affects results
**Solution**: Try multiple references and validate with known patterns

### Issue: Autoreject removes too many epochs
**Solution**: Adjust consensus parameter or use local rejection

## References

- Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python
- Jas et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG
- MNE-Python Documentation: https://mne.tools/stable/index.html