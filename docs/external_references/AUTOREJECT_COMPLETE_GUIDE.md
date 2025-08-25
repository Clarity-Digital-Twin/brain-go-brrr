# Autoreject Complete Guide

*Based on official documentation and source code analysis*

## Overview

Autoreject automatically cleans epochs by learning optimal rejection thresholds from the data. It provides:
- Automated artifact rejection for MEG/EEG
- Data-driven threshold selection
- Repair vs reject decisions
- Cross-validation for robustness

## Installation

```bash
pip install -U autoreject
# or
conda install -c conda-forge autoreject
```

## Core Concepts

### 1. Local vs Global Rejection

**Local Rejection**: Channel-specific thresholds
- Different threshold for each channel
- Adapts to channel-specific noise characteristics

**Global Rejection**: Single threshold for all channels
- Simpler but less adaptive
- Useful for consistent noise across channels

### 2. Repair vs Reject

Autoreject makes two decisions for each epoch:
1. **Repair**: Interpolate bad channels if only a few are bad
2. **Reject**: Discard entire epoch if too many channels are bad

## Basic Usage

### Simple Example
```python
from autoreject import AutoReject
import mne

# Load your data
raw = mne.io.read_raw_edf('your_file.edf', preload=True)
events = mne.make_fixed_length_events(raw, duration=2.0)
epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)

# Apply autoreject
ar = AutoReject(random_state=42, n_jobs=1)
epochs_clean = ar.fit_transform(epochs)

# Check what was rejected
print(f"Rejected {ar.reject_log.bad_epochs.sum()} epochs")
```

## AutoReject Class

### Key Parameters

```python
AutoReject(
    n_interpolate=None,      # List of values to try for channel interpolation
    consensus=None,          # Consensus parameter values to try  
    thresh_method='bayesian_optimization',  # How to find optimal threshold
    cv=10,                  # Cross-validation folds (default=10)
    picks=None,             # Channels to use
    random_state=None,      # For reproducibility
    n_jobs=1,              # Parallel processing
    verbose=True           # Print progress
)
```

### Parameter Details

#### n_interpolate
```python
# Default (per AutoReject 0.4.2)
n_interpolate = [1, 4, 32]  # Default grid: try interpolating 1, 4, or 32 channels

# For faster processing
n_interpolate = [1, 2]  # Only try 1-2 channels

# For cleaner data
n_interpolate = [0, 1]  # Sometimes no interpolation needed

# TUAB recommended grid (not default)
n_interpolate = [1, 2, 3, 4]  # For 20-channel data
```

#### consensus  
```python
# Default (per AutoReject 0.4.2)
import numpy as np
consensus = np.linspace(0, 1.0, 11)  # Default: 11 values from 0 to 1
# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Stricter cleaning
consensus = [0.7, 0.8, 0.9]  # Require 70-90% good channels

# More permissive
consensus = [0.3, 0.4, 0.5]  # Allow more bad channels
```

#### thresh_method
```python
# Options:
'bayesian_optimization'  # Default, finds optimal via Bayesian opt
'random_search'         # Random search over parameter space
```

### Advanced Usage

```python
# Custom parameters for TUAB dataset (not defaults!)
ar = AutoReject(
    n_interpolate=[1, 2, 3, 4],  # TUAB-specific grid for 20 channels
    consensus=[0.3, 0.5, 0.7],   # TUAB-specific consensus values
    thresh_method='bayesian_optimization',
    cv=5,  # Reduced from default 10 for speed
    random_state=42,
    n_jobs=4,  # Use parallel processing
    verbose=True
)

# Fit on training data
ar.fit(epochs_train)

# Apply to test data
epochs_test_clean = ar.transform(epochs_test)

# Access the learned parameters
# Note: n_interpolate_ and consensus_ are dicts by channel type
print(f"Optimal n_interpolate (EEG): {ar.n_interpolate_.get('eeg', 'N/A')}")
print(f"Optimal consensus (EEG): {ar.consensus_.get('eeg', 'N/A')}")
```

## Reject Log

The reject log tracks what happened to each epoch:

```python
# After fitting
reject_log = ar.reject_log

# Boolean array of bad epochs
bad_epochs = reject_log.bad_epochs  # Shape: (n_epochs,)

# Labels for each epoch/channel
labels = reject_log.labels  # Shape: (n_epochs, n_channels)
# 0 = good
# 1 = bad (not interpolated)
# 2 = bad & interpolated (repaired)

# Visualize
reject_log.plot()  # Heat map of rejection
reject_log.plot_epochs(epochs)  # Show rejected epochs
```

## RANSAC for Bad Channel Detection

```python
from autoreject import Ransac

# Detect bad channels
ransac = Ransac(
    n_resample=50,      # Number of resamples
    min_channels=0.25,  # Minimum channels to use (25%)
    min_corr=0.75,      # Minimum correlation for good channel
    unbroken_time=0.4,  # Minimum unbroken time (seconds)
    n_jobs=1,
    random_state=42
)

# Fit and transform
epochs_clean = ransac.fit_transform(epochs)

# Get bad channels
print(f"Bad channels: {ransac.bad_chs_}")

# Mark in raw data
raw.info['bads'] = ransac.bad_chs_
```

## Validation Curves

Find optimal parameters using validation curves:

```python
from autoreject import validation_curve

# Test different threshold values for global AR
param_range = None  # Will use default range from min to max PTP
train_scores, test_scores = validation_curve(
    epochs,  # epochs is first argument, not ar
    param_name='thresh',  # For global AR, varies threshold
    param_range=param_range,
    cv=5
)

# Plot results
import matplotlib.pyplot as plt
plt.plot(param_range, train_scores.mean(axis=1), label='train')
plt.plot(param_range, test_scores.mean(axis=1), label='test')
plt.xlabel('Threshold')  # Changed from 'Consensus' to 'Threshold'
plt.ylabel('Score')
plt.legend()
```

## Computing Thresholds

Get channel-specific thresholds:

```python
from autoreject import compute_thresholds

# Compute thresholds for each channel
thresholds = compute_thresholds(
    epochs,
    picks=None,  # Use all channels
    method='bayesian_optimization',
    random_state=42,
    n_jobs=1
)

# Returns dict with channel names as keys
print(f"Threshold for Fp1: {thresholds['Fp1']}")
```

## Global Rejection Threshold

For simpler global rejection:

```python
from autoreject import get_rejection_threshold

# Get single threshold for all channels
threshold = get_rejection_threshold(
    epochs,
    ch_types=['eeg'],
    cv=5,
    random_state=42
)

# Use with standard MNE rejection
epochs_clean = epochs.copy().drop_bad(reject={'eeg': threshold})
```

## Integration with MNE Pipeline

### Complete Preprocessing Pipeline

```python
import mne
from autoreject import AutoReject, Ransac

def preprocess_with_autoreject(raw_file):
    # 1. Load and filter
    raw = mne.io.read_raw_edf(raw_file, preload=True)
    raw.filter(l_freq=0.5, h_freq=50.0)
    
    # 2. Create epochs
    events = mne.make_fixed_length_events(raw, duration=4.0)
    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=4.0,
        baseline=(0, 0.5),
        preload=True
    )
    
    # 3. Detect bad channels with RANSAC
    ransac = Ransac(random_state=42, n_jobs=1)
    epochs = ransac.fit_transform(epochs)
    print(f"Bad channels: {ransac.bad_chs_}")
    
    # 4. Apply Autoreject
    ar = AutoReject(
        n_interpolate=[1, 2, 3],
        consensus=[0.5, 0.7],
        random_state=42,
        n_jobs=1
    )
    epochs_clean = ar.fit_transform(epochs)
    
    # 5. Report results
    print(f"Kept {epochs_clean.get_data().shape[0]}/{len(epochs)} epochs")
    # n_interpolate_ and consensus_ are dicts by channel type
    print(f"Interpolated up to {ar.n_interpolate_.get('eeg', 'N/A')} channels (EEG)")
    print(f"Used consensus threshold: {ar.consensus_.get('eeg', 'N/A')} (EEG)")
    
    return epochs_clean, ar
```

## Tips for TUAB Dataset

### Recommended Parameters

```python
# For TUAB abnormality detection
ar_tuab = AutoReject(
    n_interpolate=[1, 2, 3, 4],  # TUAB-specific: 20 channels, can interpolate more
    consensus=[0.3, 0.5, 0.7],    # Clinical data may have more artifacts
    thresh_method='bayesian_optimization',
    cv=5,
    random_state=42,
    n_jobs=4  # Use parallel processing for speed
)
```

### Handling Clinical Data

```python
# Clinical data often has more artifacts
# Be more aggressive with cleaning

# 1. Pre-filter more aggressively
raw.filter(l_freq=1.0, h_freq=40.0)  # Narrower band

# 2. Use stricter consensus
ar = AutoReject(
    consensus=[0.6, 0.7, 0.8],  # Require more good channels
    n_interpolate=[2, 3, 4, 5],  # Allow more interpolation
)

# 3. Consider two-stage cleaning
# First pass: remove very bad epochs
ar_aggressive = AutoReject(consensus=[0.8, 0.9])
epochs_rough = ar_aggressive.fit_transform(epochs)

# Second pass: fine cleaning
ar_fine = AutoReject(consensus=[0.5, 0.6])
epochs_clean = ar_fine.fit_transform(epochs_rough)
```

## Memory and Performance

### Memory-Efficient Processing

```python
# Process in batches for large datasets
def process_in_batches(epochs, batch_size=1000):
    ar = AutoReject(random_state=42)
    
    # Fit on subset
    ar.fit(epochs[:batch_size])
    
    # Apply to all data in chunks
    results = []
    for i in range(0, len(epochs), batch_size):
        batch = epochs[i:i+batch_size]
        clean_batch = ar.transform(batch)
        results.append(clean_batch)
    
    return mne.concatenate_epochs(results)
```

### Speed Optimization

```python
# Faster processing with fewer CV folds
ar_fast = AutoReject(
    cv=3,  # Fewer folds
    n_interpolate=[1, 2],  # Fewer options
    consensus=[0.5],  # Single value
    n_jobs=-1  # Use all cores
)

# Even faster: use random search
ar_fastest = AutoReject(
    thresh_method='random_search',
    cv=2
)
```

## Visualization

### Plotting Results

```python
# 1. Plot reject log
ar.reject_log.plot()  # Heatmap
ar.reject_log.plot_epochs(epochs)  # Show bad epochs

# 2. Before/after comparison
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# Before
epochs.average().plot(axes=axes[0], show=False)
axes[0].set_title('Before Autoreject')

# After
epochs_clean.average().plot(axes=axes[1], show=False)
axes[1].set_title('After Autoreject')

plt.tight_layout()
plt.show()

# 3. Check data quality
epochs_clean.plot_psd()  # Power spectral density
epochs_clean.plot_image()  # Time-frequency image
```

## Saving and Loading

```python
# Save fitted Autoreject object
import pickle

# Option 1: Pickle (always works)
# Save
with open('autoreject_model.pkl', 'wb') as f:
    pickle.dump(ar, f)

# Load
with open('autoreject_model.pkl', 'rb') as f:
    ar_loaded = pickle.load(f)

# Option 2: HDF5 format (requires h5io package)
# NOTE: Requires `pip install h5io` for HDF5 support
from autoreject import read_auto_reject

# Save (requires h5io)
ar.save('ar_model.hdf5')  # Will error if h5io not installed

# Load (requires h5io)
ar_loaded = read_auto_reject('ar_model.hdf5')
```

## Common Issues and Solutions

### Issue: Too many epochs rejected
```python
# Solution: Adjust consensus
ar = AutoReject(consensus=[0.3, 0.4, 0.5])  # More permissive
```

### Issue: Processing too slow
```python
# Solution: Reduce parameter search
ar = AutoReject(
    n_interpolate=[1, 2],  # Fewer options
    consensus=[0.5],       # Single value
    cv=3,                  # Fewer folds
    n_jobs=-1             # Use all cores
)
```

### Issue: Different results each run
```python
# Solution: Set random state
ar = AutoReject(random_state=42)  # Reproducible
```

### Issue: Memory errors with large data
```python
# Solution: Process in smaller chunks
# See batch processing example above
```

## References

1. Jas et al. (2017). "Autoreject: Automated artifact rejection for MEG and EEG data." NeuroImage, 159, 417-429.

2. Jas et al. (2016). "Automated rejection and repair of bad trials in MEG/EEG." 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI).

3. Official Documentation: https://autoreject.github.io/