# Autoreject Parameters for TUAB Dataset

## TUAB Dataset Characteristics

- **Channels**: 20 standard 10-20 EEG channels
- **Sampling Rate**: 256 Hz
- **Population**: Clinical data (normal and abnormal EEGs)
- **Artifacts**: Higher artifact rate due to clinical setting
- **Goal**: Binary classification (normal/abnormal)

## Recommended Autoreject Configuration

### Quick Start Configuration

```python
from autoreject import AutoReject

# Recommended for TUAB (not AutoReject defaults!)
ar_tuab = AutoReject(
    n_interpolate=[1, 2, 3, 4],  # TUAB-specific: Try 1-4 channels (default is [1, 4, 32])
    consensus=[0.3, 0.5, 0.7],   # TUAB-specific: 30-70% good (default is np.linspace(0, 1, 11))
    thresh_method='bayesian_optimization',
    cv=5,  # Reduced from default=10 for speed
    random_state=42,
    n_jobs=4  # Parallel processing
)
```

### Rationale for Parameters

#### n_interpolate = [1, 2, 3, 4]
- TUAB has 20 channels total
- Can afford to interpolate up to 20% (4 channels)
- Clinical data often has 1-2 bad channels
- Range allows algorithm to find optimal value

#### consensus = [0.3, 0.5, 0.7]
- Clinical data is noisier than research-grade
- 0.3 = Keep epochs with 30% good channels (permissive)
- 0.5 = Keep epochs with 50% good channels (balanced)
- 0.7 = Keep epochs with 70% good channels (strict)
- Algorithm will choose based on cross-validation

## Integration with TUAB Training Pipeline

### Complete Preprocessing Function

```python
import mne
from autoreject import AutoReject, Ransac
import numpy as np
from pathlib import Path

def preprocess_tuab_file(edf_path, window_duration=4.0, window_stride=2.0):
    """
    Preprocess TUAB EDF file with MNE and Autoreject.
    
    Args:
        edf_path: Path to TUAB EDF file
        window_duration: Window size in seconds (4s for EEGPT)
        window_stride: Stride between windows (2s = 50% overlap)
    
    Returns:
        clean_windows: Array of clean EEG windows
        labels: Binary labels (0=normal, 1=abnormal)
    """
    
    # 1. Load raw data
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    # 2. Handle TUAB channel naming (OLD to MODERN)
    channel_mapping = {
        'EEG T3-REF': 'T7',
        'EEG T4-REF': 'T8', 
        'EEG T5-REF': 'P7',
        'EEG T6-REF': 'P8',
        # Add other mappings as needed
    }
    raw.rename_channels(channel_mapping)
    
    # 3. Set standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    
    # 4. Filter (important for clinical data)
    raw.filter(
        l_freq=0.5,   # Remove DC drift
        h_freq=50.0,  # Remove high-freq noise
        fir_design='firwin'
    )
    
    # 5. Remove line noise
    raw.notch_filter(freqs=60)  # 60 Hz for US
    
    # 6. Create fixed-length epochs
    events = mne.make_fixed_length_events(
        raw, 
        duration=window_duration,
        overlap=window_duration - window_stride
    )
    
    epochs = mne.Epochs(
        raw, events,
        tmin=0, 
        tmax=window_duration,
        baseline=None,  # No baseline for EEGPT
        preload=True,
        reject_by_annotation=True
    )
    
    # 7. Detect bad channels with RANSAC
    ransac = Ransac(
        n_resample=50,
        min_channels=0.25,  # Need at least 25% channels
        min_corr=0.75,
        unbroken_time=0.4,
        n_jobs=1,
        random_state=42
    )
    epochs = ransac.fit_transform(epochs)
    
    if ransac.bad_chs_:
        print(f"Bad channels detected: {ransac.bad_chs_}")
    
    # 8. Apply Autoreject
    ar = AutoReject(
        n_interpolate=[1, 2, 3, 4],  # TUAB-specific grid
        consensus=[0.3, 0.5, 0.7],
        thresh_method='bayesian_optimization',
        cv=5,
        random_state=42,
        n_jobs=4,
        verbose=False
    )
    
    epochs_clean = ar.fit_transform(epochs)
    
    # 9. Report cleaning statistics
    n_epochs_before = len(epochs)
    n_epochs_after = len(epochs_clean)
    print(f"Epochs: {n_epochs_before} → {n_epochs_after} "
          f"({100*n_epochs_after/n_epochs_before:.1f}% kept)")
    
    # 10. Extract clean windows
    clean_windows = epochs_clean.get_data()  # Shape: (n_epochs, n_channels, n_samples)
    
    # 11. Get label from filename
    label = 1 if 'abnormal' in str(edf_path) else 0
    labels = np.full(len(clean_windows), label)
    
    return clean_windows, labels
```

### Batch Processing for Training

```python
def process_tuab_dataset(data_dir, cache_dir, split='train'):
    """
    Process entire TUAB dataset with Autoreject.
    
    Args:
        data_dir: Path to TUAB EDF files
        cache_dir: Where to save processed data
        split: 'train' or 'eval'
    """
    import torch
    from tqdm import tqdm
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get file list
    edf_files = list(Path(data_dir).glob('**/*.edf'))
    
    # Split files (80/20 or as needed)
    if split == 'train':
        files = edf_files[:int(0.8 * len(edf_files))]
    else:
        files = edf_files[int(0.8 * len(edf_files)):]
    
    # Fit Autoreject on subset first
    print("Fitting Autoreject on subset...")
    subset_files = files[:50]  # Use 50 files to fit
    
    all_epochs = []
    for f in tqdm(subset_files, desc="Loading subset"):
        raw = mne.io.read_raw_edf(f, preload=True)
        # ... preprocessing steps ...
        epochs = create_epochs(raw)
        all_epochs.append(epochs)
    
    # Concatenate and fit
    combined_epochs = mne.concatenate_epochs(all_epochs)
    
    ar = AutoReject(
        n_interpolate=[1, 2, 3, 4],  # TUAB-specific grid
        consensus=[0.3, 0.5, 0.7],
        random_state=42,
        n_jobs=4
    )
    ar.fit(combined_epochs)
    
    # Note: n_interpolate_ and consensus_ are dicts by channel type
    print(f"Autoreject fitted: n_interpolate={ar.n_interpolate_.get('eeg', 'N/A')}, "
          f"consensus={ar.consensus_.get('eeg', 'N/A')}")
    
    # Process all files with fitted AR
    for i, edf_file in enumerate(tqdm(files, desc=f"Processing {split}")):
        # Load and preprocess
        raw = mne.io.read_raw_edf(edf_file, preload=True)
        # ... preprocessing ...
        epochs = create_epochs(raw)
        
        # Apply fitted Autoreject
        epochs_clean = ar.transform(epochs)
        
        # Save to cache
        data = epochs_clean.get_data()
        label = 1 if 'abnormal' in str(edf_file) else 0
        
        cache_file = cache_dir / f"{split}_{i:05d}.pt"
        torch.save({
            'x': torch.FloatTensor(data),
            'y': label,
            'file': str(edf_file),
            'n_windows': len(data)
        }, cache_file)
    
    # Save fitted AR model (requires h5io package for HDF5 format)
    # Note: Install with `pip install h5io` if using HDF5 format
    ar.save(cache_dir / 'autoreject_model.hdf5')  # Requires h5io
    print(f"Saved Autoreject model to {cache_dir}/autoreject_model.hdf5")
```

## Performance Considerations

### Memory-Efficient Settings

```python
# For WSL2 with limited memory
ar_memory_efficient = AutoReject(
    n_interpolate=[1, 2],    # Fewer options
    consensus=[0.5],          # Single value
    cv=3,                     # Fewer CV folds
    n_jobs=1,                 # Single thread (less memory)
    verbose=False
)
```

### Speed-Optimized Settings

```python
# For faster processing (may be less optimal)
ar_fast = AutoReject(
    thresh_method='random_search',  # Faster than Bayesian
    n_interpolate=[2],              # Fixed value
    consensus=[0.5],                # Fixed value
    cv=2,                           # Minimal CV
    n_jobs=-1                       # All cores
)
```

## Quality Metrics

### Track Preprocessing Quality

```python
def compute_quality_metrics(epochs_before, epochs_after, ar):
    """
    Compute metrics to track preprocessing quality.
    """
    metrics = {
        'n_epochs_in': len(epochs_before),
        'n_epochs_out': len(epochs_after),
        'retention_rate': len(epochs_after) / len(epochs_before),
        'n_interpolate_used': ar.n_interpolate_.get('eeg', 'N/A'),  # Dict by channel type
        'consensus_used': ar.consensus_.get('eeg', 'N/A'),  # Dict by channel type
        'bad_epochs': ar.reject_log.bad_epochs.sum(),
        'interpolated_channels': (ar.reject_log.labels == 2).sum()  # labels==2 means interpolated
    }
    
    # Signal quality improvement
    snr_before = compute_snr(epochs_before)
    snr_after = compute_snr(epochs_after)
    metrics['snr_improvement'] = snr_after - snr_before
    
    return metrics

def compute_snr(epochs):
    """Simple SNR calculation."""
    data = epochs.get_data()
    signal_power = np.mean(data ** 2)
    noise = data - np.mean(data, axis=-1, keepdims=True)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)
```

## Validation Strategy

### A/B Testing with and without Autoreject

```python
def compare_with_without_autoreject(test_files):
    """
    Compare model performance with and without Autoreject.
    """
    results = {'with_ar': [], 'without_ar': []}
    
    for edf_file in test_files:
        # Without Autoreject
        raw = mne.io.read_raw_edf(edf_file, preload=True)
        raw.filter(0.5, 50.0)
        epochs_raw = create_epochs(raw)
        score_without = evaluate_model(epochs_raw)
        results['without_ar'].append(score_without)
        
        # With Autoreject
        ar = AutoReject(
            n_interpolate=[1, 2, 3, 4],  # TUAB-specific grid
            consensus=[0.3, 0.5, 0.7]
        )
        epochs_clean = ar.fit_transform(epochs_raw)
        score_with = evaluate_model(epochs_clean)
        results['with_ar'].append(score_with)
    
    print(f"Without AR: {np.mean(results['without_ar']):.3f}")
    print(f"With AR: {np.mean(results['with_ar']):.3f}")
    print(f"Improvement: {np.mean(results['with_ar']) - np.mean(results['without_ar']):.3f}")
```

## Expected Improvements

Based on literature and similar clinical datasets:

### Without Autoreject
- Raw accuracy: ~56%
- High variance in predictions
- Model learns artifact patterns

### With Autoreject
- Expected accuracy: 65-70% (first improvement)
- Lower variance
- Model learns EEG patterns

### With Full MNE + Autoreject Pipeline
- Target accuracy: 75-87%
- Stable training
- Robust predictions

## Troubleshooting

### Issue: Too much data rejected (>50%)

```python
# Solution: More permissive settings
ar = AutoReject(
    n_interpolate=[2, 3, 4, 5],  # Allow more interpolation
    consensus=[0.2, 0.3, 0.4],    # Lower consensus threshold
)
```

### Issue: Still noisy after Autoreject

```python
# Solution: Two-stage cleaning
# Stage 1: Aggressive
ar1 = AutoReject(consensus=[0.7, 0.8, 0.9])
epochs_stage1 = ar1.fit_transform(epochs)

# Stage 2: Fine-tuning
ar2 = AutoReject(consensus=[0.4, 0.5, 0.6])
epochs_final = ar2.fit_transform(epochs_stage1)
```

### Issue: Different results per file

```python
# Solution: Fit once, apply to all
# See batch processing example above
```

## References

- TUAB Dataset: Temple University Abnormal EEG Corpus
- Target performance: 87% AUROC (from EEGPT paper)
- Autoreject paper: Jas et al. (2017)