# MNE-Autoreject Integration Guide

## Overview

This guide shows how to optimally combine MNE-Python preprocessing with Autoreject for maximum data quality. The integration leverages MNE's preprocessing capabilities with Autoreject's adaptive artifact rejection.

## Why Both Tools?

### MNE-Python Provides
- Data I/O (EDF, FIF, etc.)
- Filtering (bandpass, notch)
- Channel operations (montage, referencing)
- Basic artifact detection
- Epoching and segmentation

### Autoreject Adds
- Adaptive threshold learning
- Repair vs reject decisions
- Cross-validated parameter selection
- Channel-specific thresholds
- Automatic quality control

## Integration Workflow

```
Raw EDF → MNE Loading → Filtering → Epoching → Autoreject → Clean Data
```

## Complete Integration Pipeline

### Step-by-Step Implementation

```python
import mne
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA
import numpy as np

def complete_mne_autoreject_pipeline(edf_path, 
                                     use_ica=True,
                                     window_duration=4.0):
    """
    Complete preprocessing pipeline combining MNE and Autoreject.
    
    Args:
        edf_path: Path to EDF file
        use_ica: Whether to apply ICA for artifact removal
        window_duration: Epoch duration in seconds
    
    Returns:
        epochs_clean: Cleaned epochs ready for analysis
        preprocessing_info: Dictionary with preprocessing details
    """
    
    # ============= MNE PREPROCESSING =============
    
    # 1. Load raw data
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    sampling_rate = raw.info['sfreq']
    
    # 2. Channel setup
    # Set channel types if needed
    if 'EOG' in raw.ch_names:
        raw.set_channel_types({'EOG': 'eog'})
    if 'ECG' in raw.ch_names:
        raw.set_channel_types({'ECG': 'ecg'})
    
    # 3. Set montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='warn')
    
    # 4. Filtering (MNE)
    # High-pass filter to remove drift
    raw.filter(l_freq=0.5, h_freq=None, fir_design='firwin')
    
    # Low-pass filter to remove high-frequency noise
    raw.filter(l_freq=None, h_freq=50.0, fir_design='firwin')
    
    # Notch filter for line noise
    raw.notch_filter(freqs=60, notch_widths=2)
    
    # 5. Bad channel detection with RANSAC (Autoreject)
    # Create temporary epochs for RANSAC
    events_temp = mne.make_fixed_length_events(raw, duration=2.0)
    epochs_temp = mne.Epochs(raw, events_temp, tmin=0, tmax=2.0, 
                             baseline=None, preload=True)
    
    ransac = Ransac(
        n_resample=50,
        min_channels=0.25,
        min_corr=0.75,
        unbroken_time=0.4,
        n_jobs=1,
        random_state=42
    )
    ransac.fit(epochs_temp)
    
    # Mark bad channels in raw
    if ransac.bad_chs_:
        raw.info['bads'] = ransac.bad_chs_
        print(f"Bad channels detected: {ransac.bad_chs_}")
        
        # Interpolate bad channels (MNE)
        raw.interpolate_bads(reset_bads=True)
    
    # 6. Re-reference (MNE)
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    
    # 7. Optional: ICA for artifact removal (MNE)
    if use_ica:
        # Filter copy for ICA
        raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None)
        
        # Fit ICA
        ica = ICA(
            n_components=min(20, len(raw.ch_names) - 1),
            method='fastica',
            random_state=42,
            max_iter=800
        )
        ica.fit(raw_ica)
        
        # Find EOG components
        if 'EOG' in raw.ch_names or any('EOG' in ch for ch in raw.ch_names):
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            ica.exclude = eog_indices
            print(f"Excluding EOG components: {eog_indices}")
        
        # Find ECG components
        if 'ECG' in raw.ch_names or any('ECG' in ch for ch in raw.ch_names):
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
            ica.exclude.extend(ecg_indices)
            print(f"Excluding ECG components: {ecg_indices}")
        
        # Apply ICA
        raw = ica.apply(raw)
    
    # 8. Create epochs (MNE)
    events = mne.make_fixed_length_events(
        raw, 
        duration=window_duration,
        overlap=window_duration/2  # 50% overlap
    )
    
    epochs = mne.Epochs(
        raw, events,
        tmin=0, 
        tmax=window_duration,
        baseline=(0, 0.2),  # Baseline correction
        preload=True,
        reject_by_annotation=True
    )
    
    # ============= AUTOREJECT CLEANING =============
    
    # 9. Apply Autoreject
    ar = AutoReject(
        n_interpolate=[1, 2, 3, 4],
        consensus=[0.3, 0.5, 0.7],
        thresh_method='bayesian_optimization',
        cv=5,
        random_state=42,
        n_jobs=4,
        verbose=True
    )
    
    epochs_clean = ar.fit_transform(epochs)
    
    # 10. Collect preprocessing info
    preprocessing_info = {
        'n_channels': len(raw.ch_names),
        'sampling_rate': sampling_rate,
        'bad_channels': ransac.bad_chs_ if ransac.bad_chs_ else [],
        'n_epochs_before': len(epochs),
        'n_epochs_after': len(epochs_clean),
        'retention_rate': len(epochs_clean) / len(epochs),
        'autoreject_n_interpolate': ar.n_interpolate_,
        'autoreject_consensus': ar.consensus_,
        'ica_components_removed': len(ica.exclude) if use_ica else 0
    }
    
    print("\n=== Preprocessing Summary ===")
    print(f"Channels: {preprocessing_info['n_channels']}")
    print(f"Bad channels: {preprocessing_info['bad_channels']}")
    print(f"Epochs: {preprocessing_info['n_epochs_before']} → "
          f"{preprocessing_info['n_epochs_after']} "
          f"({preprocessing_info['retention_rate']*100:.1f}% kept)")
    print(f"Autoreject: n_interpolate={ar.n_interpolate_}, "
          f"consensus={ar.consensus_}")
    
    return epochs_clean, preprocessing_info
```

## Order of Operations (Critical!)

The order matters for optimal results:

1. **Load & Setup** (MNE)
   - Load raw data
   - Set channel types
   - Apply montage

2. **Initial Filtering** (MNE)
   - High-pass filter (remove drift)
   - Low-pass filter (remove HF noise)
   - Notch filter (remove line noise)

3. **Bad Channel Detection** (Autoreject RANSAC)
   - Detect bad channels
   - Interpolate them (MNE)

4. **Re-reference** (MNE)
   - Apply after bad channel interpolation
   - Common average reference typical

5. **ICA** (MNE, optional)
   - Remove EOG/ECG artifacts
   - Apply before epoching

6. **Epoching** (MNE)
   - Segment continuous data
   - Apply baseline correction

7. **Autoreject** (Final cleaning)
   - Adaptive artifact rejection
   - Repair/reject decisions

## Advanced Integration Patterns

### Pattern 1: Two-Stage Autoreject

```python
def two_stage_cleaning(epochs):
    """
    Apply Autoreject in two stages for better cleaning.
    """
    # Stage 1: Aggressive cleaning to remove very bad epochs
    ar_aggressive = AutoReject(
        consensus=[0.7, 0.8, 0.9],
        n_interpolate=[0, 1],
        cv=3,
        random_state=42
    )
    epochs_stage1 = ar_aggressive.fit_transform(epochs)
    print(f"Stage 1: {len(epochs)} → {len(epochs_stage1)} epochs")
    
    # Stage 2: Fine cleaning with more interpolation
    ar_fine = AutoReject(
        consensus=[0.4, 0.5, 0.6],
        n_interpolate=[2, 3, 4],
        cv=5,
        random_state=42
    )
    epochs_final = ar_fine.fit_transform(epochs_stage1)
    print(f"Stage 2: {len(epochs_stage1)} → {len(epochs_final)} epochs")
    
    return epochs_final
```

### Pattern 2: Quality-Based Processing

```python
def quality_based_pipeline(raw):
    """
    Apply different preprocessing based on data quality.
    """
    # Assess initial quality
    quality_score = assess_data_quality(raw)
    
    if quality_score > 0.7:
        # Good quality: light preprocessing
        raw.filter(0.5, 50)
        ar = AutoReject(consensus=[0.5, 0.6])
        
    elif quality_score > 0.4:
        # Medium quality: standard preprocessing
        raw.filter(1.0, 40)
        ar = AutoReject(
            consensus=[0.4, 0.5, 0.6],
            n_interpolate=[1, 2, 3]
        )
        
    else:
        # Poor quality: aggressive preprocessing
        raw.filter(1.0, 30)
        # Apply ICA first
        apply_ica_cleaning(raw)
        ar = AutoReject(
            consensus=[0.3, 0.4, 0.5],
            n_interpolate=[2, 3, 4, 5]
        )
    
    return ar

def assess_data_quality(raw):
    """Simple quality assessment."""
    data = raw.get_data()
    
    # Check for flat channels
    flat_channels = np.sum(np.std(data, axis=1) < 1e-6)
    
    # Check for high amplitude
    high_amp = np.sum(np.max(np.abs(data), axis=1) > 200e-6)
    
    # Simple quality score
    quality = 1.0 - (flat_channels + high_amp) / len(raw.ch_names)
    return quality
```

### Pattern 3: Adaptive Processing

```python
def adaptive_preprocessing(raw, target_retention=0.7):
    """
    Adaptively adjust parameters to achieve target epoch retention.
    """
    epochs = create_epochs(raw)
    
    # Try different consensus values
    consensus_options = [
        [0.7, 0.8, 0.9],  # Strict
        [0.5, 0.6, 0.7],  # Moderate
        [0.3, 0.4, 0.5],  # Permissive
    ]
    
    for consensus in consensus_options:
        ar = AutoReject(
            consensus=consensus,
            n_interpolate=[1, 2, 3],
            cv=3,
            random_state=42
        )
        
        # Test on subset
        test_epochs = epochs[:100]
        ar.fit(test_epochs)
        clean = ar.transform(test_epochs)
        
        retention = len(clean) / len(test_epochs)
        print(f"Consensus {consensus}: {retention:.2f} retention")
        
        if retention >= target_retention:
            # Use these parameters
            return ar.fit_transform(epochs)
    
    # If no parameters achieve target, use most permissive
    return ar.fit_transform(epochs)
```

## Synergy Benefits

### 1. Complementary Strengths

```python
# MNE handles deterministic preprocessing
raw.filter(0.5, 50)  # Physics-based filtering
raw.set_eeg_reference('average')  # Mathematical referencing

# Autoreject handles adaptive, data-driven decisions
ar.fit(epochs)  # Learns from your specific data
epochs_clean = ar.transform(epochs)  # Applies learned thresholds
```

### 2. Quality Checkpoints

```python
def preprocessing_with_checkpoints(raw):
    """
    Add quality checks between MNE and Autoreject steps.
    """
    results = {}
    
    # Checkpoint 1: After filtering
    raw.filter(0.5, 50)
    results['snr_after_filter'] = calculate_snr(raw)
    
    # Checkpoint 2: After bad channel removal
    detect_and_interpolate_bad_channels(raw)
    results['n_bad_channels'] = len(raw.info['bads'])
    
    # Checkpoint 3: After ICA
    if apply_ica:
        raw = apply_ica_cleaning(raw)
        results['ica_components_removed'] = n_components
    
    # Checkpoint 4: After epoching
    epochs = create_epochs(raw)
    results['n_epochs_created'] = len(epochs)
    
    # Checkpoint 5: After Autoreject
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    results['n_epochs_kept'] = len(epochs_clean)
    results['retention_rate'] = len(epochs_clean) / len(epochs)
    
    # Quality decision
    if results['retention_rate'] < 0.5:
        print("Warning: Low retention rate. Consider adjusting parameters.")
    
    return epochs_clean, results
```

### 3. Ensemble Approach

```python
def ensemble_preprocessing(raw):
    """
    Use multiple preprocessing strategies and combine.
    """
    strategies = []
    
    # Strategy 1: MNE basic + Autoreject
    raw1 = raw.copy()
    raw1.filter(0.5, 50)
    epochs1 = create_epochs(raw1)
    ar1 = AutoReject(consensus=[0.5])
    clean1 = ar1.fit_transform(epochs1)
    strategies.append(('basic', clean1))
    
    # Strategy 2: MNE + ICA + Autoreject
    raw2 = raw.copy()
    raw2.filter(0.5, 50)
    apply_ica_cleaning(raw2)
    epochs2 = create_epochs(raw2)
    ar2 = AutoReject(consensus=[0.6])
    clean2 = ar2.fit_transform(epochs2)
    strategies.append(('ica', clean2))
    
    # Strategy 3: Aggressive MNE + Autoreject
    raw3 = raw.copy()
    raw3.filter(1.0, 40)
    raw3.set_eeg_reference('REST')
    epochs3 = create_epochs(raw3)
    ar3 = AutoReject(consensus=[0.4], n_interpolate=[2,3,4])
    clean3 = ar3.fit_transform(epochs3)
    strategies.append(('aggressive', clean3))
    
    # Select best based on quality metric
    best_quality = -np.inf
    best_epochs = None
    
    for name, epochs in strategies:
        quality = calculate_quality_score(epochs)
        print(f"{name}: quality={quality:.3f}, n_epochs={len(epochs)}")
        
        if quality > best_quality:
            best_quality = quality
            best_epochs = epochs
    
    return best_epochs
```

## Optimization Tips

### For Speed

```python
# Minimal pipeline for speed
def fast_pipeline(raw):
    # Basic filtering only
    raw.filter(0.5, 50, n_jobs=4)
    
    # Simple epoching
    epochs = create_fixed_epochs(raw, duration=4.0)
    
    # Fast Autoreject
    ar = AutoReject(
        consensus=[0.5],  # Single value
        n_interpolate=[2],  # Single value
        cv=2,  # Minimal CV
        n_jobs=-1
    )
    
    return ar.fit_transform(epochs)
```

### For Quality

```python
# Maximum quality pipeline
def quality_pipeline(raw):
    # Careful filtering
    raw.filter(0.5, 50, method='fir', fir_design='firwin2')
    
    # RANSAC for bad channels
    detect_bad_channels_ransac(raw)
    raw.interpolate_bads()
    
    # ICA cleaning
    apply_comprehensive_ica(raw)
    
    # Careful epoching with overlap
    epochs = create_overlapping_epochs(raw)
    
    # Thorough Autoreject
    ar = AutoReject(
        consensus=[0.3, 0.4, 0.5, 0.6, 0.7],
        n_interpolate=[1, 2, 3, 4, 5],
        cv=10,
        thresh_method='bayesian_optimization'
    )
    
    return ar.fit_transform(epochs)
```

### For Memory Efficiency

```python
# Memory-efficient pipeline
def memory_efficient_pipeline(edf_path):
    # Load without preloading
    raw = mne.io.read_raw_edf(edf_path, preload=False)
    
    # Filter in place
    raw.load_data()
    raw.filter(0.5, 50, n_jobs=1)
    
    # Process in chunks
    chunk_size = 1000  # epochs
    all_clean = []
    
    for start_idx in range(0, n_total_epochs, chunk_size):
        # Load chunk
        epochs_chunk = load_epochs_chunk(raw, start_idx, chunk_size)
        
        # Clean chunk
        ar = AutoReject(n_jobs=1)  # Single thread
        clean_chunk = ar.fit_transform(epochs_chunk)
        
        # Save and clear
        all_clean.append(clean_chunk)
        del epochs_chunk
    
    return concatenate_epochs(all_clean)
```

## Validation

### Compare Pipelines

```python
def validate_preprocessing(test_files):
    """
    Compare different preprocessing combinations.
    """
    results = {
        'mne_only': [],
        'ar_only': [],
        'mne_ar': [],
        'full_pipeline': []
    }
    
    for file in test_files:
        raw = mne.io.read_raw_edf(file, preload=True)
        
        # MNE only
        raw1 = raw.copy()
        raw1.filter(0.5, 50)
        epochs1 = create_epochs(raw1)
        results['mne_only'].append(evaluate(epochs1))
        
        # Autoreject only
        epochs2 = create_epochs(raw.copy())
        ar = AutoReject()
        clean2 = ar.fit_transform(epochs2)
        results['ar_only'].append(evaluate(clean2))
        
        # MNE + Autoreject
        raw3 = raw.copy()
        raw3.filter(0.5, 50)
        epochs3 = create_epochs(raw3)
        clean3 = ar.fit_transform(epochs3)
        results['mne_ar'].append(evaluate(clean3))
        
        # Full pipeline
        clean4, _ = complete_mne_autoreject_pipeline(file)
        results['full_pipeline'].append(evaluate(clean4))
    
    # Print results
    for method, scores in results.items():
        print(f"{method}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

## Common Integration Issues

### Issue: Order matters

```python
# WRONG: Autoreject before filtering
epochs = create_epochs(raw)
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs)
raw.filter(0.5, 50)  # Too late!

# RIGHT: Filter first
raw.filter(0.5, 50)
epochs = create_epochs(raw)
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs)
```

### Issue: Incompatible parameters

```python
# WRONG: ICA after epoching
epochs = create_epochs(raw)
ica = ICA()
ica.fit(epochs)  # ICA needs continuous data

# RIGHT: ICA on continuous data
ica = ICA()
ica.fit(raw)
raw = ica.apply(raw)
epochs = create_epochs(raw)
```

### Issue: Double cleaning

```python
# WRONG: Reject in both MNE and Autoreject
epochs = mne.Epochs(raw, reject=dict(eeg=100e-6))  # MNE rejection
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs)  # Double rejection

# RIGHT: Let Autoreject handle rejection
epochs = mne.Epochs(raw, reject=None)  # No MNE rejection
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs)  # Single rejection
```

## References

- MNE-Python: https://mne.tools/
- Autoreject: https://autoreject.github.io/
- Integration examples: https://autoreject.github.io/stable/auto_examples/