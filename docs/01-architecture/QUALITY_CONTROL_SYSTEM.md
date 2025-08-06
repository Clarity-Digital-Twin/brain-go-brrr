# Quality Control System

## Overview

This document details the integration of AutoReject for automated artifact rejection in EEG data processing. AutoReject provides automated, data-driven artifact rejection that matches human expert agreement (~87.5%) while being fully reproducible.

## Key Performance Metrics

### AutoReject Benchmarks
- **Human Agreement**: 87.5% match with expert annotations
- **Processing Speed**: <5 minutes for 1-hour recording
- **Bad Channel Detection**: >95% sensitivity
- **Artifact Types**: Eye blinks, muscle, heartbeat, environmental
- **Interpolation Quality**: SNR improvement of 3-5 dB

## Core Algorithm

### 1. Cross-Validation Based Threshold Estimation
```python
class AutoRejectOptimizer:
    """
    Automatically find optimal rejection thresholds using cross-validation
    """
    
    def __init__(self, n_interpolate=None, cv=10):
        if n_interpolate is None:
            n_interpolate = [1, 2, 3, 4]  # Try different interpolation counts
        
        self.n_interpolate = n_interpolate
        self.cv = cv
        self.thresholds_ = None
        self.consensus_ = None
    
    def fit(self, epochs):
        """
        Find optimal parameters through cross-validation
        
        Key insight: Use held-out data to evaluate threshold performance
        """
        # Grid search over thresholds
        candidate_thresholds = self._compute_thresholds(epochs)
        
        # Cross-validation to find optimal threshold
        best_threshold = None
        best_score = -np.inf
        
        for threshold in candidate_thresholds:
            scores = []
            
            # K-fold cross-validation
            for train_idx, val_idx in KFold(n_splits=self.cv).split(epochs):
                # Apply threshold on training set
                train_epochs = epochs[train_idx]
                good_epochs = self._apply_threshold(train_epochs, threshold)
                
                # Evaluate on validation set
                val_epochs = epochs[val_idx]
                score = self._evaluate_threshold(good_epochs, val_epochs)
                scores.append(score)
            
            # Average score across folds
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = threshold
        
        self.thresholds_ = best_threshold
        return self
```

### 2. Local (Per-Sensor) Threshold Estimation
```python
def compute_local_thresholds(epochs, n_jobs=1):
    """
    Estimate threshold for each sensor independently
    
    This mimics human expert behavior who evaluate each channel
    """
    n_epochs, n_channels, n_times = epochs.get_data().shape
    thresholds = np.zeros((n_channels,))
    
    # Parallel processing per channel
    def _compute_channel_threshold(ch_idx):
        channel_data = epochs.get_data()[:, ch_idx, :]
        
        # Compute peak-to-peak for each epoch
        peak_to_peak = np.ptp(channel_data, axis=1)
        
        # Use robust statistics to find threshold
        # Avoid influence of already bad epochs
        q75, q25 = np.percentile(peak_to_peak, [75, 25])
        iqr = q75 - q25
        
        # Tukey's method for outlier detection
        threshold = q75 + 1.5 * iqr
        
        return threshold
    
    # Process channels in parallel
    from joblib import Parallel, delayed
    thresholds = Parallel(n_jobs=n_jobs)(
        delayed(_compute_channel_threshold)(ch) 
        for ch in range(n_channels)
    )
    
    return np.array(thresholds)
```

### 3. Consensus Algorithm
```python
class ConsensusReject:
    """
    Decide whether to reject or repair epochs based on bad sensors
    """
    
    def __init__(self, n_interpolate=None, consensus=None):
        self.n_interpolate = n_interpolate or [1, 2, 3, 4]
        self.consensus = consensus or [0.1, 0.2, 0.3, 0.4]
    
    def fit_transform(self, epochs, thresholds):
        """
        Main AutoReject algorithm combining rejection and repair
        """
        n_epochs, n_channels, n_times = epochs.get_data().shape
        
        # Find bad sensors for each epoch
        bad_sensors = self._find_bad_sensors(epochs, thresholds)
        
        # Optimize consensus and interpolation parameters
        best_params = self._optimize_consensus(epochs, bad_sensors)
        
        # Apply optimal strategy
        epochs_clean = epochs.copy()
        reject_log = RejectLog(bad_sensors)
        
        for epoch_idx in range(n_epochs):
            n_bad = np.sum(bad_sensors[epoch_idx])
            
            if n_bad == 0:
                # Good epoch, keep as is
                continue
            elif n_bad <= best_params['n_interpolate']:
                # Few bad sensors, interpolate
                self._interpolate_sensors(
                    epochs_clean, epoch_idx, 
                    bad_sensors[epoch_idx]
                )
                reject_log.labels[epoch_idx] = 1  # Repaired
            else:
                # Too many bad sensors, reject epoch
                reject_log.labels[epoch_idx] = 2  # Rejected
        
        # Remove rejected epochs
        epochs_clean = epochs_clean[reject_log.labels != 2]
        
        return epochs_clean, reject_log
    
    def _find_bad_sensors(self, epochs, thresholds):
        """Identify bad sensors per epoch using local thresholds"""
        data = epochs.get_data()
        n_epochs, n_channels = data.shape[:2]
        
        bad_sensors = np.zeros((n_epochs, n_channels), dtype=bool)
        
        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                # Peak-to-peak amplitude
                ptp = np.ptp(data[epoch_idx, ch_idx])
                
                # Compare to threshold
                if ptp > thresholds[ch_idx]:
                    bad_sensors[epoch_idx, ch_idx] = True
        
        return bad_sensors
    
    def _interpolate_sensors(self, epochs, epoch_idx, bad_mask):
        """Interpolate bad sensors using spherical splines"""
        from mne.channels.interpolation import _interpolate_bads_eeg
        
        # Get sensor positions
        pos = epochs.info['chs']
        good_idx = np.where(~bad_mask)[0]
        bad_idx = np.where(bad_mask)[0]
        
        # Interpolate each bad sensor
        epoch_data = epochs._data[epoch_idx]
        
        for bad_ch in bad_idx:
            # Use spherical spline interpolation
            weights = _compute_interpolation_weights(
                pos[bad_ch], pos[good_idx]
            )
            
            # Apply weights
            epoch_data[bad_ch] = np.dot(weights, epoch_data[good_idx])
```

## Integration with EEGPT Pipeline

### 1. Enhanced Dataset with AutoReject
```python
class TUABEnhancedDataset(TUABDataset):
    """TUAB dataset with integrated AutoReject cleaning"""
    
    def __init__(self, use_autoreject=True, ar_cache_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.use_autoreject = use_autoreject
        self.ar_cache_dir = ar_cache_dir
        self.ar = None
        
        if use_autoreject:
            # Initialize AutoReject with optimal parameters
            self.ar = AutoReject(
                n_interpolate=[1, 2, 3, 4],
                cv=5,
                n_jobs=4,
                verbose=False
            )
    
    def __getitem__(self, idx):
        # Get base window
        window, label = super().__getitem__(idx)
        
        if not self.use_autoreject:
            return window, label
        
        # Apply AutoReject
        window_clean = self._apply_autoreject(window, idx)
        
        return window_clean, label
    
    def _apply_autoreject(self, window, idx):
        """Apply AutoReject with caching"""
        # Check cache first
        cache_key = f"ar_{self.file_path}_{idx}"
        cached = self._load_from_cache(cache_key)
        
        if cached is not None:
            return cached
        
        # Create MNE epochs object
        epochs = self._create_epochs(window)
        
        # Fit and transform
        if self.ar.thresholds_ is None:
            # First time, need to fit
            epochs_clean, reject_log = self.ar.fit_transform(epochs)
        else:
            # Already fitted, just transform
            epochs_clean, reject_log = self.ar.transform(epochs)
        
        # Extract cleaned data
        if len(epochs_clean) > 0:
            window_clean = epochs_clean.get_data()[0]
        else:
            # Epoch was fully rejected, return zeros
            window_clean = np.zeros_like(window)
        
        # Cache result
        self._save_to_cache(cache_key, window_clean)
        
        return window_clean
```

### 2. Quality Metrics Integration
```python
class QualityAwareTraining:
    """Training that adapts based on data quality"""
    
    def __init__(self, model, quality_threshold=0.7):
        self.model = model
        self.quality_threshold = quality_threshold
    
    def training_step(self, batch):
        windows, labels, quality_scores = batch
        
        # Weight loss by quality
        loss = self.model.compute_loss(windows, labels)
        
        # Down-weight low quality samples
        quality_weights = torch.sigmoid(
            (quality_scores - self.quality_threshold) * 10
        )
        
        weighted_loss = (loss * quality_weights).mean()
        
        # Log quality metrics
        self.log('avg_quality', quality_scores.mean())
        self.log('low_quality_ratio', (quality_scores < 0.5).float().mean())
        
        return weighted_loss
```

## Artifact Types and Strategies

### 1. Eye Blink Detection
```python
def detect_eog_artifacts(epochs, threshold=150e-6):
    """Detect eye movement artifacts using EOG channels"""
    
    # Get EOG channels if available
    eog_picks = mne.pick_types(epochs.info, eog=True)
    
    if len(eog_picks) == 0:
        # Use frontal EEG channels as proxy
        eog_picks = mne.pick_channels(
            epochs.ch_names, 
            ['Fp1', 'Fp2', 'F7', 'F8']
        )
    
    # Detect blinks
    eog_data = epochs.get_data()[:, eog_picks, :]
    
    # Compute peak amplitudes
    peaks = np.max(np.abs(eog_data), axis=(1, 2))
    
    # Mark epochs with blinks
    blink_epochs = peaks > threshold
    
    return blink_epochs
```

### 2. Muscle Artifact Detection
```python
def detect_muscle_artifacts(epochs, freq_range=(70, 100)):
    """Detect muscle artifacts using high-frequency power"""
    
    # Compute PSD in muscle frequency range
    psds, freqs = mne.time_frequency.psd_multitaper(
        epochs, 
        fmin=freq_range[0], 
        fmax=freq_range[1],
        n_jobs=1
    )
    
    # Average power in muscle band
    muscle_power = psds.mean(axis=2)  # Average over frequencies
    
    # Z-score normalization
    z_scores = (muscle_power - muscle_power.mean(axis=0)) / muscle_power.std(axis=0)
    
    # Detect outliers
    muscle_epochs = np.any(z_scores > 3, axis=1)
    
    return muscle_epochs
```

### 3. Environmental Artifact Detection
```python
def detect_environmental_artifacts(epochs, powerline_freq=50):
    """Detect environmental artifacts (powerline, electromagnetic)"""
    
    artifacts = {}
    
    # Powerline interference
    psds, freqs = mne.time_frequency.psd_welch(
        epochs,
        fmin=powerline_freq - 2,
        fmax=powerline_freq + 2
    )
    
    # Check for peaks at powerline frequency
    freq_idx = np.argmin(np.abs(freqs - powerline_freq))
    powerline_power = psds[:, :, freq_idx]
    
    # Flag epochs with strong powerline
    artifacts['powerline'] = powerline_power.max(axis=1) > np.percentile(
        powerline_power.max(axis=1), 95
    )
    
    # Broadband noise (electromagnetic interference)
    broadband_std = epochs.get_data().std(axis=2)
    artifacts['broadband'] = np.any(
        broadband_std > 3 * np.median(broadband_std, axis=0),
        axis=1
    )
    
    return artifacts
```

## Performance Optimization

### 1. Caching Strategy
```python
class AutoRejectCache:
    """Efficient caching for AutoReject results"""
    
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Memory cache for current session
        self.memory_cache = {}
    
    def get_or_compute(self, epochs, key):
        """Get from cache or compute AutoReject"""
        
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            self.memory_cache[key] = result
            return result
        
        # Compute AutoReject
        ar = AutoReject(n_jobs=4)
        epochs_clean, reject_log = ar.fit_transform(epochs)
        
        # Cache results
        result = {
            'thresholds': ar.thresholds_,
            'reject_log': reject_log,
            'n_interpolated': ar.n_interpolated_,
            'consensus': ar.consensus_
        }
        
        # Save to disk
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        # Save to memory
        self.memory_cache[key] = result
        
        return result
```

### 2. Batch Processing
```python
def batch_autoreject(file_paths, n_jobs=-1):
    """Process multiple files in parallel"""
    
    def process_single_file(file_path):
        # Load data
        raw = mne.io.read_raw_edf(file_path, preload=True)
        
        # Create epochs
        epochs = create_fixed_length_epochs(raw, duration=8.0)
        
        # Apply AutoReject
        ar = AutoReject()
        epochs_clean, reject_log = ar.fit_transform(epochs)
        
        # Return summary
        return {
            'file': file_path,
            'n_epochs': len(epochs),
            'n_rejected': reject_log.bad_epochs.sum(),
            'n_interpolated': (reject_log.labels == 1).sum(),
            'quality_score': 1 - (reject_log.bad_epochs.sum() / len(epochs))
        }
    
    # Process in parallel
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(fp) for fp in file_paths
    )
    
    return pd.DataFrame(results)
```

## Clinical Integration

### 1. Quality Report Generation
```python
def generate_quality_report(raw, reject_log):
    """Generate comprehensive quality control report"""
    
    report = {
        'summary': {
            'total_duration_min': raw.times[-1] / 60,
            'n_channels': len(raw.ch_names),
            'sampling_rate': raw.info['sfreq'],
            'n_epochs_analyzed': len(reject_log.labels),
            'n_epochs_rejected': (reject_log.labels == 2).sum(),
            'n_epochs_repaired': (reject_log.labels == 1).sum(),
            'overall_quality_score': compute_quality_score(reject_log)
        },
        
        'channel_quality': analyze_channel_quality(raw, reject_log),
        'temporal_quality': analyze_temporal_quality(raw, reject_log),
        'artifact_summary': summarize_artifacts(raw, reject_log)
    }
    
    return report

def analyze_channel_quality(raw, reject_log):
    """Analyze quality per channel"""
    
    bad_sensors = reject_log.bad_sensors  # [n_epochs, n_channels]
    
    channel_stats = []
    for ch_idx, ch_name in enumerate(raw.ch_names):
        stats = {
            'channel': ch_name,
            'bad_epoch_ratio': bad_sensors[:, ch_idx].mean(),
            'interpolation_count': np.sum(
                bad_sensors[reject_log.labels == 1, ch_idx]
            ),
            'quality_grade': grade_channel_quality(
                bad_sensors[:, ch_idx].mean()
            )
        }
        channel_stats.append(stats)
    
    return pd.DataFrame(channel_stats)

def grade_channel_quality(bad_ratio):
    """Grade channel quality A-F"""
    if bad_ratio < 0.05:
        return 'A'
    elif bad_ratio < 0.10:
        return 'B'
    elif bad_ratio < 0.20:
        return 'C'
    elif bad_ratio < 0.30:
        return 'D'
    else:
        return 'F'
```

### 2. Visualization Tools
```python
def plot_quality_summary(raw, reject_log):
    """Create quality control visualization"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Bad epochs over time
    ax = axes[0]
    time_points = np.arange(len(reject_log.labels)) * 8 / 60  # minutes
    
    # Color code: green=good, yellow=repaired, red=rejected
    colors = ['green', 'yellow', 'red']
    labels = ['Good', 'Repaired', 'Rejected']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = reject_log.labels == i
        ax.scatter(time_points[mask], np.ones(mask.sum()) * i, 
                  c=color, label=label, s=20)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Epoch Status')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(labels)
    ax.legend()
    ax.set_title('Epoch Quality Over Time')
    
    # 2. Channel quality heatmap
    ax = axes[1]
    bad_by_channel = reject_log.bad_sensors.mean(axis=0)
    
    im = ax.imshow(bad_by_channel.reshape(1, -1), 
                   aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(raw.ch_names)))
    ax.set_xticklabels(raw.ch_names, rotation=45)
    ax.set_ylabel('Bad Epoch %')
    ax.set_title('Channel Quality')
    plt.colorbar(im, ax=ax)
    
    # 3. Quality metrics over time
    ax = axes[2]
    window_size = 50  # epochs
    quality_curve = []
    
    for i in range(0, len(reject_log.labels) - window_size):
        window = reject_log.labels[i:i+window_size]
        quality = (window == 0).sum() / window_size
        quality_curve.append(quality)
    
    ax.plot(np.arange(len(quality_curve)) * 8 / 60, quality_curve)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Local Quality Score')
    ax.set_title('Recording Quality Over Time')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig
```

## Best Practices

### 1. Parameter Selection
```python
# Conservative settings (minimize false rejections)
ar_conservative = AutoReject(
    n_interpolate=[1, 2],  # Interpolate max 2 channels
    consensus=[0.3, 0.4],  # Require 30-40% bad channels to reject
    cv=10,  # More folds for stability
    thresh_method='bayesian'  # Bayesian optimization
)

# Aggressive settings (maximize artifact removal)
ar_aggressive = AutoReject(
    n_interpolate=[1, 2, 3, 4, 5],  # Interpolate up to 5 channels
    consensus=[0.1, 0.2, 0.3],  # Reject with 10% bad channels
    cv=5,
    thresh_method='random_search'
)

# Clinical settings (balanced)
ar_clinical = AutoReject(
    n_interpolate=[1, 2, 3, 4],
    consensus=[0.2, 0.3, 0.4],
    cv=5,
    random_state=42  # Reproducibility
)
```

### 2. Integration Guidelines
1. **Always run AutoReject before EEGPT feature extraction**
2. **Cache AutoReject results to avoid recomputation**
3. **Log quality metrics for clinical review**
4. **Validate on subset before batch processing**
5. **Use conservative settings for clinical data**

## Performance Impact

### Expected Improvements with AutoReject
- **AUROC**: +3-5% improvement in abnormality detection
- **Sensitivity**: +5-10% for subtle abnormalities
- **Robustness**: Consistent performance across sites
- **Reliability**: Reduced variance in predictions

### Computational Overhead
- **Time**: +20-30% processing time
- **Memory**: +500MB for caching
- **Storage**: +10% for quality logs

## References

- AutoReject Paper: "Autoreject: Automated artifact rejection for MEG and EEG data"
- Implementation: https://autoreject.github.io
- MNE Integration: https://mne.tools/stable/auto_tutorials/preprocessing/plot_40_autoreject.html
- Our Integration: brain_go_brrr/data/tuab_enhanced_dataset.py