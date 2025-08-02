# 🚀 AUTOREJECT TDD IMPLEMENTATION PLAN

## 🎯 MISSION: INTEGRATE AUTOREJECT INTO EEGPT TRAINING PIPELINE

Despite significant architectural challenges, we CAN make this work with creative solutions. This document outlines a comprehensive TDD approach to integrate AutoReject while maintaining performance.

## 📊 CURRENT SITUATION ANALYSIS

### The Core Problem
1. **AutoReject needs epochs** → EEGPT uses continuous windows
2. **AutoReject needs channel positions** → TUAB lacks them
3. **AutoReject is memory intensive** → We're already constrained
4. **AutoReject uses CV optimization** → Slow for 3000+ files

### But We Can Solve This! 💪

## 🔧 PROPOSED SOLUTION: HYBRID WINDOW-EPOCH APPROACH

### Key Insight
We can treat windows AS epochs for AutoReject purposes, then convert back!

```python
# Current: Raw → Windows → EEGPT
# Proposed: Raw → Windows → Pseudo-Epochs → AutoReject → Clean Windows → EEGPT
```

## 📝 TDD IMPLEMENTATION PLAN

### PHASE 1: WRITE THE TESTS FIRST (Week 1)

#### Test 1: Window-to-Epoch Converter
```python
# tests/test_autoreject_integration.py
def test_window_to_epoch_conversion():
    """Test converting sliding windows to MNE epochs for AutoReject."""
    # Given: 10s windows with 50% overlap
    windows = create_test_windows(n_windows=100, duration=10, overlap=0.5)
    
    # When: Converting to epochs
    epochs = WindowToEpochConverter.convert(windows)
    
    # Then: Epochs should maintain data integrity
    assert epochs.get_data().shape == (100, 19, 2560)  # 10s @ 256Hz
    assert_array_almost_equal(windows[0], epochs.get_data()[0])
```

#### Test 2: Synthetic Channel Positions
```python
def test_synthetic_channel_positions():
    """Test adding synthetic but realistic channel positions."""
    # Given: TUAB channels without positions
    raw = create_tuab_raw_without_positions()
    
    # When: Adding synthetic positions
    raw_with_pos = add_synthetic_positions(raw, layout='standard_1020')
    
    # Then: Positions should be valid for AutoReject
    assert all(ch['loc'][:3].any() for ch in raw_with_pos.info['chs'])
    assert AutoReject()._check_positions(raw_with_pos)  # Should not raise
```

#### Test 3: Memory-Efficient AutoReject
```python
def test_memory_efficient_autoreject():
    """Test chunked AutoReject processing for large datasets."""
    # Given: Large dataset that would OOM
    large_dataset = create_large_test_dataset(n_files=1000)
    
    # When: Processing in chunks
    ar_processor = ChunkedAutoRejectProcessor(chunk_size=100)
    results = ar_processor.fit_transform(large_dataset)
    
    # Then: Should process without OOM
    assert len(results) == 1000
    assert get_memory_usage() < 8 * 1024**3  # Less than 8GB
```

#### Test 4: AutoReject Parameter Caching
```python
def test_autoreject_parameter_caching():
    """Test caching AutoReject parameters to avoid recomputation."""
    # Given: AutoReject parameters from subset
    subset = load_tuab_subset(n=100)
    ar_params = fit_autoreject_on_subset(subset)
    
    # When: Applying to full dataset
    full_dataset = load_tuab_full()
    cleaned = apply_cached_autoreject(full_dataset, ar_params)
    
    # Then: Should be fast and effective
    assert processing_time < 0.1  # 100ms per file max
    assert artifact_reduction_rate > 0.8
```

#### Test 5: Integration with EEGPT Pipeline
```python
def test_eegpt_with_autoreject():
    """Test full pipeline with AutoReject integrated."""
    # Given: Noisy TUAB data
    noisy_data = load_noisy_tuab_sample()
    
    # When: Processing through pipeline
    pipeline = EEGPTPipeline(use_autoreject=True)
    features = pipeline.process(noisy_data)
    
    # Then: Features should be cleaner
    assert features.artifact_score < 0.2
    assert features.snr > 10  # dB
```

### PHASE 2: IMPLEMENTATION ARCHITECTURE (Week 2)

#### 1. Window-Epoch Adapter Pattern
```python
class WindowEpochAdapter:
    """Adapter to make windows compatible with AutoReject."""
    
    def __init__(self, window_duration=10.0, overlap=0.5):
        self.window_duration = window_duration
        self.overlap = overlap
    
    def windows_to_epochs(self, raw, windows):
        """Convert sliding windows to MNE Epochs."""
        # Create artificial events at window starts
        events = self._create_window_events(raw, windows)
        
        # Create epochs with proper metadata
        epochs = mne.Epochs(
            raw, events,
            tmin=0, tmax=self.window_duration,
            baseline=None,
            preload=True,
            metadata=self._create_metadata(windows)
        )
        return epochs
    
    def epochs_to_windows(self, epochs_clean):
        """Convert cleaned epochs back to windows."""
        # Reconstruct overlapping windows from epochs
        return self._reconstruct_windows(epochs_clean)
```

#### 2. Synthetic Position Generator
```python
class SyntheticPositionGenerator:
    """Generate realistic channel positions for TUAB data."""
    
    STANDARD_POSITIONS = {
        'FP1': [-0.0270, 0.0866, 0.0150],
        'FP2': [0.0270, 0.0866, 0.0150],
        # ... full 10-20 system
        'T7': [-0.0860, 0.0, -0.0150],  # Modern name
        'T3': [-0.0860, 0.0, -0.0150],  # Old name (same position)
    }
    
    def add_positions_to_raw(self, raw):
        """Add synthetic but valid positions."""
        montage = self._create_tuab_montage()
        raw.set_montage(montage, on_missing='ignore')
        return raw
```

#### 3. Chunked AutoReject Processor
```python
class ChunkedAutoRejectProcessor:
    """Memory-efficient AutoReject for large datasets."""
    
    def __init__(self, chunk_size=100, cache_path='autoreject_cache'):
        self.chunk_size = chunk_size
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(exist_ok=True)
    
    def fit_on_subset(self, file_paths, n_samples=200):
        """Fit AutoReject on representative subset."""
        # Sample files stratified by label
        subset = self._stratified_sample(file_paths, n_samples)
        
        # Fit AutoReject
        ar = AutoReject(n_interpolate=[1, 4], n_jobs=1)
        ar.fit(subset)
        
        # Cache parameters
        self._cache_parameters(ar)
        return ar
    
    def transform_dataset(self, dataset):
        """Apply fitted AutoReject to full dataset."""
        ar = self._load_cached_parameters()
        
        for chunk in self._chunk_dataset(dataset):
            cleaned_chunk = ar.transform(chunk)
            yield cleaned_chunk
```

#### 4. Enhanced TUAB Dataset with AutoReject
```python
class TUABDatasetWithAutoReject(TUABEnhancedDataset):
    """TUAB dataset with integrated AutoReject cleaning."""
    
    def __init__(self, *args, use_autoreject=True, ar_cache_dir='ar_cache', **kwargs):
        super().__init__(*args, **kwargs)
        self.use_autoreject = use_autoreject
        self.ar_processor = None
        
        if use_autoreject:
            self._init_autoreject(ar_cache_dir)
    
    def _load_and_clean_file(self, file_path):
        """Load EDF and apply AutoReject if enabled."""
        # Load raw data
        raw = mne.io.read_raw_edf(file_path, preload=True)
        
        # Add synthetic positions
        raw = self.position_generator.add_positions_to_raw(raw)
        
        # Standard preprocessing
        raw = self._preprocess_raw(raw)
        
        if self.use_autoreject:
            # Convert to pseudo-epochs
            windows = self._extract_windows(raw)
            epochs = self.adapter.windows_to_epochs(raw, windows)
            
            # Apply AutoReject
            epochs_clean = self.ar_processor.transform(epochs)
            
            # Convert back to windows
            windows_clean = self.adapter.epochs_to_windows(epochs_clean)
            
            return windows_clean
        else:
            return self._extract_windows(raw)
```

### PHASE 3: PERFORMANCE OPTIMIZATIONS (Week 3)

#### 1. Parallel AutoReject Processing
```python
class ParallelAutoRejectProcessor:
    """Use multiprocessing for AutoReject."""
    
    def process_batch(self, file_paths, n_workers=4):
        """Process files in parallel."""
        with mp.Pool(n_workers) as pool:
            results = pool.map(self._process_single, file_paths)
        return results
```

#### 2. GPU-Accelerated Artifact Detection
```python
class GPUArtifactDetector:
    """Use GPU for fast artifact detection pre-screening."""
    
    def pre_screen(self, windows):
        """Quick GPU-based artifact detection."""
        # Move to GPU
        windows_gpu = torch.tensor(windows).cuda()
        
        # Fast artifact metrics
        variance = torch.var(windows_gpu, dim=-1)
        kurtosis = self._gpu_kurtosis(windows_gpu)
        
        # Flag obvious artifacts
        artifact_mask = (variance > threshold) | (kurtosis > 8)
        
        return artifact_mask.cpu().numpy()
```

#### 3. Incremental Learning
```python
class IncrementalAutoReject:
    """Update AutoReject parameters incrementally."""
    
    def update_parameters(self, new_data, learning_rate=0.1):
        """Incrementally update rejection thresholds."""
        # Compute new thresholds on batch
        new_thresh = self._compute_thresholds(new_data)
        
        # Exponential moving average update
        self.thresholds = (1 - learning_rate) * self.thresholds + learning_rate * new_thresh
```

### PHASE 4: FALLBACK STRATEGIES (Week 4)

#### 1. Graceful Degradation
```python
class AutoRejectWithFallback:
    """AutoReject with automatic fallback to simpler methods."""
    
    def process(self, data):
        try:
            # Try full AutoReject
            return self.autoreject.fit_transform(data)
        except MemoryError:
            logger.warning("AutoReject OOM - falling back to chunk processing")
            return self._chunked_process(data)
        except RuntimeError as e:
            if "channel positions" in str(e):
                logger.warning("No positions - using amplitude rejection")
                return self._amplitude_rejection(data)
            raise
```

#### 2. Hybrid Approach
```python
class HybridQualityControl:
    """Combine AutoReject with fast methods."""
    
    def clean_data(self, raw):
        # Fast pre-screening with amplitude
        bad_channels = self._amplitude_screening(raw)
        raw.info['bads'] = bad_channels
        
        # AutoReject on remaining channels
        if len(raw.ch_names) - len(bad_channels) >= 5:
            epochs = self._create_epochs(raw)
            epochs_clean = self.autoreject.transform(epochs)
            return self._epochs_to_raw(epochs_clean)
        else:
            return raw
```

## 🚨 CRITICAL IMPLEMENTATION DETAILS

### 1. Channel Mapping Consistency
```python
# ALWAYS map old to new names BEFORE AutoReject
OLD_TO_NEW_MAPPING = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}

def ensure_modern_channel_names(raw):
    mapping = {ch: OLD_TO_NEW_MAPPING.get(ch, ch) for ch in raw.ch_names}
    raw.rename_channels(mapping)
    return raw
```

### 2. Memory Management
```python
# Clear GPU cache after each batch
torch.cuda.empty_cache()

# Use context managers
with torch.no_grad():
    process_batch()

# Explicit garbage collection
import gc
gc.collect()
```

### 3. Error Handling
```python
# Comprehensive error handling
try:
    epochs_clean = autoreject.fit_transform(epochs)
except np.linalg.LinAlgError:
    logger.error("SVD convergence failure - data too noisy")
    epochs_clean = amplitude_rejection(epochs)
except ValueError as e:
    if "n_interpolate" in str(e):
        logger.error("Too many bad channels for interpolation")
        epochs_clean = epochs  # Return unmodified
    else:
        raise
```

## 📊 EXPECTED IMPROVEMENTS

### With AutoReject Integration:
- **Artifact reduction**: 70-90% of eye blinks, muscle artifacts
- **Bad channel handling**: Automatic interpolation instead of zeroing
- **AUROC improvement**: +5-10% expected (0.85 → 0.90-0.93)
- **Training stability**: Reduced gradient noise from artifacts

### Performance Trade-offs:
- **Training time**: +30-50% (acceptable for quality gain)
- **Memory usage**: +2-3GB during processing (manageable with chunking)
- **Complexity**: Higher, but encapsulated in adapter classes

## 🎬 IMPLEMENTATION TIMELINE

### Week 1: TDD Foundation
- [ ] Write all test cases
- [ ] Set up test infrastructure
- [ ] Create mock data generators

### Week 2: Core Implementation
- [ ] Window-Epoch adapter
- [ ] Synthetic position generator
- [ ] Basic AutoReject integration

### Week 3: Optimization
- [ ] Chunked processing
- [ ] Parameter caching
- [ ] Parallel processing

### Week 4: Testing & Refinement
- [ ] Integration testing
- [ ] Performance benchmarking
- [ ] Documentation

## 🔐 RISK MITIGATION

### Risk 1: Memory Overflow
**Mitigation**: Chunked processing, parameter caching, GPU management

### Risk 2: Position Data Issues
**Mitigation**: Synthetic positions, fallback to amplitude methods

### Risk 3: Performance Degradation
**Mitigation**: Parallel processing, pre-trained parameters, GPU acceleration

### Risk 4: Compatibility Issues
**Mitigation**: Adapter pattern, extensive testing, gradual rollout

## 🎯 SUCCESS CRITERIA

1. **All tests pass** (100% coverage)
2. **AUROC improvement** ≥ 5%
3. **Training time increase** < 50%
4. **Memory usage** < 10GB peak
5. **No breaking changes** to existing API

## 💡 INNOVATIVE SOLUTIONS

### 1. Window-Epoch Duality
Treat sliding windows as pseudo-epochs, maintaining compatibility with both systems.

### 2. Transfer Learning for AutoReject
Pre-train AutoReject parameters on a subset, then apply to full dataset.

### 3. Hybrid Processing
Combine fast amplitude screening with selective AutoReject application.

### 4. Synthetic but Valid Positions
Generate positions that satisfy AutoReject requirements while being anatomically plausible.

## 🚀 LET'S BUILD THIS!

Despite the challenges, we CAN integrate AutoReject successfully. The key is:
1. **Creative adaptation** (windows as epochs)
2. **Smart optimization** (chunking, caching)
3. **Robust fallbacks** (graceful degradation)
4. **Thorough testing** (TDD all the way)

This plan turns obstacles into opportunities. AutoReject WILL improve our model performance!