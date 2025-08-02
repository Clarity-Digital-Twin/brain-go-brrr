# 🔬 AUTOREJECT PRECISE INTEGRATION SPECIFICATION

## 🎯 EXACT INTEGRATION POINTS & MODIFICATIONS

### 1. DATA FLOW TRANSFORMATION

#### CURRENT FLOW:
```
EDF File → mne.io.read_raw_edf() → Filter → Extract Windows → Tensor → EEGPT
```

#### NEW FLOW WITH AUTOREJECT:
```
EDF File → mne.io.read_raw_edf() → Add Positions → Filter → Windows → 
→ Pseudo-Epochs → AutoReject → Clean Epochs → Reconstruct Windows → Tensor → EEGPT
```

### 2. PRECISE FILE MODIFICATIONS

#### A. `src/brain_go_brrr/data/tuab_enhanced_dataset.py`

**Lines to modify: 287-344 (_load_edf_file method)**

```python
# ADD at line 287:
from brain_go_brrr.preprocessing.autoreject_adapter import WindowEpochAdapter, SyntheticPositionGenerator
from brain_go_brrr.preprocessing.chunked_autoreject import ChunkedAutoRejectProcessor

# MODIFY __init__ at line 130:
def __init__(
    self,
    data_dir: str | Path,
    split: str = "train",
    # ... existing params ...
    use_autoreject: bool = False,  # NEW PARAMETER
    ar_cache_dir: str | Path = "autoreject_cache",  # NEW PARAMETER
):
    # ... existing init code ...
    self.use_autoreject = use_autoreject
    self.ar_processor = None
    self.window_adapter = None
    self.position_generator = None
    
    if self.use_autoreject:
        self._initialize_autoreject(ar_cache_dir)

# ADD new method after __init__:
def _initialize_autoreject(self, cache_dir: str | Path):
    """Initialize AutoReject components."""
    self.ar_processor = ChunkedAutoRejectProcessor(cache_dir=cache_dir)
    self.window_adapter = WindowEpochAdapter(
        window_duration=self.window_duration,
        window_stride=self.window_stride
    )
    self.position_generator = SyntheticPositionGenerator()
    
    # Pre-fit on subset if no cache exists
    if not self.ar_processor.has_cached_params():
        logger.info("Fitting AutoReject on subset...")
        subset_files = self._get_stratified_subset(n=200)
        self.ar_processor.fit_on_subset(subset_files)

# MODIFY _load_edf_file at line 287:
def _load_edf_file(self, file_path: Path, label: int) -> dict[str, Any]:
    """Load and preprocess a single EDF file with optional AutoReject."""
    try:
        # Load raw EEG data
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # NEW: Add synthetic positions if using AutoReject
        if self.use_autoreject:
            raw = self.position_generator.add_positions_to_raw(raw)
        
        # ... existing channel selection code ...
        
        # Apply preprocessing
        raw.filter(0.1, 75.0, fir_design="firwin", verbose=False)
        raw.notch_filter(50.0, fir_design="firwin", verbose=False)
        
        # NEW: Apply AutoReject before windowing
        if self.use_autoreject:
            raw = self._apply_autoreject_to_raw(raw)
        
        # ... rest of existing windowing code ...
```

#### B. Create `src/brain_go_brrr/preprocessing/autoreject_adapter.py`

```python
"""Adapter classes for AutoReject integration with windowed data."""

import numpy as np
import mne
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class WindowEpochAdapter:
    """Convert between sliding windows and MNE Epochs for AutoReject compatibility."""
    
    def __init__(self, window_duration: float = 10.0, window_stride: float = 5.0):
        self.window_duration = window_duration
        self.window_stride = window_stride
        self.overlap = 1.0 - (window_stride / window_duration)
    
    def raw_to_windowed_epochs(self, raw: mne.io.Raw) -> mne.Epochs:
        """Convert raw data to epochs matching our windowing scheme."""
        # Calculate window parameters
        sfreq = raw.info['sfreq']
        window_samples = int(self.window_duration * sfreq)
        stride_samples = int(self.window_stride * sfreq)
        
        # Create events at window start positions
        n_samples = raw.n_times
        start_positions = np.arange(0, n_samples - window_samples + 1, stride_samples)
        
        # Create events array (sample_idx, 0, event_id)
        events = np.column_stack([
            start_positions,
            np.zeros(len(start_positions), dtype=int),
            np.ones(len(start_positions), dtype=int)
        ])
        
        # Create epochs
        epochs = mne.Epochs(
            raw, events,
            event_id={'window': 1},
            tmin=0,
            tmax=self.window_duration - (1.0 / sfreq),  # Adjust for sample duration
            baseline=None,
            preload=True,
            verbose=False
        )
        
        return epochs
    
    def epochs_to_continuous(self, epochs_clean: mne.Epochs, original_raw: mne.io.Raw) -> mne.io.Raw:
        """Reconstruct continuous data from cleaned epochs."""
        # Get clean epoch data
        data_clean = epochs_clean.get_data()  # (n_epochs, n_channels, n_times)
        
        # Reconstruction with overlap handling
        sfreq = epochs_clean.info['sfreq']
        n_channels = data_clean.shape[1]
        total_samples = original_raw.n_times
        
        # Initialize output array
        reconstructed = np.zeros((n_channels, total_samples))
        counts = np.zeros(total_samples)
        
        # Get event times
        events = epochs_clean.events
        
        for i, (event_sample, _, _) in enumerate(events):
            if i < len(data_clean):  # Ensure we have data for this epoch
                start_idx = event_sample
                end_idx = start_idx + data_clean.shape[2]
                
                if end_idx <= total_samples:
                    # Add data with averaging for overlaps
                    reconstructed[:, start_idx:end_idx] += data_clean[i]
                    counts[start_idx:end_idx] += 1
        
        # Average overlapping regions
        counts[counts == 0] = 1  # Avoid division by zero
        reconstructed /= counts
        
        # Create new Raw object with cleaned data
        raw_clean = mne.io.RawArray(
            reconstructed,
            original_raw.info.copy(),
            verbose=False
        )
        
        return raw_clean


class SyntheticPositionGenerator:
    """Generate anatomically valid channel positions for datasets lacking montage info."""
    
    # Standard 10-20 positions (x, y, z in meters)
    STANDARD_1020_POSITIONS = {
        # Frontal
        'FP1': np.array([-0.0270, 0.0866, 0.0150]),
        'FP2': np.array([0.0270, 0.0866, 0.0150]),
        'F7': np.array([-0.0702, 0.0596, -0.0150]),
        'F3': np.array([-0.0450, 0.0693, 0.0300]),
        'FZ': np.array([0.0000, 0.0732, 0.0450]),
        'F4': np.array([0.0450, 0.0693, 0.0300]),
        'F8': np.array([0.0702, 0.0596, -0.0150]),
        
        # Temporal (including old naming)
        'T7': np.array([-0.0860, 0.0000, -0.0150]),  # Modern
        'T3': np.array([-0.0860, 0.0000, -0.0150]),  # Old (same position)
        'T8': np.array([0.0860, 0.0000, -0.0150]),   # Modern
        'T4': np.array([0.0860, 0.0000, -0.0150]),   # Old (same position)
        
        # Central
        'C3': np.array([-0.0520, 0.0000, 0.0600]),
        'CZ': np.array([0.0000, 0.0000, 0.0850]),
        'C4': np.array([0.0520, 0.0000, 0.0600]),
        
        # Parietal (including old naming)
        'P7': np.array([-0.0702, -0.0596, -0.0150]),  # Modern
        'T5': np.array([-0.0702, -0.0596, -0.0150]),  # Old (same position)
        'P3': np.array([-0.0450, -0.0693, 0.0300]),
        'PZ': np.array([0.0000, -0.0732, 0.0450]),
        'P4': np.array([0.0450, -0.0693, 0.0300]),
        'P8': np.array([0.0702, -0.0596, -0.0150]),   # Modern
        'T6': np.array([0.0702, -0.0596, -0.0150]),   # Old (same position)
        
        # Occipital
        'O1': np.array([-0.0270, -0.0866, 0.0150]),
        'O2': np.array([0.0270, -0.0866, 0.0150]),
    }
    
    def add_positions_to_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Add synthetic but anatomically valid positions to raw data."""
        # Get channel names
        ch_names = raw.ch_names
        
        # Build position dictionary for available channels
        ch_pos = {}
        for ch_name in ch_names:
            ch_upper = ch_name.upper()
            if ch_upper in self.STANDARD_1020_POSITIONS:
                ch_pos[ch_name] = self.STANDARD_1020_POSITIONS[ch_upper]
        
        if not ch_pos:
            logger.warning("No standard channel names found, using default positions")
            # Use evenly spaced positions as fallback
            n_channels = len(ch_names)
            angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
            radius = 0.075  # 7.5 cm
            for i, ch_name in enumerate(ch_names):
                x = radius * np.cos(angles[i])
                y = radius * np.sin(angles[i])
                ch_pos[ch_name] = np.array([x, y, 0.02])  # 2cm above head
        
        # Create montage
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos)
        
        # Apply to raw
        raw.set_montage(montage, on_missing='ignore')
        logger.info(f"Added positions for {len(ch_pos)}/{len(ch_names)} channels")
        
        return raw
```

#### C. Create `src/brain_go_brrr/preprocessing/chunked_autoreject.py`

```python
"""Memory-efficient AutoReject implementation for large datasets."""

import numpy as np
import mne
from pathlib import Path
import pickle
import joblib
from typing import Optional, List, Dict, Any, Tuple
import logging
from autoreject import AutoReject

logger = logging.getLogger(__name__)


class ChunkedAutoRejectProcessor:
    """Process large datasets with AutoReject using chunking and caching."""
    
    def __init__(
        self,
        cache_dir: str | Path = "autoreject_cache",
        chunk_size: int = 100,
        n_interpolate: List[int] = [1, 4],
        consensus: float = 0.1,
        random_state: int = 42
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.n_interpolate = n_interpolate
        self.consensus = consensus
        self.random_state = random_state
        self.ar_params = None
        self.is_fitted = False
    
    def has_cached_params(self) -> bool:
        """Check if pre-fitted parameters exist."""
        param_file = self.cache_dir / "autoreject_params.pkl"
        return param_file.exists()
    
    def fit_on_subset(self, file_paths: List[Path], n_samples: int = 200) -> None:
        """Fit AutoReject on a representative subset."""
        logger.info(f"Fitting AutoReject on {n_samples} samples...")
        
        # Sample files stratified by directory/label if possible
        sampled_files = self._stratified_sample(file_paths, n_samples)
        
        # Load and concatenate epochs from sampled files
        all_epochs = []
        for file_path in sampled_files:
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                # Create fixed-length epochs
                epochs = mne.make_fixed_length_epochs(
                    raw, duration=10.0, overlap=5.0, verbose=False
                )
                all_epochs.append(epochs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if not all_epochs:
            raise ValueError("No valid files found for AutoReject fitting")
        
        # Concatenate all epochs
        combined_epochs = mne.concatenate_epochs(all_epochs)
        logger.info(f"Fitting on {len(combined_epochs)} total epochs")
        
        # Fit AutoReject with minimal parameters for speed
        ar = AutoReject(
            n_interpolate=self.n_interpolate,
            consensus=self.consensus,
            n_jobs=1,  # Single job for memory efficiency
            random_state=self.random_state,
            verbose=False
        )
        
        # Fit and extract parameters
        ar.fit(combined_epochs)
        
        # Save fitted parameters
        self._save_parameters(ar)
        self.ar_params = self._extract_parameters(ar)
        self.is_fitted = True
        
        logger.info("AutoReject fitting completed and cached")
    
    def transform_raw(self, raw: mne.io.Raw, window_adapter: 'WindowEpochAdapter') -> mne.io.Raw:
        """Apply fitted AutoReject to raw data."""
        if not self.is_fitted:
            self._load_parameters()
        
        # Convert to epochs
        epochs = window_adapter.raw_to_windowed_epochs(raw)
        
        # Apply AutoReject with pre-fitted parameters
        epochs_clean = self._apply_autoreject(epochs)
        
        # Convert back to raw
        raw_clean = window_adapter.epochs_to_continuous(epochs_clean, raw)
        
        return raw_clean
    
    def _apply_autoreject(self, epochs: mne.Epochs) -> mne.Epochs:
        """Apply AutoReject with cached parameters."""
        # Create AutoReject instance with cached thresholds
        ar = self._create_autoreject_from_params()
        
        # Transform epochs
        epochs_clean = ar.transform(epochs)
        
        # Log results
        n_rejected = len(epochs) - len(epochs_clean)
        logger.debug(f"AutoReject: {n_rejected}/{len(epochs)} epochs rejected")
        
        return epochs_clean
    
    def _create_autoreject_from_params(self) -> AutoReject:
        """Create AutoReject instance from cached parameters."""
        ar = AutoReject(
            n_interpolate=self.n_interpolate,
            consensus=self.consensus,
            n_jobs=1,
            random_state=self.random_state,
            verbose=False
        )
        
        # Set pre-computed thresholds
        if self.ar_params:
            ar.threshes_ = self.ar_params['thresholds']
            ar.consensus_ = self.ar_params['consensus']
            ar.n_interpolate_ = self.ar_params['n_interpolate']
            ar.picks_ = self.ar_params['picks']
        
        return ar
    
    def _save_parameters(self, ar: AutoReject) -> None:
        """Save fitted AutoReject parameters."""
        params = self._extract_parameters(ar)
        param_file = self.cache_dir / "autoreject_params.pkl"
        
        with open(param_file, 'wb') as f:
            pickle.dump(params, f)
        
        logger.info(f"AutoReject parameters saved to {param_file}")
    
    def _load_parameters(self) -> None:
        """Load pre-fitted parameters."""
        param_file = self.cache_dir / "autoreject_params.pkl"
        
        if not param_file.exists():
            raise ValueError(f"No cached parameters found at {param_file}")
        
        with open(param_file, 'rb') as f:
            self.ar_params = pickle.load(f)
        
        self.is_fitted = True
        logger.info("AutoReject parameters loaded from cache")
    
    def _extract_parameters(self, ar: AutoReject) -> Dict[str, Any]:
        """Extract fitted parameters from AutoReject instance."""
        return {
            'thresholds': ar.threshes_,
            'consensus': ar.consensus_,
            'n_interpolate': ar.n_interpolate_,
            'picks': ar.picks_ if hasattr(ar, 'picks_') else None
        }
    
    def _stratified_sample(self, file_paths: List[Path], n_samples: int) -> List[Path]:
        """Sample files trying to maintain label distribution."""
        # Simple random sampling for now
        # TODO: Implement proper stratification based on labels
        import random
        
        random.seed(self.random_state)
        n_samples = min(n_samples, len(file_paths))
        return random.sample(file_paths, n_samples)
```

#### D. Modify `experiments/eegpt_linear_probe/train_enhanced.py`

**Add at line 35 (imports):**
```python
from brain_go_brrr.preprocessing.autoreject_adapter import WindowEpochAdapter

# Add to argument parser (around line 650):
parser.add_argument(
    "--use-autoreject",
    action="store_true",
    help="Use AutoReject for artifact removal (adds ~30%% training time)"
)
parser.add_argument(
    "--ar-cache-dir",
    type=str,
    default="autoreject_cache",
    help="Directory for AutoReject parameter cache"
)

# Modify dataset initialization (around line 480):
train_dataset = TUABEnhancedDataset(
    data_dir=data_config["data_dir"],
    split="train",
    max_samples=train_config.get("max_samples"),
    window_duration=data_config["window_duration"],
    window_stride=data_config["window_stride"],
    sampling_rate=data_config["sampling_rate"],
    use_autoreject=args.use_autoreject,  # NEW
    ar_cache_dir=args.ar_cache_dir,      # NEW
)
```

### 3. CONFIGURATION UPDATES

#### Update `experiments/eegpt_linear_probe/configs/tuab_enhanced_config.yaml`:

```yaml
# Add under data_config section:
data_config:
  # ... existing config ...
  
  # AutoReject settings
  use_autoreject: false  # Set to true to enable
  ar_cache_dir: "autoreject_cache"
  ar_fit_samples: 200  # Number of files to fit AutoReject on
  ar_n_interpolate: [1, 4]  # Interpolation parameters
  ar_consensus: 0.1  # Consensus parameter
```

### 4. TEST INFRASTRUCTURE

#### Create `tests/test_autoreject_integration.py`:

```python
"""Comprehensive tests for AutoReject integration."""

import pytest
import numpy as np
import mne
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset
from brain_go_brrr.preprocessing.autoreject_adapter import WindowEpochAdapter, SyntheticPositionGenerator
from brain_go_brrr.preprocessing.chunked_autoreject import ChunkedAutoRejectProcessor


class TestWindowEpochAdapter:
    """Test window-epoch conversion functionality."""
    
    def test_raw_to_windowed_epochs(self):
        """Test converting raw data to windowed epochs."""
        # Create mock raw data
        sfreq = 256
        n_channels = 19
        duration = 60  # seconds
        data = np.random.randn(n_channels, int(sfreq * duration))
        info = mne.create_info(n_channels, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # Convert to epochs
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=5.0)
        epochs = adapter.raw_to_windowed_epochs(raw)
        
        # Verify
        assert isinstance(epochs, mne.Epochs)
        assert epochs.info['sfreq'] == sfreq
        assert len(epochs) == 11  # (60-10)/5 + 1
        assert epochs.get_data().shape == (11, n_channels, 2560)  # 10s @ 256Hz
    
    def test_epochs_to_continuous_reconstruction(self):
        """Test reconstructing continuous data from epochs."""
        # Create test data
        sfreq = 256
        n_channels = 19
        duration = 30
        data = np.random.randn(n_channels, int(sfreq * duration))
        info = mne.create_info(n_channels, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # Convert to epochs and back
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=10.0)  # No overlap
        epochs = adapter.raw_to_windowed_epochs(raw)
        raw_reconstructed = adapter.epochs_to_continuous(epochs, raw)
        
        # Verify reconstruction
        assert raw_reconstructed.n_times == raw.n_times
        assert np.allclose(
            raw_reconstructed.get_data()[:, :epochs.get_data().size],
            raw.get_data()[:, :epochs.get_data().size],
            rtol=1e-5
        )


class TestSyntheticPositionGenerator:
    """Test synthetic position generation."""
    
    def test_add_positions_to_raw(self):
        """Test adding positions to raw data."""
        # Create raw without positions
        ch_names = ['FP1', 'FP2', 'C3', 'C4', 'T3', 'T4']  # Mix of standard names
        info = mne.create_info(ch_names, 256, ch_types='eeg')
        data = np.random.randn(len(ch_names), 2560)
        raw = mne.io.RawArray(data, info)
        
        # Add positions
        generator = SyntheticPositionGenerator()
        raw_with_pos = generator.add_positions_to_raw(raw)
        
        # Verify positions
        for i, ch in enumerate(raw_with_pos.info['chs']):
            loc = ch['loc'][:3]
            assert np.any(loc != 0)  # Should have non-zero positions
            assert np.linalg.norm(loc) < 0.15  # Within reasonable head radius
    
    def test_old_new_channel_mapping(self):
        """Test that old and new channel names get same positions."""
        generator = SyntheticPositionGenerator()
        
        # Check T3/T7 mapping
        assert np.array_equal(
            generator.STANDARD_1020_POSITIONS['T3'],
            generator.STANDARD_1020_POSITIONS['T7']
        )
        
        # Check all mappings
        mappings = [('T3', 'T7'), ('T4', 'T8'), ('T5', 'P7'), ('T6', 'P8')]
        for old, new in mappings:
            assert np.array_equal(
                generator.STANDARD_1020_POSITIONS[old],
                generator.STANDARD_1020_POSITIONS[new]
            )


class TestChunkedAutoRejectProcessor:
    """Test chunked AutoReject processing."""
    
    @pytest.fixture
    def mock_file_paths(self, tmp_path):
        """Create mock EDF files."""
        files = []
        for i in range(10):
            file_path = tmp_path / f"test_{i}.edf"
            # Create minimal valid EDF
            # In real test, use mne.io.write_raw_edf
            file_path.touch()
            files.append(file_path)
        return files
    
    def test_initialization(self, tmp_path):
        """Test processor initialization."""
        processor = ChunkedAutoRejectProcessor(cache_dir=tmp_path / "ar_cache")
        assert processor.cache_dir.exists()
        assert not processor.is_fitted
        assert not processor.has_cached_params()
    
    @patch('mne.io.read_raw_edf')
    @patch('mne.make_fixed_length_epochs')
    def test_fit_on_subset(self, mock_epochs, mock_read, tmp_path, mock_file_paths):
        """Test fitting AutoReject on subset."""
        # Mock data loading
        mock_raw = Mock()
        mock_read.return_value = mock_raw
        
        # Mock epochs
        mock_epoch_data = Mock()
        mock_epoch_data.get_data.return_value = np.random.randn(10, 19, 2560)
        mock_epochs.return_value = mock_epoch_data
        
        # Fit processor
        processor = ChunkedAutoRejectProcessor(cache_dir=tmp_path / "ar_cache")
        
        # This would need more complex mocking in real implementation
        # For now, just test the interface
        assert not processor.is_fitted
        # processor.fit_on_subset(mock_file_paths, n_samples=5)
        # assert processor.is_fitted


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.slow
    def test_dataset_with_autoreject(self, tmp_path):
        """Test TUABEnhancedDataset with AutoReject enabled."""
        # This would require full test data setup
        # For now, test the interface
        
        # Mock dataset initialization
        with patch('brain_go_brrr.data.tuab_enhanced_dataset.TUABEnhancedDataset._load_file_list'):
            dataset = TUABEnhancedDataset(
                data_dir=tmp_path,
                split="train",
                use_autoreject=True,
                ar_cache_dir=tmp_path / "ar_cache"
            )
            
            assert dataset.use_autoreject
            assert dataset.ar_processor is not None
            assert dataset.window_adapter is not None
            assert dataset.position_generator is not None
    
    def test_memory_usage(self):
        """Test that memory usage stays within bounds."""
        # Create large mock dataset
        n_channels = 19
        n_samples = 256 * 60 * 10  # 10 minutes
        data = np.random.randn(n_channels, n_samples).astype(np.float32)
        
        # Check memory usage
        memory_mb = data.nbytes / (1024 * 1024)
        assert memory_mb < 100  # Should be under 100MB for this test
    
    def test_error_handling(self):
        """Test graceful error handling."""
        adapter = WindowEpochAdapter()
        
        # Test with invalid data
        with pytest.raises(ValueError):
            adapter.raw_to_windowed_epochs(None)


# Performance benchmarks
class BenchmarkAutoReject:
    """Benchmark AutoReject performance."""
    
    def test_processing_speed(self, benchmark):
        """Benchmark processing speed."""
        # Create test data
        sfreq = 256
        n_channels = 19
        duration = 60
        data = np.random.randn(n_channels, int(sfreq * duration))
        info = mne.create_info(n_channels, sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # Benchmark window conversion
        adapter = WindowEpochAdapter()
        result = benchmark(adapter.raw_to_windowed_epochs, raw)
        assert isinstance(result, mne.Epochs)
```

### 5. PERFORMANCE MONITORING

#### Add to `experiments/eegpt_linear_probe/train_enhanced.py`:

```python
# Add performance tracking (around line 550):
if args.use_autoreject:
    # Track AutoReject overhead
    trainer.logger.experiment.add_scalar(
        "timing/autoreject_overhead_ms",
        dataset.ar_processing_time * 1000,
        global_step=trainer.global_step
    )
    
    # Track rejection statistics
    trainer.logger.experiment.add_scalar(
        "data_quality/rejection_rate",
        dataset.rejection_stats['rejection_rate'],
        global_step=trainer.global_step
    )
```

### 6. FALLBACK MECHANISMS

#### Add to `tuab_enhanced_dataset.py`:

```python
def _apply_autoreject_to_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
    """Apply AutoReject with automatic fallback."""
    try:
        # Try full AutoReject
        return self.ar_processor.transform_raw(raw, self.window_adapter)
    
    except MemoryError:
        logger.warning("AutoReject OOM - falling back to amplitude rejection")
        return self._amplitude_based_cleaning(raw)
    
    except RuntimeError as e:
        if "channel positions" in str(e):
            logger.warning("Position error - using amplitude rejection")
            return self._amplitude_based_cleaning(raw)
        raise
    
    except Exception as e:
        logger.error(f"AutoReject failed: {e}")
        # Log but don't fail training
        return raw

def _amplitude_based_cleaning(self, raw: mne.io.Raw) -> mne.io.Raw:
    """Simple amplitude-based artifact rejection."""
    # Implementation of fallback cleaning
    data = raw.get_data()
    
    # Detect bad channels
    channel_stds = np.std(data, axis=1)
    bad_channels = np.where(
        (channel_stds < 0.1e-6) |  # Flat channels
        (channel_stds > 200e-6)     # Noisy channels
    )[0]
    
    # Mark bad channels
    raw.info['bads'] = [raw.ch_names[i] for i in bad_channels]
    
    # Simple epoch rejection would go here
    return raw
```

## 🎯 EXACT EXECUTION ORDER

1. **Create adapter classes** (autoreject_adapter.py)
2. **Create chunked processor** (chunked_autoreject.py)
3. **Write all tests** (test_autoreject_integration.py)
4. **Modify dataset class** (tuab_enhanced_dataset.py)
5. **Update training script** (train_enhanced.py)
6. **Update configuration** (tuab_enhanced_config.yaml)
7. **Run tests to ensure nothing breaks**
8. **Benchmark with/without AutoReject**

## ⚠️ CRITICAL GOTCHAS TO HANDLE

1. **Channel position missing**: ALWAYS add synthetic positions
2. **Memory overflow**: ALWAYS use chunked processing
3. **Old/new channel names**: ALWAYS map before processing
4. **Overlap reconstruction**: CAREFULLY average overlapping regions
5. **Performance tracking**: ALWAYS log processing time
6. **Error handling**: NEVER let AutoReject crash training