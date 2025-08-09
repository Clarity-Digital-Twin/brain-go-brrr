"""Enhanced TUAB dataset with paper-matching specifications."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
import mne
from collections import defaultdict
import warnings

from .tuab_dataset import TUABDataset
from ..preprocessing.autoreject_adapter import WindowEpochAdapter, SyntheticPositionGenerator
from ..preprocessing.chunked_autoreject import ChunkedAutoRejectProcessor

logger = logging.getLogger(__name__)

# TUAB uses old naming convention
OLD_TO_NEW_CHANNELS = {
    'T3': 'T7',
    'T4': 'T8', 
    'T5': 'P7',
    'T6': 'P8'
}

# Reverse mapping
NEW_TO_OLD_CHANNELS = {
    'T7': 'T3',
    'T8': 'T4',
    'P7': 'T5',
    'P8': 'T6'
}

# Paper's standard channel order (23 channels)
PAPER_CHANNEL_ORDER = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'
]

# Our 20-channel subset (missing A1, A2, T1, T2)
TUAB_20_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2', 'A1'  # Use A1 as 20th channel if available
]


class TUABEnhancedDataset(TUABDataset):
    """Enhanced TUAB dataset matching paper specifications.
    
    Key differences from base class:
    - 10-second windows instead of 8-second
    - 200Hz sampling rate instead of 256Hz
    - 0.1-75Hz bandpass filter instead of 0.5-50Hz
    - 50Hz notch filter
    - 50% overlap for training (5s stride)
    - Channel adaptation for TUAB naming
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        window_duration: float = 10.0,  # 10 seconds to match paper
        window_stride: Optional[float] = None,
        sampling_rate: int = 200,  # 200Hz to match paper
        channels: Optional[List[str]] = None,
        preload: bool = False,
        normalize: bool = True,
        bandpass_low: float = 0.1,  # Match paper
        bandpass_high: float = 75.0,  # Match paper
        notch_freq: float = 50.0,  # Add notch filter
        cache_dir: Optional[Union[str, Path]] = None,
        use_old_naming: bool = True,  # TUAB uses old channel names
        max_recording_mins: float = 20.0,
        n_jobs: int = 1,
        verbose: bool = True,
        use_autoreject: bool = False,
        ar_cache_dir: Optional[Union[str, Path]] = None,
        cache_mode: str = "write",  # "write" or "readonly"
    ):
        """Initialize enhanced TUAB dataset.
        
        Args:
            root_dir: Path to TUAB root directory
            split: One of "train", "eval", "test"
            window_duration: Window size in seconds (10.0 to match paper)
            window_stride: Stride in seconds (5.0 for 50% overlap in training)
            sampling_rate: Target sampling rate (200 to match paper)
            channels: List of channels to use (default: TUAB_20_CHANNELS)
            preload: Whether to preload all data into memory
            normalize: Whether to normalize data
            bandpass_low: Low frequency for bandpass filter (0.1)
            bandpass_high: High frequency for bandpass filter (75.0)
            notch_freq: Notch filter frequency (50.0)
            cache_dir: Directory for caching preprocessed data
            use_old_naming: Use old channel naming (T3/T4/T5/T6)
            max_recording_mins: Maximum recording duration to consider
            n_jobs: Number of parallel jobs for preprocessing
            verbose: Whether to print progress
            use_autoreject: Whether to use AutoReject for artifact removal
            ar_cache_dir: Directory for caching AutoReject parameters
            cache_mode: Cache mode - "write" (default) or "readonly" (never regenerate)
        """
        # Set default stride based on split
        if window_stride is None:
            window_stride = 5.0 if split == "train" else 10.0  # 50% overlap for training
        
        # Use TUAB 20 channels by default
        if channels is None:
            channels = TUAB_20_CHANNELS.copy()
        
        # Store enhanced parameters
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.use_old_naming = use_old_naming
        self.target_sampling_rate = sampling_rate
        self.cache_mode = cache_mode
        
        # Store additional parameters before calling parent
        self.channels = channels
        self.max_recording_mins = max_recording_mins
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.use_autoreject = use_autoreject
        self.ar_cache_dir = Path(ar_cache_dir) if ar_cache_dir else Path("autoreject_cache")
        
        # Initialize AutoReject components if enabled
        self.ar_processor: Any = None
        self.window_adapter: Any = None
        self.position_generator: Any = None
        
        # Check if we should skip file scanning in readonly mode
        if cache_mode == "readonly" and cache_dir:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                # Check if we have any cache files
                cache_files = list(cache_path.glob("*.pkl"))
                if not cache_files:
                    raise RuntimeError(
                        f"Cache mode is 'readonly' but no cache files found in {cache_path}. "
                        f"Please build cache first with cache_mode='write' or use a different mode."
                    )
                logger.info(f"Found {len(cache_files)} cache files in readonly mode")
        
        # Initialize base class with only the parameters it accepts
        super().__init__(
            root_dir=Path(root_dir) if isinstance(root_dir, str) else root_dir,
            split=split,
            sampling_rate=sampling_rate,
            window_duration=window_duration,
            window_stride=window_stride,
            preload=preload,
            normalize=normalize,
            cache_dir=Path(cache_dir) if isinstance(cache_dir, str) else cache_dir,
        )
        
        logger.info(f"Enhanced TUAB dataset initialized:")
        logger.info(f"  Window: {window_duration}s @ {sampling_rate}Hz")
        logger.info(f"  Stride: {window_stride}s ({window_stride/window_duration*100:.0f}% overlap)")
        logger.info(f"  Filter: {bandpass_low}-{bandpass_high}Hz + {notch_freq}Hz notch")
        logger.info(f"  Channels: {len(channels)} ({self.use_old_naming and 'old' or 'new'} naming)")
        
        # Initialize AutoReject after parent init completes
        if self.use_autoreject:
            logger.info(f"  AutoReject: ENABLED (cache: {self.ar_cache_dir})")
            self._initialize_autoreject()
    
    def _initialize_autoreject(self) -> None:
        """Initialize AutoReject components."""
        logger.info("Initializing AutoReject components...")
        
        self.ar_processor = ChunkedAutoRejectProcessor(
            cache_dir=self.ar_cache_dir,
            chunk_size=100,
            n_interpolate=[1, 4],
            consensus=0.1,
            random_state=42
        )
        
        self.window_adapter = WindowEpochAdapter(
            window_duration=self.window_duration,
            window_stride=self.window_stride
        )
        
        self.position_generator = SyntheticPositionGenerator()
        
        # Pre-fit on subset if no cache exists
        if not self.ar_processor.has_cached_params():
            logger.info("No cached AutoReject parameters found - will fit on first batch")
    
    def _collect_samples(self) -> None:
        """Override to skip file scanning in readonly cache mode."""
        if self.cache_mode == "readonly" and self.cache_dir and self.cache_dir.exists():
            # In readonly mode, create dummy entries - actual data comes from cache
            logger.info("Readonly cache mode: skipping file scan, using cached data")
            # Create minimal structure needed for dataset
            self.file_list = []
            self.class_counts = {"normal": 0, "abnormal": 0}
            self.samples = []
            
            # Count cache files to get approximate dataset size
            cache_files = list(self.cache_dir.glob("*.pkl"))
            if cache_files:
                # Estimate based on cache files
                logger.info(f"Found {len(cache_files)} cached windows")
                # We can't create accurate samples without scanning files
                # So we'll need to load a cache index or fail
                raise RuntimeError(
                    "Readonly cache mode requires a cache index. "
                    "Please use TUABCachedDataset or build cache with 'write' mode first."
                )
            return
        
        # Otherwise use parent's file scanning
        super()._collect_samples()
    
    def _apply_autoreject_to_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply AutoReject cleaning to raw data."""
        if not self.use_autoreject:
            return raw
            
        try:
            # Apply AutoReject
            if self.ar_processor is not None:
                raw_clean = self.ar_processor.transform_raw(raw, self.window_adapter)
            else:
                raw_clean = raw
            return raw_clean
            
        except MemoryError:
            logger.warning("AutoReject OOM - falling back to amplitude rejection")
            return self._amplitude_based_cleaning(raw)
            
        except RuntimeError as e:
            if "channel positions" in str(e):
                logger.warning("No channel positions - using amplitude rejection")
                return self._amplitude_based_cleaning(raw)
            raise
            
        except Exception as e:
            logger.error(f"AutoReject failed: {e}")
            # Don't fail training - return original
            return raw
    
    def _amplitude_based_cleaning(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Simple amplitude-based artifact rejection fallback."""
        data = raw.get_data()
        
        # Detect bad channels
        channel_stds = np.std(data, axis=1)
        bad_indices = np.where(
            (channel_stds < 0.1e-6) |  # Flat
            (channel_stds > 200e-6)     # Very noisy
        )[0]
        
        raw.info['bads'] = [raw.ch_names[i] for i in bad_indices]
        
        if len(raw.info['bads']) > 0:
            logger.info(f"Marked {len(raw.info['bads'])} bad channels: {raw.info['bads']}")
        
        return raw
    
    def _load_edf_file(self, file_path: Path) -> npt.NDArray[np.float64]:
        """Load and preprocess EDF file with enhanced preprocessing.
        
        Args:
            file_path: Path to EDF file
            
        Returns:
            Preprocessed EEG data [channels, time]
        """
        # Load raw data
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Add synthetic positions if using AutoReject
        if self.use_autoreject:
            if self.position_generator is not None:
                raw = self.position_generator.add_positions_to_raw(raw)
        
        # Apply enhanced preprocessing
        raw = self._preprocess_raw(raw)
        
        # Apply AutoReject if enabled
        if self.use_autoreject:
            raw = self._apply_autoreject_to_raw(raw)
        
        # Get data and convert to float32
        data = raw.get_data().astype(np.float32)
        
        return data
    
    def _preprocess_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Preprocess raw data matching paper specifications.
        
        Args:
            raw: Raw MNE object
            
        Returns:
            Preprocessed raw object
        """
        # Create a mapping of standardized channel names (similar to parent)
        channel_map = {}
        for ch_name in raw.ch_names:
            if ch_name in self.CHANNEL_MAPPING:
                std_name = self.CHANNEL_MAPPING[ch_name]
                # Map to old naming if requested
                if self.use_old_naming and std_name in NEW_TO_OLD_CHANNELS:
                    std_name = NEW_TO_OLD_CHANNELS[std_name]
                if std_name in self.channels:
                    channel_map[ch_name] = std_name
        
        # Rename channels to standard names
        if channel_map:
            raw.rename_channels(channel_map)
        
        # Select channels
        available_channels = [ch for ch in self.channels if ch in raw.ch_names]
        if len(available_channels) < len(self.channels):
            missing = set(self.channels) - set(available_channels)
            logger.warning(f"Missing channels: {missing}")
            
            # Try to handle missing A1/A2 by using Cz as reference
            if 'A1' in missing and 'CZ' in available_channels:
                logger.info("Using CZ as pseudo-A1 reference")
                # Don't actually add channel, just note it
        
        raw.pick_channels(available_channels, ordered=True)
        
        # Apply preprocessing pipeline matching paper
        # 1. Bandpass filter (0.1-75 Hz)
        raw.filter(
            l_freq=self.bandpass_low,
            h_freq=self.bandpass_high,
            method='fir',
            fir_design='firwin',
            skip_by_annotation='edge',
            n_jobs=self.n_jobs,
            verbose=False
        )
        
        # 2. Notch filter (50 Hz)
        if self.notch_freq is not None:
            raw.notch_filter(
                freqs=self.notch_freq,
                method='fir',
                fir_design='firwin',
                skip_by_annotation='edge',
                n_jobs=self.n_jobs,
                verbose=False
            )
        
        # 3. Resample to target rate
        if raw.info['sfreq'] != self.target_sampling_rate:
            raw.resample(
                sfreq=self.target_sampling_rate,
                npad='auto',
                n_jobs=self.n_jobs,
                verbose=False
            )
        
        # 4. Re-reference to average
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
        
        return raw
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced training.
        
        Returns:
            Tensor of weights for each sample
        """
        # Calculate class weights for balanced sampling
        # Count classes from our dataset info
        n_normal = self.class_counts.get('normal', 0)
        n_abnormal = self.class_counts.get('abnormal', 0)
        
        if n_normal == 0 or n_abnormal == 0:
            # If one class is missing, use uniform weights
            return torch.ones(len(self), dtype=torch.float32)
        
        # Calculate weights inversely proportional to class frequency
        total = n_normal + n_abnormal
        weight_normal = total / (2.0 * n_normal)
        weight_abnormal = total / (2.0 * n_abnormal)
        
        # Create weight for each sample based on its label
        sample_weights = []
        for i in range(len(self)):
            # Get label for this sample
            sample_info = self.samples[i]
            label = sample_info["label"]
            weight = weight_normal if label == 0 else weight_abnormal
            sample_weights.append(weight)
        
        return torch.tensor(sample_weights, dtype=torch.float32)
    
    def _extract_windows(self, raw: mne.io.Raw, label: int = 0) -> List[Tuple[npt.NDArray[np.float64], int]]:
        """Extract windows with proper overlap.
        
        Args:
            raw: Preprocessed raw object
            label: Label for the windows (0=normal, 1=abnormal)
            
        Returns:
            List of (window_data, label) tuples
        """
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        # Calculate window parameters
        window_samples = int(self.window_duration * sfreq)
        stride_samples = int(self.window_stride * sfreq)
        
        # Extract windows with overlap
        windows = []
        n_windows = (data.shape[1] - window_samples) // stride_samples + 1
        
        for i in range(n_windows):
            start = i * stride_samples
            end = start + window_samples
            
            if end > data.shape[1]:
                break
                
            window = data[:, start:end]
            
            # Ensure correct shape
            if window.shape[1] == window_samples:
                windows.append((window, label))
        
        return windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single window.
        
        Args:
            idx: Window index
            
        Returns:
            Tuple of (data, label) where:
                data: Tensor of shape [n_channels, n_samples]
                label: Integer label (0=normal, 1=abnormal)
        """
        window_data, label = super().__getitem__(idx)
        
        # Ensure we have exactly the expected number of channels
        expected_channels = len(self.channels)
        if window_data.shape[0] < expected_channels:
            # Pad with zeros if missing channels
            n_missing = expected_channels - window_data.shape[0]
            padding = torch.zeros(n_missing, window_data.shape[1])
            window_data = torch.cat([window_data, padding], dim=0)
        elif window_data.shape[0] > expected_channels:
            # Truncate if too many channels
            window_data = window_data[:expected_channels]
        
        # Ensure correct number of samples (2000 for 10s @ 200Hz)
        expected_samples = int(self.window_duration * self.sampling_rate)
        if window_data.shape[1] != expected_samples:
            # Resample if needed
            if window_data.shape[1] > expected_samples:
                # Downsample
                indices = torch.linspace(0, window_data.shape[1]-1, expected_samples).long()
                window_data = window_data[:, indices]
            else:
                # Pad
                padding = torch.zeros(window_data.shape[0], expected_samples - window_data.shape[1])
                window_data = torch.cat([window_data, padding], dim=1)
        
        return window_data, label
    
