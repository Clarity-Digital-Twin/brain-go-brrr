"""
TUEV dataset with MNE+Autoreject preprocessing.
Multi-class event detection (6 classes) with 23→20 channel mapping.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# The 6 TUEV classes
CLASS_MAPPING = {
    'spsw': 0,  # Spike and Sharp Wave (epileptiform)
    'gped': 1,  # Generalized Periodic Epileptiform Discharges
    'pled': 2,  # Periodic Lateralized Epileptiform Discharges
    'eyem': 3,  # Eye Movement (artifact)
    'artf': 4,  # Other Artifacts
    'bckg': 5,  # Background (normal)
}


class TUEVMNEDataset(Dataset):
    """TUEV dataset with MNE+Autoreject preprocessing.

    This dataset handles:
    1. 23→20 channel mapping (TUEV-specific)
    2. Multi-class labels from .lab files
    3. MNE+Autoreject preprocessing
    """

    # Cache version - bump this when preprocessing pipeline changes
    CACHE_VERSION = "mne-ar-v3"  # v3: fixed-grid windows, argmax labeling, gentle AR

    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        cache_dir: Path | None = None,
        force_rebuild: bool = False,
    ):
        """Initialize MNE-preprocessed TUEV dataset.

        Args:
            root_dir: Root directory containing TUEV EDF files
            split: 'train' or 'eval' split
            cache_dir: Directory for cached preprocessed data
            force_rebuild: Force rebuilding cache even if it exists
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / 'edf' / split

        if not self.split_dir.exists():
            raise ValueError(f"Dataset not found at {self.split_dir}")

        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.root_dir / 'cache' / 'tuev_mne_preprocessed'
        )

        # Check if we need to import preprocessor (only for building)
        self.preprocessor = None

        # Load or build cache
        if force_rebuild or not self._cache_exists():
            logger.info(f"Cache not found or rebuild forced. Building cache at {self.cache_dir}")
            self._build_cache()

        # Load cache index
        self._load_cache_index()

    def _cache_exists(self) -> bool:
        """Check if cache exists for this split."""
        index_file = self.cache_dir / f"index_{self.split}_{self.CACHE_VERSION}.json"
        return index_file.exists()

    def _load_cache_index(self):
        """Load the cache index for this split."""
        index_file = self.cache_dir / f"index_{self.split}_{self.CACHE_VERSION}.json"

        if not index_file.exists():
            raise FileNotFoundError(f"Cache index not found at {index_file}")

        with open(index_file) as f:
            self.index = json.load(f)

        # Create flat list of samples
        self.samples = []
        for window_id in range(self.index['total_windows']):
            if str(window_id) in self.index['windows']:
                self.samples.append(self.index['windows'][str(window_id)])

        logger.info(f"Loaded {len(self.samples)} windows for {self.split} split")
        logger.info(
            f"From {self.index['n_files']} files, {self.index['n_rejected']} epochs rejected"
        )

        # Log class distribution
        if 'class_counts' in self.index:
            logger.info(f"Class distribution: {self.index['class_counts']}")

    def _build_cache(self):
        """Build preprocessed cache with MNE+Autoreject."""
        # Import here to avoid dependency if just using cache
        from ..mne_integration.tuev_preprocessor import TUEVPreprocessor

        logger.info(f"Building MNE-preprocessed cache for {self.split}...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize preprocessor
        self.preprocessor = TUEVPreprocessor()

        # Get list of EDF files
        edf_files = self._get_edf_files()
        if not edf_files:
            raise ValueError(f"No EDF files found for split '{self.split}'")

        # Process files and build cache
        cache_index = {
            'split': self.split,
            'windows': {},
            'n_files': 0,
            'total_windows': 0,
            'n_rejected': 0,
            'class_counts': {k: 0 for k in CLASS_MAPPING},
            'version': self.CACHE_VERSION,
            # CRITICAL: Expected shape for TUEV
            'expected_shape': [20, 1024],  # Exactly 20 channels, 1024 samples
            'channel_info': {
                'n_channels': 20,
                'channel_order': 'Standard 10-20 with Fz, without Fpz',
                'note': 'TUEV uses 20 channels (Fz included) per EEGPT Table 13',
            },
        }

        window_id = 0

        for edf_path in tqdm(edf_files, desc=f"Building {self.split} cache"):
            # Load annotations for this file
            annotations = self._load_annotations(edf_path)

            if not annotations:
                logger.warning(f"No annotations found for {edf_path.name}")
                continue

            try:
                # Apply MNE preprocessing with fixed-grid windows
                epochs_clean, info, window_labels = self.preprocessor.process_raw_with_annotations(
                    edf_path,
                    annotations,
                    window_overlap=0.0,  # Default no overlap
                )

                # CRITICAL: Use epochs.selection to maintain correct label alignment after AR
                # epochs_clean.selection contains indices of kept epochs from original
                if hasattr(epochs_clean, 'selection'):
                    kept_indices = epochs_clean.selection
                else:
                    # Fallback if selection not available (shouldn't happen with modern MNE)
                    logger.warning("epochs.selection not available, using sequential mapping")
                    kept_indices = range(len(epochs_clean))

                # Validate selection is sorted and within bounds (critical for correct labeling)
                sel = np.asarray(kept_indices)
                assert np.all(sel[:-1] <= sel[1:]), f"Selection not sorted: {sel[:10]}"
                assert sel.max() < len(window_labels), (
                    f"Selection out of bounds: max={sel.max()}, n_labels={len(window_labels)}"
                )
                assert len(sel) == len(epochs_clean), (
                    f"Selection size mismatch: {len(sel)} != {len(epochs_clean)}"
                )

                # Save each epoch with its correctly aligned label
                for epoch_idx, original_idx in enumerate(kept_indices):
                    epoch_data = epochs_clean.get_data()[epoch_idx]
                    label = window_labels[original_idx]  # Use original index for correct label
                    
                    # CRITICAL FIX: NORMALIZE THE DATA!
                    # MNE outputs in Volts (1e-5 scale), EEGPT needs ~N(0,1)
                    epoch_mean = epoch_data.mean()
                    epoch_std = epoch_data.std()
                    epoch_data = (epoch_data - epoch_mean) / (epoch_std + 1e-8)
                    logger.info(f"Normalized: mean={epoch_mean:.2e}→0, std={epoch_std:.2e}→1")

                    # CRITICAL: Enforce exactly 20 channels for TUEV
                    expected_channels = 20  # TUEV uses 20 channels (with Fz, without Fpz)
                    expected_samples = 1024  # 4s @ 256Hz

                    if epoch_data.shape[0] != expected_channels:
                        logger.error(
                            f"CHANNEL COUNT ERROR in {edf_path.name}: "
                            f"Got {epoch_data.shape[0]} channels, expected exactly {expected_channels}. "
                            f"Shape: {epoch_data.shape}. This window will be SKIPPED."
                        )
                        # SKIP this window to prevent cache corruption
                        continue

                    if epoch_data.shape[1] != expected_samples:
                        logger.warning(
                            f"Sample count mismatch: {epoch_data.shape}, expected ({expected_channels}, {expected_samples})"
                        )
                        continue

                    # epoch_data shape: (n_channels=20, n_samples=1024)
                    cache_file = self.cache_dir / f"window_{window_id:06d}_{self.CACHE_VERSION}.pt"

                    # Get numeric label from string label
                    label_int = CLASS_MAPPING.get(label, 5)  # Default to background

                    # Save preprocessed window
                    torch.save(
                        {
                            'x': torch.from_numpy(epoch_data).float(),
                            'y': torch.tensor(label_int, dtype=torch.long),
                            'file': edf_path.name,
                        },
                        cache_file,
                    )

                    # Update index
                    cache_index['windows'][str(window_id)] = {
                        'cache_file': cache_file.name,
                        'label': label_int,
                        'label_str': label,
                        'source_file': edf_path.name,
                    }

                    # Update class counts
                    cache_index['class_counts'][label] += 1

                    window_id += 1

                cache_index['n_files'] += 1
                cache_index['n_rejected'] += info['n_rejected']

                # Add reject rate to index
                if 'reject_rates' not in cache_index:
                    cache_index['reject_rates'] = []
                cache_index['reject_rates'].append(info['reject_rate'])

                # Track missing channels per file for QC
                if 'missing_channels' in info:
                    if 'file_missing_channels' not in cache_index:
                        cache_index['file_missing_channels'] = {}
                    cache_index['file_missing_channels'][edf_path.name] = info['missing_channels']

                # Track AR learned parameters for auditing
                if info.get('ar_learned_params'):
                    if 'ar_learned_params_summary' not in cache_index:
                        cache_index['ar_learned_params_summary'] = []
                    cache_index['ar_learned_params_summary'].append(
                        {'file': edf_path.name, 'params': info['ar_learned_params']}
                    )

                # Store first file's metadata as reference
                if cache_index['n_files'] == 1:
                    cache_index['sfreq_after'] = info.get('sfreq_after', 256)
                    cache_index['window_overlap'] = info.get('window_overlap', 0.0)

            except Exception as e:
                logger.error(f"Failed to process {edf_path.name}: {e}")
                continue

        cache_index['total_windows'] = window_id

        # Save index
        index_file = self.cache_dir / f"index_{self.split}_{self.CACHE_VERSION}.json"
        with open(index_file, 'w') as f:
            json.dump(cache_index, f, indent=2)

        logger.info(
            f"Cache build complete: {cache_index['total_windows']} windows from {cache_index['n_files']} files"
        )
        logger.info(f"Class distribution: {cache_index['class_counts']}")

    def _get_edf_files(self) -> list[Path]:
        """Get list of EDF files for this split."""
        return sorted(self.split_dir.glob('**/*.edf'))

    def _load_annotations(self, edf_path: Path) -> list[dict]:
        """Load and parse TUEV annotations from .lab files.

        Args:
            edf_path: Path to EDF file

        Returns:
            List of annotation dicts with 'start', 'end', 'label' keys
        """
        annotations = []

        # Find corresponding .lab files (one per channel)
        base_name = edf_path.stem
        lab_dir = edf_path.parent
        lab_files = sorted(lab_dir.glob(f"{base_name}_*.lab"))

        if not lab_files:
            return annotations

        # Parse annotations from .lab files
        for lab_file in lab_files:
            with open(lab_file) as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            try:
                                start_us = float(parts[0])
                                end_us = float(parts[1])
                                label = parts[2].lower()

                                # Convert microseconds to seconds
                                start_sec = start_us / 1e6
                                end_sec = end_us / 1e6

                                # Only use known labels
                                if label in CLASS_MAPPING:
                                    # Create 4-second windows from annotation
                                    window_duration = 4.0  # seconds
                                    current_start = start_sec

                                    while current_start < end_sec:
                                        window_end = min(current_start + window_duration, end_sec)

                                        annotations.append(
                                            {
                                                'start': current_start,
                                                'end': window_end,
                                                'label': label,
                                            }
                                        )

                                        current_start += window_duration

                            except (ValueError, IndexError):
                                continue

        return annotations

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get preprocessed window and label.

        Returns:
            Tuple of (preprocessed EEG tensor, label)
            EEG tensor shape: (20, 1024) - 20 channels, 4 seconds @ 256Hz
        """
        sample = self.samples[idx]
        cache_file = self.cache_dir / sample['cache_file']

        # Load cached preprocessed data
        data = torch.load(cache_file, map_location='cpu')

        return data['x'], data['y']
