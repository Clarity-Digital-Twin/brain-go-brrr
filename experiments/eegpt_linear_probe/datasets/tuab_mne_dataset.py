"""
TUAB dataset with MNE+Autoreject preprocessing.
This dataset applies the full preprocessing pipeline during cache building.
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TUABMNEDataset(Dataset):
    """TUAB dataset with MNE+Autoreject preprocessing.

    This dataset can work in two modes:
    1. Cache mode: Load pre-preprocessed data from cache
    2. Build mode: Build cache by preprocessing raw EDF files
    """

    # Cache version - bump this when preprocessing pipeline changes
    CACHE_VERSION = "mne-ar-v2"  # v2: fixed epoch boundaries, configurable notch
    # Note: v2 cache already has 19 channels consistently, channel enforcement added for future builds

    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        cache_dir: Path | None = None,
        force_rebuild: bool = False,
    ):
        """Initialize MNE-preprocessed TUAB dataset.

        Args:
            root_dir: Root directory containing TUAB EDF files
            split: 'train' or 'eval' split
            cache_dir: Directory for cached preprocessed data
            force_rebuild: Force rebuilding cache even if it exists
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.root_dir / 'cache' / 'mne_preprocessed'
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

    def _build_cache(self):
        """Build preprocessed cache with MNE+Autoreject."""
        # Import here to avoid dependency if just using cache
        from ..mne_integration.preprocessor import TUABPreprocessor

        logger.info(f"Building MNE-preprocessed cache for {self.split}...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize preprocessor
        self.preprocessor = TUABPreprocessor()

        # Get list of EDF files
        edf_files = self._get_edf_files()
        if not edf_files:
            raise ValueError(f"No EDF files found for split '{self.split}'")

        # Process files and build cache
        cache_index = {
            'split': self.split,
            'windows': {},
            'total_windows': 0,
            'n_files': len(edf_files),
            'n_rejected': 0,
            'preprocessing_params': {
                'sampling_rate': 256,
                'window_duration': 4.0,
                'bandpass': [0.5, 45],
                'notch': 60,
                'ar_n_interpolate': [1, 2, 3, 4],
                'ar_consensus': [0.3, 0.5, 0.7],
                'ar_cv': 5,
            },
            # CRITICAL: Expected shape for validation
            'expected_shape': [19, 1024],  # Exactly 19 channels, 1024 samples
            'cache_version': self.CACHE_VERSION,
            'channel_info': {
                'n_channels': 19,
                'channel_order': 'Standard 10-20 without Fz',
                'note': 'TUAB uses 19 channels (Fz excluded) to match EEGPT requirements',
            },
        }

        window_idx = 0
        total_epochs_before = 0
        total_epochs_after = 0

        for edf_path, label in edf_files:
            try:
                logger.info(f"Processing {edf_path.name}")

                # Apply full preprocessing
                epochs_clean, preprocessing_info = self.preprocessor.process_raw(edf_path)

                # Track rejection statistics
                total_epochs_before += preprocessing_info['n_epochs_before']
                total_epochs_after += preprocessing_info['n_epochs_after']

                # Save each epoch as separate cache file
                for epoch_idx, epoch_data in enumerate(epochs_clean.get_data()):
                    # Convert to float32 for training
                    epoch_data = epoch_data.astype('float32')

                    # CRITICAL FIX: NORMALIZE THE DATA!
                    # MNE outputs in Volts (1e-5 scale), EEGPT needs ~N(0,1)
                    epoch_mean = epoch_data.mean()
                    epoch_std = epoch_data.std()
                    epoch_data = (epoch_data - epoch_mean) / (epoch_std + 1e-8)
                    logger.info(f"Normalized: mean={epoch_mean:.2e}→0, std={epoch_std:.2e}→1")

                    # CRITICAL: Enforce exactly 19 channels for TUAB
                    # This prevents the 20-channel bug that occurred with aaaaakfo_s00X files
                    expected_channels = 19  # TUAB uses 19 channels (no Fz)
                    expected_samples = int(4.0 * 256)  # 4s @ 256Hz = 1024 samples

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

                    # Save cache file with channel info for validation
                    cache_file = f"window_{window_idx:06d}_{self.CACHE_VERSION}.pt"
                    cache_path = self.cache_dir / cache_file

                    torch.save(
                        {
                            'x': torch.from_numpy(epoch_data),
                            'y': torch.tensor(label, dtype=torch.float32),
                            'source_file': str(edf_path),
                            'epoch_idx': epoch_idx,
                            'n_channels': epoch_data.shape[0],  # Store for validation
                            'n_samples': epoch_data.shape[1],    # Store for validation
                        },
                        cache_path,
                    )

                    # Update index with shape info
                    cache_index['windows'][str(window_idx)] = {
                        'cache_file': cache_file,
                        'label': label,
                        'source': str(edf_path.name),
                        'shape': list(epoch_data.shape),  # Add shape to index for validation
                    }

                    window_idx += 1

            except Exception as e:
                logger.error(f"Error processing {edf_path}: {e}")
                continue

        # Update final statistics
        cache_index['total_windows'] = window_idx
        cache_index['n_rejected'] = total_epochs_before - total_epochs_after

        # Save index
        index_file = self.cache_dir / f"index_{self.split}_{self.CACHE_VERSION}.json"
        with open(index_file, 'w') as f:
            json.dump(cache_index, f, indent=2)

        logger.info(f"Cache built: {window_idx} windows from {len(edf_files)} files")
        logger.info(f"Rejection rate: {cache_index['n_rejected']} epochs")

    def _get_edf_files(self) -> list[tuple[Path, int]]:
        """Get list of EDF files for this split.

        Returns:
            List of (file_path, label) tuples
        """
        edf_files = []

        # TUAB has structure: root/train/normal|abnormal/01_tcp_ar/*.edf
        # or root/eval/normal|abnormal/01_tcp_ar/*.edf
        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            # Fallback: maybe we're already at the split level
            normal_dir = self.root_dir / 'normal'
            abnormal_dir = self.root_dir / 'abnormal'
        else:
            normal_dir = split_dir / 'normal'
            abnormal_dir = split_dir / 'abnormal'

        if normal_dir.exists():
            # Handle nested structure: normal/01_tcp_ar/*.edf
            normal_files = sorted(normal_dir.glob('**/*.edf'))
            edf_files.extend([(f, 0) for f in normal_files])
            logger.info(f"Found {len(normal_files)} normal EDF files")

        if abnormal_dir.exists():
            # Handle nested structure: abnormal/01_tcp_ar/*.edf
            abnormal_files = sorted(abnormal_dir.glob('**/*.edf'))
            edf_files.extend([(f, 1) for f in abnormal_files])
            logger.info(f"Found {len(abnormal_files)} abnormal EDF files")

        logger.info(f"Found {len(edf_files)} total EDF files for {self.split} split")
        return edf_files

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get preprocessed sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (eeg_data, label)
        """
        sample_info = self.samples[idx]
        cache_path = self.cache_dir / sample_info['cache_file']

        # Load cached data
        data = torch.load(cache_path, map_location='cpu')

        return data['x'], data['y']
