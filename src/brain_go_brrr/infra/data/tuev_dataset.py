"""TUEV dataset with MNE+Autoreject preprocessing.
Multi-class event detection (6 classes) with 20 channels (Fz included, Fpz excluded).
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from brain_go_brrr.infra.data.channels import CHANNELS_TUEV_20

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


class TUEVMNEDataset(Dataset[tuple[torch.Tensor, int]]):
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

    def _load_cache_index(self) -> None:
        """Load the cache index for this split."""
        index_file = self.cache_dir / f"index_{self.split}_{self.CACHE_VERSION}.json"

        if not index_file.exists():
            raise FileNotFoundError(f"Cache index not found at {index_file}")

        with index_file.open() as f:
            self.index = json.load(f)

        # Validate META.json cache contract
        meta_file = self.cache_dir / "META.json"
        if meta_file.exists():
            with meta_file.open() as f:
                meta = json.load(f)

            # Assert critical cache properties
            assert meta['sr'] == 256, f"Cache sample rate mismatch: {meta['sr']} != 256"
            assert meta['unit'] == 'mV', f"Cache unit mismatch: {meta['unit']} != mV"
            assert meta['window'] == 1024, f"Cache window mismatch: {meta['window']} != 1024"
            assert meta['norm'] == 'wrapper', f"Cache norm mismatch: {meta['norm']} != wrapper"

            # Validate channels - support both old and new key
            if 'channels' in meta:
                assert (
                    meta['channels'] == CHANNELS_TUEV_20
                ), "Cache channels mismatch! Expected TUEV 20 channels (with FZ, no FPZ)"
            elif 'channels20' in meta:  # Backward compat
                logger.warning("META.json uses deprecated 'channels20' key, should use 'channels'")
                assert (
                    meta['channels20'] == CHANNELS_TUEV_20
                ), "Cache channels mismatch! Expected TUEV 20 channels (with FZ, no FPZ)"

            logger.info(
                f"Cache validated: sr={meta['sr']}, unit={meta['unit']}, norm={meta['norm']}, commit={meta.get('commit', 'unknown')}"
            )
        else:
            logger.warning(f"META.json not found at {meta_file} - cache may be outdated")

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

    def _build_cache(self) -> None:
        """Build preprocessed cache with MNE+Autoreject.

        NOTE: This requires the TUEVPreprocessor module which is not currently
        implemented. Use pre-built cache instead.
        """
        raise NotImplementedError(
            "Building TUEV cache requires TUEVPreprocessor which is not implemented. "
            "Please use a pre-built cache. The cache_dir parameter should point to "
            "an existing cache directory with META.json and window files."
        )

    def _get_edf_files(self) -> list[Path]:
        """Get list of EDF files for this split."""
        return sorted(self.split_dir.glob('**/*.edf'))

    def _load_annotations(self, edf_path: Path) -> list[dict[str, Any]]:
        """Load and parse TUEV annotations from .lab files.

        Args:
            edf_path: Path to EDF file

        Returns:
            List of annotation dicts with 'start', 'end', 'label' keys
        """
        annotations: list[dict[str, Any]] = []

        # Find corresponding .lab files (one per channel)
        base_name = edf_path.stem
        lab_dir = edf_path.parent
        lab_files = sorted(lab_dir.glob(f"{base_name}_*.lab"))

        if not lab_files:
            return annotations

        # Parse annotations from .lab files
        for lab_file in lab_files:
            with lab_file.open() as f:
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
