"""TUEV Dataset Implementation - Following Table 13 with EEGPT Compatibility.

Based on TUEV_UNIFIED_SPECS.md:
- Input: 23 × 1024 samples (4.0 seconds @ 256Hz)
- Classes: 6 (SPSW, GPED, PLED, EYEM, ARTF, BCKG)
- Channel reduction: 23 → 20 standard channels
- Actual data: 250Hz, needs resampling to 256Hz
- EEGPT requirement: window size must be divisible by 64 (patch size)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# CRITICAL: Must be divisible by 64 for EEGPT patch embedding
# Paper Table 13 shows ~1000 samples but EEGPT requires multiples of 64
# [Decision] Use 1024 samples (4.0s) for EEGPT compatibility
WINDOW_SAMPLES = 1024  # 16 patches of 64 samples each
TARGET_SAMPLING_RATE = 256  # Hz
WINDOW_SECONDS = WINDOW_SAMPLES / TARGET_SAMPLING_RATE  # 4.0s

# The 6 TUEV classes
CLASS_MAPPING = {
    'spsw': 0,  # Spike and Sharp Wave (epileptiform)
    'gped': 1,  # Generalized Periodic Epileptiform Discharges
    'pled': 2,  # Periodic Lateralized Epileptiform Discharges
    'eyem': 3,  # Eye Movement (artifact)
    'artf': 4,  # Other Artifacts
    'bckg': 5   # Background (normal)
}

# The 20 target channels (from paper line 615)
TARGET_CHANNELS = [
    'FP1', 'FPZ', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2'
]

# TCP montage channels that we need to select from
TCP_CHANNELS = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FZ-CZ', 'CZ-PZ',
    'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4',
    'A1-T3', 'T4-A2'
]


class TUEVDataset(Dataset):
    """TUEV Dataset with exact paper specifications from Table 13."""

    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        cache_dir: Optional[Path] = None,
        resample: bool = True,
        normalize: bool = True
    ):
        """Initialize TUEV dataset.

        Args:
            root_dir: Path to TUEV v2.0.1 directory
            split: 'train' or 'eval'
            cache_dir: Optional directory for cached windows
            resample: Whether to resample 250Hz → 256Hz
            normalize: Whether to z-score normalize
        """
        self.root_dir = Path(root_dir)
        self.split = split
        # TUEV has edf/train and edf/eval structure
        self.split_dir = self.root_dir / 'edf' / split
        self.resample = resample
        self.normalize = normalize

        # Verify dataset exists
        if not self.split_dir.exists():
            raise ValueError(f"Dataset not found at {self.split_dir}")

        # Load file list and annotations
        self.samples = self._load_annotations()
        logger.info(f"Loaded {len(self.samples)} windows for {split} split")

        # Count classes
        class_counts = {}
        for sample in self.samples:
            label = sample['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        logger.info(f"Class distribution: {class_counts}")

        # Setup cache if provided
        self.cache_dir = None
        if cache_dir:
            self.cache_dir = Path(cache_dir) / f"tuev_{split}_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_annotations(self) -> List[Dict]:
        """Load and parse TUEV annotations from .lab files."""
        samples = []

        # Get all EDF files
        edf_files = sorted(self.split_dir.glob("**/*.edf"))
        logger.info(f"Found {len(edf_files)} EDF files in {self.split}")

        for edf_path in tqdm(edf_files, desc=f"Loading {self.split} annotations"):
            # Find corresponding .lab files (one per channel)
            base_name = edf_path.stem
            lab_pattern = edf_path.parent / f"{base_name}*.lab"
            lab_files = sorted(edf_path.parent.glob(f"{base_name}*.lab"))

            if not lab_files:
                logger.warning(f"No .lab files found for {edf_path}")
                continue

            # Parse annotations from .lab files
            annotations = self._parse_lab_files(lab_files)

            # Create windows from annotations
            for ann in annotations:
                samples.append({
                    'edf_path': str(edf_path),
                    'start_sec': ann['start'],
                    'end_sec': ann['end'],
                    'label': ann['label'],
                    'channels': ann.get('channels', [])
                })

        return samples

    def _parse_lab_files(self, lab_files: List[Path]) -> List[Dict]:
        """Parse .lab annotation files.

        Lab files contain:
        - Start and end times in microseconds
        - Event labels (spsw, gped, pled, eyem, artf, bckg)
        """
        annotations = []

        for lab_file in lab_files:
            with open(lab_file, 'r') as f:
                lines = f.readlines()

            # Skip header lines
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    # Format: start_us end_us label [confidence]
                    start_us = float(parts[0])
                    end_us = float(parts[1])
                    label = parts[2].lower()

                    # Convert microseconds to seconds
                    start_sec = start_us / 1e6
                    end_sec = end_us / 1e6

                    # Only use known labels
                    if label in CLASS_MAPPING:
                        # Create window-sized segments
                        duration = end_sec - start_sec

                        # If annotation is longer than window, split it
                        while duration > 0:
                            window_end = min(start_sec + WINDOW_SECONDS, end_sec)

                            annotations.append({
                                'start': start_sec,
                                'end': window_end,
                                'label': label,
                                'channel': lab_file.stem.split('_')[-1]  # Channel name
                            })

                            start_sec = window_end
                            duration = end_sec - start_sec

        return annotations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get preprocessed window following Table 13 specs."""
        sample = self.samples[idx]

        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"sample_{idx:06d}.pt"
            if cache_file.exists():
                data = torch.load(cache_file, weights_only=True)
                return data['x'], data['y']

        # Load EDF segment
        raw = mne.io.read_raw_edf(sample['edf_path'], preload=False, verbose=False)

        # Extract window
        start = sample['start_sec']
        end = min(start + WINDOW_SECONDS, sample['end_sec'])

        # Crop to window
        raw.crop(tmin=start, tmax=end)
        raw.load_data()

        # Get data
        data = raw.get_data()
        original_sfreq = raw.info['sfreq']

        # Resample if needed (250Hz → 256Hz)
        if self.resample and original_sfreq != TARGET_SAMPLING_RATE:
            n_samples_orig = data.shape[1]
            n_samples_target = int(n_samples_orig * TARGET_SAMPLING_RATE / original_sfreq)
            data = signal.resample(data, n_samples_target, axis=1)

        # Select 23 channels (or pad if fewer)
        if data.shape[0] < 23:
            # Pad with zeros if fewer channels
            padding = np.zeros((23 - data.shape[0], data.shape[1]))
            data = np.vstack([data, padding])
        elif data.shape[0] > 23:
            # Select first 23 channels
            data = data[:23]

        # Ensure exactly 1000 samples (crop or pad)
        if data.shape[1] > WINDOW_SAMPLES:
            data = data[:, :WINDOW_SAMPLES]
        elif data.shape[1] < WINDOW_SAMPLES:
            padding = np.zeros((data.shape[0], WINDOW_SAMPLES - data.shape[1]))
            data = np.hstack([data, padding])

        # Normalize
        if self.normalize:
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            data = (data - mean) / (std + 1e-6)

        # Convert to tensor
        x = torch.tensor(data, dtype=torch.float32)
        y = CLASS_MAPPING[sample['label']]

        # Cache if directory provided
        if self.cache_dir:
            cache_file = self.cache_dir / f"sample_{idx:06d}.pt"
            torch.save({'x': x, 'y': y}, cache_file)

        return x, y

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced loss."""
        counts = torch.zeros(6)
        for sample in self.samples:
            label = CLASS_MAPPING[sample['label']]
            counts[label] += 1

        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(counts)

        return weights


class TUEVCachedDataset(Dataset):
    """Memory-mapped TUEV dataset for efficient loading."""

    def __init__(self, cache_dir: Path, split: str = 'train'):
        """Load pre-cached TUEV dataset.

        Args:
            cache_dir: Directory with cached .pt files
            split: 'train' or 'eval'
        """
        self.cache_dir = Path(cache_dir) / f"tuev_{split}_cache"

        # Load index
        index_file = self.cache_dir / "index.json"
        if not index_file.exists():
            raise ValueError(f"Cache index not found at {index_file}")

        with open(index_file, 'r') as f:
            self.index = json.load(f)

        self.samples = self.index['samples']
        self.n_classes = 6

        logger.info(f"Loaded cached dataset with {len(self.samples)} samples")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load cached sample from disk."""
        sample_info = self.samples[idx]
        cache_file = self.cache_dir / sample_info['cache_file']

        # Load without keeping in memory
        data = torch.load(cache_file, map_location='cpu', weights_only=True)

        return data['x'], data['y']

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def get_class_weights(self) -> torch.Tensor:
        """Get class weights from index."""
        if 'class_weights' in self.index:
            return torch.tensor(self.index['class_weights'])

        # Compute from samples
        counts = torch.zeros(6)
        for sample in self.samples:
            counts[sample['label']] += 1

        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(counts)

        return weights


def verify_dataset():
    """Verify TUEV dataset matches paper specifications."""
    root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuh_eeg/TUEV/v2.0.1")

    # Create dataset
    dataset = TUEVDataset(root, split='train', resample=True)

    # Get first sample
    x, y = dataset[0]

    # Verify shape matches Table 13
    assert x.shape == (23, 1000), f"Wrong shape: {x.shape}, expected (23, 1000)"
    assert y in range(6), f"Wrong label: {y}, expected 0-5"

    print(f"✓ Dataset shape correct: {x.shape}")
    print(f"✓ Label range correct: {y}")
    print(f"✓ Total samples: {len(dataset)}")

    # Check class distribution
    weights = dataset.get_class_weights()
    print(f"✓ Class weights: {weights}")

    return True


if __name__ == "__main__":
    verify_dataset()
