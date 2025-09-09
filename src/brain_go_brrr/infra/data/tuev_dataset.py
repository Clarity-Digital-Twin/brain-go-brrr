"""TUEV dataset with MNE+Autoreject preprocessing.

Multi-class event detection (6 classes).

Modes:
- Paper parity (recommended): Keep 23 raw channels (incl. A1/A2/T1/T2), cached as Volts; use a learnable 23→20 mapper before EEGPT.
- Legacy: Preprocess/mask to a canonical 20-channel interface (Fz & Fpz included, Oz excluded). Not paper-parity.
"""

import json
import logging
import subprocess
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
    CACHE_VERSION = "mne-ar-v4"  # v4: RANSAC disabled by default, EEG types set pre-montage

    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        cache_dir: Path | None = None,
        force_rebuild: bool = False,
        use_paper_parity: bool = False,
    ):
        """Initialize MNE-preprocessed TUEV dataset.

        Args:
            root_dir: Root directory containing TUEV EDF files
            split: 'train' or 'eval' split
            cache_dir: Directory for cached preprocessed data
            force_rebuild: Force rebuilding cache even if it exists
            use_paper_parity: If True, use 23 channels for paper parity.
                            If False, use existing 20-channel approach.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / 'edf' / split
        self.use_paper_parity = use_paper_parity
        self.n_channels = 23 if use_paper_parity else 20

        if not self.split_dir.exists():
            raise ValueError(f"Dataset not found at {self.split_dir}")

        # Modify cache directory based on mode
        base_cache = Path(cache_dir) if cache_dir else self.root_dir / 'cache'

        if use_paper_parity:
            self.cache_dir = base_cache / 'tuev_23ch_paper_parity' / split
        else:
            self.cache_dir = base_cache / 'tuev_mne_preprocessed'

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
            assert meta['unit'] == 'V', f"Cache unit mismatch: {meta['unit']} != V"
            assert meta['window'] == 1024, f"Cache window mismatch: {meta['window']} != 1024"
            assert meta['norm'] == 'wrapper', f"Cache norm mismatch: {meta['norm']} != wrapper"

            # Validate channels - support both old and new key
            from brain_go_brrr.infra.preprocessing.tuev_preprocessor import (
                CHANNELS_TUEV_23_CANONICAL,
            )

            expected_channels = (
                CHANNELS_TUEV_23_CANONICAL if self.use_paper_parity else CHANNELS_TUEV_20
            )
            if 'channels' in meta:
                assert meta['channels'] == expected_channels, (
                    f"Cache channels mismatch! Expected {self.n_channels} channels"
                )
            elif 'channels20' in meta and not self.use_paper_parity:  # Backward compat
                logger.warning("META.json uses deprecated 'channels20' key, should use 'channels'")
                assert meta['channels20'] == CHANNELS_TUEV_20, (
                    "Cache channels mismatch! Expected TUEV 20 channels (with FPZ, no OZ)"
                )

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
        """Build preprocessed cache with MNE+Autoreject."""
        from tqdm import tqdm  # type: ignore[import-untyped]

        from brain_go_brrr.infra.data.channels import CHANNELS_TUEV_20
        from brain_go_brrr.infra.preprocessing.tuev_preprocessor import (
            CHANNELS_TUEV_23_CANONICAL,
            TUEVPreprocessor,
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        preprocessor = TUEVPreprocessor(use_paper_parity=self.use_paper_parity)

        # Track all windows globally
        global_window_id = 0
        windows_dict = {}
        class_counts = dict.fromkeys(range(6), 0)
        n_rejected_total = 0

        edf_files = self._get_edf_files()
        logger.info(f"Building cache for {len(edf_files)} TUEV files in {self.split} split")

        for edf_path in tqdm(edf_files, desc="Processing TUEV files"):
            try:
                annotations = self._load_annotations(edf_path)

                # Call EXISTING method with CORRECT signature
                epochs_clean, info, window_labels = preprocessor.process_raw_with_annotations(
                    edf_path,
                    annotations,
                    window_overlap=0.5,  # 50% overlap for 2s stride on 4s windows
                )

                # Extract data from MNE Epochs object
                epoch_data = epochs_clean.get_data()  # Shape: (n_epochs, n_channels, 1024)

                # Process each clean epoch
                for epoch_idx in range(len(epochs_clean)):
                    # Get single epoch data (n_channels, 1024)
                    x_volts = epoch_data[epoch_idx]  # In Volts from MNE

                    # CRITICAL: Keep in Volts (SI units) for SSOT compliance
                    # Wrapper expects Volts for normalization
                    # x_volts is already in Volts from MNE

                    # Get label for this window
                    label_str = window_labels[epoch_idx]
                    label_int = CLASS_MAPPING[label_str]

                    # Ensure correct tensor types (channels x time)
                    x_tensor = torch.tensor(
                        x_volts, dtype=torch.float32
                    )  # (n_channels, 1024) in Volts
                    y_tensor = torch.tensor(
                        label_int, dtype=torch.long
                    )  # Long for CrossEntropyLoss

                    # Save individual window
                    cache_file = f"window_{global_window_id}.pt"
                    torch.save(
                        {'x': x_tensor, 'y': y_tensor},
                        self.cache_dir / cache_file,
                        _use_new_zipfile_serialization=True,
                    )

                    # Track in index
                    windows_dict[str(global_window_id)] = {
                        'cache_file': cache_file,
                        'label': int(label_int),
                        'file': str(edf_path.relative_to(self.root_dir)),
                    }

                    class_counts[label_int] += 1
                    global_window_id += 1

                n_rejected_total += info.get('n_rejected', 0)

            except Exception as e:
                logger.warning(f"Error processing {edf_path.name}: {e}")
                continue

        # Write index JSON
        index_data = {
            'total_windows': global_window_id,
            'windows': windows_dict,
            'n_files': len(edf_files),
            'n_rejected': n_rejected_total,
            'class_counts': {str(k): v for k, v in class_counts.items()},
        }

        index_path = self.cache_dir / f'index_{self.split}_{self.CACHE_VERSION}.json'
        with index_path.open('w') as f:
            json.dump(index_data, f, indent=2)

        # Write META JSON
        channels_list = CHANNELS_TUEV_23_CANONICAL if self.use_paper_parity else CHANNELS_TUEV_20
        meta_data = {
            'sr': 256,
            'unit': 'V',  # SI units (Volts) per SSOT
            'window': 1024,
            'channels': channels_list,
            'n_channels': self.n_channels,
            'norm': 'wrapper',
            'paper_parity': self.use_paper_parity,
            'commit': subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'], cwd=Path(__file__).parent
            )
            .decode()
            .strip(),
            'split': self.split,
            'dataset': 'TUEV',
        }

        with (self.cache_dir / 'META.json').open('w') as f:
            json.dump(meta_data, f, indent=2)

        logger.info(f"Cache built: {global_window_id} windows, class dist: {class_counts}")

        # Fail fast if cache is empty
        if global_window_id == 0:
            raise ValueError(
                "Cache building failed: 0 windows produced. "
                "Check preprocessing logs for 'Valid channel positions' errors. "
                "Likely cause: montage not set after channel synthesis."
            )

        # Future enhancement: Track success/failure rate
        # if n_rejected_total / len(edf_files) > 0.5:
        #     logger.error(f"Too many failures: {n_rejected_total}/{len(edf_files)}")
        #     raise ValueError("Cache building failed: >50% of files failed")

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
        # Try with weights_only=True for security (torch >= 2.4), fallback if not supported
        try:
            # Check torch version for better error messaging
            import torch
            torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
            if torch_version >= (2, 4):
                data = torch.load(cache_file, map_location='cpu', weights_only=True)  # nosec:weights_only
            else:
                # Older torch versions don't support weights_only
                logger.debug(f"Using torch {torch.__version__} - weights_only not supported")
                data = torch.load(cache_file, map_location='cpu')  # nosec:weights_only - pre-2.4 torch
        except TypeError:
            # Fallback for edge cases or dev versions
            data = torch.load(cache_file, map_location='cpu')  # nosec:weights_only - fallback for edge cases

        return data['x'], data['y']


# Backwards-compatible alias expected by training/build scripts
# Thin alias to avoid import mismatches without duplicating implementation
TUEVDataset = TUEVMNEDataset
