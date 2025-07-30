"""TUAB (TUH Abnormal) Dataset for EEG abnormality detection.

Dataset from Temple University Hospital EEG Corpus.
Paper: https://www.isip.piconepress.com/projects/tuh_eeg/
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import mne
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TUABDataset(Dataset):
    """TUAB Dataset for abnormality detection.

    Expects data organized as:
    root_dir/
        train/
            normal/
                *.edf
            abnormal/
                *.edf
        val/
            normal/
                *.edf
            abnormal/
                *.edf
        test/
            ...
    """

    LABEL_MAP = {"normal": 0, "abnormal": 1}

    # Standard TUAB channels (23 channels)
    STANDARD_CHANNELS = [
        "FP1",
        "FP2",
        "F7",
        "F3",
        "FZ",
        "F4",
        "F8",
        "T3",
        "C3",
        "CZ",
        "C4",
        "T4",
        "T5",
        "P3",
        "PZ",
        "P4",
        "T6",
        "O1",
        "O2",
        "A1",
        "A2",  # Reference electrodes
        "FPZ",
        "OZ",
    ]

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        sampling_rate: int = 256,
        window_duration: float = 30.0,
        window_stride: float = 30.0,
        preload: bool = False,
        normalize: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize TUAB dataset.

        Args:
            root_dir: Root directory containing train/val/test splits
            split: Which split to use ('train', 'val', 'test')
            sampling_rate: Target sampling rate in Hz
            window_duration: Window duration in seconds
            window_stride: Stride between windows in seconds
            preload: Whether to preload all data into memory
            normalize: Whether to apply z-score normalization
            cache_dir: Optional directory for caching preprocessed data
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.sampling_rate = sampling_rate
        self.window_duration = window_duration
        self.window_stride = window_stride
        self.window_samples = int(window_duration * sampling_rate)
        self.stride_samples = int(window_stride * sampling_rate)
        self.preload = preload
        self.normalize = normalize
        self.cache_dir = cache_dir

        # Validate directory structure
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")

        # Collect all EDF files
        self.samples: list[dict[str, Any]] = []
        self._collect_samples()

        logger.info(
            f"Loaded TUAB {split} split: {len(self.samples)} windows from "
            f"{len(self.file_list)} files ({self.class_counts})"
        )

        # Preload if requested
        if self.preload:
            logger.info("Preloading all data into memory...")
            self._preload_data()

    def _collect_samples(self) -> None:
        """Collect all samples from the dataset."""
        self.file_list: list[dict[str, Any]] = []
        self.class_counts: dict[str, int] = defaultdict(int)

        # Iterate through normal/abnormal directories
        for class_name in ["normal", "abnormal"]:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            label = self.LABEL_MAP[class_name]

            # Find all EDF files
            edf_files = list(class_dir.glob("*.edf"))
            logger.info(f"Found {len(edf_files)} {class_name} files")

            for edf_file in edf_files:
                try:
                    # Get file info without loading
                    info = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
                    duration = info.n_times / info.info["sfreq"]

                    # Calculate number of windows
                    n_windows = int((duration - self.window_duration) / self.window_stride) + 1

                    if n_windows > 0:
                        self.file_list.append(
                            {
                                "path": edf_file,
                                "label": label,
                                "class_name": class_name,
                                "n_windows": n_windows,
                            }
                        )

                        # Add window indices
                        for window_idx in range(n_windows):
                            self.samples.append(
                                {
                                    "file_idx": len(self.file_list) - 1,
                                    "window_idx": window_idx,
                                    "label": label,
                                    "class_name": class_name,
                                }
                            )

                        self.class_counts[class_name] += n_windows

                except Exception as e:
                    logger.warning(f"Error reading {edf_file}: {e}")

    def _preload_data(self) -> None:
        """Preload all data into memory."""
        self.preloaded_data = {}

        for file_info in self.file_list:
            try:
                data = self._load_edf_file(Path(file_info["path"]))
                self.preloaded_data[file_info["path"]] = data
            except Exception as e:
                logger.error(f"Error preloading {file_info['path']}: {e}")

    def _load_edf_file(self, file_path: Path) -> np.ndarray:
        """Load and preprocess EDF file.

        Args:
            file_path: Path to EDF file

        Returns:
            Preprocessed EEG data [channels, time]
        """
        # Load raw data
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Select channels (handle missing channels)
        available_channels = [ch for ch in self.STANDARD_CHANNELS if ch in raw.ch_names]
        if len(available_channels) < len(self.STANDARD_CHANNELS):
            logger.warning(
                f"Missing channels in {file_path.name}: "
                f"{set(self.STANDARD_CHANNELS) - set(available_channels)}"
            )

        raw.pick_channels(available_channels, ordered=True)

        # Resample if needed
        if raw.info["sfreq"] != self.sampling_rate:
            raw.resample(self.sampling_rate)

        # Apply basic preprocessing
        raw.filter(0.5, 50.0, fir_design="firwin", verbose=False)
        raw.notch_filter(60.0, fir_design="firwin", verbose=False)

        # Get data
        data = raw.get_data()

        # Pad with zeros if we have fewer channels
        if data.shape[0] < len(self.STANDARD_CHANNELS):
            padding = np.zeros((len(self.STANDARD_CHANNELS) - data.shape[0], data.shape[1]))
            data = np.vstack([data, padding])

        return data.astype(np.float32)  # type: ignore

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (eeg_data, label)
            eeg_data: [channels, time] tensor
            label: 0 (normal) or 1 (abnormal)
        """
        sample_info = self.samples[idx]
        file_info = self.file_list[sample_info["file_idx"]]

        # Load data (from cache or file)
        if self.preload:
            data = self.preloaded_data[file_info["path"]]
        else:
            data = self._load_edf_file(file_info["path"])

        # Extract window
        start_idx = sample_info["window_idx"] * self.stride_samples
        end_idx = start_idx + self.window_samples

        # Handle edge case
        if end_idx > data.shape[1]:
            # Pad with zeros
            window = np.zeros((data.shape[0], self.window_samples), dtype=np.float32)
            available = data.shape[1] - start_idx
            window[:, :available] = data[:, start_idx:]
        else:
            window = data[:, start_idx:end_idx]

        # Normalize if requested
        if self.normalize:
            mean = window.mean(axis=1, keepdims=True)
            std = window.std(axis=1, keepdims=True) + 1e-6
            window = (window - mean) / std

        # Convert to tensor
        eeg_tensor = torch.from_numpy(window).float()
        label = sample_info["label"]

        return eeg_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for balanced training."""
        total = sum(self.class_counts.values())
        weights = []

        for class_name in ["normal", "abnormal"]:
            count = self.class_counts[class_name]
            weight = total / (len(self.class_counts) * count) if count > 0 else 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)
