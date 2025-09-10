"""TUEV Event Dataset for paper parity - loads pre-extracted event segments."""

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore[import-untyped]

from brain_go_brrr.infra.preprocessing.tuev_event_extractor import TUEVEventExtractor


class TUEVEventDataset(Dataset[tuple[torch.Tensor, int]]):
    """TUEV event segment dataset for paper parity.

    This is DIFFERENT from our sliding window TUEVMNEDataset.
    This extracts ONLY event segments for classification, matching
    the EEGPT reference implementation exactly.

    Key differences from sliding window approach:
    - Extracts 5-second segments at 200Hz (not 4s at 256Hz)
    - Only event segments (no sliding windows over full recording)
    - Natural class balance (not 99.5% background)
    - Matches EEGPT paper results (62.32% BAC)
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        cache_dir: Path | None = None,
        force_rebuild: bool = False,
    ):
        """Initialize TUEV event dataset.

        Args:
            root_dir: Root directory containing edf/train, edf/eval structure
            split: Dataset split ('train', 'eval', 'test')
            cache_dir: Directory for caching extracted segments
            force_rebuild: Whether to force rebuilding the cache
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.cache_dir = cache_dir or self.root_dir / 'cache' / 'tuev_event_segments'

        # TUEV class mapping (6 classes)
        self.class_mapping = {
            'spsw': 0,  # spike and slow wave
            'gped': 1,  # generalized periodic epileptiform discharge
            'pled': 2,  # periodic lateralized epileptiform discharge
            'eyem': 3,  # eye movement
            'artf': 4,  # artifact
            'bckg': 5,  # background
        }

        # Build or load cache
        if force_rebuild or not self._cache_exists():
            self._build_cache()

        self._load_cache()

    def _cache_exists(self) -> bool:
        """Check if cache exists for this split."""
        index_file = self.cache_dir / self.split / 'index.json'
        return index_file.exists()

    def _parse_annotations(self, edf_path: Path) -> list[dict[str, float | int]]:
        """Parse annotations from .rec.lab file.

        Args:
            edf_path: Path to EDF file

        Returns:
            List of annotation dicts with 'start', 'end', 'label' keys
        """
        # Find corresponding .rec.lab file
        # Pattern: if EDF is xxx.edf, lab file is xxx.rec.lab
        lab_path = edf_path.parent / f"{edf_path.stem}.rec.lab"

        if not lab_path.exists():
            return []

        annotations = []
        with lab_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                # Parse microseconds to seconds
                start_us = float(parts[0])
                end_us = float(parts[1])
                label_str = parts[2].lower()

                # Convert to seconds
                start_sec = start_us / 1e6
                end_sec = end_us / 1e6

                # Map label to integer
                if label_str in self.class_mapping:
                    label = self.class_mapping[label_str]
                    annotations.append({'start': start_sec, 'end': end_sec, 'label': label})

        return annotations

    def _build_cache(self) -> None:
        """Build cache using our event extractor."""
        print(f"Building {self.split} cache for TUEV event segments...")

        extractor = TUEVEventExtractor()

        # Get all EDF files for this split
        split_dir = self.root_dir / 'edf' / self.split
        if not split_dir.exists():
            print(f"Warning: Split directory {split_dir} does not exist")
            # Create empty cache
            cache_dir = self.cache_dir / self.split
            cache_dir.mkdir(parents=True, exist_ok=True)
            index_file = cache_dir / 'index.json'
            with index_file.open('w') as f:
                json.dump(
                    {
                        'segments': [],
                        'n_segments': 0,
                        'class_counts': {},
                        'n_subjects': 0,
                        'fs': 200,
                        'duration': 5.0,
                        'channels': 23,
                        'samples': 1000,
                    },
                    f,
                    indent=2,
                )
            return

        edf_files = list(split_dir.rglob('*.edf'))

        # Process each file
        all_segments = []
        for edf_path in tqdm(edf_files, desc=f"Building {self.split} cache"):
            # Parse annotations
            annotations = self._parse_annotations(edf_path)

            if not annotations:
                continue

            # Extract segments
            try:
                segments = extractor.extract_segments(edf_path, annotations)
            except Exception as e:
                print(f"Error processing {edf_path}: {e}")
                continue

            # Save each segment
            for i, (segment, label) in enumerate(segments):
                # Extract subject ID (first part before underscore)
                subject_id = edf_path.stem.split('_')[0]
                segment_id = f"{edf_path.stem}_{i}"

                cache_file = self.cache_dir / self.split / f"{segment_id}.pt"
                cache_file.parent.mkdir(parents=True, exist_ok=True)

                # Convert to tensor in Volts (SI units per our SSOT)
                segment_tensor = torch.from_numpy(segment).float()

                # Save with weights_only compatible format
                torch.save(
                    {
                        'x': segment_tensor,  # (23, 1000)
                        'y': label,
                        'id': segment_id,
                    },
                    cache_file,
                )

                all_segments.append(
                    {'file': cache_file.name, 'label': label, 'subject': subject_id}
                )

        # Calculate statistics
        class_counts = Counter([s['label'] for s in all_segments])
        n_subjects = len({s['subject'] for s in all_segments})

        # Save index with metadata
        index_file = self.cache_dir / self.split / 'index.json'
        with index_file.open('w') as f:
            json.dump(
                {
                    'segments': all_segments,
                    'n_segments': len(all_segments),
                    'class_counts': {str(k): v for k, v in class_counts.items()},
                    'n_subjects': n_subjects,
                    'fs': 200,  # Paper-specified sampling rate
                    'duration': 5.0,  # Paper-specified duration
                    'channels': 23,  # Paper-specified channels
                    'samples': 1000,  # 5s * 200Hz
                },
                f,
                indent=2,
            )

        print(f"Built cache with {len(all_segments)} segments from {n_subjects} subjects")
        print(f"Class distribution: {dict(class_counts)}")

    def _load_cache(self) -> None:
        """Load cached segments index."""
        index_file = self.cache_dir / self.split / 'index.json'

        if not index_file.exists():
            # Create empty dataset
            self.segments = []
            self.metadata = {'fs': 200, 'duration': 5.0, 'channels': 23, 'samples': 1000}
            return

        with index_file.open() as f:
            index_data = json.load(f)

        self.segments = index_data['segments']
        self.metadata = {
            'fs': index_data.get('fs', 200),
            'duration': index_data.get('duration', 5.0),
            'channels': index_data.get('channels', 23),
            'samples': index_data.get('samples', 1000),
            'n_segments': index_data.get('n_segments', 0),
            'n_subjects': index_data.get('n_subjects', 0),
            'class_counts': index_data.get('class_counts', {}),
        }

        print(f"Loaded {self.split} cache: {len(self.segments)} segments")

    def __len__(self) -> int:
        """Return number of segments in dataset."""
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a segment and its label.

        Args:
            idx: Index of segment to retrieve

        Returns:
            Tuple of (segment, label) where:
                segment: Tensor of shape (23, 1000) in Volts
                label: Integer class label (0-5)
        """
        segment_info = self.segments[idx]
        cache_file = self.cache_dir / self.split / segment_info['file']

        # Load with weights_only=True for safety
        data = torch.load(cache_file, weights_only=True)

        return data['x'], data['y']
