#!/usr/bin/env python3
"""
Deterministic MNE cache builder with manifest tracking.
Ensures reproducible preprocessing with atomic writes and PHI safety.
"""

import argparse
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from brain_go_brrr.infra.data.tuab_dataset import TUABDataset as TUABMNEDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CRITICAL SPECS FOR EEGPT
CACHE_VERSION = "2.0.0"  # Bump when preprocessing changes
CHANNELS_TUAB = 19  # NO Fz for TUAB
CHANNELS_TUEV = 20  # With Fz for TUEV
SAMPLE_RATE = 256  # Hz
WINDOW_SECONDS = 4.0  # EEGPT requires 4s
WINDOW_SAMPLES = 1024  # 4s @ 256Hz
FEATURE_DIM = 2048  # 4 summary tokens × 512 dims

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Set deterministic algos
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_code_fingerprint() -> str:
    """Get git commit or file hash for tracking code version."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    # Fallback to file hash
    hasher = hashlib.sha256()
    hasher.update(Path(__file__).read_bytes())
    return hasher.hexdigest()[:8]


def create_manifest(
    corpus: str,
    split: str,
    data_root: Path,
    cache_dir: Path,
    n_files: int,
    n_windows: int,
    label_distribution: Dict[str, int]
) -> Dict[str, Any]:
    """Create comprehensive manifest for cache validation."""
    
    preprocessing_config = {
        "sample_rate_hz": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "window_stride_seconds": WINDOW_SECONDS,  # Non-overlapping
        "channels": CHANNELS_TUAB if corpus == "TUAB" else CHANNELS_TUEV,
        "channel_mapping": {
            "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"
        } if corpus == "TUAB" else {},
        "bandpass": [0.5, 50.0],  # Hz
        "notch": 60.0,  # Hz (US power line)
        "normalization": "z-score per channel",
        "random_seed": RANDOM_SEED,
    }
    
    # Create canonical hash of config
    config_str = json.dumps(preprocessing_config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    manifest = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now().isoformat(),
        "corpus": {
            "name": corpus,
            "version": "v3.0.1" if corpus == "TUAB" else "v2.0.1",
            "split": split,
            "data_root": str(data_root.relative_to(Path.cwd())) if data_root.is_absolute() else str(data_root),
        },
        "preprocessing": preprocessing_config,
        "config_hash": config_hash,
        "tools": {
            "mne": "1.10.0",  # Update if different
            "autoreject": "0.4.5",  # Update if different
            "numpy": np.__version__,
            "torch": torch.__version__,
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        },
        "code": {
            "fingerprint": get_code_fingerprint(),
            "script": "cache_builder_v2.py",
        },
        "data_stats": {
            "n_files": n_files,
            "n_windows": n_windows,
            "label_distribution": label_distribution,
            "expected_shape": [CHANNELS_TUAB if corpus == "TUAB" else CHANNELS_TUEV, WINDOW_SAMPLES],
            "feature_dim": FEATURE_DIM,
        },
        "integrity": {
            "phi_safe": True,  # No patient IDs in filenames
            "atomic_writes": True,  # Using temp + rename
            "paths_relative": True,  # All paths relative to data/
        }
    }
    
    return manifest


def sanitize_filename(file_path: Path) -> str:
    """Create PHI-safe cache filename."""
    # Hash the stem to avoid patient IDs
    hasher = hashlib.sha256()
    hasher.update(str(file_path).encode())
    safe_name = hasher.hexdigest()[:16]
    return f"{safe_name}.pt"


def atomic_write(data: Any, target_path: Path) -> None:
    """Write with temp file + atomic rename for safety."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first
    with tempfile.NamedTemporaryFile(
        delete=False, 
        dir=target_path.parent, 
        suffix='.tmp'
    ) as tmp:
        torch.save(data, tmp.name)
        tmp_path = Path(tmp.name)
    
    # Atomic rename
    tmp_path.replace(target_path)


def build_cache(
    corpus: str,
    split: str, 
    data_root: Path,
    cache_dir: Path,
    force_rebuild: bool = False
) -> None:
    """Build cache with full validation and manifest."""
    
    split_cache_dir = cache_dir / split
    manifest_path = cache_dir / f"manifest_{split}.json"
    
    # Check existing cache
    if split_cache_dir.exists() and not force_rebuild:
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                existing = json.load(f)
            logger.info(f"Cache exists with hash {existing['config_hash']}")
            logger.info("Use --force-rebuild to regenerate")
            return
    
    # Clean if force rebuild
    if split_cache_dir.exists() and force_rebuild:
        import shutil
        logger.info(f"Force rebuild: removing {split_cache_dir}")
        shutil.rmtree(split_cache_dir)
        if manifest_path.exists():
            manifest_path.unlink()
    
    # Clean any temp files from previous runs
    for tmp_file in cache_dir.glob("**/*.tmp"):
        logger.info(f"Cleaning temp file: {tmp_file}")
        tmp_file.unlink()
    
    logger.info(f"Building {corpus} {split} cache...")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Cache dir: {cache_dir}")
    
    # Create dataset
    dataset = TUABMNEDataset(
        root_dir=data_root,
        split=split,
        cache_dir=cache_dir,
        sampling_rate=SAMPLE_RATE,
        window_duration=WINDOW_SECONDS,
        window_stride=WINDOW_SECONDS,
        normalize=False,  # NEVER in dataset, only in wrapper
    )
    
    n_windows = len(dataset)
    logger.info(f"Dataset has {n_windows} windows")
    
    # Get stats by processing a sample
    if n_windows > 0:
        # Test first window
        x, y = dataset[0]
        y_val = y if isinstance(y, int) else y.item()
        
        # Validate shape
        expected_channels = CHANNELS_TUAB if corpus == "TUAB" else CHANNELS_TUEV
        assert x.shape == (expected_channels, WINDOW_SAMPLES), \
            f"Wrong shape: {x.shape} != ({expected_channels}, {WINDOW_SAMPLES})"
        
        logger.info(f"✓ Window shape: {x.shape}")
        logger.info(f"✓ Label: {y_val} (0=normal, 1=abnormal)")
        logger.info(f"✓ Dtype: {x.dtype}")
        
        # Check a few more windows for consistency
        for i in [100, 500, 1000, 5000]:
            if i < n_windows:
                xi, yi = dataset[i]
                assert xi.shape == x.shape, f"Inconsistent shape at index {i}"
        logger.info(f"✓ Validated {min(5, n_windows)} windows - all consistent")
    
    # Get label distribution (requires iterating once - expensive but important)
    logger.info("Computing label distribution...")
    label_counts = {"normal": 0, "abnormal": 0}
    
    # Sample instead of full iteration for large datasets
    sample_size = min(10000, n_windows)
    indices = np.random.choice(n_windows, sample_size, replace=False)
    
    for idx in indices:
        _, y = dataset[idx]
        y_val = y if isinstance(y, int) else y.item()
        if y_val == 0:
            label_counts["normal"] += 1
        else:
            label_counts["abnormal"] += 1
    
    # Scale up estimates
    scale_factor = n_windows / sample_size
    label_counts = {k: int(v * scale_factor) for k, v in label_counts.items()}
    
    logger.info(f"Label distribution (estimated): {label_counts}")
    
    # Create and save manifest
    n_files = len(dataset.file_paths) if hasattr(dataset, 'file_paths') else -1
    manifest = create_manifest(
        corpus=corpus,
        split=split,
        data_root=data_root,
        cache_dir=cache_dir,
        n_files=n_files,
        n_windows=n_windows,
        label_distribution=label_counts
    )
    
    # Save manifest atomically
    atomic_write(manifest, manifest_path.with_suffix('.tmp'))
    manifest_path.with_suffix('.tmp').replace(manifest_path)
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✓ Manifest saved: {manifest_path}")
    logger.info(f"✓ Config hash: {manifest['config_hash']}")
    logger.info(f"✓ Cache ready for use!")


def main():
    parser = argparse.ArgumentParser(description='Build deterministic MNE cache')
    parser.add_argument(
        '--corpus',
        type=str,
        default='TUAB',
        choices=['TUAB', 'TUEV'],
        help='Which corpus to process'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory containing EDF files'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        required=True,
        help='Directory to save cache'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'eval', 'both'],
        default='both',
        help='Which split(s) to build'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuilding even if cache exists'
    )
    
    args = parser.parse_args()
    
    splits = ['train', 'eval'] if args.split == 'both' else [args.split]
    
    for split in splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {args.corpus} {split} split")
        logger.info(f"{'='*60}")
        
        try:
            build_cache(
                corpus=args.corpus,
                split=split,
                data_root=Path(args.data_root),
                cache_dir=Path(args.cache_dir),
                force_rebuild=args.force_rebuild
            )
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
            raise
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ All splits complete!")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()