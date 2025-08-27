"""Build TUEV Cache - DEPRECATED LEGACY SCRIPT

⚠️ WARNING: This script uses the OLD TUEVDataset without MNE preprocessing.
   USE INSTEAD: ./scripts/build_tuev_mne_cache.sh (MNE+Autoreject path)

This legacy script:
1. Loads raw TUEV EDF files (NO MNE preprocessing)
2. Outputs 23 channels (not the standard 20)
3. Does NOT apply Autoreject
4. Does NOT match the training pipeline

CORRECT WORKFLOW:
  ./scripts/build_tuev_mne_cache.sh  # Build with MNE preprocessing
  python train_tuev_mne.py           # Train with MNE cache
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Imports assume PYTHONPATH is set to repository root
from experiments.eegpt_linear_probe.datasets.tuev_dataset import TUEVDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_cache(config_path: str, output_dir: str):
    """Build cached TUEV dataset.

    Args:
        config_path: Path to TUEV config file
        output_dir: Directory to save cache
    """
    # Set environment variable if not set
    if 'BGB_DATA_ROOT' not in os.environ:
        os.environ['BGB_DATA_ROOT'] = (
            '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data'
        )
        logger.info(f"Set BGB_DATA_ROOT to {os.environ['BGB_DATA_ROOT']}")

    # Load config
    config = OmegaConf.load(config_path)

    # Setup cache directory
    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building TUEV cache at {cache_dir}")
    logger.info(f"Config: {config_path}")

    # Process both splits
    for split in ['train', 'eval']:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing {split} split")
        logger.info(f"{'=' * 50}")

        # Create split cache directory
        split_cache = cache_dir / f"tuev_{split}_cache"
        split_cache.mkdir(parents=True, exist_ok=True)

        # Create dataset (without cache to force loading)
        dataset = TUEVDataset(
            root_dir=Path(config.data.root_dir),
            split=split,
            cache_dir=None,  # Don't use cache during building
            resample=True,
            normalize=True,
        )

        logger.info(f"Found {len(dataset)} windows in {split} split")

        # Process and cache all samples
        samples_info = []
        class_counts = dict.fromkeys(range(6), 0)

        for idx in tqdm(range(len(dataset)), desc=f"Caching {split}"):
            # Get sample
            x, y = dataset[idx]

            # Verify shape (EEGPT requires 1024 samples divisible by 64)
            assert x.shape == (23, 1024), f"Wrong shape: {x.shape}, expected (23, 1024)"
            assert y in range(6), f"Wrong label: {y}"

            # Save to disk
            cache_file = f"sample_{idx:06d}.pt"
            cache_path = split_cache / cache_file

            torch.save({'x': x, 'y': y}, cache_path)

            # Track info
            samples_info.append({'idx': idx, 'cache_file': cache_file, 'label': int(y)})

            class_counts[int(y)] += 1

        # Compute class weights
        total = sum(class_counts.values())
        class_weights = []
        for i in range(6):
            count = class_counts[i]
            weight = 1.0 / (count + 1e-6) if count > 0 else 0.0
            class_weights.append(weight)

        # Normalize weights
        class_weights = np.array(class_weights)
        class_weights = class_weights / class_weights.sum() * len(class_weights)

        # Save index
        index = {
            'split': split,
            'n_samples': len(samples_info),
            'n_classes': 6,
            'class_names': ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG'],
            'class_counts': class_counts,
            'class_weights': class_weights.tolist(),
            'samples': samples_info,
            'config': OmegaConf.to_container(config),
        }

        index_path = split_cache / 'index.json'
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

        # Report statistics
        logger.info(f"\n{split} Split Statistics:")
        logger.info(f"  Total samples: {len(samples_info)}")
        logger.info("  Class distribution:")
        for i, name in enumerate(index['class_names']):
            count = class_counts[i]
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"    {name}: {count} ({pct:.1f}%)")
        logger.info(f"  Class weights: {class_weights}")
        logger.info(f"  Cache size: {split_cache}")

    logger.info(f"\n{'=' * 50}")
    logger.info("Cache building complete!")
    logger.info(f"Cache directory: {cache_dir}")

    # Verify cache
    logger.info("\nVerifying cache...")
    for split in ['train', 'eval']:
        split_cache = cache_dir / f"tuev_{split}_cache"
        index_path = split_cache / 'index.json'

        with open(index_path) as f:
            index = json.load(f)

        # Check a few samples
        for i in range(min(3, len(index['samples']))):
            sample_info = index['samples'][i]
            cache_file = split_cache / sample_info['cache_file']

            if not cache_file.exists():
                logger.error(f"Missing cache file: {cache_file}")
                continue

            data = torch.load(cache_file, map_location='cpu')
            assert data['x'].shape == (
                23,
                1024,
            ), f"Wrong shape in {cache_file}, expected (23, 1024)"
            assert data['y'] in range(6), f"Wrong label in {cache_file}"

        logger.info(f"✓ {split} cache verified")

    logger.info("\n✅ Cache building successful!")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("TUEV CACHE BUILT SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nCache location: {cache_dir}")
    print("\nTo use this cache in training:")
    print("  python train_tuev.py \\")
    print("    --config configs/tuev.yaml \\")
    print("    --use-cache \\")
    print("    --device cuda")
    print("\nExpected performance (from paper):")
    print("  Balanced Accuracy: 0.6232 ± 0.0114")
    print("  Weighted F1:       0.8187 ± 0.0063")
    print("  Cohen's Kappa:     0.6351 ± 0.0134")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    # Print deprecation warning
    print("\n" + "="*70)
    print("⚠️  DEPRECATION WARNING")
    print("="*70)
    print("This script uses the LEGACY TUEVDataset without MNE preprocessing.")
    print("It outputs 23 channels and does NOT match the training pipeline.")
    print("\nUSE INSTEAD:")
    print("  ./scripts/build_tuev_mne_cache.sh  # For MNE preprocessing (20 channels)")
    print("  python train_tuev_mne.py           # For training")
    print("="*70)
    
    response = input("\nContinue with LEGACY script anyway? (y/N): ")
    if response.lower() != 'y':
        print("Aborted. Please use the MNE path instead.")
        exit(0)

    parser = argparse.ArgumentParser(description="Build TUEV cache")
    parser.add_argument(
        '--config', type=str, default='../configs/tuev.yaml', help='Path to config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_4s_256hz_v2',
        help='Output cache directory (v2 = 1024 samples)',
    )

    args = parser.parse_args()

    # Set environment variable if not set
    if 'BGB_DATA_ROOT' not in os.environ:
        os.environ['BGB_DATA_ROOT'] = (
            '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data'
        )

    build_cache(args.config, args.output)
