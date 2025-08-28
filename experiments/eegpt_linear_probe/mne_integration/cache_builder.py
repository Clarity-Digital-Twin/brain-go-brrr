#!/usr/bin/env python3
"""
Build MNE-preprocessed cache for TUAB dataset.
This script preprocesses all TUAB EDF files with MNE+Autoreject and saves them to cache.
"""

import argparse
import logging
from pathlib import Path

from brain_go_brrr.infra.data.tuab_dataset import TUABDataset as TUABMNEDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Build MNE-preprocessed cache for TUAB dataset')
    parser.add_argument(
        '--data-root',
        type=str,
        default='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuab',
        help='Root directory containing TUAB EDF files',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuab_mne_preprocessed',
        help='Directory to save preprocessed cache',
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'eval', 'both'],
        default='both',
        help='Which split(s) to build cache for',
    )
    parser.add_argument(
        '--force-rebuild', action='store_true', help='Force rebuilding cache even if it exists'
    )

    args = parser.parse_args()

    # Determine which splits to process
    splits = ['train', 'eval'] if args.split == 'both' else [args.split]

    # Build cache for each split
    for split in splits:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Building cache for {split} split")
        logger.info(f"{'=' * 60}")

        try:
            dataset = TUABMNEDataset(
                root_dir=Path(args.data_root),
                split=split,
                cache_dir=Path(args.cache_dir),
                force_rebuild=args.force_rebuild,
            )

            logger.info(f"Successfully built cache for {split} split")
            logger.info(f"Dataset contains {len(dataset)} windows")

            # Test loading a sample
            if len(dataset) > 0:
                x, y = dataset[0]
                logger.info(f"Sample shape: {x.shape}, Label: {y.item()}")

        except Exception as e:
            logger.error(f"Failed to build cache for {split} split: {e}")
            raise

    logger.info(f"\n{'=' * 60}")
    logger.info("Cache building complete!")
    logger.info(f"Cache saved to: {args.cache_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
