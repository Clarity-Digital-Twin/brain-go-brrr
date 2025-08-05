#!/bin/bash
set -e

# Build 4-second window cache for paper-aligned training
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1

echo "Building 4-second window cache for TUAB dataset..."
echo "This will take ~30-60 minutes depending on disk speed"

# Create cache directory
CACHE_DIR="$BGB_DATA_ROOT/cache_4s"
mkdir -p "$CACHE_DIR"

# Build cache with 4-second windows
uv run python -c "
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path('$BGB_DATA_ROOT').parent
sys.path.insert(0, str(project_root))

from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset

print('Building training set cache...')
train_dataset = TUABCachedDataset(
    root_dir=Path('$BGB_DATA_ROOT/datasets/external/tuab'),
    split='train',
    window_duration=4.0,  # 4 seconds as per paper
    window_stride=2.0,    # 50% overlap
    sampling_rate=256,
    cache_dir=Path('$CACHE_DIR'),
    cache_index_path=Path('$BGB_DATA_ROOT/cache/tuab_index.json')
)
print(f'Training samples: {len(train_dataset)}')

print('Building validation set cache...')
val_dataset = TUABCachedDataset(
    root_dir=Path('$BGB_DATA_ROOT/datasets/external/tuab'),
    split='eval',
    window_duration=4.0,
    window_stride=4.0,  # No overlap for validation
    sampling_rate=256,
    cache_dir=Path('$CACHE_DIR'),
    cache_index_path=Path('$BGB_DATA_ROOT/cache/tuab_index.json')
)
print(f'Validation samples: {len(val_dataset)}')

print('Cache building complete!')
print(f'Cache location: {CACHE_DIR}')
"

echo "4-second window cache built successfully!"