#!/bin/bash
# Build TUEV 23-channel cache for paper parity

set -e

# Set environment variables
export BGB_DATA_ROOT="${BGB_DATA_ROOT:-/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data}"
export BGB_CACHE_DIR="${BGB_CACHE_DIR:-$BGB_DATA_ROOT/cache}"

echo "=============================================="
echo "Building TUEV 23-channel cache for paper parity"
echo "=============================================="
echo "Data root: $BGB_DATA_ROOT"
echo "Cache dir: $BGB_CACHE_DIR/tuev_23ch_paper_parity"
echo "This will take 4-6 hours. Run in tmux!"
echo ""

# Check if TUEV data exists
if [ ! -d "$BGB_DATA_ROOT/datasets/tuev/edf" ]; then
    echo "ERROR: TUEV dataset not found at $BGB_DATA_ROOT/datasets/tuev"
    echo "Please download TUEV first"
    exit 1
fi

# Build cache for both train and eval splits
for SPLIT in train eval; do
    echo "=============================================="
    echo "Building $SPLIT split with 23 channels..."
    echo "=============================================="
    
    uv run python -c "
import sys
import os
os.environ['BGB_DATA_ROOT'] = '$BGB_DATA_ROOT'
os.environ['BGB_CACHE_DIR'] = '$BGB_CACHE_DIR'

# Add project root to path
sys.path.insert(0, '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr')

from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset
from pathlib import Path

print(f'Building 23-channel cache for {SPLIT} split...')
dataset = TUEVMNEDataset(
    root_dir=Path('$BGB_DATA_ROOT/datasets/tuev'),
    split='$SPLIT',
    cache_dir=Path('$BGB_CACHE_DIR'),
    force_rebuild=True,
    use_paper_parity=True  # 23 channels for paper parity
)

print(f'✓ Built {len(dataset)} windows for $SPLIT split')
print(f'  Channels: {dataset.n_channels}')
print(f'  Cache location: {dataset.cache_dir}')

# Verify first sample has 23 channels
if len(dataset) > 0:
    x, y = dataset[0]
    print(f'  Sample shape: {x.shape} (should be (23, 1024))')
    assert x.shape[0] == 23, f'Expected 23 channels, got {x.shape[0]}'
"
done

echo ""
echo "=============================================="
echo "✓ Cache build complete!"
echo "=============================================="
echo "Next steps:"
echo "1. Check cache integrity:"
echo "   ls -la $BGB_CACHE_DIR/tuev_23ch_paper_parity/"
echo ""
echo "2. Launch training with paper parity:"
echo "   ./launch_tuev_paper_parity.sh"
echo "=============================================="