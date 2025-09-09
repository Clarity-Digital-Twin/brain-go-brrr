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
    
    # Discover repo root and ensure src/ is on PYTHONPATH for imports
    ROOT_DIR=$(git rev-parse --show-toplevel)
    export PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH"
    export CURRENT_SPLIT="$SPLIT"

    uv run python - << 'PYCODE'
import os
from pathlib import Path
from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset

data_root = os.environ['BGB_DATA_ROOT']
cache_root = os.environ['BGB_CACHE_DIR']
split = os.environ['CURRENT_SPLIT']

print(f'Building 23-channel cache for {split} split...')
dataset = TUEVMNEDataset(
    root_dir=Path(f"{data_root}/datasets/tuev"),
    split=split,
    cache_dir=Path(cache_root),
    force_rebuild=True,
    use_paper_parity=True,
)

print(f'✓ Built {len(dataset)} windows for {split} split')
print(f'  Channels: {dataset.n_channels}')
print(f'  Cache location: {dataset.cache_dir}')

if len(dataset) > 0:
    x, _ = dataset[0]
    print(f'  Sample shape: {x.shape} (should be (23, 1024))')
    assert x.shape[0] == 23, f'Expected 23 channels, got {x.shape[0]}'
PYCODE
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
