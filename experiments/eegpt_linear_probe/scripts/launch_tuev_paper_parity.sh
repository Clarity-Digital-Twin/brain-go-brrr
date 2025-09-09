#!/bin/bash
# Launch TUEV training with paper parity (23 channels + learned mapper)
# Target: 62.32% BAC as reported in EEGPT paper Table 3

set -e

# Set environment variables
export BGB_DATA_ROOT="${BGB_DATA_ROOT:-/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data}"
export BGB_CACHE_DIR="${BGB_CACHE_DIR:-$BGB_DATA_ROOT/cache}"

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate run name with timestamp
RUN_NAME="tuev_paper_parity_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "TUEV Paper Parity Training"
echo "========================================="
echo "Run name: $RUN_NAME"
echo "Data root: $BGB_DATA_ROOT"
echo "Cache dir (base): $BGB_CACHE_DIR (dataset writes to tuev_23ch_paper_parity/)"
echo "Config: configs/tuev_paper_parity.yaml"
echo "Expected: 62.32% BAC (vs current ~22%)"
echo "========================================="
echo ""

# Check if cache exists
CACHE_TRAIN="$BGB_CACHE_DIR/tuev_23ch_paper_parity/train"
CACHE_EVAL="$BGB_CACHE_DIR/tuev_23ch_paper_parity/eval"

if [ ! -d "$CACHE_TRAIN" ] || [ ! -d "$CACHE_EVAL" ]; then
    echo "WARNING: 23-channel cache not found!"
    echo "Expected directories:"
    echo "  - $CACHE_TRAIN"
    echo "  - $CACHE_EVAL"
    echo ""
    echo "Please build the cache first with:"
    echo "  ./scripts/build_tuev_23ch_cache.sh"
    echo ""
    echo "Proceeding anyway - dataset will try to build cache on the fly..."
    echo ""
fi

# Launch training
echo "Starting training with paper parity configuration..."
# Use uv to ensure project environment and pass BASE cache dir (dataset adds subdir)
uv run python train_tuev_mne.py \
    --config configs/tuev_paper_parity.yaml \
    --cache_dir "$BGB_CACHE_DIR" \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo ""
echo "========================================="
echo "Training complete!"
echo "Log saved to: logs/${RUN_NAME}.log"
echo "========================================="
