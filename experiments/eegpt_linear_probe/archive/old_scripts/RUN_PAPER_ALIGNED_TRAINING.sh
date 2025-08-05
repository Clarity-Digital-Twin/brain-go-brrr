#!/bin/bash
# Execute paper-aligned training for TUAB abnormality detection

set -e  # Exit on error

echo "=========================================="
echo "EEGPT Linear Probe Training (Paper-Aligned)"
echo "Target: AUROC > 0.85 (Paper: 0.87)"
echo "=========================================="

# Set environment
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Step 1: Build 4-second window dataset (if not exists)
CACHE_DIR="$BGB_DATA_ROOT/cache/tuab_4s_windows"

if [ ! -d "$CACHE_DIR" ]; then
    echo ""
    echo "Step 1: Building 4-second window dataset..."
    echo "This will take ~30-60 minutes"
    echo ""
    
    cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr
    uv run python experiments/eegpt_linear_probe/build_tuab_4s_cache.py
else
    echo ""
    echo "Step 1: 4-second cache already exists at $CACHE_DIR"
    echo "Skipping dataset building..."
fi

# Step 2: Train with paper-aligned settings
echo ""
echo "Step 2: Training linear probe with paper settings..."
echo "- 4-second windows (matches EEGPT pretraining)"
echo "- OneCycle LR schedule (2.5e-4 → 5e-4 → 3.13e-5)"
echo "- 200 epochs with early stopping"
echo "- Batch size 256"
echo "- Gradient clipping at 5.0"
echo ""

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/paper_aligned_$TIMESTAMP"

# Launch training in tmux for monitoring
tmux new-session -d -s eegpt_paper_aligned \
    "uv run python train_paper_aligned.py \
        --config configs/tuab_4s_paper_aligned.yaml \
        --output_dir $OUTPUT_DIR \
        --device cuda \
    2>&1 | tee $OUTPUT_DIR/training.log"

echo ""
echo "Training launched in tmux session 'eegpt_paper_aligned'"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Monitor with:"
echo "  tmux attach -t eegpt_paper_aligned"
echo ""
echo "Check progress with:"
echo "  tail -f $OUTPUT_DIR/training.log | grep -E 'Epoch|AUROC'"
echo ""
echo "Expected training time: ~2-3 hours on RTX 3090"
echo "Expected AUROC: >0.85 (closing gap to paper's 0.87)"