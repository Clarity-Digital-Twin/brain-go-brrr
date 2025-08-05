#!/bin/bash
# Execute training with existing 8s cache (temporary)

set -e  # Exit on error

echo "=========================================="
echo "EEGPT Linear Probe Training (8s windows)"
echo "Target: AUROC > 0.85"
echo "=========================================="

# Set environment
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Create temporary config with 8s windows
CONFIG_FILE="configs/tuab_8s_temp.yaml"

# Copy and modify config
cp configs/tuab_4s_paper_aligned.yaml $CONFIG_FILE
sed -i 's/window_duration: 4.0/window_duration: 8.0/g' $CONFIG_FILE
sed -i 's/window_stride: 2.0/window_stride: 4.0/g' $CONFIG_FILE
sed -i 's|cache/tuab_4s_windows|cache/tuab_enhanced|g' $CONFIG_FILE

echo ""
echo "Training with 8s windows (temporary)"
echo ""

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/training_8s_$TIMESTAMP"
mkdir -p $OUTPUT_DIR

# Launch training in tmux for monitoring
tmux new-session -d -s eegpt_training \
    "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_paper_aligned.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR \
        --device cuda \
    2>&1 | tee $OUTPUT_DIR/training.log"

echo ""
echo "Training launched in tmux session 'eegpt_training'"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Monitor with:"
echo "  tmux attach -t eegpt_training"
echo ""
echo "Check progress with:"
echo "  tail -f $OUTPUT_DIR/training.log | grep -E 'Epoch|AUROC'"
echo ""
echo "Expected training time: ~2-3 hours on RTX 4090"
echo "Expected AUROC: >0.85"