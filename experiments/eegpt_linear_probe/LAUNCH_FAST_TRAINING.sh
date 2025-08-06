#!/bin/bash
# Fast training with proper settings for 4s windows

set -e

echo "=== LAUNCHING FAST 4S WINDOW TRAINING ==="
echo "This uses optimized settings to avoid the slow loading issue"
echo ""

# Setup environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data

# Create logs directory
mkdir -p logs

# Log file with timestamp
LOG_FILE="logs/fast_4s_training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training with paper-aligned 4s config..."
echo "Log file: $LOG_FILE"
echo ""

# Launch in tmux with the paper-aligned script which handles data better
tmux new-session -d -s eegpt_fast \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe && \
     /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_paper_aligned.py \
     --config configs/tuab_4s_paper_target.yaml \
     --device cuda \
     2>&1 | tee $LOG_FILE"

echo "✅ Training launched in tmux session 'eegpt_fast'"
echo ""
echo "Commands:"
echo "  Watch progress:  tmux attach -t eegpt_fast"
echo "  Check logs:      tail -f $LOG_FILE"
echo "  Monitor GPU:     watch -n 1 nvidia-smi"
echo ""
echo "Expected speed: ~1-2 seconds per iteration (not 124s!)"