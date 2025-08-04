#!/bin/bash
# Professional launch script for EEGPT training
# Uses pure PyTorch due to Lightning 2.5.2 dataloader hang bug

set -e

# Navigate to experiment directory
cd "$(dirname "$0")"

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Create logs directory
mkdir -p logs

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "Starting EEGPT Linear Probe Training"
echo "===================================="
echo "Log file: $LOG_FILE"
echo ""

# Launch training in tmux for monitoring
tmux new-session -d -s eegpt_training \
    "../../.venv/bin/python train_pytorch_stable.py 2>&1 | tee $LOG_FILE"

echo "Training launched in tmux session 'eegpt_training'"
echo ""
echo "Commands:"
echo "  Watch live:  tmux attach -t eegpt_training"
echo "  Detach:      Ctrl+B then D"
echo "  Kill:        tmux kill-session -t eegpt_training"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE | grep -E 'Epoch|AUROC'"