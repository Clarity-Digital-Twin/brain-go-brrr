#!/bin/bash
set -e

# Fixed launch script for EEGPT training
# Resolves PyTorch Lightning hang with deterministic=True + limit_train_batches

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr

# Environment
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export EEGPT_CONFIG=configs/tuab_stable.yaml

# Create log directory
mkdir -p logs

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/rescue_${TIMESTAMP}.log"

echo "Starting EEGPT training with fixes applied..."
echo "Config: $EEGPT_CONFIG"
echo "Log file: $LOG_FILE"

# Launch in tmux
tmux new-session -d -s eegpt_training \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe && ../../.venv/bin/python train_enhanced.py 2>&1 | tee ../../$LOG_FILE"

echo "Training launched in tmux session 'eegpt_training'"
echo "Monitor with: tmux attach -t eegpt_training"
echo "View logs: tail -f $LOG_FILE"