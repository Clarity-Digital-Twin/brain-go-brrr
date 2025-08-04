#!/bin/bash
set -e

# Launch STABLE training with fixes for NaN issues
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export EEGPT_CONFIG="configs/tuab_stable.yaml"  # Use stable config

LOG_FILE="logs/eegpt_stable_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Starting STABLE EEGPT training with NaN fixes..."
echo "Config: $EEGPT_CONFIG"
echo "Key changes for stability:"
echo "  - Using fp32 precision (no mixed precision)"
echo "  - Removed label smoothing for binary classification"
echo "  - Reduced learning rate to 2e-4"
echo "  - Reduced gradient accumulation to 2 (64 effective batch)"
echo "  - Using cosine scheduler instead of onecycle"
echo "Log file: $LOG_FILE"

# Launch in tmux
tmux new-session -d -s eegpt_stable \
    ".venv/bin/python experiments/eegpt_linear_probe/train_enhanced.py 2>&1 | tee $LOG_FILE"

echo ""
echo "Training launched in tmux session 'eegpt_stable'"
echo "Commands:"
echo "  - Watch live: tmux attach -t eegpt_stable"
echo "  - Check status: bash MONITOR_TRAINING.sh"
echo "  - View logs: tail -f $LOG_FILE"
echo "  - Kill if needed: tmux kill-session -t eegpt_stable"