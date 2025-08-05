#!/bin/bash
set -e

# Launch paper-aligned training with 4-second windows
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export EEGPT_CONFIG=configs/tuab_4s_paper_target.yaml

LOG_FILE="logs/paper_aligned_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Starting EEGPT paper-aligned training (4-second windows)..."
echo "Target AUROC: 0.869 ± 0.005 (paper performance)"
echo "Config: $EEGPT_CONFIG"
echo "Log file: $LOG_FILE"

# Launch in tmux for robustness
tmux new-session -d -s eegpt_paper_aligned \
    "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_paper_aligned.py 2>&1 | tee $LOG_FILE"

echo "Training launched in tmux session 'eegpt_paper_aligned'"
echo "Monitor with: tmux attach -t eegpt_paper_aligned"
echo "View logs: tail -f $LOG_FILE"