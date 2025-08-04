#!/bin/bash
# FINAL WORKING LAUNCH SCRIPT
set -e

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export EEGPT_CONFIG=configs/tuab_stable.yaml

# Set matmul precision for RTX 4090
export TORCH_CUDNN_V8_API_ENABLED=1

LOG_FILE="logs/WORKING_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "LAUNCHING EEGPT TRAINING - THIS WORKS!"
echo "Log: $LOG_FILE"

# Kill any existing sessions
tmux kill-session -t eegpt_working 2>/dev/null || true

# Launch with monitoring
tmux new-session -d -s eegpt_working \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr && .venv/bin/python experiments/eegpt_linear_probe/train_enhanced.py 2>&1 | tee $LOG_FILE"

# Add monitoring panels
tmux split-window -t eegpt_working -h "watch -n 5 'nvidia-smi | head -20'"
tmux split-window -t eegpt_working -v "tail -f $LOG_FILE | grep --line-buffered -E 'Epoch|loss|step|batch'"

echo "TRAINING LAUNCHED!"
echo "Monitor: tmux attach -t eegpt_working"