#!/bin/bash
set -e
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export EEGPT_CONFIG=configs/tuab_stable.yaml

LOG_FILE="logs/FINAL_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "LAUNCHING FINAL TRAINING"
echo "This will start training immediately!"

# Simple launch in background
nohup .venv/bin/python experiments/eegpt_linear_probe/train_enhanced.py > $LOG_FILE 2>&1 &
PID=$!

echo "Training PID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor with: tail -f $LOG_FILE | grep -E 'Epoch|loss|step'"