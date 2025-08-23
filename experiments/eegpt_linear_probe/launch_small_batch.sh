#!/bin/bash
set -e

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/small_batch_${TIMESTAMP}.log"
OUTPUT_DIR="output/tuab_small_${TIMESTAMP}"

echo "Starting with batch_size=64 (instead of 256)..."
echo "This should use 4x less memory"

# Run with smaller batch size
tmux new-session -d -s tuab_small \
    "\
     /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_tuab.py \
     --config configs/tuab.yaml \
     --output_dir ${OUTPUT_DIR} \
     --batch_size 64 \
     2>&1 | tee ${LOG_FILE}"

echo "Started in tmux session 'tuab_small'"
echo "Monitor with: tmux attach -t tuab_small"
