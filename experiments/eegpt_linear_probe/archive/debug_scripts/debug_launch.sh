#!/bin/bash
set -e

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1  # Better error messages

LOG_FILE="logs/debug_training_$(date +%Y%m%d_%H%M%S).log"
OUTPUT_DIR="output/tuab_debug_$(date +%Y%m%d_%H%M%S)"

echo "Starting debug training with CUDA_LAUNCH_BLOCKING=1..."
echo "This will give us better error messages if it crashes"

# Run directly (not in tmux) so we can see the error
timeout 7200 /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_tuab.py \
    --config configs/tuab.yaml \
    --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${LOG_FILE}

echo "Exit code: $?"
