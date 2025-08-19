#!/bin/bash
# Launch TUAB training

set -e

PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
cd "$PROJECT_ROOT/experiments/eegpt_linear_probe"

export BGB_DATA_ROOT="$PROJECT_ROOT/data"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "Starting TUAB training..."
echo "WARNING: Line 67 in train_tuab.py has averaging bug!"
echo "WARNING: Config has wrong input_dim (512 instead of 63488)!"

$PROJECT_ROOT/.venv/bin/python train_tuab.py --config configs/tuab.yaml