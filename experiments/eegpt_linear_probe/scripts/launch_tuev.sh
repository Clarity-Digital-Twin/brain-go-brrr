#!/bin/bash
# Launch TUEV training

set -e

PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
cd "$PROJECT_ROOT/experiments/eegpt_linear_probe"

export BGB_DATA_ROOT="$PROJECT_ROOT/data"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "Starting TUEV training..."
echo "WARNING: Line 96 in train_tuev.py has wrong dimensions (2048 instead of 30720)!"

$PROJECT_ROOT/.venv/bin/python train_tuev.py