#!/bin/bash
# Launch TUEV event detection training in tmux session

set -e

# Configuration
PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/eegpt_linear_probe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$EXPERIMENT_DIR/output/tuev_${TIMESTAMP}"
LOG_FILE="$EXPERIMENT_DIR/logs/tuev_training_${TIMESTAMP}.log"

# Setup environment
export BGB_DATA_ROOT="$PROJECT_ROOT/data"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create directories
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

echo "Launching TUEV training..."
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_FILE"
echo "Monitor with: tmux attach -t tuev_training"

# Launch in tmux
tmux new-session -d -s tuev_training \
    "cd $EXPERIMENT_DIR && \
     $PROJECT_ROOT/.venv/bin/python train_tuev.py \
     --config configs/tuev.yaml \
     --output_dir $OUTPUT_DIR \
     2>&1 | tee $LOG_FILE"

echo "Training launched successfully!"
