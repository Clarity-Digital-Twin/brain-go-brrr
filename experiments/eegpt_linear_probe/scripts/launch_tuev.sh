#!/bin/bash
set -euo pipefail  # Exit on error, undefined variables, and pipe failures
# Launch TUEV event detection training in tmux session

# Configuration
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT"

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

# Guard: ensure no conflicting tmux session is running
if tmux has-session -t tuev_training 2>/dev/null; then
    echo "❌ ERROR: tmux session 'tuev_training' already exists!" | tee -a "$LOG_FILE"
    echo "Kill it with: tmux kill-session -t tuev_training" | tee -a "$LOG_FILE"
    exit 1
fi

# Launch in tmux with automatic restart on crash
tmux new-session -d -s tuev_training \
    "cd $EXPERIMENT_DIR && \
     while true; do \
         echo 'Starting/Resuming TUEV training at \$(date)' | tee -a $LOG_FILE; \
         $PROJECT_ROOT/.venv/bin/python train_tuev.py \
         --config configs/tuev.yaml \
         --output_dir $OUTPUT_DIR \
         2>&1 | tee -a $LOG_FILE; \
         EXIT_CODE=\$?; \
         if [ \$EXIT_CODE -eq 0 ]; then \
             echo 'Training completed successfully!' | tee -a $LOG_FILE; \
             break; \
         else \
             echo 'Training crashed with code '\$EXIT_CODE'! Restarting in 10 seconds...' | tee -a $LOG_FILE; \
             sleep 10; \
         fi; \
     done"

echo "Training launched successfully!"
