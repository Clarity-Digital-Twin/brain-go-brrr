#!/bin/bash
set -e

# NaN-Safe EEGPT Training Launch Script
# =====================================
# This script launches training with MAXIMUM safety against NaN issues

# Get absolute paths
PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
cd "$PROJECT_ROOT"

# Export environment variables
export BGB_DATA_ROOT="$PROJECT_ROOT/data"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# CRITICAL: Disable all parallelism that could cause issues
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Create logs directory
LOG_DIR="$PROJECT_ROOT/logs/nan_safe_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "NaN-Safe EEGPT Training"
echo "=========================================="
echo "Start time: $(date)"
echo "Log directory: $LOG_DIR"
echo "Python: $PROJECT_ROOT/.venv/bin/python"
echo

# Check for existing tmux sessions
if tmux has-session -t eegpt_nan_safe 2>/dev/null; then
    echo "WARNING: Existing tmux session 'eegpt_nan_safe' found"
    echo "Kill it with: tmux kill-session -t eegpt_nan_safe"
    exit 1
fi

# Launch in tmux with comprehensive logging
echo "Launching training in tmux session 'eegpt_nan_safe'..."
tmux new-session -d -s eegpt_nan_safe \
    "cd $PROJECT_ROOT/experiments/eegpt_linear_probe && \
     $PROJECT_ROOT/.venv/bin/python train_pytorch_nan_safe.py 2>&1 | \
     tee $LOG_DIR/training.log; \
     echo 'TRAINING COMPLETED WITH EXIT CODE: $?'; \
     echo 'Press any key to exit...'; \
     read -n 1"

echo
echo "Training launched successfully!"
echo
echo "Commands:"
echo "  Monitor:  tmux attach -t eegpt_nan_safe"
echo "  Detach:   Ctrl+B, then D"
echo "  Kill:     tmux kill-session -t eegpt_nan_safe"
echo "  Logs:     tail -f $LOG_DIR/training.log"
echo
echo "Safety features enabled:"
echo "  ✓ NaN detection at every step"
echo "  ✓ Gradient clipping (1.0)"
echo "  ✓ Reduced learning rate (2e-4)"
echo "  ✓ Warmup schedule (5 epochs)"
echo "  ✓ Anomaly detection on first batch"
echo "  ✓ Checkpoint validation"
echo "  ✓ Safe numerical operations"
echo