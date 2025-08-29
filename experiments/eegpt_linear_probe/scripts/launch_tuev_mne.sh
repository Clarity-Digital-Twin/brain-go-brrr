#!/bin/bash
# Launch TUEV training with MNE+Autoreject preprocessing

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
DATA_ROOT="${BGB_DATA_ROOT:-$PROJECT_ROOT/data}"

# Training parameters
BATCH_SIZE=${BATCH_SIZE:-128}  # Paper uses 500, but reduced for memory safety
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$EXPERIMENT_DIR/output/tuev_mne_$TIMESTAMP"
LOG_FILE="$EXPERIMENT_DIR/logs/tuev_mne_$TIMESTAMP.log"

echo "=============================================="
echo "Launching TUEV Training with MNE Preprocessing"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Batch size: $BATCH_SIZE"
echo ""
echo "Target metrics (Table 13):"
echo "  - Balanced Accuracy: 62.32%"
echo "  - Weighted F1: 81.87%"
echo "  - Cohen's Kappa: 0.635"
echo "=============================================="

# Create directories
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Virtual environment activated"
fi

# Change to experiment directory
cd "$EXPERIMENT_DIR"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Check cache exists
CACHE_DIR="$DATA_ROOT/cache/tuev_mne_v2"
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: MNE cache not found at $CACHE_DIR"
    echo "Please run: ./scripts/launch_tuev_cache.sh first"
    exit 1
fi

# Launch training
echo ""
echo "Starting training..."
echo "To monitor: tail -f $LOG_FILE"
echo ""

# Use tmux or direct execution based on preference
if command -v tmux &> /dev/null; then
    # Launch in tmux session
    SESSION_NAME="tuev_mne_training"

    # Kill existing session if it exists
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

    # Start new session
    tmux new-session -d -s "$SESSION_NAME" \
        "cd $EXPERIMENT_DIR && \
         python train_tuev_mne.py \
            --config configs/tuev.yaml \
            --output-dir $OUTPUT_DIR \
            --cache-dir $CACHE_DIR \
            2>&1 | tee $LOG_FILE"

    echo "Training launched in tmux session: $SESSION_NAME"
    echo "To attach: tmux attach -t $SESSION_NAME"
    echo "To detach: Ctrl+B, then D"
else
    # Direct execution
    python train_tuev_mne.py \
        --config configs/tuev.yaml \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        2>&1 | tee "$LOG_FILE"
fi
