#!/bin/bash
# Launch TUEV cache building with the deterministic cache builder

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
DATA_ROOT="${BGB_DATA_ROOT:-$PROJECT_ROOT/data}"

# TUEV paths
TUEV_ROOT="$DATA_ROOT/datasets/tuev"
CACHE_DIR="$DATA_ROOT/cache/tuev_mne_v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$EXPERIMENT_DIR/logs/tuev_cache_$TIMESTAMP.log"

echo "=============================================="
echo "Launching TUEV Cache Build in tmux"
echo "=============================================="
echo "TUEV root: $TUEV_ROOT"
echo "Cache dir: $CACHE_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Check if TUEV data exists
if [ ! -d "$TUEV_ROOT/edf" ]; then
    echo "ERROR: TUEV dataset not found at $TUEV_ROOT"
    echo "Please download TUEV first"
    exit 1
fi

# Count files
TRAIN_FILES=$(find "$TUEV_ROOT/edf/train" -name "*.edf" 2>/dev/null | wc -l || echo 0)
EVAL_FILES=$(find "$TUEV_ROOT/edf/eval" -name "*.edf" 2>/dev/null | wc -l || echo 0)
TOTAL_FILES=$((TRAIN_FILES + EVAL_FILES))

echo "Found $TOTAL_FILES EDF files ($TRAIN_FILES train, $EVAL_FILES eval)"
echo ""

# Create log directory
mkdir -p "$EXPERIMENT_DIR/logs"

# Launch in tmux using our deterministic cache builder
tmux new-session -d -s tuev_cache \
    "cd $PROJECT_ROOT && \
     BGB_DATA_ROOT=$DATA_ROOT uv run python experiments/eegpt_linear_probe/mne_integration/cache_builder.py \
     --corpus TUEV \
     --data-root $TUEV_ROOT/edf \
     --cache-dir $CACHE_DIR \
     --split both \
     2>&1 | tee $LOG_FILE"

echo "✓ Cache build launched in tmux session 'tuev_cache'"
echo ""
echo "Commands:"
echo "  Monitor:  tmux attach -t tuev_cache"
echo "  Detach:   Ctrl+B then D"
echo "  Kill:     tmux kill-session -t tuev_cache"
echo ""
echo "Log file: $LOG_FILE"
echo "=============================================="
