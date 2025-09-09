#!/bin/bash
# Launch TUEV training with paper parity (23 channels + learned mapper)
# Target: 62.32% BAC as reported in EEGPT paper Table 3
# Enhanced with auto-recovery, tmux support, and checkpoint resumption

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
DATA_ROOT="${BGB_DATA_ROOT:-$PROJECT_ROOT/data}"
BGB_CACHE_DIR="${BGB_CACHE_DIR:-$DATA_ROOT/cache}"

# Training parameters
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$EXPERIMENT_DIR/output/tuev_paper_parity_$TIMESTAMP"
LOG_FILE="$EXPERIMENT_DIR/logs/tuev_paper_parity_$TIMESTAMP.log"
RUN_NAME="tuev_paper_parity_$TIMESTAMP"

echo "========================================="
echo "TUEV Paper Parity Training (Enhanced)"
echo "========================================="
echo "Run name: $RUN_NAME"
echo "Data root: $DATA_ROOT"
echo "Cache dir (base): $BGB_CACHE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Config: configs/tuev_paper_parity.yaml"
echo "Expected: 62.32% BAC (Table 13)"
echo "========================================="
echo ""

# Create directories
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

# Check if cache exists
CACHE_TRAIN="$BGB_CACHE_DIR/tuev_23ch_paper_parity/train"
CACHE_EVAL="$BGB_CACHE_DIR/tuev_23ch_paper_parity/eval"

if [ ! -d "$CACHE_TRAIN" ] || [ ! -d "$CACHE_EVAL" ]; then
    echo "ERROR: 23-channel cache not found!"
    echo "Expected directories:"
    echo "  - $CACHE_TRAIN"
    echo "  - $CACHE_EVAL"
    echo ""
    echo "Please build the cache first with:"
    echo "  ./scripts/build_tuev_23ch_cache.sh"
    exit 1
fi

# Verify cache has content
TRAIN_WINDOWS=$(find "$CACHE_TRAIN" -name "window_*.pt" 2>/dev/null | wc -l)
EVAL_WINDOWS=$(find "$CACHE_EVAL" -name "window_*.pt" 2>/dev/null | wc -l)
echo "Cache found: $TRAIN_WINDOWS train windows, $EVAL_WINDOWS eval windows"

if [ "$TRAIN_WINDOWS" -eq 0 ] || [ "$EVAL_WINDOWS" -eq 0 ]; then
    echo "ERROR: Cache directories exist but are empty!"
    exit 1
fi

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Virtual environment activated"
fi

# Change to experiment directory
cd "$EXPERIMENT_DIR"

# Set Python path and environment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT="$DATA_ROOT"
export BGB_CACHE_DIR="$BGB_CACHE_DIR"

# Auto-recovery settings
MAX_RETRIES=${MAX_RETRIES:-10}
RETRY_DELAY=${RETRY_DELAY:-30}
ENABLE_RECOVERY=${ENABLE_RECOVERY:-true}

echo ""
echo "Starting training..."
echo "Auto-recovery: $ENABLE_RECOVERY (max retries: $MAX_RETRIES)"
echo "To monitor: tail -f $LOG_FILE"
echo ""

# Function to run training with recovery
run_with_recovery() {
    local retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        echo "[$(date)] Training attempt $((retry_count + 1))/$MAX_RETRIES"

        # Find latest checkpoint for resume
        RESUME_ARG=""
        LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint_*.pt" -type f 2>/dev/null | sort -V | tail -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo "Found checkpoint: $LATEST_CHECKPOINT"
            RESUME_ARG="--resume $LATEST_CHECKPOINT"
        fi

        # Run training with proper argument names
        uv run python train_tuev_mne.py \
            --config configs/tuev_paper_parity.yaml \
            --output-dir "$OUTPUT_DIR" \
            --cache-dir "$BGB_CACHE_DIR" \
            $RESUME_ARG \
            2>&1 | tee -a "$LOG_FILE"

        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$(date)] Training completed successfully!"
            return 0
        else
            echo "[$(date)] Training failed with exit code $EXIT_CODE"
            
            # Check for specific errors in log
            if grep -q "CUDA out of memory" "$LOG_FILE"; then
                echo "GPU memory error detected. Consider reducing batch size."
            fi
            if grep -q "FileNotFoundError.*cache" "$LOG_FILE"; then
                echo "Cache access error. Verify cache directories are accessible."
            fi
            
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                echo "Waiting $RETRY_DELAY seconds before retry..."
                sleep $RETRY_DELAY
            fi
        fi
    done
    echo "Training failed after $MAX_RETRIES attempts"
    return 1
}

# Use tmux or direct execution based on preference
if command -v tmux &> /dev/null; then
    # Launch in tmux session
    SESSION_NAME="tuev_paper_parity"

    # Kill existing session if it exists
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

    if [ "$ENABLE_RECOVERY" = "true" ]; then
        # Start new session with recovery wrapper
        echo "Launching in tmux with auto-recovery enabled..."
        tmux new-session -d -s "$SESSION_NAME" \
            "bash -lc 'cd \"$EXPERIMENT_DIR\" && export BGB_DATA_ROOT=\"$DATA_ROOT\" && export BGB_CACHE_DIR=\"$BGB_CACHE_DIR\" && export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" && export OUTPUT_DIR=\"$OUTPUT_DIR\" && export MAX_RETRIES=\"$MAX_RETRIES\" && export RETRY_DELAY=\"$RETRY_DELAY\" && export LOG_FILE=\"$LOG_FILE\" && $(declare -f run_with_recovery); run_with_recovery'"
    else
        # Single-run command
        echo "Launching in tmux (single run)..."
        tmux new-session -d -s "$SESSION_NAME" \
            "bash -lc 'cd \"$EXPERIMENT_DIR\" && export BGB_DATA_ROOT=\"$DATA_ROOT\" && export BGB_CACHE_DIR=\"$BGB_CACHE_DIR\" && export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" && uv run python train_tuev_mne.py --config configs/tuev_paper_parity.yaml --output-dir \"$OUTPUT_DIR\" --cache-dir \"$BGB_CACHE_DIR\" 2>&1 | tee \"$LOG_FILE\"'"
    fi

    echo ""
    echo "========================================="
    echo "Training launched in tmux session: $SESSION_NAME"
    echo "Commands:"
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Detach: Ctrl+B, then D"
    echo "  Kill:   tmux kill-session -t $SESSION_NAME"
    echo "  Monitor: tail -f $LOG_FILE"
    echo "========================================="
else
    # Direct execution (no tmux)
    echo "tmux not found, running directly..."
    if [ "$ENABLE_RECOVERY" = "true" ]; then
        run_with_recovery
    else
        uv run python train_tuev_mne.py \
            --config configs/tuev_paper_parity.yaml \
            --output-dir "$OUTPUT_DIR" \
            --cache-dir "$BGB_CACHE_DIR" \
            2>&1 | tee "$LOG_FILE"
    fi
fi

echo ""
echo "========================================="
echo "Script complete!"
echo "Log saved to: $LOG_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "========================================="