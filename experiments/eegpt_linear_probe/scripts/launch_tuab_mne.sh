#!/bin/bash
# Launch TUAB training with MNE+Autoreject preprocessing

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
DATA_ROOT="${BGB_DATA_ROOT:-$PROJECT_ROOT/data}"

# Training parameters
BATCH_SIZE=${BATCH_SIZE:-256}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$EXPERIMENT_DIR/output/tuab_mne_$TIMESTAMP"
LOG_FILE="$EXPERIMENT_DIR/logs/tuab_mne_$TIMESTAMP.log"

echo "=============================================="
echo "Launching TUAB Training with MNE Preprocessing"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Cache dir: $DATA_ROOT/cache/tuab_mne_v2"
echo "Model checkpoint: $DATA_ROOT/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
echo "Batch size: $BATCH_SIZE"
echo ""
echo "Expected improvement: 56% → 75-87% AUROC"
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

# Set Python path and environment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT="$DATA_ROOT"  # Ensure config can resolve ${BGB_DATA_ROOT}

# Check cache exists
CACHE_DIR="$DATA_ROOT/cache/tuab_mne_v2"
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: MNE cache not found at $CACHE_DIR"
    echo "Please run: ./scripts/build_mne_cache.sh first"
    exit 1
fi

# Auto-recovery settings
MAX_RETRIES=${MAX_RETRIES:-10}
RETRY_DELAY=${RETRY_DELAY:-30}
ENABLE_RECOVERY=${ENABLE_RECOVERY:-true}

# Launch training
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
        
        # Run training
        uv run python train_tuab_mne.py \
            --config configs/tuab.yaml \
            --output-dir "$OUTPUT_DIR" \
            --cache-dir "$CACHE_DIR" \
            $RESUME_ARG \
            2>&1 | tee -a "$LOG_FILE"
        
        EXIT_CODE=${PIPESTATUS[0]}
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$(date)] Training completed successfully!"
            return 0
        else
            echo "[$(date)] Training failed with exit code $EXIT_CODE"
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
    SESSION_NAME="tuab_mne_training"

    # Kill existing session if it exists
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

    if [ "$ENABLE_RECOVERY" = "true" ]; then
        # Start new session with recovery wrapper
        tmux new-session -d -s "$SESSION_NAME" \
            "bash -lc 'cd \"$EXPERIMENT_DIR\" && export BGB_DATA_ROOT=\"$DATA_ROOT\" && export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" && export OUTPUT_DIR=\"$OUTPUT_DIR\" && export CACHE_DIR=\"$CACHE_DIR\" && export MAX_RETRIES=\"$MAX_RETRIES\" && export RETRY_DELAY=\"$RETRY_DELAY\" && $(declare -f run_with_recovery); run_with_recovery'"
    else
        # Original single-run command
        tmux new-session -d -s "$SESSION_NAME" \
            "bash -lc 'cd \"$EXPERIMENT_DIR\" && export BGB_DATA_ROOT=\"$DATA_ROOT\" && export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\" && uv run python train_tuab_mne.py --config configs/tuab.yaml --output-dir \"$OUTPUT_DIR\" --cache-dir \"$CACHE_DIR\" 2>&1 | tee \"$LOG_FILE\"'"
    fi

    echo "Training launched in tmux session: $SESSION_NAME"
    echo "To attach: tmux attach -t $SESSION_NAME"
    echo "To detach: Ctrl+B, then D"
else
    # Direct execution
    if [ "$ENABLE_RECOVERY" = "true" ]; then
        run_with_recovery
    else
        uv run python train_tuab_mne.py \
            --config configs/tuab.yaml \
            --output-dir "$OUTPUT_DIR" \
            --cache-dir "$CACHE_DIR" \
            2>&1 | tee "$LOG_FILE"
    fi
fi
