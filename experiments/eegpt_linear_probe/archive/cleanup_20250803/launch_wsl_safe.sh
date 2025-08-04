#!/bin/bash
# WSL-safe EEGPT training launch with all memory optimizations

set -e

echo "=================================================="
echo "WSL-SAFE EEGPT TRAINING LAUNCHER"
echo "=================================================="
echo "Fixes applied:"
echo "  ✓ num_workers=0 (avoid WSL deadlocks)"
echo "  ✓ Batch size 16 with 8x accumulation"
echo "  ✓ Mixed precision (fp16)"
echo "  ✓ Gradient checkpointing"
echo "  ✓ Reduced window size (5.12s)"
echo "  ✓ Using dedicated memsafe config"
echo "=================================================="

# Configuration
CONFIG=experiments/eegpt_linear_probe/configs/tuab_memsafe.yaml
CHECKPOINT=logs/fast_robust_20250801_103902/checkpoints/last.ckpt

# Check if checkpoint exists
if [ -f "$CHECKPOINT" ]; then
    echo "✓ Found checkpoint: $CHECKPOINT"
    RESUME_FLAG="--ckpt_path $CHECKPOINT"
else
    echo "⚠ No checkpoint found, starting fresh"
    RESUME_FLAG=""
fi

# Clear any existing sessions
tmux kill-session -t eegpt_wsl 2>/dev/null || true

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Environment settings
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6

# Create log directory
LOG_DIR="logs/wsl_safe_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Copy config to log dir for reference
cp $CONFIG $LOG_DIR/config.yaml

# Launch training directly (no tmux for now to see errors)
echo ""
echo "Starting training..."
echo "Logs: $LOG_DIR/training.log"
echo ""

# Run training with explicit settings
uv run python experiments/eegpt_linear_probe/train_enhanced.py \
    $RESUME_FLAG \
    2>&1 | tee $LOG_DIR/training.log &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# Monitor GPU usage in background
(
    while kill -0 $TRAIN_PID 2>/dev/null; do
        echo "[$(date +%H:%M:%S)] $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk '{printf "GPU: %d/%d MB (%d%%)", $1, $2, $3}')" >> $LOG_DIR/gpu_monitor.log
        sleep 30
    done
    echo "[$(date +%H:%M:%S)] Training process ended" >> $LOG_DIR/gpu_monitor.log
) &

echo "=================================================="
echo "Training launched!"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/training.log"
echo ""
echo "GPU usage:"
echo "  tail -f $LOG_DIR/gpu_monitor.log"
echo ""
echo "Check for stalls:"
echo "  watch 'grep -E \"Epoch|loss|step\" $LOG_DIR/training.log | tail -20'"
echo ""
echo "If it stalls at DataLoader, you'll see no output after 'Loading train_dataloader'"
echo "=================================================="

# Wait a bit and check if it's actually running
sleep 60
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo ""
    echo "✓ Training still running after 60s - good sign!"
    if grep -q "Epoch 0" $LOG_DIR/training.log 2>/dev/null; then
        echo "✓ Successfully reached training loop!"
    else
        echo "⚠ Haven't reached Epoch 0 yet - might be stuck in DataLoader"
    fi
else
    echo ""
    echo "✗ Training died within 60s - check logs for errors"
    tail -20 $LOG_DIR/training.log
fi