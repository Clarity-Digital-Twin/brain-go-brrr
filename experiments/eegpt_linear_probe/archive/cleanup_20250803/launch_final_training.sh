#!/bin/bash
# Final EEGPT training launch with conservative memory settings

set -e

echo "=================================================="
echo "LAUNCHING FINAL EEGPT ABNORMALITY DETECTION TRAINING"
echo "=================================================="
echo "Configuration:"
echo "  ✓ Batch size: 32 (with gradient accumulation = 128 effective)"
echo "  ✓ Workers: 2"
echo "  ✓ Mixed precision: disabled for stability"
echo "  ✓ Memory monitoring enabled"
echo "=================================================="

# Clear any existing sessions
tmux kill-session -t eegpt_final 2>/dev/null || true

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Create log directory
LOG_DIR="logs/final_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# First, update the config to use smaller batch size
cp experiments/eegpt_linear_probe/configs/tuab_enhanced_config.yaml $LOG_DIR/config.yaml
sed -i 's/batch_size: 64/batch_size: 32/g' $LOG_DIR/config.yaml
sed -i 's/accumulate_grad_batches: 2/accumulate_grad_batches: 4/g' $LOG_DIR/config.yaml

# Set environment for better debugging
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Launch training
echo "Starting training in tmux session 'eegpt_final'..."
tmux new -d -s eegpt_final \
    "cd $(pwd) && \
     echo '=== Final Training Started ===' && \
     echo 'GPU monitoring active...' && \
     watch -n 10 'nvidia-smi | head -20' &> $LOG_DIR/gpu_monitor.log & \
     GPU_PID=\$! && \
     uv run python experiments/eegpt_linear_probe/train_enhanced.py 2>&1 | tee $LOG_DIR/training.log && \
     echo '=== Training completed successfully! ===' || \
     echo '=== Training failed - check logs ===' && \
     kill \$GPU_PID 2>/dev/null || true"

echo "=================================================="
echo "Training launched successfully!"
echo ""
echo "Monitor training:"
echo "  tmux attach -t eegpt_final"
echo ""
echo "View logs:"
echo "  tail -f $LOG_DIR/training.log"
echo ""
echo "Check GPU:"
echo "  watch nvidia-smi"
echo ""
echo "If OOM occurs, reduce batch_size in:"
echo "  $LOG_DIR/config.yaml"
echo "=================================================="

# Also start a separate monitor to track progress
(
    sleep 30
    echo ""
    echo "Quick status check:"
    if tmux has-session -t eegpt_final 2>/dev/null; then
        echo "✓ Training is running"
        echo "Latest log entries:"
        tail -5 $LOG_DIR/training.log 2>/dev/null || echo "Waiting for logs..."
    else
        echo "✗ Training session ended"
        echo "Check logs for errors:"
        echo "  grep -i error $LOG_DIR/training.log"
    fi
) &