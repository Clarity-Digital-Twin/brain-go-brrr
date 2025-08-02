#!/bin/bash
# Launch robust EEGPT training with memory management and error recovery

set -e  # Exit on error

echo "=================================================="
echo "LAUNCHING ROBUST EEGPT TRAINING"
echo "=================================================="
echo "Safety features:"
echo "  ✓ CUDA_LAUNCH_BLOCKING for better error tracking"
echo "  ✓ Gradient accumulation if OOM"
echo "  ✓ Automatic checkpoint recovery"
echo "  ✓ Memory monitoring"
echo "=================================================="

# Create log directory
LOG_DIR="logs/robust_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Check if previous training is running
if tmux has-session -t eegpt_robust 2>/dev/null; then
    echo "Previous training session found. Killing it..."
    tmux kill-session -t eegpt_robust
fi

# Set environment variables for better debugging
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Launch training in tmux with memory monitoring
echo "Starting robust training in tmux session 'eegpt_robust'..."
tmux new -d -s eegpt_robust \
    "cd $(pwd) && \
     echo 'Starting training with memory monitoring...' && \
     watch -n 5 'nvidia-smi | head -20' &> $LOG_DIR/gpu_monitor.log & \
     GPU_PID=\$! && \
     uv run python experiments/eegpt_linear_probe/train_enhanced.py 2>&1 | tee $LOG_DIR/training.log; \
     kill \$GPU_PID 2>/dev/null || true"

echo "=================================================="
echo "Training started with robust configuration!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t eegpt_robust"
echo ""
echo "Check logs:"
echo "  tail -f $LOG_DIR/training.log"
echo ""
echo "GPU usage:"
echo "  tail -f $LOG_DIR/gpu_monitor.log"
echo ""
echo "If it crashes, check:"
echo "  grep -i error $LOG_DIR/training.log"
echo "=================================================="