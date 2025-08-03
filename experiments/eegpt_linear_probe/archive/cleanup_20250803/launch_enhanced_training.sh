#!/bin/bash
# Launch enhanced EEGPT training with all paper-matching improvements

set -e  # Exit on error

echo "=================================================="
echo "LAUNCHING ENHANCED EEGPT TRAINING"
echo "=================================================="
echo "Improvements implemented:"
echo "  ✓ 10-second windows @ 200Hz"
echo "  ✓ Two-layer probe with dropout"
echo "  ✓ Channel adaptation (20→22→19)"
echo "  ✓ Batch size 100"
echo "  ✓ 50 epochs with warmup"
echo "  ✓ Layer decay 0.65"
echo "  ✓ OneCycle LR schedule"
echo "  ✓ 0.1-75Hz filter + 50Hz notch"
echo "=================================================="

# Create log directory
LOG_DIR="logs/enhanced_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Check if previous training is running
if tmux has-session -t eegpt_enhanced 2>/dev/null; then
    echo "Previous training session found. Killing it..."
    tmux kill-session -t eegpt_enhanced
fi

# Launch training in tmux
echo "Starting enhanced training in tmux session 'eegpt_enhanced'..."
tmux new -d -s eegpt_enhanced \
    "cd $(pwd) && \
     uv run python experiments/eegpt_linear_probe/train_enhanced.py 2>&1 | tee $LOG_DIR/training.log"

echo "=================================================="
echo "Training started!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t eegpt_enhanced"
echo ""
echo "Check logs:"
echo "  tail -f $LOG_DIR/training.log"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir logs/"
echo "=================================================="