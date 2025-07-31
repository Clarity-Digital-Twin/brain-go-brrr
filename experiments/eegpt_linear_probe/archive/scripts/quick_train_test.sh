#!/bin/bash
# Quick training test script for EEGPT linear probe

set -e  # Exit on error

echo "🚀 Starting EEGPT Linear Probe Quick Test"
echo "========================================="

# Set environment
export BGB_DATA_ROOT="/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/data"
export PYTHONUNBUFFERED=1  # Force unbuffered output

# Create necessary directories
mkdir -p checkpoints logs

echo "📊 Configuration:"
echo "- Data root: $BGB_DATA_ROOT"
echo "- Using 2% of training data"
echo "- Using 10% of validation data"
echo "- 1 epoch for quick test"
echo ""

# Start training
echo "🏃 Starting training..."
uv run python train_tuab_probe.py 2>&1 | tee training_quick_$(date +%Y%m%d_%H%M%S).log &
TRAIN_PID=$!

echo "Training started with PID: $TRAIN_PID"
echo ""

# Monitor for a bit
echo "📈 Monitoring progress..."
sleep 10

# Check if still running
if ps -p $TRAIN_PID > /dev/null; then
    echo "✅ Training is running!"
    echo ""
    echo "To monitor:"
    echo "  - Watch logs: tail -f training_quick_*.log"
    echo "  - Check process: ps aux | grep train_tuab_probe"
    echo "  - Run monitor: python monitor_training.py"
    echo "  - TensorBoard: tensorboard --logdir lightning_logs"
else
    echo "❌ Training stopped unexpectedly"
    echo "Check the log file for errors"
fi
