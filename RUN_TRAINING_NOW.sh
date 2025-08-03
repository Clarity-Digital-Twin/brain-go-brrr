#!/bin/bash
# SIMPLE FUCKING TRAINING THAT WORKS

set -e

echo "🚀 STARTING SIMPLE TRAINING - NO CACHE BULLSHIT"
echo "============================================"

# Kill any existing sessions
tmux kill-session -t simple_train 2>/dev/null || true

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Create log directory
LOG_DIR="logs/simple_working_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Launch training
tmux new -d -s simple_train \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr && \
     uv run python experiments/eegpt_linear_probe/SIMPLE_WORKING_TRAIN.py 2>&1 | tee $LOG_DIR/training.log"

echo "✅ Training started in tmux session 'simple_train'"
echo ""
echo "Monitor with: tmux attach -t simple_train"
echo "Logs at: $LOG_DIR/training.log"
echo ""

# Wait and check
sleep 10
if tmux has-session -t simple_train 2>/dev/null; then
    echo "✅ Still running after 10 seconds - looking good!"
else
    echo "❌ Training crashed! Check logs:"
    tail -50 $LOG_DIR/training.log
fi