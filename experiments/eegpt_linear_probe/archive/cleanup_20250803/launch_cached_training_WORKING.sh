#!/bin/bash
# WORKING CACHED TRAINING - USES CACHE PROPERLY

set -e

echo "========================================"
echo "CACHED EEGPT TRAINING - FAST LOADING"
echo "========================================"
echo "Using TUABCachedDataset for instant loading"
echo "Cache index: data/cache/tuab_index.json"
echo "========================================"

# Kill any existing sessions
tmux kill-session -t cached_train 2>/dev/null || true

# Clear GPU
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Create log directory
LOG_DIR="logs/cached_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Launch training
tmux new -d -s cached_train \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr && \
     uv run python experiments/eegpt_linear_probe/train_enhanced.py \
         2>&1 | tee $LOG_DIR/training.log"

echo "Training started in tmux session 'cached_train'"
echo "Monitor with: tmux attach -t cached_train"
echo "Logs at: $LOG_DIR/training.log"
echo ""

# Check after 20 seconds
sleep 20
if tmux has-session -t cached_train 2>/dev/null; then
    echo "Training running successfully!"
    echo "First log lines:"
    head -30 $LOG_DIR/training.log | grep -E "(windows|Epoch|GPU|Loading)" || echo "Waiting for output..."
else
    echo "Training failed! Check logs:"
    tail -50 $LOG_DIR/training.log
fi