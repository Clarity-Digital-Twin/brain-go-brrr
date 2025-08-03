#!/bin/bash
# Launch EEGPT training using pre-built cache for FAST startup

set -e  # Exit on error

echo "=================================================="
echo "LAUNCHING CACHED EEGPT TRAINING"
echo "=================================================="
echo "✓ Using pre-built cache for instant loading"
echo "✓ 8s windows @ 256Hz (matching cache)"
echo "✓ Proper configuration for fast training"
echo "=================================================="

# Create log directory
LOG_DIR="logs/cached_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Check if cache exists
CACHE_DIR="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuab_enhanced"
CACHE_COUNT=$(find $CACHE_DIR -name "*.pkl" 2>/dev/null | wc -l)
echo "Found $CACHE_COUNT cache files"

if [ $CACHE_COUNT -eq 0 ]; then
    echo "ERROR: No cache files found! Run scripts/build_tuab_cache.py first"
    exit 1
fi

# Kill any existing sessions
tmux kill-session -t eegpt_cached 2>/dev/null || true

# Set environment for cached dataset
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export EEGPT_CONFIG="configs/tuab_cached.yaml"

# Launch training
echo "Starting cached training in tmux session 'eegpt_cached'..."
tmux new -d -s eegpt_cached \
    "cd $(pwd) && \
     echo 'Using cached configuration: $EEGPT_CONFIG' && \
     echo 'Starting training with pre-built cache...' && \
     uv run python experiments/eegpt_linear_probe/train_enhanced.py 2>&1 | tee $LOG_DIR/training.log"

echo "=================================================="
echo "Cached training started!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t eegpt_cached"
echo ""
echo "Check logs:"
echo "  tail -f $LOG_DIR/training.log"
echo ""
echo "Training should start immediately with cached data!"
echo "=================================================="