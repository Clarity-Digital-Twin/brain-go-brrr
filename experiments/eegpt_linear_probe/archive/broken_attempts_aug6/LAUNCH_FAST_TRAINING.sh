#\!/bin/bash
# LAUNCH FAST 4S TRAINING - THIS FUCKING WORKS

set -e

echo "=========================================="
echo "🚀 LAUNCHING FAST 4S TRAINING"
echo "=========================================="

# Kill any existing sessions
echo "Cleaning up old sessions..."
tmux kill-session -t fast_4s 2>/dev/null || true

# Set environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data

# Navigate to project root
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

# Check cache exists
if [ -d "$BGB_DATA_ROOT/cache_4s_working" ]; then
    echo "✅ Cache directory exists"
    echo "   Files: $(find $BGB_DATA_ROOT/cache_4s_working -name "*.pt" | wc -l)"
else
    echo "❌ Cache directory not found\!"
    exit 1
fi

# Launch in tmux
echo ""
echo "Starting training in tmux session 'fast_4s'..."
tmux new-session -d -s fast_4s \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr && \
     /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python \
     experiments/eegpt_linear_probe/FAST_4S_TRAINER.py 2>&1 | tee logs/fast_4s_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "✅ TRAINING LAUNCHED\!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  tmux attach -t fast_4s"
echo ""
echo "Check GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "EXPECTED:"
echo "  - Iteration time: <2 seconds"
echo "  - Target AUROC: 0.869"
echo "  - Training time: 3-4 hours"
echo ""
echo "If iteration time >5s, cache is NOT working\!"
echo "=========================================="
