#!/bin/bash
set -e

# Final launch script with all fixes and monitoring
PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
cd "$PROJECT_ROOT"

# Set up environment
export BGB_DATA_ROOT="$PROJECT_ROOT/data"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export EEGPT_CONFIG=configs/tuab_stable.yaml

# Set PyTorch optimizations
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0

LOG_FILE="$PROJECT_ROOT/logs/final_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================="
echo "FINAL EEGPT TRAINING WITH ALL FIXES"
echo "=========================================="
echo "Config: $EEGPT_CONFIG"
echo "Log file: $LOG_FILE"
echo "Working directory: $(pwd)"
echo ""
echo "Key fixes applied:"
echo "  ✓ Custom collate for variable channel counts"
echo "  ✓ Identity normalization (no bad stats)"
echo "  ✓ FP32 precision for stability"
echo "  ✓ Gradient clipping enabled"
echo "  ✓ NaN detection safeguards"
echo "  ✓ Cached dataset with proper index"
echo ""

# Kill any existing sessions
tmux kill-session -t eegpt_final 2>/dev/null || true

# Launch training in tmux
tmux new-session -d -s eegpt_final \
    "cd $PROJECT_ROOT && $PROJECT_ROOT/.venv/bin/python experiments/eegpt_linear_probe/train_enhanced.py 2>&1 | tee $LOG_FILE"

echo "Training launched in tmux session 'eegpt_final'"
echo ""
echo "Monitor with:"
echo "  tmux attach -t eegpt_final"
echo "  tail -f $LOG_FILE"
echo ""

# Launch monitoring in separate tmux panes
tmux split-window -t eegpt_final -v "watch -n 10 'nvidia-smi | grep python'"
tmux split-window -t eegpt_final -h "watch -n 30 'tail -20 $LOG_FILE | grep -E \"Epoch|loss|step\" || echo \"Waiting for training...\"'"

echo "Monitoring panels created in tmux session"
echo "=========================================="