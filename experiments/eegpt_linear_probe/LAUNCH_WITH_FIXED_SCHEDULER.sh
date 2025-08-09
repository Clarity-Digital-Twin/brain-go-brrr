#!/bin/bash
# Launch training with FIXED OneCycleLR scheduler (per-batch stepping)

set -e

# Configuration
PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
cd "$PROJECT_ROOT/experiments/eegpt_linear_probe"

# Environment
export BGB_DATA_ROOT="$PROJECT_ROOT/data"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Log file with timestamp
LOG_FILE="logs/fixed_scheduler_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "==============================================="
echo "LAUNCHING TRAINING WITH FIXED SCHEDULER"
echo "==============================================="
echo "Config: configs/tuab_4s_paper_aligned.yaml"
echo "Script: train_paper_aligned.py (with per-batch scheduler stepping)"
echo "Log: $LOG_FILE"
echo ""
echo "SCHEDULER FIX APPLIED:"
echo "  ✅ OneCycleLR steps per batch (not per epoch)"
echo "  ✅ Total steps = batches_per_epoch * num_epochs"
echo "  ✅ LR will change every batch: 0.00012 → 0.003 → 0.000003"
echo "==============================================="

# Launch in tmux for monitoring
tmux new-session -d -s eegpt_fixed \
    "$PROJECT_ROOT/.venv/bin/python train_paper_aligned.py \
    --config configs/tuab_4s_paper_aligned.yaml \
    2>&1 | tee $LOG_FILE"

echo ""
echo "✅ Training launched in tmux session 'eegpt_fixed'"
echo ""
echo "MONITOR WITH:"
echo "  tmux attach -t eegpt_fixed"
echo ""
echo "CHECK SCHEDULER:"
echo "  tail -f $LOG_FILE | grep 'LR:'"
echo ""
echo "VERIFY LR IS CHANGING:"
echo "  Watch for LR values to increase during warmup (first ~10% of training)"
echo "  Then stay at max_lr (0.003) for ~40% of training"
echo "  Then decrease during annealing (last ~50% of training)"
echo ""
echo "If LR stays constant, scheduler is still broken!"