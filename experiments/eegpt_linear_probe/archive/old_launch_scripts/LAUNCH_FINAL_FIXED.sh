#!/bin/bash
# FINAL launch script with ALL scheduler fixes applied
# This version includes all critical fixes discovered through deep analysis

set -e

# Configuration
PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
cd "$PROJECT_ROOT/experiments/eegpt_linear_probe"

# Environment
export BGB_DATA_ROOT="$PROJECT_ROOT/data"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Log file with timestamp
LOG_FILE="logs/FINAL_fixed_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "================================================================"
echo "LAUNCHING TRAINING WITH ALL CRITICAL FIXES"
echo "================================================================"
echo ""
echo "✅ FIXES APPLIED:"
echo "  1. OneCycleLR steps per OPTIMIZER step (not per batch)"
echo "  2. Gradient accumulation aware total_steps calculation"
echo "  3. Global step tracking for proper resume"
echo "  4. No start_epoch reset bug"
echo "  5. Optimizer LR logging (not scheduler.get_last_lr)"
echo "  6. Proper checkpoint saving with global_step"
echo ""
echo "📊 EXPECTED BEHAVIOR:"
echo "  - LR starts at ~0.00012 (warmup)"
echo "  - LR increases to 0.003 (peak) by 10% of training"
echo "  - LR decreases to ~0.000003 (final) in last 50%"
echo "  - AUROC should reach 0.869+ (paper target)"
echo ""
echo "Config: configs/tuab_4s_paper_aligned.yaml"
echo "Script: train_paper_aligned_FINAL.py"
echo "Log: $LOG_FILE"
echo "================================================================"

# Dry run test first (optional - uncomment to test scheduler)
# echo "Running scheduler dry test..."
# $PROJECT_ROOT/.venv/bin/python -c "
# import torch
# from torch.optim.lr_scheduler import OneCycleLR
# opt = torch.optim.Adam([torch.randn(10)], lr=0.003)
# sched = OneCycleLR(opt, max_lr=0.003, total_steps=1000, pct_start=0.1)
# lrs = []
# for i in range(100):
#     opt.step()
#     sched.step()
#     lr = opt.param_groups[0]['lr']
#     if i % 10 == 0:
#         print(f'Step {i}: LR = {lr:.6f}')
#     lrs.append(lr)
# assert lrs[0] < lrs[50], 'Warmup not working!'
# print('✅ Scheduler test passed!')
# "

# Launch training in tmux
tmux new-session -d -s eegpt_final_fixed \
    "$PROJECT_ROOT/.venv/bin/python train_paper_aligned_FINAL.py \
    --config configs/tuab_4s_paper_aligned.yaml \
    2>&1 | tee $LOG_FILE"

echo ""
echo "✅ Training launched in tmux session 'eegpt_final_fixed'"
echo ""
echo "MONITOR WITH:"
echo "  tmux attach -t eegpt_final_fixed"
echo ""
echo "VERIFY SCHEDULER IS WORKING:"
echo "  # Watch for changing LR values"
echo "  tail -f $LOG_FILE | grep -E '(LR:|lr=)'"
echo ""
echo "  # First 100 steps should show LR increasing (warmup)"
echo "  # Steps 100-500 should show LR ~0.003 (peak)"
echo "  # After step 2900+ should show LR decreasing (annealing)"
echo ""
echo "CHECK PROGRESS:"
echo "  grep 'Global Step:' $LOG_FILE | tail -5"
echo ""
echo "If LR is constant, something is still wrong!"
echo "Expected to see LR change EVERY optimizer step."