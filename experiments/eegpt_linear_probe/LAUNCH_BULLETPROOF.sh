#!/bin/bash
# BULLETPROOF launch script - All 10 bugs fixed!

set -e

# Configuration
PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
cd "$PROJECT_ROOT/experiments/eegpt_linear_probe"

# Environment
export BGB_DATA_ROOT="$PROJECT_ROOT/data"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/BULLETPROOF_${TIMESTAMP}.log"
mkdir -p logs

echo "================================================================"
echo "🛡️ BULLETPROOF TRAINING LAUNCH"
echo "================================================================"
echo ""
echo "✅ ALL 10 BUGS FIXED:"
echo "  1. cycle_momentum=False for AdamW"
echo "  2. Per-batch scheduler stepping"
echo "  3. No start_epoch reset"
echo "  4. Accumulation-aware total_steps"
echo "  5. Global step tracking"
echo "  6. Optimizer LR logging"
echo "  7. RNG state saving"
echo "  8. Extensive sanity checks"
echo "  9. Non-blocking transfers"
echo "  10. Gradient norm monitoring"
echo ""
echo "📊 EXPECTED BEHAVIOR:"
echo "  Step 1: LR ~0.00012 (initial)"
echo "  Step 100: LR increasing (warmup)"
echo "  Step 500: LR ~0.003 (peak)"
echo "  Step 2900+: LR decreasing (annealing)"
echo "  Final: LR ~0.000003"
echo ""
echo "🎯 TARGET: AUROC ≥ 0.869"
echo "================================================================"

# First run dry test
echo ""
echo "🧪 Running 100-step dry test first..."
$PROJECT_ROOT/.venv/bin/python train_paper_aligned_BULLETPROOF.py \
    --config configs/tuab_4s_paper_aligned.yaml \
    --dry_run

echo ""
echo "✅ Dry test complete. Check above that LR is changing!"
echo ""
read -p "Continue with full training? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Launch training in tmux
echo ""
echo "🚀 Launching BULLETPROOF training..."
tmux new-session -d -s eegpt_bulletproof \
    "$PROJECT_ROOT/.venv/bin/python train_paper_aligned_BULLETPROOF.py \
    --config configs/tuab_4s_paper_aligned.yaml \
    2>&1 | tee $LOG_FILE"

echo ""
echo "✅ Training launched in tmux session 'eegpt_bulletproof'"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "MONITOR WITH:"
echo "  tmux attach -t eegpt_bulletproof"
echo ""
echo "VERIFY LR IS CHANGING:"
echo "  tail -f $LOG_FILE | grep 'Step.*LR:'"
echo ""
echo "CHECK KEY MILESTONES:"
echo "  # After 100 steps (should show warmup)"
echo "  grep 'Step 100' $LOG_FILE"
echo ""
echo "  # After 500 steps (should be at peak ~0.003)"
echo "  grep 'Step 500' $LOG_FILE"
echo ""
echo "  # After 2900 steps (should be annealing)"
echo "  grep 'Step 2900' $LOG_FILE"
echo ""
echo "If LR is NOT changing as expected, STOP immediately!"
echo "================================================================"