#!/bin/bash
set -e

# EEGPT Resume Training - FIXED VERSION
# Continues from epoch 15 (AUROC 0.7916)

echo "🚀 RESUMING EEGPT TRAINING FROM EPOCH 16"
echo "📊 Current AUROC: 0.7916"
echo "🎯 Target AUROC: 0.869"
echo ""

# Set environment
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Paths
VENV_PATH="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python"
SCRIPT_PATH="train_paper_aligned_resume.py"
CONFIG_PATH="configs/tuab_4s_wsl_safe.yaml"
CHECKPOINT_PATH="output/tuab_4s_paper_target_20250806_132743/best_model.pt"
LOG_FILE="logs/resume_fixed_$(date +%Y%m%d_%H%M%S).log"

# Check checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "✅ Checkpoint found: $CHECKPOINT_PATH"
echo "📝 Logging to: $LOG_FILE"
echo ""

# Kill any existing session
tmux kill-session -t eegpt_resume 2>/dev/null || true

# Launch training in tmux
echo "Launching in tmux session 'eegpt_resume'..."
tmux new-session -d -s eegpt_resume \
    "$VENV_PATH $SCRIPT_PATH \
     --config $CONFIG_PATH \
     --resume $CHECKPOINT_PATH \
     2>&1 | tee $LOG_FILE"

echo ""
echo "✅ Training resumed successfully!"
echo ""
echo "📊 Monitor with: tmux attach -t eegpt_resume"
echo "📈 Check metrics: tail -f $LOG_FILE | grep AUROC"
echo "🔥 GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "Press Ctrl+B then D to detach from tmux without stopping training"
