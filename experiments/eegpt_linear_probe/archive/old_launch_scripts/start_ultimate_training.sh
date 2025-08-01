#!/bin/bash
# ULTIMATE EEGPT TRAINING SCRIPT - PRODUCTION READY

set -e

echo "🚀🚀🚀 STARTING ULTIMATE EEGPT TRAINING 🚀🚀🚀"
echo "============================================"
echo "Dataset: TUAB v3.0.1 (5,986 files)"
echo "Model: EEGPT 10M parameters"
echo "GPU: RTX 4090"
echo ""

# Project paths
PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
cd "$PROJECT_ROOT"

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION="eegpt_ultimate_$TIMESTAMP"

# Create directories
mkdir -p experiments/eegpt_linear_probe/logs
mkdir -p experiments/eegpt_linear_probe/checkpoints

# Kill any existing Python processes that might interfere
echo "🧹 Cleaning up any stuck processes..."
pkill -f "train_robust.py" || true
pkill -f "train_paper_aligned.py" || true
sleep 2

# Create new tmux session
echo "📺 Creating tmux session: $SESSION"
tmux new-session -d -s $SESSION

# Send environment setup
tmux send-keys -t $SESSION "cd $PROJECT_ROOT" C-m
tmux send-keys -t $SESSION "export BGB_DATA_ROOT='$PROJECT_ROOT/data'" C-m
tmux send-keys -t $SESSION "export PYTHONUNBUFFERED=1" C-m
tmux send-keys -t $SESSION "export CUDA_VISIBLE_DEVICES=0" C-m

# Add monitoring in background
tmux send-keys -t $SESSION "echo '🔥 STARTING PAPER-ALIGNED TRAINING WITH FULL DATASET 🔥'" C-m
tmux send-keys -t $SESSION "echo 'This will take 4-6 hours. Monitor with: nvidia-smi -l 1'" C-m
tmux send-keys -t $SESSION "echo ''" C-m

# Start the training with output to both terminal and log file
tmux send-keys -t $SESSION "uv run python experiments/eegpt_linear_probe/train_paper_aligned.py 2>&1 | tee experiments/eegpt_linear_probe/logs/ultimate_training_$TIMESTAMP.log" C-m

echo ""
echo "✅ TRAINING LAUNCHED SUCCESSFULLY!"
echo ""
echo "📊 MONITOR YOUR TRAINING:"
echo "   1. Attach to tmux:     tmux attach -t $SESSION"
echo "   2. Watch GPU:          watch -n 1 nvidia-smi"
echo "   3. Tail logs:          tail -f experiments/eegpt_linear_probe/logs/ultimate_training_$TIMESTAMP.log"
echo "   4. TensorBoard:        cd $PROJECT_ROOT && tensorboard --logdir experiments/eegpt_linear_probe/logs"
echo ""
echo "⚡ DETACH: Ctrl+B then D"
echo "🛑 KILL: tmux kill-session -t $SESSION"
echo ""
echo "🎯 Expected completion: 4-6 hours"
echo "📈 Target AUROC: ≥0.93"