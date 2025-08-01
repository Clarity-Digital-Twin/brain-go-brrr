#!/bin/bash
# Start robust EEGPT Linear Probe training in tmux session

set -e

echo "🚀 Starting Robust EEGPT Linear Probe Training in tmux"
echo "=================================================="
echo "Dataset: TUAB v3.0.1 (5,986 EDF files)"
echo "Model: EEGPT 10M parameters"
echo ""

# Set environment
PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
export BGB_DATA_ROOT="$PROJECT_ROOT/data"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Session name with timestamp
SESSION="eegpt_training_$(date +%Y%m%d_%H%M%S)"

# Create necessary directories
cd "$PROJECT_ROOT/experiments/eegpt_linear_probe"
mkdir -p checkpoints logs

# Create tmux session
tmux new-session -d -s $SESSION

# Send commands to session
tmux send-keys -t $SESSION "cd $PROJECT_ROOT/experiments/eegpt_linear_probe" C-m
tmux send-keys -t $SESSION "export BGB_DATA_ROOT='$BGB_DATA_ROOT'" C-m
tmux send-keys -t $SESSION "export PYTHONUNBUFFERED=1" C-m
tmux send-keys -t $SESSION "export CUDA_VISIBLE_DEVICES=0" C-m
tmux send-keys -t $SESSION "echo '🏃 Starting robust training with full dataset...'" C-m
tmux send-keys -t $SESSION "python train_robust.py 2>&1 | tee logs/training_robust_$(date +%Y%m%d_%H%M%S).log" C-m

echo "✅ Training started in tmux session: $SESSION"
echo ""
echo "📋 Useful commands:"
echo "   - Attach to session: tmux attach -t $SESSION"
echo "   - Detach: Ctrl+B, then D"
echo "   - Monitor progress: tail -f logs/training_robust_*.log"
echo "   - Check GPU usage: nvidia-smi -l 1"
echo "   - Kill session: tmux kill-session -t $SESSION"
echo ""
echo "📊 TensorBoard: cd $PROJECT_ROOT/experiments/eegpt_linear_probe && tensorboard --logdir logs"
echo ""
echo "Dataset status:"
echo "   Train: $(find $BGB_DATA_ROOT/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/train -name "*.edf" | wc -l) files"
echo "   Eval: $(find $BGB_DATA_ROOT/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/eval -name "*.edf" | wc -l) files"