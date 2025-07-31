#!/bin/bash
# Start EEGPT Linear Probe training in tmux session

set -e

echo "🚀 Starting EEGPT Linear Probe Training in tmux"
echo "=============================================="

# Set environment
export BGB_DATA_ROOT="/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/data"
export PYTHONUNBUFFERED=1

# Session name
SESSION="eegpt_training"

# Check if session exists
tmux has-session -t $SESSION 2>/dev/null && {
    echo "⚠️  Session '$SESSION' already exists"
    echo "   Attach with: tmux attach -t $SESSION"
    echo "   Kill with: tmux kill-session -t $SESSION"
    exit 1
}

# Create necessary directories
mkdir -p checkpoints logs

# Create tmux session
tmux new-session -d -s $SESSION

# Send commands to session
tmux send-keys -t $SESSION "cd $(pwd)" C-m
tmux send-keys -t $SESSION "export BGB_DATA_ROOT='$BGB_DATA_ROOT'" C-m
tmux send-keys -t $SESSION "export PYTHONUNBUFFERED=1" C-m
tmux send-keys -t $SESSION "echo '🏃 Starting training...'" C-m
tmux send-keys -t $SESSION "uv run python train_tuab_probe.py 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log" C-m

echo "✅ Training started in tmux session: $SESSION"
echo ""
echo "📋 Useful commands:"
echo "   - Attach to session: tmux attach -t $SESSION"
echo "   - Detach: Ctrl+B, then D"
echo "   - Monitor logs: tail -f training_*.log"
echo "   - Check GPU: watch -n 1 'ps aux | grep python | head -5'"
echo "   - Kill session: tmux kill-session -t $SESSION"
echo ""
echo "📊 TensorBoard: tensorboard --logdir lightning_logs"
