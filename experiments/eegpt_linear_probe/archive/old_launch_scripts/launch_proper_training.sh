#!/bin/bash
# Launch proper EEGPT training with all fixes

SESSION_NAME="eegpt_proper"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION_NAME

# Send commands to session
tmux send-keys -t $SESSION_NAME "cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/experiments/eegpt_linear_probe" C-m
tmux send-keys -t $SESSION_NAME "source /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/.venv/bin/activate" C-m

# Create log directory
tmux send-keys -t $SESSION_NAME "mkdir -p logs" C-m

# Run training with output to both console and file
tmux send-keys -t $SESSION_NAME "python train_final_proper.py 2>&1 | tee logs/training_proper_${TIMESTAMP}.log" C-m

echo "Training launched in tmux session: $SESSION_NAME"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "Commands:"
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Detach:  Ctrl+B then D"
echo "  List:    tmux ls"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/training_proper_${TIMESTAMP}.log | grep -E 'Epoch|train_loss|val_loss|val_auroc'"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir logs/run_*"
