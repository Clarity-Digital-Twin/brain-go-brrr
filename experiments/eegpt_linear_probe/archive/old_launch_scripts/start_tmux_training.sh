#!/bin/bash
# Start EEGPT training in tmux session

SESSION_NAME="eegpt_training"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION_NAME

# Send commands to session
tmux send-keys -t $SESSION_NAME "cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/experiments/eegpt_linear_probe" C-m
tmux send-keys -t $SESSION_NAME "source /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/.venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "python train_fresh.py" C-m

echo "Training started in tmux session: $SESSION_NAME"
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Ctrl+B then D"
echo "To check: tmux ls"
