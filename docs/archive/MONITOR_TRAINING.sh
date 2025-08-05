#!/bin/bash
# Quick monitoring script for EEGPT training

echo "====================================="
echo "EEGPT Training Monitor"
echo "====================================="
echo

# Find latest log file
LATEST_LOG=$(ls -t logs/nan_safe_training_*/training.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    LATEST_LOG=$(ls -t logs/pytorch_training_*.log 2>/dev/null | head -1)
fi

if [ -z "$LATEST_LOG" ]; then
    echo "No training logs found!"
    exit 1
fi

echo "Monitoring: $LATEST_LOG"
echo

# Show last few lines and grep for important metrics
echo "Latest progress:"
tail -20 "$LATEST_LOG" | grep -E "Epoch|loss|acc|grad_norm|AUROC|NaN" || tail -20 "$LATEST_LOG"

echo
echo "Training metrics:"
grep -E "Train Loss:|Val Loss:|AUROC|Balanced Acc" "$LATEST_LOG" | tail -10

echo
echo "To attach to tmux session: tmux attach -t eegpt_nan_safe"
echo "To watch live: tail -f $LATEST_LOG"