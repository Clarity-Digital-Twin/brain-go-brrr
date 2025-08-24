#!/bin/bash
# Monitor current training progress

echo "========================================="
echo "TUAB Training Monitor"
echo "========================================="

# Find latest log file
LOG_FILE=$(ls -t logs/tuab_training_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No training log found!"
    exit 1
fi

echo "Monitoring: $LOG_FILE"
echo "-----------------------------------------"

# Check if training is running
if tmux list-sessions 2>/dev/null | grep -q tuab_training; then
    echo "✓ Training session is active"
else
    echo "✗ No active training session"
fi

# Get current progress
echo -e "\nCurrent Progress:"
tail -1 "$LOG_FILE" | grep -o "Training:.*" || echo "Waiting for training to start..."

# Check for errors
echo -e "\nRecent Errors/Warnings:"
tail -100 "$LOG_FILE" | grep -E "ERROR|WARNING|Exception|OOM|CUDA" | tail -5 || echo "No recent errors"

# Check batch timing
echo -e "\nBatch Processing Speed:"
tail -20 "$LOG_FILE" | grep -o "[0-9.]*it/s" | tail -5 | while read speed; do
    echo "  • $speed"
done

# Check memory cleanup logs
echo -e "\nMemory Management:"
grep "clearing cache\|Saved checkpoint" "$LOG_FILE" | tail -3 || echo "No memory cleanup logs yet"

echo "========================================="
echo "Live monitor: tail -f $LOG_FILE"
echo "Attach to session: tmux attach -t tuab_training"
