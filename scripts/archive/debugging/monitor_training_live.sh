#!/bin/bash
# Live monitoring of EEGPT training

echo "=== EEGPT TRAINING MONITOR ==="
echo "Time: $(date)"
echo ""

# Check if tmux session exists
if tmux has-session -t eegpt_stable 2>/dev/null; then
    echo "✅ Training session ACTIVE"
else
    echo "❌ Training session NOT FOUND"
    exit 1
fi

# Check GPU usage
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  Memory: %s MB | Utilization: %s%% | Temp: %s°C\n", $1, $2, $3}'

# Check latest log
LOG=$(ls -t logs/stable_training_*.log 2>/dev/null | head -1)
if [ -f "$LOG" ]; then
    echo ""
    echo "Latest log: $LOG"
    echo "Log size: $(wc -l < "$LOG") lines"
    
    # Check for training progress
    echo ""
    echo "Training Progress:"
    tail -50 "$LOG" | grep -E "Epoch|step.*loss|val_" | tail -5
    
    # Check for errors
    echo ""
    if tail -100 "$LOG" | grep -q "NaN"; then
        echo "⚠️  WARNING: NaN detected in recent logs!"
    elif tail -100 "$LOG" | grep -q "Error\|Exception"; then
        echo "⚠️  WARNING: Error detected in recent logs!"
    else
        echo "✅ No NaN or errors in recent logs"
    fi
fi

echo ""
echo "To attach: tmux attach -t eegpt_stable"
echo "To detach: Ctrl+B then D"