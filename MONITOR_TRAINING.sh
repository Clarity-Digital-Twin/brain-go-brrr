#!/bin/bash
# MONITORING SCRIPT - RUN THIS EVERY 30 MINUTES

echo "================================================================"
echo "TRAINING MONITOR CHECK - $(date)"
echo "================================================================"

# 1. Check if tmux session is still running
if tmux has-session -t eegpt_training 2>/dev/null; then
    echo "✅ Training session ACTIVE"
else
    echo "❌ TRAINING CRASHED - SESSION NOT FOUND!"
    exit 1
fi

# 2. Get last 20 lines from tmux
echo -e "\n📊 LATEST TRAINING OUTPUT:"
echo "--------------------------------"
tmux capture-pane -t eegpt_training -p | tail -20

# 3. Check GPU utilization
echo -e "\n🖥️ GPU STATUS:"
echo "--------------------------------"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "GPU Util: %s%%\nMemory: %s/%s MB\nTemp: %s°C\n", $1, $2, $3, $4}'

# 4. Check for errors in log
LOG_FILE=$(ls -t logs/eegpt_training_*/training.log 2>/dev/null | head -1)
if [ -f "$LOG_FILE" ]; then
    echo -e "\n⚠️ CHECKING FOR ERRORS:"
    echo "--------------------------------"
    ERROR_COUNT=$(grep -c "ERROR\|Exception\|Traceback\|nan\|inf" "$LOG_FILE" || echo "0")
    echo "Error count: $ERROR_COUNT"
    
    if [ "$ERROR_COUNT" -gt "0" ]; then
        echo "RECENT ERRORS:"
        grep -n "ERROR\|Exception\|Traceback\|nan\|inf" "$LOG_FILE" | tail -5
    fi
fi

# 5. Check training progress
echo -e "\n📈 TRAINING PROGRESS:"
echo "--------------------------------"
if [ -f "$LOG_FILE" ]; then
    # Get epoch info
    grep "Epoch" "$LOG_FILE" | tail -5
    
    # Get loss info
    echo -e "\nRecent losses:"
    grep -E "loss.*[0-9]" "$LOG_FILE" | tail -5
    
    # Get metrics
    echo -e "\nRecent metrics:"
    grep -E "auroc|accuracy" "$LOG_FILE" | tail -5
fi

# 6. Check if still using cache properly
echo -e "\n💾 CACHE STATUS:"
echo "--------------------------------"
if [ -f "$LOG_FILE" ]; then
    CACHE_MSG=$(grep "Loaded TUAB.*from cache" "$LOG_FILE" | tail -1)
    if [ -n "$CACHE_MSG" ]; then
        echo "✅ $CACHE_MSG"
    else
        echo "⚠️ No cache loading message found"
    fi
    
    # Check for file scanning
    SCAN_COUNT=$(grep -c "Scanning\|Found.*\.edf" "$LOG_FILE" || echo "0")
    if [ "$SCAN_COUNT" -gt "0" ]; then
        echo "❌ WARNING: File scanning detected! ($SCAN_COUNT occurrences)"
    else
        echo "✅ No file scanning detected"
    fi
fi

echo -e "\n================================================================"
echo "Next check in 30 minutes: $(date -d '+30 minutes')"
echo "================================================================"