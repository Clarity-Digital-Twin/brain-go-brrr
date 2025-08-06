#!/bin/bash
# Monitor training every 30 minutes

LOG_FILE="monitoring_log_$(date +%Y%m%d).txt"

while true; do
    echo "=== MONITORING CHECK: $(date) ===" | tee -a $LOG_FILE
    
    # Check tmux session
    if tmux has-session -t eegpt_final 2>/dev/null; then
        echo "✅ Session ACTIVE" | tee -a $LOG_FILE
    else
        echo "❌ SESSION DEAD!" | tee -a $LOG_FILE
        exit 1
    fi
    
    # GPU status
    echo "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader)" | tee -a $LOG_FILE
    
    # Check for training progress
    PROGRESS=$(tmux capture-pane -t eegpt_final -p | grep -E "Epoch|step.*loss" | tail -5)
    if [ -n "$PROGRESS" ]; then
        echo "TRAINING PROGRESS:" | tee -a $LOG_FILE
        echo "$PROGRESS" | tee -a $LOG_FILE
    else
        echo "⚠️  NO TRAINING PROGRESS YET" | tee -a $LOG_FILE
    fi
    
    # Check for errors
    ERRORS=$(tmux capture-pane -t eegpt_final -p | grep -E "Error|Exception|NaN|ENDED" | tail -5)
    if [ -n "$ERRORS" ]; then
        echo "⚠️  ERRORS FOUND:" | tee -a $LOG_FILE
        echo "$ERRORS" | tee -a $LOG_FILE
    fi
    
    echo "" | tee -a $LOG_FILE
    
    # Wait 30 minutes
    sleep 1800
done