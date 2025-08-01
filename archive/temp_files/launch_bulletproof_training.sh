#!/bin/bash
# BULLETPROOF EEGPT TRAINING LAUNCHER

# Project root
PROJECT_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr"
cd "$PROJECT_ROOT"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION="eegpt_bulletproof_$TIMESTAMP"
LOG_FILE="$PROJECT_ROOT/experiments/eegpt_linear_probe/logs/bulletproof_$TIMESTAMP.log"

# Create log directory
mkdir -p experiments/eegpt_linear_probe/logs

echo "🚀 BULLETPROOF EEGPT TRAINING LAUNCHER"
echo "====================================="
echo "Session: $SESSION"
echo "Log: $LOG_FILE"
echo ""

# Kill any stuck processes
pkill -f "train_paper_aligned.py" || true
sleep 2

# Create tmux session with proper shell and logging
tmux new-session -d -s $SESSION "bash -lc '
    cd $PROJECT_ROOT
    export BGB_DATA_ROOT=$PROJECT_ROOT/data
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=0
    echo \"Starting EEGPT training at $(date)\"
    echo \"Log file: $LOG_FILE\"
    echo \"\"
    uv run python experiments/eegpt_linear_probe/train_paper_aligned.py 2>&1 | tee $LOG_FILE
    echo \"\"
    echo \"Training finished at $(date)\"
    echo \"Press Enter to close...\"
    read
'"

echo "✅ Training launched in tmux!"
echo ""
echo "Monitor with:"
echo "  - Attach: tmux attach -t $SESSION"
echo "  - Logs:   tail -f $LOG_FILE"
echo "  - GPU:    watch -n 1 nvidia-smi"
echo ""
echo "Detach: Ctrl+B then D"
echo "Kill:   tmux kill-session -t $SESSION"