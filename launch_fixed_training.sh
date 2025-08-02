#!/bin/bash
# FINAL FIXED EEGPT TRAINING LAUNCH SCRIPT

echo "🚀 LAUNCHING FIXED EEGPT TRAINING WITH ALL PATCHES 🚀"
echo "=================================================="
echo "Fixes applied:"
echo "  ✅ Cached dataset for instant loading (0.2s vs 4+ hours)"
echo "  ✅ Custom collate function for channel/window mismatches"
echo "  ✅ Proper tmux session management"
echo "  ✅ Continuous logging to file"
echo ""

# Create logs directory
mkdir -p logs

# Kill any existing sessions
tmux kill-session -t eegpt_fixed 2>/dev/null || true

# Set timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/eegpt_fixed_${TIMESTAMP}.log"

echo "Starting tmux session: eegpt_fixed"
echo "Log file: ${LOG_FILE}"
echo ""

# Launch in tmux with all environment variables and logging
tmux new-session -d -s eegpt_fixed \
    "export PYTHONUNBUFFERED=1 && \
     export CUDA_VISIBLE_DEVICES=0 && \
     export EEGPT_CONFIG=configs/tuab_cached_fast.yaml && \
     export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data && \
     echo '🔥 EEGPT Training Started at $(date)' | tee ${LOG_FILE} && \
     .venv/bin/python experiments/eegpt_linear_probe/train_enhanced.py 2>&1 | tee -a ${LOG_FILE}"

echo "✅ Training launched in tmux session 'eegpt_fixed'"
echo ""
echo "Commands:"
echo "  Watch logs:    tail -f ${LOG_FILE}"
echo "  Attach tmux:   tmux attach -t eegpt_fixed"
echo "  Check status:  tmux ls"
echo ""
echo "Waiting 10 seconds to check initial status..."
sleep 10

# Check if session is still running
if tmux has-session -t eegpt_fixed 2>/dev/null; then
    echo "✅ Training is running!"
    echo ""
    echo "First 20 lines of log:"
    echo "======================"
    head -n 20 ${LOG_FILE}
else
    echo "❌ Training session died! Check logs:"
    echo "tail -n 50 ${LOG_FILE}"
fi