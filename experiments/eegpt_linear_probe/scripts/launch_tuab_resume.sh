#!/bin/bash
# Robust training with automatic resume from last checkpoint

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
# Add memory monitoring
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/tuab_resume_${TIMESTAMP}.log"

echo "===================================="
echo "RESUMING TUAB training from last checkpoint"
echo "===================================="

# Find the most recent checkpoint
LATEST_OUTPUT=$(ls -td output/tuab_* 2>/dev/null | head -1)
if [ -z "$LATEST_OUTPUT" ]; then
    echo "ERROR: No previous training found!"
    exit 1
fi

# Find the latest checkpoint in that directory
LATEST_CHECKPOINT=$(ls -t ${LATEST_OUTPUT}/checkpoint_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in ${LATEST_OUTPUT}"
    exit 1
fi

echo "Found checkpoint: ${LATEST_CHECKPOINT}"
echo "Output dir: ${LATEST_OUTPUT}"
echo "Log: ${LOG_FILE}"
echo "===================================="

# Create logs directory if needed
mkdir -p logs

# Run with automatic restart and resume
tmux new-session -d -s tuab_resume \
    "while true; do \
     echo '===================================' | tee -a ${LOG_FILE}; \
     echo 'Starting/Resuming at $(date)' | tee -a ${LOG_FILE}; \
     echo 'Memory before: $(free -h | grep Mem)' | tee -a ${LOG_FILE}; \
     echo '===================================' | tee -a ${LOG_FILE}; \
     /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python -u train_tuab.py \
     --config configs/tuab.yaml \
     --output_dir ${LATEST_OUTPUT} \
     --resume ${LATEST_CHECKPOINT} \
     2>&1 | tee -a ${LOG_FILE}; \
     EXIT_CODE=\$?; \
     echo '===================================' | tee -a ${LOG_FILE}; \
     echo 'Training exited with code: '\$EXIT_CODE | tee -a ${LOG_FILE}; \
     echo 'Memory after: $(free -h | grep Mem)' | tee -a ${LOG_FILE}; \
     echo '===================================' | tee -a ${LOG_FILE}; \
     if [ \$EXIT_CODE -eq 0 ]; then \
       echo 'Training completed successfully!' | tee -a ${LOG_FILE}; \
       break; \
     else \
       echo 'Training crashed/interrupted! Will auto-restart in 30s...' | tee -a ${LOG_FILE}; \
       # Update checkpoint path for next resume
       LATEST_CHECKPOINT=\$(ls -t ${LATEST_OUTPUT}/checkpoint_*.pt 2>/dev/null | head -1); \
       echo 'Next resume from: '\$LATEST_CHECKPOINT | tee -a ${LOG_FILE}; \
       sleep 30; \
     fi; \
     done"

echo "Started in tmux session 'tuab_resume'"
echo "Monitor with: tmux attach -t tuab_resume"
echo "View logs: tail -f ${LOG_FILE}"
echo ""
echo "NOTE: Training will auto-restart if it crashes!"
echo "To stop: tmux kill-session -t tuab_resume"