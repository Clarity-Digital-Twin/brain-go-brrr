#!/bin/bash
# FIXED launcher with true auto-restart and automatic resume

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/tuab_training_${TIMESTAMP}.log"
OUTPUT_DIR="output/tuab_${TIMESTAMP}"

# BUT if we're resuming, use existing output dir
LATEST_OUTPUT=$(ls -td output/tuab_* 2>/dev/null | head -1)
if [ -n "$LATEST_OUTPUT" ]; then
    OUTPUT_DIR="$LATEST_OUTPUT"
    echo "Using existing output dir: $OUTPUT_DIR"
fi

echo "===================================="
echo "Starting TUAB training with FIXED auto-restart"
echo "- TRUE auto-restart on any crash"
echo "- Automatic resume from latest checkpoint"
echo "- Batch size: 64"
echo "- Output: ${OUTPUT_DIR}"
echo "- Log: ${LOG_FILE}"
echo "===================================="

mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# FIXED: True auto-restart loop with automatic checkpoint resume
tmux new-session -d -s tuab_training \
    "while true; do \
     echo '====================================' | tee -a ${LOG_FILE}; \
     echo 'Starting/Resuming at $(date)' | tee -a ${LOG_FILE}; \
     echo 'Memory: $(free -h | grep Mem)' | tee -a ${LOG_FILE}; \
     LATEST_CKPT=\$(ls -1t ${OUTPUT_DIR}/checkpoint_epoch*_batch*.pt 2>/dev/null | head -1); \
     RESUME_ARG=''; \
     if [ -n \"\$LATEST_CKPT\" ]; then \
       echo 'Resuming from: '\$LATEST_CKPT | tee -a ${LOG_FILE}; \
       RESUME_ARG=\"--resume \$LATEST_CKPT\"; \
     else \
       echo 'Starting fresh training' | tee -a ${LOG_FILE}; \
     fi; \
     /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python -u train_tuab_fixed.py \
     --config configs/tuab.yaml \
     --output_dir ${OUTPUT_DIR} \
     \$RESUME_ARG \
     2>&1 | tee -a ${LOG_FILE}; \
     EXIT_CODE=\$?; \
     echo '====================================' | tee -a ${LOG_FILE}; \
     echo 'Training exited with code: '\$EXIT_CODE | tee -a ${LOG_FILE}; \
     if [ \$EXIT_CODE -eq 0 ]; then \
       echo 'Training TRULY completed successfully!' | tee -a ${LOG_FILE}; \
       break; \
     else \
       echo 'Training crashed/interrupted. AUTO-RESTARTING in 10s...' | tee -a ${LOG_FILE}; \
       sleep 10; \
     fi; \
     done"

echo "Started in tmux session 'tuab_training'"
echo "Monitor with: tmux attach -t tuab_training"
echo "View logs: tail -f ${LOG_FILE}"
echo ""
echo "NOTE: Training will ALWAYS auto-restart on crash!"
echo "To stop: tmux kill-session -t tuab_training"