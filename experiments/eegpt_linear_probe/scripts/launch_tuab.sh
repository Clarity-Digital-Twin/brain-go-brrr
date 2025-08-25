#!/bin/bash
# Safe training with crash recovery and better monitoring

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
# Add memory monitoring
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/tuab_training_${TIMESTAMP}.log"
OUTPUT_DIR="output/tuab_${TIMESTAMP}"

echo "===================================="
echo "Starting TUAB training with improvements:"
echo "- Batch size: 64 (reduced from 256)"
echo "- Periodic checkpointing every 500 batches"
echo "- Memory cleanup every 100 batches"
echo "- Better error handling and logging"
echo "- Output: ${OUTPUT_DIR}"
echo "- Log: ${LOG_FILE}"
echo "===================================="

# Create output directories
mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# Run with error catching and auto-restart capability
tmux new-session -d -s tuab_training \
    "while true; do \
     echo 'Starting/Resuming training at $(date)' | tee -a ${LOG_FILE}; \
     LATEST_CKPT=\$(ls -t ${OUTPUT_DIR}/checkpoint_epoch*_batch*.pt 2>/dev/null | head -1); \
     RESUME_ARG=''; \
     if [ -n \"\$LATEST_CKPT\" ]; then \
       echo 'Found checkpoint: '\$LATEST_CKPT | tee -a ${LOG_FILE}; \
       RESUME_ARG=\"--resume \$LATEST_CKPT\"; \
     fi; \
     /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python -u train_tuab.py \
     --config configs/tuab.yaml \
     --output_dir ${OUTPUT_DIR} \
     \$RESUME_ARG \
     2>&1 | tee -a ${LOG_FILE}; \
     EXIT_CODE=\$?; \
     echo 'Training exited with code: '\$EXIT_CODE | tee -a ${LOG_FILE}; \
     if [ \$EXIT_CODE -eq 0 ]; then \
       echo 'Training completed successfully!' | tee -a ${LOG_FILE}; \
       break; \
     else \
       echo 'Training crashed! AUTO-RESTARTING in 10s...' | tee -a ${LOG_FILE}; \
       sleep 10; \
     fi; \
     done"

echo "Started in tmux session 'tuab_training'"
echo "Monitor with: tmux attach -t tuab_training"
echo "View logs: tail -f ${LOG_FILE}"
