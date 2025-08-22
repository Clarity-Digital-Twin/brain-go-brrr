#!/bin/bash
# Launch TUAB abnormality detection training with verified configuration
# Target: 0.87 AUROC (paper performance)

set -e  # Exit on error

# Navigate to correct directory
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

# Set environment variables
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
OUTPUT_DIR="output/tuab_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/tuab_training_${TIMESTAMP}.log"

# Create directories
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}

# Display training info
echo "================================================"
echo "🚀 TUAB ABNORMALITY DETECTION TRAINING"
echo "================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Log file: ${LOG_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Target AUROC: 0.87 (paper performance)"
echo "------------------------------------------------"
echo "Configuration:"
echo "  - Model: EEGPT (frozen backbone)"
echo "  - Dataset: TUAB (4s windows, 50% overlap)"
echo "  - Batch size: 256"
echo "  - Features: Full temporal (32,768 dimensions)"
echo "  - Max epochs: 10 with early stopping"
echo "================================================"

# Check if tmux session exists
if tmux has-session -t tuab_training 2>/dev/null; then
    echo "❌ ERROR: tmux session 'tuab_training' already exists!"
    echo "Kill it with: tmux kill-session -t tuab_training"
    exit 1
fi

# Launch in tmux
echo ""
echo "Starting training in tmux session 'tuab_training'..."
tmux new-session -d -s tuab_training \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe && \
     /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_tuab.py \
     --config configs/tuab.yaml \
     --output_dir ${OUTPUT_DIR} \
     2>&1 | tee ${LOG_FILE}"

echo "✅ Training started!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t tuab_training"
echo ""
echo "Check logs:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Kill if needed:"
echo "  tmux kill-session -t tuab_training"
echo "================================================"
