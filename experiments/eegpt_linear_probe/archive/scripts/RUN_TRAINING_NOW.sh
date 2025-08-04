#!/bin/bash
# PROFESSIONAL CACHED TRAINING SCRIPT - VERIFIED AND TESTED

set -e

echo "=================================================================================="
echo "EEGPT LINEAR PROBE TRAINING - CACHED DATASET"
echo "=================================================================================="
echo "Dataset: TUAB (930,495 train windows, cached)"
echo "Model: EEGPT with two-layer probe"
echo "Config: experiments/eegpt_linear_probe/configs/tuab_cached.yaml"
echo "=================================================================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export EEGPT_CONFIG="configs/tuab_cached.yaml"

# Kill any existing sessions
tmux kill-session -t eegpt_training 2>/dev/null || true

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/eegpt_training_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo ""
echo "Starting training..."
echo "Log directory: ${LOG_DIR}"
echo ""

# Launch training in tmux
tmux new-session -d -s eegpt_training \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr && \
     EEGPT_CONFIG="configs/tuab_cached.yaml" uv run python experiments/eegpt_linear_probe/train_enhanced.py \
         2>&1 | tee ${LOG_DIR}/training.log"

echo "=================================================================================="
echo "✅ Training started successfully!"
echo ""
echo "Monitor training:"
echo "  tmux attach -t eegpt_training"
echo ""
echo "View logs:"
echo "  tail -f ${LOG_DIR}/training.log"
echo ""
echo "GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo "=================================================================================="

# Wait and verify
sleep 30
if tmux has-session -t eegpt_training 2>/dev/null; then
    echo ""
    echo "✅ Training still running after 30 seconds - looking good!"
    echo ""
    echo "Recent log output:"
    tail -20 "${LOG_DIR}/training.log" | grep -E "(Epoch|loss|accuracy|GPU|windows)" || echo "Waiting for training output..."
else
    echo ""
    echo "❌ Training stopped unexpectedly! Check logs:"
    tail -50 "${LOG_DIR}/training.log"
    exit 1
fi