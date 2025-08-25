#!/bin/bash
set -euo pipefail

# FIXED training with BCEWithLogitsLoss matching EEGPT paper

# Get repository root and set PYTHONPATH
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT"

cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/tuab_fixed_${TIMESTAMP}.log"
OUTPUT_DIR="output/tuab_fixed_${TIMESTAMP}"

echo "====================================="
echo "FIXED TUAB Training (BCEWithLogitsLoss)"
echo "====================================="
echo "- Binary classification (1 output neuron)"
echo "- BCEWithLogitsLoss (no class weights)"
echo "- Matching EEGPT paper exactly"
echo "- Output: ${OUTPUT_DIR}"
echo "- Log: ${LOG_FILE}"
echo "====================================="

mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# Launch in tmux
tmux new-session -d -s tuab_fixed \
    "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python -u train_tuab_FIXED.py \
    --config configs/tuab.yaml \
    --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${LOG_FILE}"

echo "Started in tmux session 'tuab_fixed'"
echo "Monitor with: tmux attach -t tuab_fixed"
echo "View logs: tail -f ${LOG_FILE}"
echo ""
echo "Expected: Loss should NOT be zero, should converge to ~0.3-0.4"
echo "Target: AUROC should reach 0.869 (paper performance)"