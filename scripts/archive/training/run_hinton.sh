#!/bin/bash
# DIRECT LAUNCHER FOR GEOFFREY HINTON - NO BS, JUST TRAINING

echo "🧠 LAUNCHING EEGPT FOR GEOFFREY HINTON 🧠"
echo "========================================"

# Environment
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
export EEGPT_CONFIG=configs/tuab_memsafe.yaml

# Fixed log dir
LOG_DIR="logs/hinton_direct"
mkdir -p $LOG_DIR

echo "Starting training..."
echo ""

# Run training directly - no pipes for now
exec uv run python experiments/eegpt_linear_probe/train_enhanced.py