#!/bin/bash
# SIMPLE CACHED TRAINING LAUNCHER

set -e

echo "=========================================="
echo "🚀 LAUNCHING SIMPLE CACHED TRAINING"
echo "=========================================="

# Kill any existing
tmux kill-session -t eegpt_cached 2>/dev/null || true

# Environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data

# Create simple config with correct paths
cat > configs/tuab_cached_simple.yaml << 'EOF'
data:
  root_dir: ${BGB_DATA_ROOT}/datasets/external/tuh_eeg_abnormal/v3.0.1/edf
  cache_dir: ${BGB_DATA_ROOT}/cache
  cache_mode: create_if_missing  # Will create cache automatically
  window_duration: 4.0  # CRITICAL: 4 seconds for EEGPT
  window_stride: 2.0
  sampling_rate: 256
  batch_size: 128
  num_workers: 4
  bandpass_low: 0.5
  bandpass_high: 50.0
  n_channels: 20
  channel_names:
    - FP1
    - FP2
    - F7
    - F3
    - FZ
    - F4
    - F8
    - T7
    - C3
    - CZ
    - C4
    - T8
    - P7
    - P3
    - PZ
    - P4
    - P8
    - O1
    - O2
    - OZ

model:
  backbone:
    name: eegpt
    checkpoint_path: ${BGB_DATA_ROOT}/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt
    freeze: true
  probe:
    input_dim: 512
    hidden_dim: 128
    dropout: 0.3
    num_classes: 1

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.01
  patience: 20
  scheduler:
    name: onecycle
    max_lr: 0.01
    pct_start: 0.3

logging:
  log_interval: 10
  save_interval: 1
  wandb: false
EOF

echo "✅ Config created: configs/tuab_cached_simple.yaml"

# Use the paper-aligned trainer which works
LOG_FILE="logs/cached_simple_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo ""
echo "Starting training..."
tmux new-session -d -s eegpt_cached \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe && \
     /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_paper_aligned.py \
     --config configs/tuab_cached_simple.yaml 2>&1 | tee $LOG_FILE"

echo ""
echo "=========================================="
echo "✅ TRAINING LAUNCHED!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  tmux attach -t eegpt_cached"
echo ""
echo "Check log:"
echo "  tail -f $LOG_FILE"
echo ""
echo "=========================================="