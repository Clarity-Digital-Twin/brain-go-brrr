#!/bin/bash
# Safe TUEV training launcher with WSL2 stability fixes
# Uses parity mode and disabled workers to prevent crashes
# Created: Sep 10, 2025 - After debugging stability issues

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Navigate to project root
cd "$PROJECT_ROOT"

echo "========================================="
echo "TUEV Safe Training Launcher"
echo "========================================="
echo "Timestamp: $TIMESTAMP"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "Using stability fixes:"
echo "  - Parity mode (1000 samples, no padding)"
echo "  - num_workers=0 (prevents WSL2 deadlock)"
echo "  - CUDA_LAUNCH_BLOCKING=1 (better errors)"
echo "  - Logging to file (crash recovery)"
echo "========================================="

# Create output directories
mkdir -p experiments/eegpt_linear_probe/output
mkdir -p experiments/eegpt_linear_probe/logs

# Launch training in tmux with all safety flags
tmux new -d -s tuev_parity "CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  uv run python experiments/eegpt_linear_probe/train_tuev_events.py \
  --data_dir data/datasets/tuev \
  --eegpt_checkpoint data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt \
  --use_parity \
  --epochs 30 \
  --batch_size 32 \
  --num_workers 0 \
  --save_dir experiments/eegpt_linear_probe/output/tuev_parity_$TIMESTAMP \
  2>&1 | tee experiments/eegpt_linear_probe/logs/tuev_parity_$TIMESTAMP.log"

echo ""
echo "✅ Training started in tmux session 'tuev_parity'"
echo ""
echo "Commands:"
echo "  Monitor:  tmux attach -t tuev_parity"
echo "  Detach:   Ctrl+B, then D"
echo "  Logs:     tail -f experiments/eegpt_linear_probe/logs/tuev_parity_$TIMESTAMP.log"
echo ""
echo "Expected milestones:"
echo "  Epoch 2:  BAC > 18%"
echo "  Epoch 5:  BAC > 25%"
echo "  Epoch 10: BAC > 40%"
echo "  Epoch 30: BAC ~ 62% (target)"