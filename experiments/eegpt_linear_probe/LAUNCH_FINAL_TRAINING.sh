#!/bin/bash
# FINAL TRAINING LAUNCHER - USES OPTIMIZED 4S CACHE

set -e

echo "=========================================="
echo "🚀 FINAL 4S TRAINING LAUNCHER"
echo "=========================================="

# Kill any existing sessions
tmux kill-session -t eegpt_final 2>/dev/null || true

# Environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data

# Check cache is ready
CACHE_INDEX="$BGB_DATA_ROOT/cache/tuab_4s_final/index.json"
if [ ! -f "$CACHE_INDEX" ]; then
    echo "❌ Cache not ready! Wait for build_4s_cache_FINAL.py to finish."
    echo "   Check progress: tail -f cache_build.log"
    exit 1
fi

# Check window count using Python instead of jq
WINDOWS=$(/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python -c "import json; print(json.load(open('$CACHE_INDEX'))['total_windows'])")
echo "✅ Cache ready with $WINDOWS windows"

if [ "$WINDOWS" -lt 1000000 ]; then
    echo "⚠️  WARNING: Only $WINDOWS windows cached (expected >10M)"
    echo "   Training may be slow!"
fi

# Create config
cat > configs/tuab_4s_final.yaml << 'EOF'
experiment:
  name: tuab_4s_final
  description: "Final training with 4s windows"
  seed: 42

data:
  cache_index: ${BGB_DATA_ROOT}/cache/tuab_4s_final/index.json
  window_duration: 4.0  # CRITICAL: 4 seconds
  window_stride: 2.0
  sampling_rate: 256
  batch_size: 256
  num_workers: 8
  n_channels: 20

model:
  backbone:
    name: eegpt
    checkpoint_path: ${BGB_DATA_ROOT}/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt
    freeze: true
  probe:
    input_dim: 512
    hidden_dim: 128
    dropout: 0.3

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.01
  patience: 20
  gradient_accumulation: 1
  mixed_precision: true
  scheduler:
    name: cosine
    warmup_epochs: 2

logging:
  log_interval: 10
  save_best_only: true
EOF

echo "✅ Config created"

# Launch training
LOG_FILE="logs/final_4s_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo ""
echo "Launching training..."
tmux new-session -d -s eegpt_final \
    "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe && \
     /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_paper_aligned.py \
     --config configs/tuab_4s_final.yaml 2>&1 | tee $LOG_FILE"

echo ""
echo "=========================================="
echo "✅ TRAINING LAUNCHED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  tmux attach -t eegpt_final"
echo ""
echo "Check speed:"
echo "  tail -f $LOG_FILE | grep it/s"
echo ""
echo "EXPECTED:"
echo "  - Speed: 400-600 it/s"
echo "  - Epoch time: ~15 minutes"
echo "  - Target AUROC: ≥0.869"
echo ""
echo "If speed <100 it/s, cache is NOT working!"
echo "=========================================="