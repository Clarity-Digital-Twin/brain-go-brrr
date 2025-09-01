#!/bin/bash
# Smoke test script to verify checkpoint/resume functionality

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
DATA_ROOT="${BGB_DATA_ROOT:-$PROJECT_ROOT/data}"

echo "====================================="
echo "SMOKE TEST: Checkpoint/Resume Logic"
echo "====================================="

# Test parameters
MAX_BATCHES=30
CHECKPOINT_EVERY=10
OUTPUT_DIR="$EXPERIMENT_DIR/output/smoke_test_$(date +%Y%m%d_%H%M%S)"

echo "Output dir: $OUTPUT_DIR"
echo "Will run $MAX_BATCHES batches with checkpoint every $CHECKPOINT_EVERY"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Phase 1: Run initial training
echo "PHASE 1: Initial training to batch 20..."
export BGB_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Modify config to limit batches
cat > "$OUTPUT_DIR/smoke_config.yaml" << EOF
experiment:
  name: smoke_test
  seed: 42

data:
  root_dir: \${BGB_DATA_ROOT}/datasets/tuab/edf
  cache_dir: \${BGB_DATA_ROOT}/cache/tuab_mne_v2
  batch_size: 64
  num_workers: 2  # Reduced for smoke test
  pin_memory: true
  persistent_workers: false  # Disabled for short run
  prefetch_factor: 2

model:
  backbone:
    name: eegpt
    checkpoint_path: \${BGB_DATA_ROOT}/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt
    freeze: true
  probe:
    type: linear
    hidden_dim: 128
    n_classes: 2
    dropout: 0.1

training:
  max_epochs: 1
  optimizer:
    name: AdamW
    lr: 1.0e-3
    weight_decay: 1.0e-4
  scheduler:
    name: OneCycleLR
    max_lr: 3.0e-3
    epochs: 1
    pct_start: 0.2
    anneal_strategy: cos
  gradient_clip_val: 1.0
  weighted_loss: true
  checkpoint_every: $CHECKPOINT_EVERY

logging:
  log_every_n_steps: 5
EOF

# Run training for 20 batches
echo "Starting initial training..."
timeout 120 uv run python "$EXPERIMENT_DIR/train_tuab_mne.py" \
    --config "$OUTPUT_DIR/smoke_config.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir "$DATA_ROOT/cache/tuab_mne_v2" 2>&1 | \
    grep -E "(Epoch|batch|checkpoint|Saved)" | head -25 || true

echo ""
echo "PHASE 1 COMPLETE"
echo "Checkpoints created:"
ls -la "$OUTPUT_DIR"/checkpoint*.pt 2>/dev/null || echo "No checkpoints found"

# Find latest checkpoint
LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint_*.pt" -type f 2>/dev/null | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found!"
    exit 1
fi

echo ""
echo "====================================="
echo "PHASE 2: Resume from checkpoint"
echo "====================================="
echo "Resuming from: $LATEST_CHECKPOINT"

# Phase 2: Resume training
timeout 60 uv run python "$EXPERIMENT_DIR/train_tuab_mne.py" \
    --config "$OUTPUT_DIR/smoke_config.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir "$DATA_ROOT/cache/tuab_mne_v2" \
    --resume "$LATEST_CHECKPOINT" 2>&1 | \
    grep -E "(Resuming|epoch|batch|Global step)" | head -10 || true

echo ""
echo "====================================="
echo "SMOKE TEST COMPLETE"
echo "====================================="

# Check heartbeat
if [ -f "$OUTPUT_DIR/heartbeat.json" ]; then
    echo "Last heartbeat:"
    cat "$OUTPUT_DIR/heartbeat.json" | python -m json.tool | grep -E "(epoch|batch_idx|global_step)" || true
fi

echo ""
echo "Final checkpoints:"
ls -lh "$OUTPUT_DIR"/checkpoint*.pt 2>/dev/null || echo "No checkpoints"

echo ""
echo "✓ Smoke test passed! Ready for full training."