#!/bin/bash
# TUEV Training Launch Script - Following Table 13 Exactly
# Based on TUEV_UNIFIED_SPECS.md

set -e

echo "=========================================="
echo "🚀 TUEV 6-CLASS EVENT DETECTION TRAINING"
echo "=========================================="
echo ""
echo "📊 ARCHITECTURE (Table 13, lines 606-613):"
echo "  - [Paper/Table] Input: 23 × 1000 (3.9 seconds @ 256Hz)"
echo "  - [Paper/Table] Channel reduction: 23 → 20"
echo "  - [Paper/Table] Temporal kernel: 55 (NOT 15!)"
echo "  - [Paper/Table] Dropout: 0.5 (NOT 0.25!)"
echo "  - [Paper] Batch size: 500"
echo "  - [Paper] Learning rate: 5e-4"
echo "  - [Decision] Optimizer: AdamW with constant schedule"
echo ""
echo "🎯 TARGET PERFORMANCE (Paper Table 3):"
echo "  - Balanced Accuracy: 0.6232 ± 0.0114"
echo "  - Weighted F1:       0.8187 ± 0.0063"
echo "  - Cohen's Kappa:     0.6351 ± 0.0134"
echo ""
echo "=========================================="

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1

# Paths
PROJECT_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr
SCRIPT_DIR=$PROJECT_ROOT/experiments/eegpt_linear_probe
CACHE_DIR=$BGB_DATA_ROOT/cache/tuev_table13

# Change to script directory
cd $SCRIPT_DIR

# Step 1: Check if cache exists
if [ ! -d "$CACHE_DIR/tuev_train_cache" ]; then
    echo "⚠️  Cache not found. Building cache first..."
    echo ""
    
    # Build cache
    python build_tuev_cache.py \
        --config configs/tuev_table13_aligned.yaml \
        --output $CACHE_DIR
    
    echo ""
    echo "✅ Cache built successfully!"
    echo ""
else
    echo "✅ Cache found at $CACHE_DIR"
    
    # Verify cache
    echo "Verifying cache integrity..."
    python -c "
import json
from pathlib import Path
cache_dir = Path('$CACHE_DIR')
for split in ['train', 'eval']:
    index_file = cache_dir / f'tuev_{split}_cache' / 'index.json'
    with open(index_file) as f:
        index = json.load(f)
    print(f'  {split}: {index[\"n_samples\"]} samples, {index[\"n_classes\"]} classes')
"
    echo ""
fi

# Step 2: Run training 3 times with different seeds (paper protocol)
echo "=========================================="
echo "📈 STARTING TRAINING (3 runs)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Results array to track performance
RESULTS_FILE="logs/tuev_results_$(date +%Y%m%d_%H%M%S).txt"

echo "Running 3 experiments with different seeds (paper protocol)..."
echo "" | tee -a $RESULTS_FILE

for SEED in 42 123 456; do
    echo "----------------------------------------"
    echo "🎲 RUN ${SEED}: Starting with seed ${SEED}"
    echo "----------------------------------------"
    
    LOG_FILE="logs/tuev_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run training
    python train_tuev_aligned.py \
        --config configs/tuev_table13_aligned.yaml \
        --device cuda \
        --seed $SEED \
        --use-cache \
        2>&1 | tee $LOG_FILE
    
    # Extract final metrics from log
    echo "Run $SEED Results:" | tee -a $RESULTS_FILE
    grep "Best Balanced Accuracy:" $LOG_FILE | tail -1 | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    
    echo "✅ Run $SEED complete!"
    echo ""
done

# Step 3: Compute mean and std
echo "=========================================="
echo "📊 FINAL RESULTS"
echo "=========================================="

python -c "
import re
with open('$RESULTS_FILE', 'r') as f:
    content = f.read()

# Extract balanced accuracy values
pattern = r'Best Balanced Accuracy: ([\d.]+)'
matches = re.findall(pattern, content)
values = [float(v) for v in matches]

if values:
    import numpy as np
    mean = np.mean(values)
    std = np.std(values)
    
    print(f'')
    print(f'Our Results (n={len(values)}):')
    print(f'  Balanced Accuracy: {mean:.4f} ± {std:.4f}')
    print(f'')
    print(f'Paper Target:')
    print(f'  Balanced Accuracy: 0.6232 ± 0.0114')
    print(f'')
    print(f'Achievement: {mean/0.6232*100:.1f}% of paper performance')
    print(f'')
    
    # Check if we beat BIOT baseline
    if mean > 0.5281:
        improvement = (mean - 0.5281) / 0.5281 * 100
        print(f'✅ Beat BIOT baseline (0.5281) by {improvement:.1f}%!')
    else:
        print(f'❌ Did not beat BIOT baseline (0.5281)')
"

echo ""
echo "=========================================="
echo "✨ TUEV TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "📁 Output locations:"
echo "  - Models: output/tuev_*/"
echo "  - Logs: logs/tuev_*.log"
echo "  - Results: $RESULTS_FILE"
echo ""

# Optional: Launch tensorboard
echo "To monitor training with TensorBoard:"
echo "  tensorboard --logdir output/"
echo ""