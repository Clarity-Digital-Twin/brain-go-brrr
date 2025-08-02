#!/bin/bash
# Benchmark AutoReject impact on EEGPT training

set -e

echo "=== EEGPT AutoReject Benchmarking ==="
echo "Comparing training with and without AutoReject"
echo ""

# Set environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data"

# Create results directory
RESULTS_DIR="experiments/eegpt_linear_probe/results/autoreject_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to extract metrics from logs
extract_metrics() {
    local log_file=$1
    local output_file=$2
    
    echo "Extracting metrics from $log_file..."
    
    # Extract final validation AUROC
    grep -E "val_auroc|val_accuracy|val_balanced_accuracy" "$log_file" | tail -5 > "$output_file"
    
    # Extract training time
    grep -E "Epoch [0-9]+/[0-9]+" "$log_file" | tail -1 >> "$output_file"
}

# Run without AutoReject (baseline)
echo "=== BASELINE: Training WITHOUT AutoReject ==="
echo "Config: tuab_enhanced_config.yaml"
echo ""

BASELINE_DIR="$RESULTS_DIR/baseline_no_autoreject"
mkdir -p "$BASELINE_DIR"

# Use python directly with venv
/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python \
    experiments/eegpt_linear_probe/train_enhanced.py \
    --config experiments/eegpt_linear_probe/configs/tuab_enhanced_config.yaml \
    --experiment.name "baseline_no_autoreject" \
    --training.epochs 10 \
    2>&1 | tee "$BASELINE_DIR/training.log"

echo ""
echo "Baseline training complete!"
echo ""

# Run with AutoReject
echo "=== ENHANCED: Training WITH AutoReject ==="
echo "Config: tuab_enhanced_autoreject.yaml"
echo ""

AUTOREJECT_DIR="$RESULTS_DIR/enhanced_autoreject"
mkdir -p "$AUTOREJECT_DIR"

/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python \
    experiments/eegpt_linear_probe/train_enhanced.py \
    --config experiments/eegpt_linear_probe/configs/tuab_enhanced_autoreject.yaml \
    --experiment.name "enhanced_autoreject" \
    --training.epochs 10 \
    2>&1 | tee "$AUTOREJECT_DIR/training.log"

echo ""
echo "AutoReject training complete!"
echo ""

# Compare results
echo "=== RESULTS COMPARISON ==="
echo ""

# Extract metrics
extract_metrics "$BASELINE_DIR/training.log" "$RESULTS_DIR/baseline_metrics.txt"
extract_metrics "$AUTOREJECT_DIR/training.log" "$RESULTS_DIR/autoreject_metrics.txt"

# Display comparison
echo "BASELINE (No AutoReject):"
cat "$RESULTS_DIR/baseline_metrics.txt"
echo ""

echo "ENHANCED (With AutoReject):"
cat "$RESULTS_DIR/autoreject_metrics.txt"
echo ""

# Calculate improvement
echo "=== EXPECTED IMPROVEMENTS ==="
echo "- AUROC: +5-10% (target: 0.85+)"
echo "- Processing overhead: <50% training time"
echo "- Memory usage: <10GB peak"
echo ""

echo "Full results saved to: $RESULTS_DIR"
echo "Benchmark complete!"