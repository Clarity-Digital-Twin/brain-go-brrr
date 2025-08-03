#!/bin/bash
# Launch EEGPT baseline training (NO AutoReject) with safeguards

set -e  # Exit on error

echo "🚀 Launching EEGPT Baseline Training (No AutoReject)"
echo "==================================================="
echo "Start time: $(date)"

# Safety checks
if [[ "$EEGPT_CONFIG" == *"cached"* ]]; then
    echo "❌ ERROR: Detected 'cached' in config - this causes looping!"
    exit 1
fi

# Set environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export EEGPT_CONFIG=configs/tuab_enhanced_config.yaml

# Create log directory
mkdir -p logs

# Log file with timestamp
LOG_FILE="logs/baseline_training_$(date +%Y%m%d_%H%M%S).log"

echo "📊 Configuration:"
echo "  - Config: $EEGPT_CONFIG"
echo "  - Data root: $BGB_DATA_ROOT"
echo "  - Log file: $LOG_FILE"
echo "  - GPU: CUDA device 0"

# Quick validation
echo -e "\n🧪 Pre-flight checks..."
if [[ ! -f ".venv/bin/python" ]]; then
    echo "❌ ERROR: Virtual environment not found!"
    exit 1
fi

if [[ ! -d "$BGB_DATA_ROOT/datasets/external/tuh_eeg_abnormal" ]]; then
    echo "❌ ERROR: TUAB dataset not found!"
    exit 1
fi

echo "✅ All checks passed!"

# Launch training
echo -e "\n🔥 Starting training..."
tmux new-session -d -s eegpt_baseline \
    ".venv/bin/python experiments/eegpt_linear_probe/train_enhanced.py 2>&1 | tee $LOG_FILE"

echo "✅ Training launched in tmux session 'eegpt_baseline'"
echo ""
echo "📝 Monitor with:"
echo "  tmux attach -t eegpt_baseline"
echo "  tail -f $LOG_FILE | grep -E '(Epoch|loss|auroc)'"
echo ""
echo "⚠️  This should take 4-8 hours for 20 epochs (not 50!)"