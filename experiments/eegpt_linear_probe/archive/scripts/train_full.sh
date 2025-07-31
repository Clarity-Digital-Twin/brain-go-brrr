#!/bin/bash
# Full EEGPT Linear Probe Training Script
set -e

echo "🚀 Starting Full EEGPT Linear Probe Training"
echo "==========================================="

# Set environment
export BGB_DATA_ROOT="/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/data"
export CUDA_VISIBLE_DEVICES="0"  # Set GPU if available

# Training parameters
BATCH_SIZE=32
EPOCHS=10
LR=1e-3

echo "Configuration:"
echo "- Data root: $BGB_DATA_ROOT"
echo "- Batch size: $BATCH_SIZE"
echo "- Epochs: $EPOCHS"
echo "- Learning rate: $LR"
echo ""

# Check if tmux session already exists
if tmux has-session -t eegpt_full_training 2>/dev/null; then
    echo "⚠️  Session 'eegpt_full_training' already exists"
    echo "   To attach: tmux attach -t eegpt_full_training"
    echo "   To kill: tmux kill-session -t eegpt_full_training"
    exit 1
fi

# Create new tmux session
tmux new-session -d -s eegpt_full_training

# Send commands to tmux session
tmux send-keys -t eegpt_full_training "cd $PWD" C-m
tmux send-keys -t eegpt_full_training "export BGB_DATA_ROOT=$BGB_DATA_ROOT" C-m

# Run training without limits
tmux send-keys -t eegpt_full_training "uv run python train_tuab_probe.py \
    training.batch_size=$BATCH_SIZE \
    training.epochs=$EPOCHS \
    training.learning_rate=$LR \
    experiment.limit_train_batches=1.0 \
    experiment.limit_val_batches=1.0 \
    2>&1 | tee training_full_$(date +%Y%m%d_%H%M%S).log" C-m

echo "✅ Training started in tmux session: eegpt_full_training"
echo ""
echo "Useful commands:"
echo "- Attach to session: tmux attach -t eegpt_full_training"
echo "- Detach from session: Ctrl+B, then D"
echo "- Kill session: tmux kill-session -t eegpt_full_training"
echo "- Monitor progress: tail -f training_full_*.log | grep -E 'Epoch|val_auroc|train_acc'"
