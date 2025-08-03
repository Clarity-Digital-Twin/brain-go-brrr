#!/bin/bash
# Baseline EEGPT training WITHOUT AutoReject

set -e

echo "==========================================================="
echo "🚀 BASELINE EEGPT TRAINING - NO AUTOREJECT 🚀"
echo "==========================================================="
echo "Configuration:"
echo "  ✓ Using cached data with readonly mode"
echo "  ✓ 8s windows @ 256Hz (standardized)"
echo "  ✓ AutoReject DISABLED for baseline"
echo "  ✓ Two-layer probe with dropout=0.5"
echo "  ✓ 20 epochs with early stopping"
echo "==========================================================="

# Create log directory
LOG_DIR="logs/baseline_noAR_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR/checkpoints

# Environment
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
export EEGPT_CONFIG=configs/tuab_enhanced_config.yaml

# Kill any existing training
tmux kill-session -t baseline_training 2>/dev/null || true

# Clear GPU
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo ""
echo "Starting training in tmux session 'baseline_training'..."
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Launch training
tmux new -d -s baseline_training \
    "cd $(pwd) && \
     echo '🎯 BASELINE TRAINING STARTED 🎯' && \
     echo 'Start time: $(date)' && \
     echo '' && \
     uv run python experiments/eegpt_linear_probe/train_enhanced.py \
         data.cache_mode=readonly \
         data.use_autoreject=false \
         trainer.default_root_dir=$LOG_DIR \
         trainer.max_epochs=20 \
         trainer.accelerator=gpu \
         trainer.devices=1 \
         trainer.precision=32 \
         trainer.accumulate_grad_batches=2 \
         trainer.gradient_clip_val=1.0 \
         trainer.log_every_n_steps=50 \
         trainer.val_check_interval=0.5 \
         2>&1 | tee $LOG_DIR/training.log"

# Monitor GPU usage
(
    echo "timestamp,gpu_mem_mb,gpu_util_pct,temp_c" > $LOG_DIR/gpu_stats.csv
    while tmux has-session -t baseline_training 2>/dev/null; do
        GPU_STATS=$(nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
        echo "$GPU_STATS" >> $LOG_DIR/gpu_stats.csv
        sleep 30
    done
    echo "Training ended at: $(date)" >> $LOG_DIR/gpu_stats.csv
) &

echo "==========================================================="
echo "✅ BASELINE TRAINING LAUNCHED!"
echo ""
echo "Monitor live:"
echo "  tmux attach -t baseline_training"
echo ""
echo "Check progress:"
echo "  tail -f $LOG_DIR/training.log | grep -E 'Epoch|auroc|loss'"
echo ""
echo "GPU usage:"
echo "  watch 'tail -10 $LOG_DIR/gpu_stats.csv'"
echo ""
echo "Expected runtime: ~45-60 minutes for 20 epochs"
echo "==========================================================="

# Wait and check status
sleep 30
if tmux has-session -t baseline_training 2>/dev/null; then
    echo ""
    echo "✅ Training running successfully after 30 seconds!"
    echo ""
    echo "First few log lines:"
    head -20 $LOG_DIR/training.log | grep -E "(Loading|windows|Epoch|GPU)" || echo "Waiting for logs..."
else
    echo ""
    echo "❌ Training failed! Check logs:"
    tail -50 $LOG_DIR/training.log
fi