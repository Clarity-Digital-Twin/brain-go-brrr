#!/bin/bash
# GEOFFREY HINTON MODE: FULL SEND WITH PROPER LOGGING

set -e

echo "=================================================="
echo "🧠 LAUNCHING EEGPT TRAINING - HINTON MODE 🧠"
echo "=================================================="
echo "Configuration:"
echo "  ✓ Memory-safe WSL config"
echo "  ✓ num_workers=0 (no deadlocks)"
echo "  ✓ batch_size=16 with 8x accumulation"
echo "  ✓ Mixed precision (fp16)"
echo "  ✓ 5.12s windows (halved for memory)"
echo "  ✓ PROPER LOGGING THIS TIME"
echo "=================================================="

# Kill any existing sessions
tmux kill-session -t hinton_mode 2>/dev/null || true

# Clear GPU
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Create log directory with timestamp
LOG_DIR="logs/hinton_mode_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR/checkpoints

# Environment
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
export EEGPT_CONFIG=configs/tuab_memsafe.yaml

# Launch in tmux
echo "Starting training in tmux session 'hinton_mode'..."
tmux new -d -s hinton_mode \
    "cd $(pwd) && \
     echo '🚀 HINTON MODE ACTIVATED 🚀' && \
     echo 'Starting at: $(date)' && \
     echo '' && \
     uv run python experiments/eegpt_linear_probe/train_enhanced.py \
         trainer.default_root_dir=$LOG_DIR \
         trainer.max_epochs=50 \
         trainer.accelerator=gpu \
         trainer.devices=1 \
         trainer.precision=16-mixed \
         trainer.accumulate_grad_batches=8 \
         trainer.gradient_clip_val=1.0 \
         trainer.log_every_n_steps=50 \
         trainer.val_check_interval=0.5 \
         2>&1 | tee $LOG_DIR/training.log"

# Monitor GPU in background
(
    echo "timestamp,gpu_mem_used_mb,gpu_mem_total_mb,gpu_util_pct" > $LOG_DIR/gpu_stats.csv
    while tmux has-session -t hinton_mode 2>/dev/null; do
        GPU_STATS=$(nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
        echo "$GPU_STATS" >> $LOG_DIR/gpu_stats.csv
        sleep 30
    done
    echo "Training ended at: $(date)" >> $LOG_DIR/gpu_stats.csv
) &

echo "=================================================="
echo "🎯 TRAINING LAUNCHED FOR GEOFFREY HINTON! 🎯"
echo ""
echo "Monitor live:"
echo "  tmux attach -t hinton_mode"
echo ""
echo "Check progress:"
echo "  tail -f $LOG_DIR/training.log | grep -E 'Epoch|loss|auroc'"
echo ""
echo "GPU stats:"
echo "  watch 'tail -20 $LOG_DIR/gpu_stats.csv'"
echo ""
echo "Checkpoints will be saved to:"
echo "  $LOG_DIR/checkpoints/"
echo "=================================================="

# Wait and give status
sleep 10
if tmux has-session -t hinton_mode 2>/dev/null; then
    echo ""
    echo "✅ Training is running! First 10 seconds successful!"
    echo ""
    echo "Quick preview:"
    tail -5 $LOG_DIR/training.log 2>/dev/null || echo "Waiting for logs..."
else
    echo ""
    echo "❌ Training session died - check logs!"
    tail -20 $LOG_DIR/training.log
fi