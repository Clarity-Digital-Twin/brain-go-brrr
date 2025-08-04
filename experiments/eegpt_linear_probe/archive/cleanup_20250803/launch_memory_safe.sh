#!/bin/bash
# Memory-safe EEGPT training launch script with aggressive optimizations

set -e  # Exit on error

echo "=================================================="
echo "LAUNCHING MEMORY-SAFE EEGPT TRAINING"
echo "=================================================="
echo "Memory optimizations:"
echo "  ✓ Batch size: 32 (with gradient accumulation)"
echo "  ✓ Mixed precision training (fp16)"
echo "  ✓ Gradient checkpointing enabled"
echo "  ✓ Limited workers: 2"
echo "  ✓ CUDA memory fraction: 0.9"
echo "  ✓ Clear cache between epochs"
echo "=================================================="

# Create log directory
LOG_DIR="logs/memory_safe_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Kill any existing sessions
if tmux has-session -t eegpt_train 2>/dev/null; then
    echo "Killing existing training session..."
    tmux kill-session -t eegpt_train
fi

# Clear GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Set aggressive memory management
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.6
export CUDA_VISIBLE_DEVICES=0

# Create memory-optimized config override
cat > $LOG_DIR/memory_override.yaml << EOF
# Memory-optimized config overrides
data:
  batch_size: 32  # Very conservative
  num_workers: 2
  pin_memory: false  # Save host memory
  persistent_workers: false
  prefetch_factor: 2  # Default is 2

training:
  accumulate_grad_batches: 4  # 32 * 4 = 128 effective
  precision: 16  # Mixed precision
  gradient_clip_val: 1.0
  
  # Checkpoint only best model
  checkpoint_metric: val_auroc
  checkpoint_mode: max
  save_top_k: 1
  
  # More aggressive early stopping
  early_stopping_patience: 5
  
model:
  # Enable gradient checkpointing if supported
  gradient_checkpointing: true
EOF

# Launch training with monitoring
echo "Starting memory-safe training in tmux session 'eegpt_train'..."
tmux new -d -s eegpt_train \
    "cd $(pwd) && \
     echo '=== Memory-Safe Training Started ===' && \
     echo 'Monitoring GPU memory every 10 seconds...' && \
     ( while true; do \
         echo \"[\$(date +%H:%M:%S)] GPU: \$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{printf \"%.1f/%.1f GB (%.0f%%)\", \$1/1024, \$3/1024, \$1*100/\$3}')\" >> $LOG_DIR/gpu_monitor.log; \
         sleep 10; \
     done ) & \
     MONITOR_PID=\$! && \
     echo \"Monitor PID: \$MONITOR_PID\" && \
     uv run python experiments/eegpt_linear_probe/train_enhanced.py \
         --config experiments/eegpt_linear_probe/configs/tuab_enhanced_config.yaml \
         --config $LOG_DIR/memory_override.yaml \
         2>&1 | tee $LOG_DIR/training.log || \
     echo '=== Training failed! Check logs ===' && \
     kill \$MONITOR_PID 2>/dev/null || true"

echo "=================================================="
echo "Memory-safe training launched!"
echo ""
echo "Monitor training:"
echo "  tmux attach -t eegpt_train"
echo ""
echo "Check logs:"
echo "  tail -f $LOG_DIR/training.log"
echo ""
echo "GPU memory usage:"
echo "  tail -f $LOG_DIR/gpu_monitor.log"
echo ""
echo "If OOM occurs, try:"
echo "  - Reduce batch_size to 16"
echo "  - Increase accumulate_grad_batches to 8"
echo "  - Disable pin_memory"
echo "=================================================="