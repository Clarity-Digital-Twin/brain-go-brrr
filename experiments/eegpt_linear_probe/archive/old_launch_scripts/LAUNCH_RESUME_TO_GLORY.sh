#!/bin/bash
# 🚀 RESUME TRAINING TO GLORY - SHOCK THE WORLD!

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}============================================${NC}"
echo -e "${PURPLE}🚀 RESUMING TRAINING FROM EPOCH 16 🚀${NC}"
echo -e "${PURPLE}============================================${NC}"
echo ""

# Check checkpoint exists
CHECKPOINT="output/tuab_4s_paper_target_20250806_132743/best_model.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}❌ Checkpoint not found at $CHECKPOINT${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Found checkpoint: $CHECKPOINT${NC}"

# Set environment for WSL safety
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONFAULTHANDLER=1

# WSL multiprocessing safety
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo ""
echo -e "${CYAN}📊 Current Status:${NC}"
echo -e "  Last completed epoch: 15"
echo -e "  Best AUROC: 0.7916 (79.16%)"
echo -e "  Target AUROC: 0.869 (86.9%)"
echo -e "  Progress: 91.1% of target"
echo -e "  Estimated epochs to target: ~10-15"
echo ""

echo -e "${YELLOW}⚙️  Configuration:${NC}"
echo "  Batch size: 32"
echo "  Learning rate: OneCycle (max 3e-3)"
echo "  num_workers: 0 (WSL safe)"
echo "  pin_memory: false (WSL safe)"
echo "  Device: CUDA (RTX 4090)"
echo ""

# Check GPU
if nvidia-smi &>/dev/null; then
    echo -e "${GREEN}✅ GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo -e "${YELLOW}⚠️  No GPU detected, will use CPU${NC}"
fi
echo ""

# Create log directory
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/resume_training_$(date +%Y%m%d_%H%M%S).log"

echo -e "${CYAN}📝 Log file: $LOG_FILE${NC}"
echo ""

# Kill any existing sessions
tmux kill-session -t eegpt_resume 2>/dev/null || true

# Change to correct directory
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

echo -e "${PURPLE}============================================${NC}"
echo -e "${PURPLE}🔥 LAUNCHING TRAINING IN TMUX 🔥${NC}"
echo -e "${PURPLE}============================================${NC}"
echo ""

# Launch in tmux
tmux new-session -d -s eegpt_resume \
    "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_paper_aligned_resume.py \
    --config configs/tuab_4s_wsl_safe.yaml \
    --resume output/tuab_4s_paper_target_20250806_132743/best_model.pt \
    --device cuda 2>&1 | tee $LOG_FILE"

echo -e "${GREEN}✅ Training resumed in tmux session: eegpt_resume${NC}"
echo ""
echo -e "${CYAN}Monitor with:${NC}"
echo "  tmux attach -t eegpt_resume"
echo ""
echo -e "${CYAN}View logs:${NC}"
echo "  tail -f $LOG_FILE"
echo ""
echo -e "${CYAN}Check progress:${NC}"
echo "  grep AUROC $LOG_FILE | tail -5"
echo ""
echo -e "${CYAN}GPU usage:${NC}"
echo "  watch -n 1 nvidia-smi"
echo ""

echo -e "${PURPLE}============================================${NC}"
echo -e "${PURPLE}🎯 TARGET: 0.869 AUROC${NC}"
echo -e "${PURPLE}🚀 LET'S SHOCK THE WORLD!${NC}"
echo -e "${PURPLE}============================================${NC}"