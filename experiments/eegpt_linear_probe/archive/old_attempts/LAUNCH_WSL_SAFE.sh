#!/bin/bash
# LAUNCH TRAINING WITH WSL-SAFE CONFIGURATION

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== LAUNCHING WSL-SAFE TRAINING ===${NC}"
echo ""

# Check if in WSL
if grep -qi microsoft /proc/version; then
    echo -e "${YELLOW}⚠️  WSL DETECTED - Using safe configuration${NC}"
    echo -e "${YELLOW}   - num_workers=0 (no multiprocessing)${NC}"
    echo -e "${YELLOW}   - pin_memory=false (WSL stability)${NC}"
else
    echo -e "${GREEN}✅ Native Linux detected${NC}"
fi

# Set environment
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Disable multiprocessing for DataLoader (WSL fix)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo ""
echo -e "${CYAN}Environment:${NC}"
echo "  BGB_DATA_ROOT: $BGB_DATA_ROOT"
echo "  CUDA: Device 0"
echo "  Config: configs/tuab_4s_wsl_safe.yaml"
echo ""

# Check cache exists
CACHE_DIR="$BGB_DATA_ROOT/cache/tuab_4s_final"
if [ ! -d "$CACHE_DIR" ]; then
    echo -e "${RED}❌ Cache not found at $CACHE_DIR${NC}"
    echo "Run: python build_mmap_cache.py first!"
    exit 1
fi

# Check for memory-mapped files
if [ ! -f "$CACHE_DIR/train_data.npy" ]; then
    echo -e "${YELLOW}⚠️  Memory-mapped arrays not found${NC}"
    echo "Building them now..."
    cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe
    /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python build_mmap_cache.py
fi

echo -e "${GREEN}✅ Cache ready${NC}"
echo ""

# Create log directory
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/training_wsl_safe_$(date +%Y%m%d_%H%M%S).log"

echo -e "${CYAN}Starting training...${NC}"
echo "Log: $LOG_FILE"
echo ""

# Launch training with WSL-safe config
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

# Use tmux for persistent session
SESSION_NAME="eegpt_wsl_safe"
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

echo -e "${CYAN}Launching in tmux session: $SESSION_NAME${NC}"
tmux new-session -d -s $SESSION_NAME \
    "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python train_paper_aligned.py \
    --config configs/tuab_4s_wsl_safe.yaml 2>&1 | tee $LOG_FILE"

echo ""
echo -e "${GREEN}=== TRAINING LAUNCHED ===${NC}"
echo ""
echo "Monitor with:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "View logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo -e "${YELLOW}NOTE: Using num_workers=0 for WSL stability${NC}"
echo -e "${YELLOW}Speed should still be 200+ it/s with memory-mapped arrays${NC}"