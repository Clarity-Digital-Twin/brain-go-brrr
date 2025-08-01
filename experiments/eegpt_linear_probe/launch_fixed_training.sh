#!/bin/bash
# Fixed training launcher - no more warmup_epochs crash!

set -e  # Exit on error

echo "🚀 Starting EEGPT Linear Probe Training (FIXED)"
echo "================================================"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_NAME="eegpt_fixed_${TIMESTAMP}"

# Create logs directory
mkdir -p logs

# Kill any existing sessions
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Launch in tmux
echo "📦 Creating tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)" "
    echo '🔧 Setting up environment...'
    source ~/.bashrc
    cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr
    source .venv/bin/activate || true
    
    echo '🏃 Starting training...'
    echo 'Session: $SESSION_NAME'
    echo 'Log: logs/fixed_${TIMESTAMP}.log'
    echo '================================'
    
    uv run python experiments/eegpt_linear_probe/train_paper_aligned.py 2>&1 | tee logs/fixed_${TIMESTAMP}.log
    
    echo ''
    echo '✅ Training completed (or crashed)'
    echo 'Check logs/fixed_${TIMESTAMP}.log for details'
    echo 'Press Enter to exit...'
    read
"

echo "✅ Training launched successfully!"
echo ""
echo "📊 Monitor with:"
echo "   tmux attach -t $SESSION_NAME"
echo "   tail -f logs/fixed_${TIMESTAMP}.log"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "🛑 To stop:"
echo "   tmux kill-session -t $SESSION_NAME"