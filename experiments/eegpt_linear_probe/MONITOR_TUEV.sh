#!/bin/bash
# Monitor TUEV training progress

echo "============================================"
echo "📊 TUEV TRAINING MONITOR"
echo "============================================"
echo
echo "Target metrics (from paper):"
echo "  Balanced Accuracy: 0.6232 ± 0.0114"
echo "  Weighted F1:       0.8187 ± 0.0063"
echo "  Cohen's Kappa:     0.6351 ± 0.0134"
echo
echo "============================================"
echo

# Check latest metrics from log
echo "📈 Latest training metrics:"
tail -50 logs/tuev_seed42_WORKING.log | grep -E "Epoch|Val" | tail -5

echo
echo "============================================"
echo

# Check if validation has started
if grep -q "Val" logs/tuev_seed42_WORKING.log; then
    echo "🎯 Best validation metrics so far:"
    grep "New best model" logs/tuev_seed42_WORKING.log | tail -1
else
    echo "⏳ Waiting for first validation epoch..."
fi

echo
echo "============================================"
echo
echo "To watch live: tmux attach -t tuev_training"
echo "To check log: tail -f logs/tuev_seed42_WORKING.log"
echo