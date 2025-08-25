#!/bin/bash
# Monitor MNE training progress

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$EXPERIMENT_DIR/logs"

echo "=============================================="
echo "MNE Training Monitor"
echo "=============================================="

# Find latest MNE log file
LATEST_LOG=$(ls -t "$LOGS_DIR"/tuab_mne_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No MNE training logs found in $LOGS_DIR"
    echo ""
    echo "Available logs:"
    ls -la "$LOGS_DIR"/*.log 2>/dev/null || echo "No log files found"
    exit 1
fi

echo "Monitoring: $LATEST_LOG"
echo ""
echo "Key metrics to watch:"
echo "- AUROC should improve from ~56% to 75-87%"
echo "- Rejection rate: ~20-40% of epochs (normal for clinical data)"
echo "- Loss should decrease steadily"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "=============================================="
echo ""

# Extract key metrics
echo "Latest metrics:"
grep -E "AUROC|Loss|Epoch" "$LATEST_LOG" | tail -5

echo ""
echo "Preprocessing statistics:"
grep -E "Autoreject:|RANSAC|bad channels|muscle artifacts" "$LATEST_LOG" | tail -10

echo ""
echo "=============================================="
echo "Live log (last 20 lines):"
echo ""

# Follow log file
tail -f "$LATEST_LOG"