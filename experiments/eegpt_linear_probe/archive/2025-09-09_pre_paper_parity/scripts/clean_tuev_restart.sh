#!/bin/bash
# Clean TUEV training and restart with fixed hyperparameters
# Created: 2025-09-08
# Purpose: Kill old training, archive logs, clear caches

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
DATA_ROOT="${BGB_DATA_ROOT:-$PROJECT_ROOT/data}"

echo "=============================================="
echo "TUEV Training Clean Restart"
echo "=============================================="

# Step 1: Kill any existing training
echo "1. Killing existing TUEV sessions..."
tmux kill-session -t tuev_mne_training 2>/dev/null || echo "  - No active training"
tmux kill-session -t tuev_cache 2>/dev/null || echo "  - No active cache build"

# Step 2: Archive old logs
ARCHIVE_DIR="$EXPERIMENT_DIR/logs/archive_$(date +%Y%m%d_%H%M%S)"
echo "2. Archiving old logs to $ARCHIVE_DIR..."
mkdir -p "$ARCHIVE_DIR"
mv "$EXPERIMENT_DIR"/logs/tuev_*.log "$ARCHIVE_DIR/" 2>/dev/null || echo "  - No logs to archive"

# Step 3: Clear old caches (keep the fixed one)
echo "3. Clearing old TUEV caches..."
rm -rf "$DATA_ROOT/cache/tuev_mne" 2>/dev/null || true
rm -rf "$DATA_ROOT/cache/tuev_mne_v2" 2>/dev/null || true
rm -rf "$DATA_ROOT/cache/tuev_mne_test" 2>/dev/null || true
rm -rf "$DATA_ROOT/cache/tuev_mne_test_fix" 2>/dev/null || true
echo "  - Keeping tuev_mne_fixed for new training"

# Step 4: Show config changes
echo ""
echo "4. Configuration changes applied:"
echo "  ✅ Fpz interpolation from Fp1/Fp2 (was zeros)"
echo "  ✅ No class weights (was weighted)"
echo "  ✅ weight_decay: 0.05 (was 0.01)"
echo "  ✅ label_smoothing: 0.1 (was none)"
echo ""

echo "=============================================="
echo "Ready to rebuild cache and retrain!"
echo ""
echo "Next steps:"
echo "  1. Rebuild cache: ./launch_tuev_cache.sh"
echo "  2. Start training: ./launch_tuev_mne.sh"
echo "=============================================="