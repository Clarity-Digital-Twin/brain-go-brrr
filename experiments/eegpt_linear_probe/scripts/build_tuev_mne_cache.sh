#!/bin/bash
# Build MNE-preprocessed cache for TUEV dataset

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
DATA_ROOT="${BGB_DATA_ROOT:-$PROJECT_ROOT/data}"

# Cache parameters
TUEV_ROOT="$DATA_ROOT/datasets/external/tuh_eeg/TUEV/v2.0.1"
CACHE_DIR="$DATA_ROOT/cache/tuev_mne_preprocessed"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$EXPERIMENT_DIR/logs/tuev_cache_build_$TIMESTAMP.log"

echo "=============================================="
echo "Building MNE Cache for TUEV Dataset"
echo "=============================================="
echo "TUEV root: $TUEV_ROOT"
echo "Cache dir: $CACHE_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "This will preprocess all TUEV EDF files with:"
echo "  - 23→20 channel mapping"
echo "  - MNE filtering (0.5-45 Hz bandpass, 60 Hz notch)"
echo "  - Autoreject artifact removal"
echo "  - 4-second window segmentation"
echo "=============================================="

# Check if TUEV data exists
if [ ! -d "$TUEV_ROOT" ]; then
    echo "ERROR: TUEV dataset not found at $TUEV_ROOT"
    echo "Please download TUEV v2.0.1 first"
    exit 1
fi

# Count EDF files
TRAIN_FILES=$(find "$TUEV_ROOT/edf/train" -name "*.edf" 2>/dev/null | wc -l)
EVAL_FILES=$(find "$TUEV_ROOT/edf/eval" -name "*.edf" 2>/dev/null | wc -l)
TOTAL_FILES=$((TRAIN_FILES + EVAL_FILES))

echo "Found $TOTAL_FILES EDF files ($TRAIN_FILES train, $EVAL_FILES eval)"
echo ""

# Estimate time (assuming ~30 seconds per file with Autoreject)
ESTIMATED_MINUTES=$((TOTAL_FILES * 30 / 60))
ESTIMATED_HOURS=$((ESTIMATED_MINUTES / 60))
echo "Estimated time: ~${ESTIMATED_HOURS} hours (${ESTIMATED_MINUTES} minutes)"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create directories
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$CACHE_DIR"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Virtual environment activated"
fi

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Change to experiment directory
cd "$EXPERIMENT_DIR"

# Create cache builder Python script
cat > "$EXPERIMENT_DIR/mne_integration/tuev_cache_builder.py" << 'EOF'
#!/usr/bin/env python3
"""
Build MNE-preprocessed cache for TUEV dataset.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.eegpt_linear_probe.datasets.tuev_mne_dataset import TUEVMNEDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Build MNE-preprocessed cache for TUEV dataset')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory containing TUEV EDF files',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        required=True,
        help='Directory to save preprocessed cache',
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'eval', 'both'],
        default='both',
        help='Which split(s) to build cache for',
    )
    parser.add_argument(
        '--force-rebuild', action='store_true', help='Force rebuilding cache even if it exists'
    )

    args = parser.parse_args()

    # Determine which splits to process
    splits = ['train', 'eval'] if args.split == 'both' else [args.split]

    # Build cache for each split
    for split in splits:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Building cache for {split} split")
        logger.info(f"{'=' * 60}")

        try:
            dataset = TUEVMNEDataset(
                root_dir=Path(args.data_root),
                split=split,
                cache_dir=Path(args.cache_dir),
                force_rebuild=args.force_rebuild,
            )

            logger.info(f"Successfully built cache for {split} split")
            logger.info(f"Dataset contains {len(dataset)} windows")

            # Test loading a sample
            if len(dataset) > 0:
                x, y = dataset[0]
                logger.info(f"Sample shape: {x.shape}, Label: {y.item()}")

        except Exception as e:
            logger.error(f"Failed to build cache for {split} split: {e}")
            raise

    logger.info(f"\n{'=' * 60}")
    logger.info("Cache building complete!")
    logger.info(f"Cache saved to: {args.cache_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
EOF

# Launch cache building
echo ""
echo "Starting cache build..."
echo "To monitor: tail -f $LOG_FILE"
echo ""

# Use tmux for long-running process
if command -v tmux &> /dev/null; then
    SESSION_NAME="tuev_cache_build"

    # Kill existing session if it exists
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

    # Start new session
    tmux new-session -d -s "$SESSION_NAME" \
        "cd $EXPERIMENT_DIR && \
         python mne_integration/tuev_cache_builder.py \
            --data-root $TUEV_ROOT \
            --cache-dir $CACHE_DIR \
            --split both \
            2>&1 | tee $LOG_FILE"

    echo "Cache building launched in tmux session: $SESSION_NAME"
    echo "To attach: tmux attach -t $SESSION_NAME"
    echo "To detach: Ctrl+B, then D"
else
    # Direct execution
    python mne_integration/tuev_cache_builder.py \
        --data-root "$TUEV_ROOT" \
        --cache-dir "$CACHE_DIR" \
        --split both \
        2>&1 | tee "$LOG_FILE"
fi
