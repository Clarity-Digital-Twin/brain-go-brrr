#!/bin/bash
# Build MNE-preprocessed cache for TUAB dataset

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
DATA_ROOT="${BGB_DATA_ROOT:-$PROJECT_ROOT/data}"

echo "=============================================="
echo "Building MNE-Preprocessed Cache for TUAB"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo ""
echo "⚠️  CRITICAL: Channel Enforcement Active ⚠️"
echo "This build enforces EXACTLY 19 channels per window."
echo "Files with 20 channels (including Fz) will have Fz dropped."
echo "This prevents the 20-channel bug found in older caches."
echo ""
echo "KNOWN ISSUE: Existing cache (mne-ar-v2) has 304 windows"
echo "with 20 channels from files aaaaakfo_s004/s005_t000.edf"
echo "See TECH_DEBT_CRITICAL.md for details."
echo "=============================================="

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Virtual environment activated"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Check if MNE and Autoreject are installed
echo ""
echo "Checking dependencies..."
python -c "import mne; print(f'MNE version: {mne.__version__}')" || {
    echo "ERROR: MNE not installed. Run: pip install mne"
    exit 1
}

python -c "import autoreject; print(f'Autoreject version: {autoreject.__version__}')" || {
    echo "ERROR: Autoreject not installed. Run: pip install autoreject"
    exit 1
}

# Build cache
echo ""
echo "Starting cache build..."
python experiments/eegpt_linear_probe/mne_integration/cache_builder.py \
    --data-root "$DATA_ROOT/datasets/external/tuab" \
    --cache-dir "$DATA_ROOT/cache/tuab_mne_preprocessed" \
    --split both \
    "$@"

echo ""
echo "=============================================="
echo "Cache build complete!"
echo "Cache location: $DATA_ROOT/cache/tuab_mne_preprocessed"
echo "=============================================="
