#!/bin/bash
# Download TUEV v2.0.1 dataset for reference EEGPT repository
# This script can be copied to reference_repos/EEGPT/ along with .env file

# Source credentials from .env file in same directory as script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

# Configuration
DATASET="tuev"
VERSION="v2.0.1"
REMOTE_PATH="data/tuh_eeg/tuh_eeg_events/${VERSION}/"
LOCAL_PATH="data/datasets/tuev/"
LOG_FILE="tuev_download.log"

# Create directory
mkdir -p ${LOCAL_PATH}

echo "================================================"
echo "Starting TUEV ${VERSION} download for reference repo"
echo "This will download the TUEV event detection dataset"
echo "Download is resumable - safe to interrupt with Ctrl+C"
echo "================================================"
echo ""
echo "Credentials loaded from .env:"
echo "  Username: $TUH_USERNAME"
echo "  Server: www.isip.piconepress.com"
echo "  Remote: ${REMOTE_PATH}"
echo "  Local: ${LOCAL_PATH}"
echo "================================================"
echo ""

# Run rsync with progress and logging
# -a: archive mode (preserves everything)
# -u: skip files that are newer on receiver
# -x: don't cross filesystem boundaries
# -v: verbose
# -L: follow symlinks
# --progress: show progress during transfer
# --partial: keep partially transferred files
rsync -auxvL \
    --progress \
    --partial \
    --log-file=${LOG_FILE} \
    ${TUH_USERNAME}@www.isip.piconepress.com:${REMOTE_PATH} \
    ${LOCAL_PATH}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ TUEV download completed successfully!"
    echo "================================================"
    
    # Create completion marker
    touch ${LOCAL_PATH}/.download_complete
    
    # Show statistics
    echo ""
    echo "Dataset statistics:"
    echo "-------------------"
    echo "Total EDF files: $(find ${LOCAL_PATH} -name "*.edf" | wc -l)"
    echo "Total CSV files: $(find ${LOCAL_PATH} -name "*.csv" | wc -l)"
    echo "Disk usage: $(du -sh ${LOCAL_PATH} | cut -f1)"
    
    # Show class distribution
    echo ""
    echo "Event class distribution:"
    echo "------------------------"
    if [ -f "${LOCAL_PATH}/v2.0.1/edf/train/events.csv" ]; then
        echo "Training set events:"
        cut -d',' -f3 ${LOCAL_PATH}/v2.0.1/edf/train/events.csv | tail -n +2 | sort | uniq -c | sort -rn
    fi
else
    echo ""
    echo "================================================"
    echo "⚠️ Download interrupted or failed"
    echo "Run this script again to resume"
    echo "Check ${LOG_FILE} for details"
    echo "================================================"
    exit 1
fi