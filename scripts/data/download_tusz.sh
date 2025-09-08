#!/bin/bash
# Download TUSZ dataset with resumable rsync and progress monitoring

# Source credentials
source .env

# Configuration
DATASET="tusz"
VERSION="v2.0.3"
REMOTE_PATH="data/tuh_eeg/tuh_eeg_seizure/${VERSION}/"
LOCAL_PATH="data/datasets/tusz/"
LOG_FILE="data/datasets/tusz_download.log"

# Create directory
mkdir -p ${LOCAL_PATH}

echo "================================================"
echo "Starting TUSZ ${VERSION} download"
echo "This will download ~40GB of seizure detection data"
echo "Download is resumable - safe to interrupt with Ctrl+C"
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
    nedc-tuh-eeg@www.isip.piconepress.com:${REMOTE_PATH} \
    ${LOCAL_PATH}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ TUSZ download completed successfully!"
    echo "================================================"
    
    # Create completion marker
    touch ${LOCAL_PATH}/.download_complete
    
    # Show statistics
    echo ""
    echo "Dataset statistics:"
    echo "-------------------"
    echo "Total EDF files: $(find ${LOCAL_PATH} -name "*.edf" | wc -l)"
    echo "Total annotation files: $(find ${LOCAL_PATH} -name "*.csv" -o -name "*.txt" | wc -l)"
    echo "Disk usage: $(du -sh ${LOCAL_PATH} | cut -f1)"
else
    echo ""
    echo "================================================"
    echo "⚠️ Download interrupted or failed"
    echo "Run this script again to resume"
    echo "================================================"
fi