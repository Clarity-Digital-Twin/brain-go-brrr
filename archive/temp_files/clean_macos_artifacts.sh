#!/bin/bash
# Clean macOS ._ metadata files from TUAB dataset

echo "=== CLEANING macOS ARTIFACTS FROM TUAB DATASET ==="
echo ""

# Safety check - count files first
TUAB_DIR="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuh_eeg_abnormal"
COUNT=$(find "$TUAB_DIR" -name "._*" -type f | wc -l)

echo "Found $COUNT macOS metadata files (._* files)"
echo "These are all 163-byte metadata files created by macOS"
echo ""

# Show a sample
echo "Sample files to be removed:"
find "$TUAB_DIR" -name "._*" -type f | head -5

echo ""
read -p "Remove all $COUNT ._ files? (y/N): " confirm

if [[ $confirm == "y" || $confirm == "Y" ]]; then
    echo "Removing macOS metadata files..."
    find "$TUAB_DIR" -name "._*" -type f -delete
    echo "✓ Removed $COUNT macOS metadata files"
    
    # Verify
    REMAINING=$(find "$TUAB_DIR" -name "._*" -type f | wc -l)
    echo "Remaining ._ files: $REMAINING"
    
    # Show new file counts
    echo ""
    echo "=== CLEAN DATASET STATS ==="
    echo "Train normal: $(find "$TUAB_DIR/v3.0.1/edf/train/normal" -name "*.edf" | wc -l)"
    echo "Train abnormal: $(find "$TUAB_DIR/v3.0.1/edf/train/abnormal" -name "*.edf" | wc -l)"
    echo "Eval normal: $(find "$TUAB_DIR/v3.0.1/edf/eval/normal" -name "*.edf" | wc -l)"
    echo "Eval abnormal: $(find "$TUAB_DIR/v3.0.1/edf/eval/abnormal" -name "*.edf" | wc -l)"
else
    echo "Cleanup cancelled"
fi