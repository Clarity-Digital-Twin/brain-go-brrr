#!/bin/bash
# Archive deprecated documentation to clean up repo root

ARCHIVE_DIR="docs/archive/deprecated_2025-07-30"
mkdir -p "$ARCHIVE_DIR"

# List of deprecated docs to move
DEPRECATED_DOCS=(
    "WARNING_OUTDATED_DOCS.md"
    "FIX_EEGPT_EMBED_DIM.md"
    "PROJECT_STATUS_UPDATED.md"
    "EEGPT_TRUE_STATUS.md"
    "TEMP_*.md"
)

echo "Archiving deprecated documentation..."

for doc in "${DEPRECATED_DOCS[@]}"; do
    if ls $doc 1> /dev/null 2>&1; then
        echo "Moving: $doc"
        mv $doc "$ARCHIVE_DIR/" 2>/dev/null || echo "  Already moved or not found"
    fi
done

# Create README in archive
cat > "$ARCHIVE_DIR/README.md" << EOF
# Deprecated Documentation Archive

This directory contains documentation that was created during the EEGPT debugging process
but is now outdated after the fixes were implemented.

**For current status, see:**
- [/EEGPT_FIXED_SUMMARY.md](/EEGPT_FIXED_SUMMARY.md) - Summary of fixes applied
- [/PROJECT_STATUS_FINAL.md](/PROJECT_STATUS_FINAL.md) - Current project status

## Archived Files

These files document the investigation and debugging process but contain outdated information:

- WARNING_OUTDATED_DOCS.md - Initial warning about conflicting docs
- FIX_EEGPT_EMBED_DIM.md - Proposed fix (partially correct)
- PROJECT_STATUS_UPDATED.md - Intermediate status
- EEGPT_TRUE_STATUS.md - Investigation findings

**Note**: These files are kept for historical reference but should not be used for
understanding the current system state.
EOF

echo "Archive complete. Created: $ARCHIVE_DIR"
echo "Don't forget to update README.md to point to current docs!"
