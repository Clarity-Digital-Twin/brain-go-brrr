# Archive Organization Plan

## Current State
- **Total files**: 94 markdown files
- **Subdirectories**: 13 (many with overlapping purposes)
- **Root files**: 30+ unsorted documents

## Proposed Structure

```
archive/
├── 01-checkpoints/           # Date-stamped progress checkpoints
├── 02-investigations/        # Bug investigations and fixes  
├── 03-planning/             # Old roadmaps and plans
├── 04-refactoring/          # Architecture refactoring history
├── 05-training/             # EEGPT training documentation
├── 06-technical-debt/       # Technical debt and cleanup
└── README.md                # Archive index and navigation
```

## Files to Consolidate/Remove

### Duplicate Checkpoints
- Keep only CHECKPOINT_2025-07-29_FINAL.md
- Archive others as they contain same info

### Redundant Investigation Files
- Merge INVESTIGATION_FINDINGS.md variations
- Keep only most comprehensive versions

### Similar Architecture Audits
- Consolidate 8+ architecture audit files
- Keep ARCHITECTURE_FINAL_STATE.md as main reference

## Action Items
1. Create new simplified structure
2. Move files to appropriate directories
3. Remove true duplicates
4. Create index for easy navigation