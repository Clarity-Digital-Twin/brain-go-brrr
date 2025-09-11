# Cleanup TODO - Technical Debt

**Created**: September 10, 2025  
**Purpose**: Track slop and technical debt to clean up later

## TUAB Dataset Slop (LOW PRIORITY)

### Files to Remove/Consolidate:
- `src/brain_go_brrr/infra/data/tuab_cached_dataset.py` - Deprecation stub (387 bytes)
- `src/brain_go_brrr/infra/data/tuab_enhanced_dataset.py` - Deprecation stub (391 bytes)

These are just aliases to `tuab_dataset.py` for backward compatibility. Can be deleted once we verify nothing uses them.

## TUEV Dataset Slop (MEDIUM PRIORITY)

### Files to Remove:
- `src/brain_go_brrr/infra/data/tuev_dataset.py` - OLD WRONG sliding window implementation (15KB)
  - Used only in archived/old test files
  - The CORRECT implementation is `tuev_event_dataset.py`
  - DELETE after TUEV training is working

## Other Technical Debt

- Multiple experiment directories with similar code
- Archive folders that could be cleaned up
- Old test files referencing deprecated implementations

---
**NOTE**: Focus on TUEV fixes first. Clean this up after achieving paper parity.