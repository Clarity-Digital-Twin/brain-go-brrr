# Documentation Cleanup Plan
Generated: September 2, 2025

## Phase 1: Preserve Valuable Content ✅

### Move to Root
- [ ] docs/archive/completed/CHANGELOG.md → /CHANGELOG.md (valuable history)

### Keep in docs/archive (but add ARCHIVED banner)
These have historical value or implementation details worth preserving:
- docs/archive/completed/ARCHITECTURE_UNIFICATION_COMPLETED.md (implementation history)
- docs/archive/completed/TUEV_IMPLEMENTATION_SPEC.md (future reference)
- docs/archive/mne_general_docs/* (MNE implementation details)

## Phase 2: Delete Truly Stale Docs 🗑️

### CI/CD Fixes (DELETE - all obsolete)
- docs/archive/ci_cd_fixes/* (6 files)
  - References old CI problems that are now fixed
  - Contains wrong test counts (790+) and coverage (66%)

### Sleep-EDF Alignment (DELETE - mostly obsolete)
- docs/archive/implementation/2025-09-sleep-edf-alignment/CRASH_*.md (3 files)
  - Old crash investigations that are resolved
  - Contains sys.path.insert hacks

### Old Status Files (DELETE)
- docs/archive/completed/SENIOR_AUDITOR_PITCH.md
- docs/archive/completed/README_CHANGES_SUMMARY.md
- docs/archive/implementation/*/ALIGNMENT_STATUS.md
- docs/archive/implementation/*/CLEANUP_STATUS.md

## Phase 3: Add ARCHIVED Banners 🏷️

Add to top of all remaining archive files:
```markdown
> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Last accurate: [DATE FROM FILE]
```

## Phase 4: Update Documentation Hub 📚

Update docs/README.md:
- [x] Add missing CI_CD_SETUP.md and TUH_CORPUS_GUIDE.md
- [x] Update date to September 2, 2025
- [ ] Add note about CHANGELOG.md moving to root

## Summary Statistics

- **Total Archive Files**: 45
- **To Delete**: ~15-20 (truly stale)
- **To Preserve with Banner**: ~25-30 (historical value)
- **To Move to Root**: 1 (CHANGELOG.md)
- **Canonical Docs**: 8 (in docs/)

## Rationale for Deletions

Delete if ANY of these are true:
1. Contains wrong information (launch_tuab.sh, 790+ tests, 66% coverage)
2. References resolved issues (CI/CD fixes, crash investigations)
3. Duplicates canonical docs without adding value
4. Contains sys.path.insert or other bad practices

Keep if:
1. Contains unique implementation history
2. Provides context for future work (TUEV spec)
3. Documents important decisions or trade-offs
4. Has educational value for understanding the codebase evolution
