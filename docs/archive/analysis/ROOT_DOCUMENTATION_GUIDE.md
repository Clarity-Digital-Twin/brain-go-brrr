# 📚 ROOT DOCUMENTATION GUIDE

## Current Root Documentation Structure (After Cleanup)

### ✅ KEEP IN ROOT (Essential)
1. **CLAUDE.md** - AI assistant guidelines (permanent)
2. **README.md** - Project overview (permanent)
3. **CHANGELOG.md** - Version history (permanent)
4. **PROJECT_STATUS.md** - Current project status (permanent)
5. **RELEASE_NOTES.md** - Current release notes (permanent)

### 🔧 ACTIVE REFACTORING DOCS (Temporary - Remove after completion)
1. **CLEANUP_PLAN.md** - Initial cleanup plan
2. **REFACTORING_INVESTIGATION.md** - Deep dive findings
3. **COMPREHENSIVE_REFACTORING_FIX_PLAN.md** - Detailed fix plan for 110+ files
4. **CRITICAL_DIVE_SAFETY_REPORT.md** - Safety verification & corrections

These will be moved to `archive/refactoring_2025_08/` after completion.

## 📁 Archive Organization

### Already Moved to Archive:
```
archive/
├── eegpt_fixes/
│   ├── EEGPT_FILES_COMPARISON.md
│   ├── EEGPT_FIX_SUMMARY.md
│   ├── EEGPT_TEMPORAL_FIXES_COMPLETE.md
│   └── EEGPT_TUEV_FIX.md
├── implementation_reports/
│   ├── FINAL_IMPLEMENTATION_REPORT.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── IMPLEMENTATION_VERIFICATION_REPORT.md
│   └── IMPACT_ANALYSIS.md
├── infrastructure_cleanup/
│   ├── INFRASTRUCTURE_CLEANUP_COMPLETE.md
│   ├── INFRA_CLEANUP_ACTION_PLAN.md
│   └── INFRA_ML_MODELS_INVESTIGATION.md
├── tuab_tuev_docs/
│   ├── TUAB_FIX.md
│   ├── TUEV_QUICK_START.md
│   └── TUEV_UNIFIED_SPECS.md
├── EXPERIMENTS_CLEANUP_SUMMARY.md
├── EXPERIMENTS_READY_STATUS.md
├── CRITICAL_ISSUES_DEEP_AUDIT.md
└── DOWNLOAD_STATUS.md
```

### Moved to docs/:
```
docs/
└── releases/
    └── RELEASE_v1.0.0.md
```

## 📝 Documentation Rules

### When to Create New Root Docs:
1. **Active work** - Only for current, active development
2. **Temporary** - Should have a clear end date
3. **Critical** - Must be immediately visible to team

### When to Archive:
1. **Work complete** - Move to archive immediately after completion
2. **Superseded** - When replaced by newer documentation
3. **Historical** - Keep for reference but not active use

### When to Use docs/:
1. **Permanent documentation** - API docs, architecture docs
2. **Guides** - User guides, developer guides
3. **Specifications** - Technical specs, requirements

## 🎯 Current Focus

We are currently working on:
1. **Cleaning up deprecated code** from two refactorings:
   - Clean Architecture refactoring (core → domain/application/infra)
   - EEGPT model unification (3 models → 1 unified probe)

2. **Active documents for this work:**
   - COMPREHENSIVE_REFACTORING_FIX_PLAN.md - The master plan
   - CRITICAL_DIVE_SAFETY_REPORT.md - Safety checks
   - REFACTORING_INVESTIGATION.md - What we found

3. **Next steps:**
   - Fix Phase 1 (3 critical production files)
   - Run tests
   - Continue with remaining phases
   - Archive these docs when complete

## 🚦 Status
- **Root cleaned:** ✅ (reduced from 28 to 9 markdown files)
- **Archives organized:** ✅
- **Ready to proceed:** ✅

---

*Last updated: 2025-08-19*
