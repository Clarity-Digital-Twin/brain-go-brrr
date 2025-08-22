# Documentation Reorganization Plan

## Current Problems
- **130 total .md files** (35 active + 95 archived) - WAY too many
- Outdated information (dates in future, non-existent features)
- Redundant architecture docs (10 files for architecture alone!)
- Misalignment with actual codebase state
- Confusing hierarchy with too many subdirectories

## Actual Codebase State (Reality Check)

### ✅ WORKING
1. **YASA Sleep Analysis** - 100% functional, 87% accuracy
2. **Autoreject QC** - 100% functional, good test coverage
3. **Basic FastAPI** - Health checks, core endpoints
4. **CI/CD Pipeline** - GitHub Actions working (after formatting fixes)
5. **Redis Caching** - Implemented and tested

### 🟡 IN PROGRESS
1. **EEGPT Abnormality Detection** - Training now (batch 292/7286)
2. **EEGPT Integration** - Model loading works, features extracting

### ❌ NOT IMPLEMENTED
1. **Event Detection** - Only architecture docs exist
2. **Authentication/Security** - No OAuth2, no JWT
3. **Kubernetes/Production** - No deployment infrastructure
4. **PostgreSQL/TimescaleDB** - Not configured
5. **Celery Queue** - Not implemented

## Proposed New Structure (LEAN & ACCURATE)

```
docs/
├── README.md                 # Main documentation hub
├── QUICK_START.md           # Get running in 5 minutes
├── ARCHITECTURE.md          # Single consolidated architecture doc
├── API.md                   # API reference (actual endpoints only)
├── TRAINING.md              # How to train models (TUAB/TUEV)
├── TESTING.md               # Testing guide
├── DEPLOYMENT.md            # Future deployment plans
└── archive/                 # Everything else goes here
```

## Consolidation Plan

### 1. Architecture (10 files → 1 file)
**KEEP**: Core concepts from DUAL_PIPELINE_ARCHITECTURE.md
**MERGE**:
- QUALITY_CONTROL_SYSTEM.md → Section in ARCHITECTURE.md
- SLEEP_ANALYSIS_INTEGRATION.md → Section in ARCHITECTURE.md
- SOLID_ARCHITECTURE_PATTERNS.md → Brief principles section

**ARCHIVE**:
- ABNORMALITY_DETECTION_PIPELINE.md (not fully implemented)
- EVENT_DETECTION_ARCHITECTURE.md (not implemented)
- HIERARCHICAL_PIPELINE_DESIGN.md (overcomplicated)
- AUTOREJECT_PRECISE_INTEGRATION_SPEC.md (already done)
- AUTOREJECT_TDD_IMPLEMENTATION_PLAN.md (already done)
- ARCHITECTURE_AUDIT_SUMMARY.md (outdated)

### 2. Implementation (8 files → absorbed into other docs)
**MERGE**:
- EEGPT_IMPLEMENTATION_GUIDE.md → TRAINING.md
- TESTING_BEST_PRACTICES.md → TESTING.md
- DOCKER_QUICKSTART.md → QUICK_START.md
- PC_SETUP_GUIDE.md → QUICK_START.md

**ARCHIVE**:
- GITHUB_ACTIONS_CLAUDE_CODE.md (move to archive/ci-cd/)
- CONFIGURATION_CHECKLIST.md (outdated)
- TRAINING_GUIDE.md (replaced by experiments/eegpt_linear_probe/README.md)

### 3. Requirements (4 files → 1 reference file)
**CREATE**: REQUIREMENTS.md (simplified, actual requirements only)
**ARCHIVE**: All existing requirement docs (BDD, PRD, TRD, SPEC)

### 4. Root Docs Update
**UPDATE**:
- `/README.md` - Simplify, point to docs/README.md
- `/CLAUDE.md` - Update with current state (remove PyTorch Lightning warning, etc.)
- `/PROJECT_STATUS.md` - Update with actual status

**ARCHIVE**:
- Redundant status files
- Old planning documents

## Action Items

1. **Create consolidated docs** (7 clean files)
2. **Archive everything else** (120+ files to archive)
3. **Update root documentation**
4. **Remove future dates and aspirational features**
5. **Add clear "NOT IMPLEMENTED" sections where needed**

## Success Metrics
- Documentation matches code 1:1
- New developer can start in <30 minutes
- No redundant information
- Clear separation of working vs planned features
- Total active docs: <10 files
