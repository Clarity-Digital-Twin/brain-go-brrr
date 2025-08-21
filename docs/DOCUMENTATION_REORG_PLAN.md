# Documentation Reorganization Plan

_Created: August 21, 2025_

## Current Documentation Assessment

### ✅ Already Updated (Accurate)
- `docs/00-overview/` - All updated in previous session
- `docs/01-architecture/` - All updated with actual implementation status
- `docs/03-api/API_REFERENCE.md` - Created fresh with actual endpoints
- `docs/CLEAN_ARCHITECTURE.md` - Updated with current structure

### 📁 docs/02-implementation/ Assessment

#### Keep and Update
1. **EEGPT_IMPLEMENTATION_GUIDE.md** - Good technical reference, needs minor updates
   - Verify against actual `infra/ml_models/eegpt_compat.py`
   - Add note about 4-second window requirement
   - Document summary token extraction issue

2. **DOCKER_QUICKSTART.md** - Docker is configured, keep as-is
   - Working docker-compose.yml exists
   - Useful for deployment

3. **TESTING_BEST_PRACTICES.md** - Good practices guide, keep
   - Aligns with actual test structure
   - References real fixtures

4. **GITHUB_ACTIONS_CLAUDE_CODE.md** - Claude bot integration guide, keep
   - Unique documentation for AI-assisted development
   - Actually implemented in `.github/workflows/claude.yml`

#### Archive (Outdated/Misleading)
1. **MVP_SUMMARY.md** - Outdated, shows old API structure
   - References old endpoint paths
   - Incorrect performance metrics

2. **ROUGH_DRAFT.md** - Product vision, not implementation
   - Aspirational features not implemented
   - Move to archive/planning/

3. **IMPLEMENTATION_PLAN.md** - Old planning document
   - References completed tasks as future work
   - Move to archive/planning/

4. **HYPERPARAMETER_OPTIMIZATION.md** - Not implemented
   - No hyperparameter tuning code exists
   - Move to archive/future-work/

5. **PC_SETUP_GUIDE.md** - Generic development setup
   - Not specific to this project
   - Could be simplified into README

6. **CONFIGURATION_CHECKLIST.md** - Likely outdated
   - Need to verify against actual configs

7. **TESTING_FIXES_SUMMARY.md** - Historical document
   - Documents past fixes, not current state
   - Move to archive/history/

### 📁 docs/03-api/ Assessment

#### Keep and Update
1. **API_REFERENCE.md** - Already updated, accurate

#### Update Needed
1. **API_DESIGN_PATTERNS.md** - Shows unimplemented patterns
   - Authentication not implemented
   - HIPAA compliance aspirational
   - Need to mark sections as "planned" vs "implemented"

### 📁 Other Documentation Issues

#### Missing Documentation
1. **Training Guide** - How to actually train the linear probe
   - Document `experiments/eegpt_linear_probe/train_paper_aligned.py`
   - Include TUAB dataset setup

2. **Deployment Guide** - Production deployment instructions
   - Beyond Docker quickstart
   - Environment variables needed

3. **Data Setup Guide** - How to prepare datasets
   - TUAB download and preprocessing
   - Sleep-EDF is downloaded but not documented

#### Redundant/Conflicting Docs
1. Multiple README-like files in different locations
2. Some docs reference old `core.*` structure (need to find all)
3. Aspirational features presented as implemented

## Proposed Actions

### 1. Create Archive Structure
```
docs/archive/
├── planning/          # Old planning docs
├── history/           # Historical changes/fixes
├── future-work/       # Not yet implemented features
└── deprecated/        # Outdated technical docs
```

### 2. Documentation Standards
- Add **"Current Status"** header to all docs
- Use consistent markers: ✅ Implemented, 🟡 In Progress, 🔴 Planned
- Reference actual code files with paths
- Remove or clearly mark aspirational features

### 3. Priority Updates
1. Fix API_DESIGN_PATTERNS.md to show what's actually implemented
2. Create TRAINING_GUIDE.md for linear probe training
3. Update EEGPT_IMPLEMENTATION_GUIDE.md with actual behavior
4. Archive all outdated planning documents

### 4. Documentation Hierarchy
```
docs/
├── 00-overview/       # Project overview (UPDATED ✅)
├── 01-architecture/   # System design (UPDATED ✅)
├── 02-guides/         # How-to guides (NEEDS WORK)
│   ├── training/      # Model training
│   ├── deployment/    # Production deployment
│   └── development/   # Development setup
├── 03-api/           # API documentation (MOSTLY DONE)
├── 04-testing/       # Testing documentation
└── archive/          # Historical/planning docs
```

## Next Steps

1. Archive outdated documents
2. Update remaining implementation docs
3. Create missing guides (training, deployment)
4. Ensure all docs reflect actual implementation
5. Remove aspirational claims without "planned" markers
