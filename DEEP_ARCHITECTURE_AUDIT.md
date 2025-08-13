# 🔍 DEEP ARCHITECTURE AUDIT - BRAIN-GO-BRRR

**Date**: 2025-08-13  
**Status**: IN PROGRESS  
**Objective**: Complete architectural audit and refactoring checklist

## 📂 Current Directory Structure Analysis

### Root-Level Files in src/brain_go_brrr/
- ✅ `__init__.py` - Package marker (CORRECT)
- ✅ `__main__.py` - Allows `python -m brain_go_brrr` (CORRECT - PROFESSIONAL)
- ✅ `_typing.py` - Type definitions (CORRECT)
- ✅ `cli.py` - Command-line interface (CORRECT)
- ✅ `mne_compat.py` - MNE compatibility layer (CORRECT)
- ✅ `py.typed` - PEP 561 marker for typed package (CORRECT - PROFESSIONAL)

**Verdict**: All root-level files are correct and professional. No changes needed.

## 🏗️ Architectural Layer Analysis

### ✅ CORRECT LAYERS (Following Clean Architecture)

#### 1. Domain Layer (`domain/`)
- **Purpose**: Pure business logic, zero external dependencies
- **Contains**: 
  - `domain/abnormal/` - Abnormality detection logic
  - `domain/quality/` - Quality control rules
  - `domain/sleep/` - Sleep analysis domain
  - `domain/channels.py` - Channel mapping
  - `domain/exceptions.py` - Domain exceptions
- **Status**: ✅ CORRECT

#### 2. Application Layer (`application/`)
- **Purpose**: Use cases, orchestration
- **Contains**:
  - `application/use_cases/` - Application services
  - `application/jobs/` - Job management
  - `application/pipeline/` - Processing pipelines
  - `application/training/` - Training workflows
  - `application/ports.py` - Port interfaces
- **Status**: ✅ CORRECT

#### 3. Infrastructure Layer (`infra/`)
- **Purpose**: External integrations, I/O
- **Contains**:
  - `infra/external/` - External adapters (YASA)
  - `infra/redis/` - Redis implementation
  - `infra/cache.py` - Cache implementations
  - `infra/serialization.py` - Serialization
- **Status**: ✅ CORRECT

#### 4. Presentation Layer (`api/`)
- **Purpose**: REST API, user interface
- **Contains**:
  - `api/app.py` - FastAPI application
  - `api/routers/` - REST endpoints
  - `api/schemas.py` - Request/response models
  - `api/auth.py` - Authentication
- **Status**: ✅ CORRECT

### ⚠️ PROBLEMATIC DIRECTORIES (Need Refactoring)

#### 1. `config/` - WRONG LOCATION
- **Files Found**:
  - `config/abnormality_config.py`
  - `config/base.py`
  - `config/__init__.py`
- **Problem**: Config should be in `application/config/` or `infra/config/`
- **Action**: Move to `application/config/`

#### 2. `tasks/` - REDUNDANT
- **Files Found**: `tasks/__init__.py` (empty)
- **Problem**: Tasks are already in `application/use_cases/tasks/`
- **Action**: DELETE (redundant)

#### 3. `training/` - REDUNDANT
- **Files Found**: `training/__init__.py` (empty)
- **Problem**: Training is already in `application/training/`
- **Action**: DELETE (redundant)

#### 4. `modules/` - VAGUE NAME
- **Files Found**: `modules/constraints.py`
- **Problem**: Unclear purpose, should be in domain or application
- **Action**: Move `constraints.py` to `domain/constraints.py`

#### 5. `core/` - COMPATIBILITY SHIMS ONLY
- **Status**: ✅ All files are redirects (CORRECT)
- **Action**: Keep as-is for backward compatibility

#### 6. `services/` - COMPATIBILITY SHIMS ONLY
- **Status**: ✅ Only contains compatibility exports
- **Action**: Keep as-is for backward compatibility

### 🔴 DIRECTORIES THAT DON'T FIT CLEAN ARCHITECTURE

#### 1. `data/` - Mixed Concerns
- **Problem**: Data loading is infrastructure concern
- **Action**: Move to `infra/data/`

#### 2. `models/` - Ambiguous
- **Problem**: ML models vs domain models confusion
- **Contains**: EEGPT models, linear probes
- **Action**: Move to `infra/ml_models/` (infrastructure concern)

#### 3. `preprocessing/` - Mixed Layer
- **Problem**: Some preprocessing is domain logic, some is infrastructure
- **Action**: 
  - Domain preprocessing → `domain/preprocessing/`
  - Infrastructure preprocessing → `infra/preprocessing/`

#### 4. `visualization/` - Presentation Concern
- **Problem**: Should be in presentation layer or separate
- **Action**: Move to `api/visualization/` or `presentation/visualization/`

#### 5. `utils/` - Grab Bag
- **Contains**: `deprecated_redirect.py`, `time.py`
- **Problem**: Utils is an anti-pattern
- **Action**: 
  - `deprecated_redirect.py` → Keep in utils (compatibility)
  - `time.py` → Move to `domain/common/` or `infra/common/`

## 📋 REFACTORING CHECKLIST

### Immediate Actions (Critical)
- [ ] Delete empty `tasks/` directory
- [ ] Delete empty `training/` directory  
- [ ] Move `config/` → `application/config/`
- [ ] Move `modules/constraints.py` → `domain/constraints.py`
- [ ] Delete empty `modules/` directory

### Layer Reorganization (Important)
- [ ] Move `data/` → `infra/data/`
- [ ] Move `models/` → `infra/ml_models/`
- [ ] Split `preprocessing/`:
  - [ ] Domain logic → `domain/preprocessing/`
  - [ ] Infrastructure → `infra/preprocessing/`
- [ ] Move `visualization/` → `presentation/visualization/`

### Clean Up (Nice to Have)
- [ ] Organize `utils/` contents properly
- [ ] Remove all `__pycache__` directories
- [ ] Update all imports after moves
- [ ] Update compatibility shims

## 🎯 Target Structure

```
src/brain_go_brrr/
├── domain/              # Pure business logic
│   ├── abnormal/
│   ├── quality/
│   ├── sleep/
│   ├── preprocessing/   # Domain preprocessing
│   ├── constraints.py
│   ├── channels.py
│   └── exceptions.py
├── application/         # Use cases & orchestration
│   ├── use_cases/
│   ├── jobs/
│   ├── pipeline/
│   ├── training/
│   ├── config/         # Application config
│   └── ports.py
├── infra/              # External & I/O
│   ├── external/       # YASA, MNE adapters
│   ├── redis/
│   ├── data/           # Data loading
│   ├── ml_models/      # EEGPT models
│   ├── preprocessing/  # Infra preprocessing
│   ├── cache.py
│   └── serialization.py
├── presentation/       # User interface
│   ├── api/           # REST API
│   │   ├── routers/
│   │   ├── app.py
│   │   └── schemas.py
│   └── visualization/ # Reports, plots
├── core/              # Compatibility shims only
├── services/          # Compatibility shims only
├── utils/             # Minimal shared utilities
├── __init__.py
├── __main__.py
├── _typing.py
├── cli.py
├── mne_compat.py
└── py.typed
```

## 🔧 Implementation Plan

### Phase 1: Delete Redundant (5 min)
1. Remove empty `tasks/` and `training/` directories
2. Remove empty `modules/` after moving constraints.py

### Phase 2: Quick Moves (10 min)
1. Move `config/` → `application/config/`
2. Move `modules/constraints.py` → `domain/constraints.py`

### Phase 3: Major Reorganization (30 min)
1. Move `data/` → `infra/data/`
2. Move `models/` → `infra/ml_models/`
3. Split and reorganize `preprocessing/`
4. Move `visualization/`

### Phase 4: Update & Test (20 min)
1. Update all imports
2. Update compatibility shims
3. Run `make lint`
4. Run `make type-check`
5. Run `make test`
6. Run `make test-all-cov`

## 📊 Current Issues Summary

| Issue | Severity | Impact | Files Affected |
|-------|----------|--------|----------------|
| Empty redundant directories | LOW | Clutter | 2 directories |
| Misplaced config | MEDIUM | Architecture violation | 3 files |
| Data in wrong layer | HIGH | Architecture violation | ~10 files |
| Models in wrong layer | HIGH | Architecture violation | ~8 files |
| Mixed preprocessing | MEDIUM | Unclear boundaries | ~15 files |

## ✅ What's Already Correct

- ✅ Domain layer is clean
- ✅ Application layer is well-organized
- ✅ Infrastructure layer is properly isolated
- ✅ API layer follows REST patterns
- ✅ Compatibility shims work correctly
- ✅ Root-level files are professional
- ✅ Type definitions are in place

## 🚀 Next Steps

1. Execute Phase 1-3 of Implementation Plan
2. Fix all imports
3. Run all tests
4. Update documentation
5. Final review

---

**Status**: Ready to execute refactoring plan. All issues identified and documented.