# Technical Debt Tracker (Single Source of Truth)
*Last Updated: September 9, 2025*
*Status: Active tracking document for all remaining technical debt*

## 🔴 Critical Issues (Block Production)

### 1. PyTorch Lightning 2.5.2 Bug
- **Impact**: Training hangs with large cached datasets (>100k samples)
- **Workaround**: Using pure PyTorch implementations
- **Fix**: Wait for Lightning fix or migrate all training to pure PyTorch
- **Affected**: All experiments if someone tries to use Lightning

## 🟡 High Priority (Active Work)

### 1. TUSZ Temporal Detection (Next Focus)
- **Status**: Architecture decided, SeizureTransformer wrapper planned
- **Next Steps**: 
  - Implement wrapper infrastructure for SeizureTransformer
  - Add NEDC evaluation metrics (FA/24h, TAES, ATWV)
  - Create EEGPT+BiLSTM alternative
- **Docs**: See TUSZ_*.md files in root

## 🟢 Medium Priority (Technical Improvements)

### 1. Mixed Precision Training
- **Impact**: Could reduce memory usage by 50%, speed up training
- **Status**: Not implemented, would help with GPU memory constraints
- **Complexity**: Medium - need to add autocast and scaler
- **Location**: Training scripts in `experiments/`

### 2. Multi-GPU Support
- **Impact**: Would dramatically speed up training
- **Status**: Not implemented
- **Complexity**: High - need DistributedDataParallel

### 3. Normalization Stats Computation
- **Current**: Using hardcoded 50μV std assumption
- **Need**: Compute actual stats from training data
- **Impact**: May improve model performance
- **Location**: `brain_go_brrr/infra/ml_models/eegpt_wrapper.py`

### 4. CI Script Validation Gap
- **Issue**: `.ci/check_script_arguments.sh` only checks TUEV launcher
- **Need**: Also validate `experiments/eegpt_linear_probe/scripts/launch_tuab_mne.sh`
- **Impact**: Could miss argument mismatches in TUAB training

## 📚 Documentation Debt

### 1. API Documentation
- **Current**: API.md shows planned endpoints, not all implemented
- **Need**: Update to reflect actual working endpoints
- **Tool**: Consider auto-generating from FastAPI schemas

### 2. Training Documentation
- **Update**: TUEV archived; training section in `/docs/TRAINING.md` now points to final verdict.

### 3. Deployment Guide
- **Status**: No production deployment documentation
- **Need**: Kubernetes manifests, PostgreSQL setup, monitoring

### 4. MkDocs Navigation
- **Update**: TUEV docs to be archived under `docs/tuev/archived/` after senior audit; update links accordingly.

## 🔧 Code Quality Debt

### 1. Test Coverage
- **Current**: Estimated ~70% coverage
- **Target**: 90%+ for critical paths
- **Missing**: Integration tests for full pipeline

### 2. Event Detection Module
- **Status**: Architecture docs only, no implementation
- **Components**: GPED/PLED detection, epileptiform discharge identification
- **Priority**: Lower than TUSZ

### 3. Deprecated Imports
- **Issue**: Some deprecation warnings in ML models
- **Fix**: Update to newer APIs
- **Location**: Check CI deprecation warnings

### 4. Error Handling
- **Issue**: Some paths don't handle edge cases
- **Example**: Missing channel handling could be more graceful

## 💾 Infrastructure Debt

### 1. Cache Versioning
- **Current**: Manual version bumps in dataset classes
- **Need**: Automated cache invalidation system

### 2. Redis Implementation
- **Status**: Redis installed but not integrated
- **Need**: Caching layer for API responses

### 3. Celery Queue
- **Status**: Architecture planned but not implemented
- **Need**: Background job processing for long-running analyses

### 4. Authentication/Authorization
- **Status**: No OAuth2/JWT implementation
- **Impact**: Required for production deployment
- **Complexity**: Medium - FastAPI has good support

## 🚨 Security Debt

### 1. Secrets Management
- **Current**: Environment variables
- **Need**: Proper secrets vault for production

### 2. PHI Handling
- **Status**: Basic checks but no comprehensive audit
- **Need**: Full HIPAA compliance review

### 3. Input Validation
- **Status**: Basic Pydantic models
- **Need**: Comprehensive validation for all endpoints

## ✅ Recently Resolved (Last Sprint)

- ✅ Channel mapping issues (T3→T7, etc.) - Fixed in preprocessing
- ✅ Balanced accuracy scoring - Removed invalid `labels` parameter
- ✅ Gradient clipping - Implemented for both probe and mapper
- ✅ Config key inconsistencies - Supports both naming conventions
- ✅ torch.load compatibility - Version detection with fallback
- ✅ Script argument mismatches - CI validation added for TUEV
- ✅ TUEV 23-channel implementation - Complete with learnable mapper
- ✅ Training infrastructure - Auto-recovery, checkpoint resumption

## 📊 Debt Metrics Summary

| Category | Count | Notes |
|----------|-------|-------|
| Critical Issues | 1 | PyTorch Lightning bug |
| Active Work | 1 | TUSZ planning |
| Medium Priority | 4 | Performance/quality improvements |
| Documentation | 4 | Accuracy and organization |
| Code Quality | 4 | Coverage and modules |
| Infrastructure | 4 | Production readiness |
| Security | 3 | Compliance and validation |

**Total Open Items**: 22

## 🎯 Recommended Priority Order

1. **Finalize TUEV archival and links** (docs updated; move files post‑audit)
2. **Start TUSZ implementation** (wrapper infrastructure ready)
3. **Fix CI script checker** (quick fix for TUAB launcher)
4. **Add mixed precision** (performance boost)
5. **Update documentation** (reflect current state)

## 📝 Notes

- REMAINING_DEBT.md has been deprecated - this document is the SSOT
- For TUEV implementation details, see `docs/tuev/`
- For active TUSZ work, see `TUSZ_*.md` files in root
- Training logs in `experiments/eegpt_linear_probe/logs/`
