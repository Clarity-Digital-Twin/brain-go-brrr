# Technical Debt Tracker
*Last Updated: September 9, 2025*

## 🔴 Critical Issues (Block Production)

### 1. PyTorch Lightning 2.5.2 Bug
- **Impact**: Training hangs with large cached datasets (>100k samples)
- **Workaround**: Using pure PyTorch implementations
- **Fix**: Wait for Lightning fix or migrate all training to pure PyTorch
- **Affected**: All experiments if someone tries to use Lightning

## 🟡 High Priority (Next Sprint)

### 1. TUSZ Temporal Detection Implementation
- **Status**: Architecture decided, SeizureTransformer wrapper planned
- **Next Steps**: 
  - Implement wrapper infrastructure for SeizureTransformer
  - Add NEDC evaluation metrics (FA/24h, TAES, ATWV)
  - Create EEGPT+BiLSTM alternative
- **Docs**: See TUSZ_*.md files in root

### 2. Mixed Precision Training
- **Impact**: Could reduce memory usage by 50%, speed up training
- **Status**: Not implemented, would help with GPU memory constraints
- **Complexity**: Medium - need to add autocast and scaler

### 3. Multi-GPU Support
- **Impact**: Would dramatically speed up training
- **Status**: Not implemented
- **Complexity**: High - need DistributedDataParallel

## 🟢 Medium Priority (Technical Improvements)

### 1. Normalization Stats Computation
- **Current**: Using hardcoded 50μV std assumption
- **Need**: Compute actual stats from training data
- **Impact**: May improve model performance
- **Location**: `brain_go_brrr/infra/ml_models/eegpt_wrapper.py`

### 2. Event Detection Module
- **Status**: Architecture docs only, no implementation
- **Components**: GPED/PLED detection, epileptiform discharge identification
- **Priority**: Lower than TUSZ

### 3. Authentication/Authorization
- **Status**: No OAuth2/JWT implementation
- **Impact**: Required for production deployment
- **Complexity**: Medium - FastAPI has good support

## 📚 Documentation Debt

### 1. API Documentation
- **Current**: API.md shows planned endpoints, not all implemented
- **Need**: Update to reflect actual working endpoints
- **Tool**: Consider auto-generating from FastAPI schemas

### 2. Training Documentation
- **Current**: TUEV guide created, but scattered across experiments
- **Need**: Centralized training documentation in `/docs/TRAINING.md`

### 3. Deployment Guide
- **Status**: No production deployment documentation
- **Need**: Kubernetes manifests, PostgreSQL setup, monitoring

## 🔧 Code Quality Debt

### 1. Test Coverage
- **Current**: ~70% coverage
- **Target**: 90%+ for critical paths
- **Missing**: Integration tests for full pipeline

### 2. Deprecated Imports
- **Issue**: Some deprecation warnings in ML models
- **Fix**: Update to newer APIs
- **Location**: Check CI deprecation warnings

### 3. Error Handling
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

## ✅ Recently Resolved (Archived)

- ✅ TUEV paper parity (62.32% target) - Training running successfully
- ✅ Channel mapping issues (T3→T7, etc.) - Fixed in preprocessing
- ✅ Balanced accuracy scoring - Removed invalid `labels` parameter
- ✅ Gradient clipping - Implemented for both probe and mapper
- ✅ Config key inconsistencies - Supports both naming conventions
- ✅ torch.load compatibility - Version detection with fallback
- ✅ Script argument mismatches - CI validation added

## 📊 Debt Metrics

- **Critical Issues**: 1
- **High Priority**: 3
- **Medium Priority**: 3
- **Documentation**: 3
- **Code Quality**: 3
- **Infrastructure**: 3
- **Security**: 3

**Total Open Items**: 19

## 🎯 Recommended Priority Order

1. **Complete TUSZ implementation** (current focus after TUEV)
2. **Add mixed precision training** (quick win for performance)
3. **Compute normalization stats** (improve model accuracy)
4. **Update API documentation** (reflect reality)
5. **Increase test coverage** (prevent regressions)

## 📝 Notes

- This document tracks ACTUAL technical debt, not completed work
- Items marked ✅ have been moved to the "Recently Resolved" section
- For TUEV-specific resolved issues, see `docs/tuev/archive/`
- For active TUSZ work, see `TUSZ_*.md` files in root