# 🔧 BEHAVIOR TEST ACTION PLAN - Fixing What's Broken

**Date**: August 13, 2025
**Status**: 1/5 Components Working
**Goal**: Get ALL components functional

## 📋 Test Results Summary

| Component | Status | Issue | Fix Required |
|-----------|--------|-------|-------------|
| YASA Sleep | ✅ WORKING | scikit-learn version warning | Update sklearn or ignore warning |
| Quality Control | ❌ BROKEN | Wrong class name | Use `EEGQualityController` not `QualityController` |
| Abnormality Detection | ❌ BROKEN | Missing model_path | Pass model path to constructor |
| API Endpoints | ❌ BROKEN | Wrong import | Import from `api.main` not `api.app` |
| PDF Generation | ❌ UNTESTED | Depends on QC | Fix QC first |

## 🛠️ Fixes Required

### 1. Quality Control Fix
```python
# WRONG:
from brain_go_brrr.domain.quality.controller import QualityController

# CORRECT:
from brain_go_brrr.domain.quality.controller import EEGQualityController
```

### 2. Abnormality Detection Fix
```python
# WRONG:
detector = AbnormalityDetector()

# CORRECT:
model_path = Path("experiments/eegpt_linear_probe/output/tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt")
detector = AbnormalityDetector(model_path=model_path)
```

### 3. API Import Fix
```python
# WRONG:
from brain_go_brrr.api.app import app

# CORRECT:
from brain_go_brrr.api.main import app
```

## 📝 Fixed Test Script

Creating `test_behavior_fixed.py` with all corrections...

## 🎯 Expected Results After Fixes

- YASA Sleep: ✅ (already working)
- Quality Control: ✅ (simple import fix)
- Abnormality Detection: ❓ (depends on model integration)
- API Endpoints: ✅ (simple import fix)
- PDF Generation: ✅ (should work after QC fix)

## ⚠️ Potential Issues

1. **Abnormality Detection**: The trained model might not be properly integrated
2. **YASA Warning**: scikit-learn version mismatch (non-critical)
3. **PDF Generation**: Might have additional dependencies

## 📊 Next Steps

1. Run fixed behavior test
2. Address any remaining issues
3. Test end-to-end workflows
4. Create integration tests
5. Update documentation

## 🚀 Clean Code Principles Applied

- **Single Responsibility**: Each test tests one component
- **Dependency Injection**: Pass required dependencies explicitly
- **Error Handling**: Catch and report specific errors
- **Clear Naming**: Use descriptive variable and function names
- **DRY**: Reuse test data across components
