# 🔍 CODEBASE AUDIT - Technical Debt & Cleanup Opportunities
**Generated**: September 4, 2025
**Last Updated**: September 8, 2025 (Post P2 Sprint 1-3)
**Status**: Active Tracking Document
**Purpose**: Track non-critical technical debt and refactoring opportunities

---

## 📊 Audit Summary

| Category | Count | Priority | Impact | Status |
|----------|-------|----------|--------|--------|
| TODO/FIXME Comments | 22 | Medium | Code clarity | Active |
| Duplicate Classes | 4 families | Low | Maintainability | Mostly resolved |
| Large Files (>500 lines) | 7 | Medium | Complexity | Active |
| Type Ignores | 34 | Low | Type safety | Active |
| Dead Code (vulture) | Periodic check | Low | Code size | Not CI-enforced |
| Deprecated Code | Tracked | Medium | Future compatibility | Re-exports cleaned |
| Hardcoded Values | 4 locations | Medium | Configuration | Sane defaults |

---

## 🔴 HIGH PRIORITY - Potential Issues

### 1. Duplicate Class Names (Mostly Resolved)
**Status**: ✅ Major duplicates resolved in P2 Sprint 1-3

**Resolved**:
- `CachePort` - ✅ Single definition in `domain/ports/cache.py`
- `LoggerPort` - ✅ Single protocol in `domain/protocols/logger.py` (@runtime_checkable)
- `YASAConfig` - ✅ Services redirect cleaned, single source
- `RedisCache` - ✅ Aliases removed, clear naming
- `FeatureExtractorPort` - ✅ Architecture violations fixed

**Remaining (intentional or minor)**:
- `JobData` - 2 instances (API DTO vs application model) - **Acceptable pattern**
- `NumpyEncoder` - 2 instances (`api/app.py`, `api/routers/qc.py`) - **Should centralize**
- `_NullModel` - 2 instances (domain-only scaffolding) - **Low priority**
- `_NullPreprocessor` - 3 instances (domain-only scaffolding) - **Low priority**

**Action**: Centralize NumpyEncoder to `api/utils/json.py`

### 2. Large Monolithic Files
**Files needing refactoring**:
- `domain/sleep/analyzer_enhanced.py` - 815 lines
- `domain/sleep/analyzer.py` - 756 lines
- `infra/ml_models/eegpt_architecture.py` - 626 lines
- `infra/preprocessing/snippets/maker.py` - 583 lines
- `infra/external/yasa_adapter.py` - 578 lines
- `domain/quality/controller.py` - 549 lines
- `api/routers/sleep.py` - 546 lines

**Impact**: Hard to navigate, test, and maintain
**Action**: Split into smaller, focused modules

---

## 🟡 MEDIUM PRIORITY - Code Quality

### 3. TODO/FIXME/HACK Comments (22 instances)
**Highest concentration**:
- `cli.py` - Multiple TODOs
- `api/app.py` - Multiple TODOs
- `api/routers/*` - Various FIXMEs
- `infra/preprocessing/snippets/` - Several optimization notes

**Sample TODOs found**:
```python
# TODO: Implement proper error handling
# FIXME: This is a temporary workaround
# HACK: Quick fix for demo, needs proper solution
# XXX: Performance bottleneck here
# REFACTOR: Extract to separate service
# OPTIMIZE: Could cache this computation
```

**Action**: Create issues for each TODO and prioritize

### 4. Hardcoded Network Configuration
**Files with hardcoded values (4 primary locations)**:
- `cli.py` - `127.0.0.1` default (acceptable CLI default)
- `infra/redis/pool.py` - `localhost` default (env override available)
- `application/config/base.py` - MLflow URI (configurable)
- `infra/cache_factory.py` - Redis URL env default (configurable)

**Patterns found**:
- `localhost`, `127.0.0.1` - Hardcoded hosts
- `8000`, `5432`, `6379` - Hardcoded ports

**Action**: Move to environment variables or config

### 5. Type Safety Issues
**Total type ignores**: 34 (acceptable for now)

**Distribution**:
- Hot paths have some ignores for performance
- Third-party library integrations
- Dynamic attribute access patterns

**Common patterns**:
```python
# type: ignore[attr-defined]
# type: ignore[import]
# type: ignore[no-untyped-def]
```

**Action**: Fix underlying type issues instead of ignoring

---

## 🟢 LOW PRIORITY - Cleanup Opportunities

### 6. Dead Code (Vulture Analysis)
**Status**: Periodic check only (not CI-enforced)

**Note**:
- Ruff already handles unused imports/variables (F401, F841)
- Vulture can be run periodically but not on every PR
- TypedDict fields in `_typing.py` are intentional for contracts

**Action**: Run vulture quarterly, not critical path

### 7. Deprecated Code Management
**Status**: ✅ Re-exports cleaned in P2 Sprint 2

**Cleaned**:
- `eegpt_compat` re-export removed from `__init__.py`
- Services redirect deleted
- EEGPTProbe usage removed from application layer

**Remaining deprecations are tracked and have migration paths**

**Action**: Continue phased deprecation per MIGRATION_GUIDE.md

### 8. Test Coverage
**Current Coverage**: ~86% (acceptable)
**Target**: 95% on critical paths (Sprint 5 - optional)

**Lower coverage areas**:
- CLI commands (`cli.py`) - Interactive, lower priority
- Redis integration - Tested in integration suite
- Some API endpoints - Covered by integration tests

**Action**: Sprint 5 optional - increase to 95%

---

## 📈 Code Metrics

### Complexity Metrics
- **Average file length**: ~200 lines (good)
- **Max file length**: 815 lines (too high)
- **Files > 500 lines**: 7 (needs refactoring)
- **Empty files**: 0 (good)

### Documentation Coverage
- Most modules have docstrings ✅
- Some TODO comments need conversion to issues
- API documentation needs update for new endpoints

### Dependencies
- All imports validated by ruff ✅
- No circular dependencies detected ✅
- Some potentially unused dependencies in pyproject.toml

---

## 🎯 Recommended Actions (Priority Order)

### Immediate (Quick Wins)
1. [x] Consolidate duplicate class definitions - ✅ MOSTLY DONE
2. [ ] Centralize NumpyEncoder to `api/utils/json.py`
3. [ ] Create GitHub issues for top 10 TODOs

### Short Term (When Touching Code)
4. [ ] Refactor large files when modifying those areas
5. [ ] Fix type ignores in hot paths only
6. [ ] Continue improving test coverage incrementally

### Long Term (Low Priority)
7. [ ] Remove deprecated code per migration timeline
8. [ ] Run vulture quarterly for cleanup
9. [ ] Convert remaining TODOs to tracked issues

---

## 🔧 Automation Opportunities

### Current Automation (✅ Already Implemented)
- **import-linter**: Architecture boundaries enforced
- **ruff**: Dead code (unused imports/vars) checked
- **mypy**: Type checking enforced
- **pip-audit**: Security scanning active
- **Pre-commit hooks**: Comprehensive suite active

### Optional Future Automation
- `vulture`: Quarterly dead code deep scan (not every PR)
- `radon`: Complexity metrics (informational only)
- `bandit`: Additional security scanning (if needed)

---

## 📝 Notes

### What Changed After P2 Sprint 1-3
- **Architecture violations**: Fixed all domain→infra imports
- **Duplicate protocols**: Consolidated to single definitions
- **Import performance**: Added <3s guard in CI
- **Warnings discipline**: CI fails on unexpected warnings
- **Deterministic testing**: Seeds and controls added

### Current State
- **Production-ready** with comprehensive quality gates
- **11/11 import-linter contracts** passing
- **CI/CD fully operational** with all guards
- Only **optional optimizations** remain (Sprint 4-5)

- This audit focused on `src/` directory only
- `experiments/` excluded as it's meant for quick iteration
- `tests/` excluded from dead code analysis (test utilities often unused)
- Some "dead code" might be public API that external code uses

---

## 🔄 Next Audit

Schedule next audit for: **October 2025**
Focus areas:
- Performance bottlenecks
- Database query optimization
- API response time analysis
- Memory usage patterns

---

**Remember**: Not all technical debt needs immediate fixing. Prioritize based on:
1. User impact
2. Developer velocity impact
3. Risk of bugs
4. Ease of fixing
