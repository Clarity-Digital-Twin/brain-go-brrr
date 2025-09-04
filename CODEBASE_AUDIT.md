# 🔍 CODEBASE AUDIT - Technical Debt & Cleanup Opportunities
**Generated**: September 4, 2025
**Status**: Active Tracking Document
**Purpose**: Track non-critical technical debt and refactoring opportunities

---

## 📊 Audit Summary

| Category | Count | Priority | Impact |
|----------|-------|----------|--------|
| TODO/FIXME Comments | 20+ | Medium | Code clarity |
| Duplicate Classes | 9 | Low | Maintainability |
| Large Files (>500 lines) | 7 | Medium | Complexity |
| Type Ignores | 30+ | Low | Type safety |
| Dead Code (vulture) | 50+ | Low | Code size |
| Deprecated Code | 14 | Medium | Future compatibility |
| Hardcoded Values | 5+ | Medium | Configuration |

---

## 🔴 HIGH PRIORITY - Potential Issues

### 1. Duplicate Class Names (Possible Namespace Conflicts)
**Files**: Multiple locations
**Classes with duplicates**:
- `CachePort(Protocol)` - Appears in multiple modules
- `FeatureExtractorPort(Protocol)` - Multiple definitions
- `JobData` - Duplicate data classes
- `LoggerPort(Protocol)` - Multiple logger interfaces
- `NumpyEncoder(json.JSONEncoder)` - Multiple JSON encoders
- `RedisCache` - Multiple cache implementations
- `YASAConfig` - Multiple config classes
- `_NullModel` - Multiple null object patterns
- `_NullPreprocessor` - Multiple preprocessor mocks

**Impact**: Confusing imports, potential wrong class usage
**Action**: Consolidate into single definitions or rename for clarity

### 2. Large Monolithic Files
**Files needing refactoring**:
- `domain/sleep/analyzer_enhanced.py` - 815 lines
- `domain/sleep/analyzer.py` - 756 lines
- `infra/ml_models/eegpt_architecture.py` - 626 lines
- `infra/preprocessing/snippets/maker.py` - 583 lines
- `infra/external/yasa_adapter.py` - 574 lines
- `domain/quality/controller.py` - 562 lines
- `api/routers/sleep.py` - 545 lines

**Impact**: Hard to navigate, test, and maintain
**Action**: Split into smaller, focused modules

---

## 🟡 MEDIUM PRIORITY - Code Quality

### 3. TODO/FIXME/HACK Comments (20+ instances)
**Highest concentration**:
- `cli.py` - 4 TODOs
- `api/routers/queue.py` - 4 FIXMEs
- `api/app.py` - 3 TODOs

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
**Files with hardcoded values**:
- `cli.py` - 2 instances (localhost/ports)
- `infra/redis/pool.py` - 2 instances (Redis connection)
- `infra/cache_factory.py` - 1 instance
- `application/config/base.py` - 1 instance
- `api/main.py` - 1 instance

**Patterns found**:
- `localhost`, `127.0.0.1` - Hardcoded hosts
- `8000`, `5432`, `6379` - Hardcoded ports

**Action**: Move to environment variables or config

### 5. Type Safety Issues
**Files with most type ignores**:
- `domain/abnormal/detector.py` - 9 ignores
- `domain/quality/controller.py` - 7 ignores
- `domain/preprocessing/features/extractor.py` - 3 ignores

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
**Unused variables in `_typing.py`** (30+ instances):
- TypedDict fields that are defined but never used
- Likely for future compatibility or documentation

**Other dead code patterns**:
- Unused imports (cleaned by ruff already)
- Unreachable code after returns
- Unused function parameters
- Never-called private methods

**Action**: Review and remove if truly unused

### 7. Deprecated Code (14 instances)
**Common patterns**:
```python
warnings.warn("X is deprecated, use Y instead", DeprecationWarning)
@deprecated("Use new_function instead")
```

**Action**: Plan migration path and remove in next major version

### 8. Missing Tests
**Modules with low/no test coverage**:
- CLI commands (`cli.py`)
- Redis integration (`infra/redis/`)
- Some API endpoints (`api/routers/`)

**Action**: Add integration tests for these modules

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

### Immediate (This Week)
1. [ ] Consolidate duplicate class definitions
2. [ ] Extract hardcoded values to config
3. [ ] Create GitHub issues for high-priority TODOs

### Short Term (This Month)
4. [ ] Refactor files >500 lines into smaller modules
5. [ ] Fix type ignores with proper typing
6. [ ] Add missing integration tests

### Long Term (Next Quarter)
7. [ ] Remove deprecated code after migration
8. [ ] Clean up dead code identified by vulture
9. [ ] Implement proper error handling where TODOs exist

---

## 🔧 Automation Opportunities

### Pre-commit Hooks to Add
- `vulture` - Dead code detection
- `radon` - Complexity checking
- `bandit` - Security scanning

### CI Checks to Add
- Maximum file size limit (500 lines)
- TODO comment limit
- Type ignore limit

---

## 📝 Notes

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
