# 🎯 REMAINING TECHNICAL DEBT

**Created**: September 8, 2025
**Status**: Two optional sprints remain
**Priority**: LOW - All critical and high-priority items complete

---

## ✅ COMPLETED DEBT (P0, P1, P2 Sprints 1-3)

**All critical technical debt has been resolved:**
- **P0**: EEGPT dimension fixes (512 vs 2048) ✅
- **P1**: Duplicate types, architecture violations ✅
- **P2 Sprint 1**: Quick wins, guards, 768-dim removal ✅
- **P2 Sprint 2**: Architecture cleanup, probe migration ✅
- **P2 Sprint 3**: Documentation, CI hardening, constants ✅

---

## 📋 REMAINING ITEMS

### Sprint 4: TUEV Channel Synthesis (4 hours) 🔬 OPTIONAL

**Problem**: TUEV dataset has 23 channels, EEGPT expects 20. Currently using zero-fill approach.

**Potential Benefit**: +1% accuracy with learnable channel mapping

**Implementation Path**:
```python
# src/brain_go_brrr/infra/ml_models/channel_mapper.py
class TUEVChannelMapper(nn.Module):
    """Learnable channel mapping from TUEV 23 → EEGPT 20 channels."""

    def __init__(self):
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv1d(23, 20, kernel_size=1),
            nn.BatchNorm1d(20),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, 23, T) → (B, 20, T)."""
        return self.channel_conv(x)
```

**Integration Points**:
1. Add before normalization in TUEV preprocessing pipeline
2. Train jointly with linear probe
3. Add config flag: `enable_tuev_channel_mapper: bool = False`

**Acceptance Criteria**:
- [ ] Shape test: (B,23,T) → (B,20,T)
- [ ] Gradients flow properly through mapper
- [ ] Config on/off works without regression
- [ ] Document channel order dependencies
- [ ] No impact on non-TUEV paths
- [ ] Benchmark shows measurable accuracy improvement

**Current TUEV Training Status**:
- Dataset caching works with new preprocessor
- Training pipeline operational
- Channel synthesis is the only optimization left

---

### Sprint 5: Polish & Hardening (3 hours) 🎨 NICE-TO-HAVE

**Remaining polish items not yet implemented:**

#### 1. Code Coverage to 95% (2 hours)
**Current**: ~86% coverage
**Target**: 95% on critical paths

**Focus Areas**:
- InMemoryCache TTL expiry and clear_pattern wildcards
- migrate_eegpt_probe_to_factory error paths
- Redis connection/timeout errors
- Config validation edge cases

#### 2. Test Parallelization Config (30 minutes)
**Tools already installed**: pytest-xdist, pytest-timeout

**Configuration needed**:
```toml
[tool.pytest.ini_options]
addopts = "-n auto --dist loadscope --timeout=60"
```

#### 3. Dead Code Detection (30 minutes)
**Using existing ruff**:
```toml
[tool.ruff]
select = ["F401", "F841"]  # Unused imports/variables
```

#### 4. Gitleaks Integration (30 minutes)
**Add secrets scanning to CI**:
```yaml
- uses: gitleaks/gitleaks-action@v2
```

---

## 🤔 SHOULD WE DO THE REMAINING WORK?

### Sprint 4 (TUEV Channel Synthesis)
**Worth doing if:**
- Training TUEV models in production
- Need every % of accuracy improvement
- Have time for experimentation

**Skip if:**
- Current zero-fill approach is sufficient
- Not actively using TUEV dataset
- Other priorities are more important

### Sprint 5 (Polish)
**Worth doing if:**
- Moving to production deployment
- Need bulletproof CI/CD
- Team is growing and needs guardrails

**Skip if:**
- Current quality gates are sufficient
- Research/prototype phase
- Limited developer time

---

## 📊 CURRENT QUALITY METRICS

**What we have now:**
- ✅ 11/11 import-linter contracts passing
- ✅ All tests green (878 passed)
- ✅ Type checking clean
- ✅ No architecture violations
- ✅ CI/CD fully operational
- ✅ Warnings as errors in CI
- ✅ Security scanning (pip-audit)
- ✅ Import performance guard (< 3s)
- ✅ Deterministic testing

**What we'd gain from remaining work:**
- +1% TUEV accuracy (Sprint 4)
- 95% code coverage (Sprint 5)
- Parallel test execution (Sprint 5)
- Secrets scanning (Sprint 5)

---

## 🎯 RECOMMENDATION

**Current State**: Production-ready with comprehensive quality gates

**Remaining Work Priority**: LOW

**Recommendation**:
1. **SKIP Sprint 4** unless actively training TUEV models
2. **DEFER Sprint 5** until moving to production or team scaling
3. **FOCUS** on feature development and model training

The codebase is now clean, maintainable, and has strong architectural boundaries. The remaining items are optimizations and polish that can be added incrementally as needed.

---

## 📝 DECISION LOG

**Date**: _____________
**Decision**: [ ] Implement Sprint 4 [ ] Implement Sprint 5 [ ] Defer Both
**Rationale**: _____________________________________________
**Assigned To**: _____________
**Target Date**: _____________
