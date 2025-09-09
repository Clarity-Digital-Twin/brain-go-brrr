# 🎯 REMAINING TECHNICAL DEBT

**Created**: September 8, 2025
**Updated**: September 9, 2025
**Status**: Sprint 4 REQUIRED for paper parity; Sprint 5 optional
**Priority**: Sprint 4 HIGH (paper parity); Sprint 5 LOW (polish)

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

### Sprint 4: TUEV Channel Mapper 🔴 REQUIRED - NOT IMPLEMENTED

**📊 Reference**: See [TUEV_METRICS_SSOT.md](TUEV_METRICS_SSOT.md) for target metrics and thresholds.

**Status**: NOT IMPLEMENTED - Required for paper parity

**Current Implementation (What We Have Now)**:
- 20-channel preprocessing approach
- Drops A1/A2 channels in tuev_preprocessor.py
- Synthesizes Fpz as (Fp1+Fp2)/2 or zeros
- Cache at `tuev_mne_fixed` with 20 channels
- This achieves partial hyperparameter parity but NOT architectural parity

**Paper Implementation (What EEGPT Actually Does)**:
- Keeps ALL 23 TUEV channels (including A1, A2)
- Uses Conv2dWithConstraint(23, 20, 1) learnable mapper
- Mapper includes BatchNorm, GELU, Dropout(0.8)
- NO preprocessing synthesis - model learns the mapping

**Impact**: This is NOT "+1% improvement" - it's THE architecture that achieved 62% BAC

**Reproducibility Seeds**: 42 (data), 123 (model init), 456 (augmentation) - as per TUEV_METRICS_SSOT.md

**Required Implementation (TO BE DONE)**:
```python
# src/brain_go_brrr/infra/ml_models/channel_mapper.py  [DOES NOT EXIST YET]
class TUEVChannelMapper(nn.Module):
    """Learnable 23→20 channel mapper for TUEV paper parity."""
    # Implementation needed based on EEGPT reference:
    # - Conv2dWithConstraint(23, 20, kernel_size=1)
    # - BatchNorm2d(20)
    # - GELU()
    # - Dropout(0.8)
    # - Conv2d(20, 20, kernel_size=(1,55), groups=20)
```
```

**Integration Required (NOT DONE)**:
1. Create channel_mapper.py module (doesn't exist)
2. Modify train_tuev_mne.py to use mapper (not integrated)
3. Add config flag (not added)
4. Rebuild cache with 23 channels (current cache is 20-ch)

**Acceptance Criteria (NONE MET YET)**:
- [ ] Create channel_mapper.py module
- [ ] Rebuild cache with 23 channels (keep A1/A2)
- [ ] Integrate mapper in training pipeline
- [ ] Shape test: (B,23,T) → (B,20,T)
- [ ] Achieve BAC ≥ 60% (paper parity target)

**Current Cache/Training Status**:
- Cache build 80% complete (289/359 files) with 20-ch approach
- Using Fpz interpolation (NOT paper parity)
- Hyperparameters match paper (lr=5e-4, wd=0.05, smoothing=0.1)
- Architecture does NOT match paper (missing mapper)

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

### Sprint 4 (TUEV Channel Mapper)
**REQUIRED for paper parity because:**
- This IS the architecture that achieved 62% BAC
- Current 20-ch approach is NOT what the paper did
- Without mapper, we're not testing the actual EEGPT approach

**Current options:**
1. **Continue current cache** → Test partial parity (hyperparams only)
2. **Kill and rebuild** → Full paper parity with 23-ch + mapper

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
- +1% TUEV BAC (Sprint 4)
- 95% code coverage (Sprint 5)
- Parallel test execution (Sprint 5)
- Secrets scanning (Sprint 5)

---

## 🎯 RECOMMENDATION

**Current State**: Production-ready with comprehensive quality gates

**Remaining Work Priority**: LOW

**Recommendation**:
1. **IMPLEMENT Sprint 4** for paper parity (required)
2. **DEFER Sprint 5** (optional polish)
3. **Decision needed NOW**: Kill current cache or let it finish?

The codebase is now clean, maintainable, and has strong architectural boundaries. The remaining items are optimizations and polish that can be added incrementally as needed.

---

## 📝 DECISION LOG

**Date**: _____________
**Decision**: [ ] Implement Sprint 4 [ ] Implement Sprint 5 [ ] Defer Both
**Rationale**: _____________________________________________
**Assigned To**: _____________
**Target Date**: _____________

---

## 🔧 Fix Plan — Actionable Checklist

### ⚠️ Current Status

```bash
# Cache build in progress (WRONG approach)
tmux attach -t tuev_cache  # 289/359 files, ~80% complete
# Building 20-ch cache with Fpz interpolation
# This is NOT paper parity

# No training currently running
# Decision needed: kill cache or let it finish?
```

### Sprint 4: TUEV Channel Mapper (REQUIRED for 62% BAC)

#### Phase 1 — Preparation (post-training)
- [ ] Backup checkpoints
  ```bash
  cp -r experiments/eegpt_linear_probe/checkpoints \
        experiments/eegpt_linear_probe/checkpoints_zerofill_backup
  ```
- [ ] Record baseline metrics
  ```bash
  grep "val_acc\|val_auroc" \
    experiments/eegpt_linear_probe/logs/latest.log > baseline_metrics.txt
  ```

#### Phase 2 — Channel Mapper Module
- [ ] Add `src/brain_go_brrr/infra/ml_models/channel_mapper.py`
  ```python
  import torch
  import torch.nn as nn

  class TUEVChannelMapper(nn.Module):
      """Map TUEV 23 channels → EEGPT 20 channels."""

      def __init__(self, dropout: float = 0.3):
          super().__init__()
          self.channel_conv = nn.Sequential(
              nn.Conv1d(23, 20, kernel_size=1, bias=True),
              nn.BatchNorm1d(20),
              nn.GELU(),
              nn.Dropout(dropout),
          )

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          return self.channel_conv(x)
  ```
- [ ] Unit tests `tests/unit/infra/ml_models/test_channel_mapper.py`
  ```python
  import torch
  from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper

  def test_channel_mapper_shapes():
      mapper = TUEVChannelMapper()
      x = torch.randn(32, 23, 1024)
      y = mapper(x)
      assert y.shape == (32, 20, 1024)

  def test_gradient_flow():
      mapper = TUEVChannelMapper()
      x = torch.randn(1, 23, 256, requires_grad=True)
      y = mapper(x)
      y.mean().backward()
      assert x.grad is not None
  ```

#### Phase 3 — Training Integration
- [ ] `experiments/eegpt_linear_probe/train_tuev_mne.py`
  ```python
  if config.get("use_channel_mapper", False):
      from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper
      channel_mapper = TUEVChannelMapper().to(device)
      optimizer.add_param_group({"params": channel_mapper.parameters()})

  # Before feeding into EEGPT
  if config.get("use_channel_mapper", False):
      x = channel_mapper(x)  # (B,23,T) → (B,20,T)
  ```
- [ ] Config flag
  ```yaml
  # experiments/eegpt_linear_probe/configs/tuev_with_mapper.yaml
  use_channel_mapper: true
  channel_mapper_dropout: 0.3
  ```

#### Phase 4 — Testing & Validation
- [ ] A/B run
  ```bash
  python train_tuev_mne.py --config configs/tuev_with_mapper.yaml --run_name tuev_mapper
  ```
- [ ] Cross-dataset safety
  ```bash
  pytest tests/integration/test_tuab_*.py -v
  ```

#### Phase 5 — Documentation
- [ ] Update TUEV_FPZ_DISCREPANCY.md with results
- [ ] Add mapper usage to TRAINING.md
- [ ] Document performance comparison

### Execution Strategy (DECISION NEEDED NOW)
1) **Immediate Decision**: 
   - **Option A**: Let current cache finish (~40 min), test partial parity
   - **Option B**: Kill cache NOW, implement full paper parity
   
2) **If Option A (partial)**:
   - Complete 20-ch cache
   - Train with hyperparams only
   - Expect BAC < 62% (not full parity)
   
3) **If Option B (full parity)**:
   - Stop cache build immediately
   - Implement 23-ch dataset + mapper
   - Rebuild cache correctly
   - This is what achieved 62% BAC

### Success Metrics
- Sprint 4: +≥1% BAC improvement on TUEV; no TUAB regression; <10% training slow-down; gradients flow
- Sprint 5: ≥95% coverage on critical paths; ~2x faster tests with parallelization; 0 secrets via gitleaks; dead code removed or justified

### Monitoring Commands
```bash
tmux attach -t tuev_training -r
watch -n 1 nvidia-smi
tail -f experiments/eegpt_linear_probe/logs/latest.log | grep -E "loss|acc|auroc"
lsof | grep "cache.*tuev"
```

### Risks & Mitigations
- Disrupting active training → wait for completion; verify `tmux ls`
- Mapper overfitting → dropout + small LR; watch val–train gap
- Dataset regressions → gate by config; run full test suite
