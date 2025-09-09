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

### Sprint 4: TUEV Channel Mapper ❌ INCORRECTLY MARKED COMPLETE

**📊 Reference**: See [TUEV_METRICS_SSOT.md](TUEV_METRICS_SSOT.md) for target metrics and thresholds.

**🔴 CRITICAL DISCOVERY (2025-09-09)**: 
**Channel mapper is NOT optional - it's HOW EEGPT achieves 62% BAC!**

**What We Did (WRONG)**:
- Fpz synthesis via interpolation (preprocessing hack)
- Dropped A1/A2 channels
- 20-channel cache

**What Paper Actually Does (VERIFIED)**:
- Keeps ALL 23 TUEV channels
- Uses Conv2dWithConstraint(23, 20, 1) learnable mapper
- Mapper includes BatchNorm, GELU, Dropout(0.8)
- NO preprocessing synthesis

**Impact**: This is NOT "+1% improvement" - it's CORE to their approach!

**Reproducibility Seeds**: 42 (data), 123 (model init), 456 (augmentation) - as per TUEV_METRICS_SSOT.md

**🔴 CORRECT Implementation (from EEGPT reference)**:
```python
# src/brain_go_brrr/infra/ml_models/channel_mapper.py
class TUEVChannelMapper(nn.Module):
    """EEGPT's actual channel mapper - REQUIRED for paper parity."""

    def __init__(self):
        super().__init__()
        self.chan_conv = nn.Sequential(
            Conv2dWithConstraint(23, 20, kernel_size=1),  # 23→20 mapping
            nn.BatchNorm2d(20),
            nn.GELU(),
            nn.Dropout(0.8),  # High dropout!
            nn.Conv2d(20, 20, kernel_size=(1,55), groups=20, padding='same')
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
- [ ] Benchmark shows measurable BAC improvement (≥1% increase)

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
- +1% TUEV BAC (Sprint 4)
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

---

## 🔧 Fix Plan — Actionable Checklist

### ⚠️ Current Constraints (Do Not Interrupt Training)

```bash
# Non-disruptive monitoring
tmux ls
tmux attach -t tuev_training -r  # read-only attach

# Training status expectations
# - TUEV event detection training in progress
# - Cache built and operational
# - Do NOT change preprocessing until training completes
```

### Sprint 4: TUEV Channel Synthesis (+1% BAC)

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

### Execution Strategy (Post-Training)
1) Save state — export metrics, backup checkpoints, document config
2) Decision (based on Balanced Accuracy, not raw accuracy):
   - If TUEV BAC < 62% ⇒ implement Sprint 4 (below EEGPT paper target of 62.32%)
   - If BAC ≥ 62% ⇒ consider skipping Sprint 4 (target achieved)
3) If doing Sprint 4 ⇒ follow Phases 1–5 and A/B test
4) Sprint 5 can be done independently (start with test parallelization)

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
