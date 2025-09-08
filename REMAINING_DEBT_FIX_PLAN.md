# 🔧 REMAINING DEBT FIX PLAN - ACTIONABLE CHECKLIST

**Created**: September 8, 2025  
**Status**: Ready for implementation when training completes  
**Priority**: OPTIONAL - All critical items already complete  
**Constraint**: TUEV training currently running in tmux - DO NOT INTERRUPT

---

## ⚠️ CURRENT CONSTRAINTS

### Active Processes (DO NOT INTERRUPT)
```bash
# Check status without disrupting:
tmux ls  # Should show active sessions
tmux attach -t tuev_training -r  # Read-only attach to monitor

# Current training status:
# - TUEV event detection training in progress
# - Cache built and operational
# - DO NOT modify preprocessing pipeline until complete
```

---

## 📋 SPRINT 4: TUEV CHANNEL SYNTHESIS (+1% Accuracy)

### Context from TUEV_FPZ_DISCREPANCY.md:
- **Problem**: TUEV has 23 channels, EEGPT expects 20 (including Fpz which TUEV lacks)
- **Current Solution**: Zero-filling Fpz (working, 99% training complete)
- **EEGPT Authors' Solution**: Learnable Conv2d(23→20) mapping
- **Potential Gain**: ~1% accuracy improvement

### Implementation Checklist:

#### Phase 1: Preparation (After training completes)
- [ ] **Backup current model checkpoints**
  ```bash
  cp -r experiments/eegpt_linear_probe/checkpoints experiments/eegpt_linear_probe/checkpoints_zerofill_backup
  ```

- [ ] **Document current performance baseline**
  ```bash
  # Record current metrics from training logs
  grep "val_acc\|val_auroc" experiments/eegpt_linear_probe/logs/latest.log > baseline_metrics.txt
  ```

#### Phase 2: Create Channel Mapper Module
- [ ] **Create new file: `src/brain_go_brrr/infra/ml_models/channel_mapper.py`**
  ```python
  """Learnable channel mapping for TUEV dataset compatibility."""
  import torch
  import torch.nn as nn
  from typing import Optional
  
  class TUEVChannelMapper(nn.Module):
      """Maps TUEV 23 channels to EEGPT expected 20 channels.
      
      Replaces zero-filling of Fpz with learnable synthesis.
      Based on EEGPT authors' approach in their reference implementation.
      """
      
      def __init__(self, dropout: float = 0.3):
          super().__init__()
          # Match EEGPT authors' architecture
          self.channel_conv = nn.Sequential(
              nn.Conv1d(23, 20, kernel_size=1, bias=True),
              nn.BatchNorm1d(20),
              nn.GELU(),
              nn.Dropout(dropout)
          )
          
      def forward(self, x: torch.Tensor) -> torch.Tensor:
          """Map (B, 23, T) → (B, 20, T).
          
          Args:
              x: Input tensor (batch, 23_channels, time)
              
          Returns:
              Mapped tensor (batch, 20_channels, time)
          """
          return self.channel_conv(x)
  ```

- [ ] **Add tests: `tests/unit/infra/ml_models/test_channel_mapper.py`**
  ```python
  def test_channel_mapper_shapes():
      mapper = TUEVChannelMapper()
      x = torch.randn(32, 23, 1024)  # batch=32, channels=23, time=1024
      y = mapper(x)
      assert y.shape == (32, 20, 1024)
      
  def test_gradient_flow():
      mapper = TUEVChannelMapper()
      x = torch.randn(1, 23, 256, requires_grad=True)
      y = mapper(x)
      loss = y.mean()
      loss.backward()
      assert x.grad is not None
  ```

#### Phase 3: Integration with Training Pipeline
- [ ] **Modify `experiments/eegpt_linear_probe/train_tuev_mne.py`**
  ```python
  # Add after model initialization (around line 200):
  if config.get('use_channel_mapper', False):
      from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper
      channel_mapper = TUEVChannelMapper().to(device)
      optimizer.add_param_group({'params': channel_mapper.parameters()})
  
  # In training loop, before EEGPT:
  if config.get('use_channel_mapper', False):
      x = channel_mapper(x)  # Map 23→20 channels
  ```

- [ ] **Add config flag to training config**
  ```yaml
  # experiments/eegpt_linear_probe/configs/tuev_with_mapper.yaml
  use_channel_mapper: true
  channel_mapper_dropout: 0.3
  ```

#### Phase 4: Testing & Validation
- [ ] **Create A/B test script**
  ```bash
  # Train with mapper
  python train_tuev_mne.py --config configs/tuev_with_mapper.yaml --run_name tuev_mapper
  
  # Compare with baseline
  python compare_runs.py --baseline tuev_zerofill --experiment tuev_mapper
  ```

- [ ] **Validate no regression on other datasets**
  ```bash
  # TUAB should be unaffected (already 20 channels)
  pytest tests/integration/test_tuab_*.py -v
  ```

#### Phase 5: Documentation
- [ ] Update TUEV_FPZ_DISCREPANCY.md with results
- [ ] Add mapper usage to TRAINING.md
- [ ] Document performance comparison

---

## 📋 SPRINT 5: POLISH & HARDENING

### 1. Code Coverage to 95% (2 hours)

#### Coverage Gap Analysis:
- [ ] **Run coverage report**
  ```bash
  make coverage
  # Focus on files < 80% coverage
  ```

- [ ] **Priority test additions**:
  ```python
  # tests/unit/infra/cache/test_memory_cache.py
  def test_ttl_expiry():
      cache = InMemoryCache(ttl=0.1)
      cache.set("key", "value")
      time.sleep(0.2)
      assert cache.get("key") is None
      
  def test_clear_pattern_wildcards():
      cache = InMemoryCache()
      cache.set("user:1", "Alice")
      cache.set("user:2", "Bob")
      cache.set("admin:1", "Charlie")
      cache.clear_pattern("user:*")
      assert cache.get("user:1") is None
      assert cache.get("admin:1") == "Charlie"
  ```

- [ ] **Redis error handling tests**:
  ```python
  # tests/unit/infra/redis/test_error_handling.py
  @patch('redis.Redis')
  def test_connection_timeout(mock_redis):
      mock_redis.side_effect = redis.TimeoutError
      # Test graceful degradation
  ```

### 2. Test Parallelization (30 min)

- [ ] **Update pyproject.toml**
  ```toml
  [tool.pytest.ini_options]
  addopts = """
  -ra
  --strict-markers
  --ignore=reference_repos
  --ignore=literature
  --ignore=data
  -n auto
  --dist loadscope
  --timeout=60
  """
  
  markers = [
      "slow: marks tests as slow",
      "integration: marks integration tests",
      "data: requires real data files",
  ]
  ```

- [ ] **Mark slow tests**
  ```python
  @pytest.mark.slow
  def test_full_preprocessing_pipeline():
      # Long-running test
  ```

- [ ] **Verify parallel execution**
  ```bash
  pytest -n 4 --dist loadscope -v
  # Should show: [gw0] [gw1] [gw2] [gw3] in output
  ```

### 3. Dead Code Detection (30 min)

- [ ] **Configure ruff for dead code**
  ```toml
  # pyproject.toml
  [tool.ruff]
  select = [
      "F401",  # Unused imports
      "F841",  # Unused variables
      "F821",  # Undefined names
      "F823",  # Local variable referenced before assignment
  ]
  ```

- [ ] **Run quarterly vulture check**
  ```bash
  # Not in CI, just periodic manual check
  pip install vulture
  vulture src/ --min-confidence 80 --exclude "*_test.py,*/tests/*"
  ```

### 4. Secrets Scanning (30 min)

- [ ] **Add to CI workflow**
  ```yaml
  # .github/workflows/security.yml
  name: Security Scan
  on: [push, pull_request]
  
  jobs:
    gitleaks:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
          with:
            fetch-depth: 0
        - uses: gitleaks/gitleaks-action@v2
          env:
            GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  ```

- [ ] **Create .gitleaks.toml**
  ```toml
  [allowlist]
  paths = [
    "literature/",
    "reference_repos/",
    "*.md"
  ]
  ```

---

## 🚀 EXECUTION STRATEGY

### When Training Completes:

1. **Save current state** (30 min)
   - [ ] Export final metrics
   - [ ] Backup model checkpoints
   - [ ] Document configuration used

2. **Decision Point**:
   - [ ] Review baseline TUEV performance
   - [ ] If < 80% accuracy → Implement Sprint 4 (channel mapper)
   - [ ] If ≥ 80% accuracy → Consider skipping Sprint 4

3. **Sprint 4 Implementation** (4 hours)
   - [ ] ONLY if accuracy improvement needed
   - [ ] Follow Phase 1-5 checklist above
   - [ ] A/B test with/without mapper

4. **Sprint 5 Implementation** (3 hours)
   - [ ] Can be done independently
   - [ ] Start with test parallelization (biggest win)
   - [ ] Add coverage incrementally

---

## 📊 SUCCESS METRICS

### Sprint 4 (Channel Mapper):
- [ ] ≥1% accuracy improvement on TUEV validation set
- [ ] No regression on TUAB dataset
- [ ] Training time increase < 10%
- [ ] Clean gradient flow through mapper

### Sprint 5 (Polish):
- [ ] Code coverage ≥ 95% on critical paths
- [ ] Test suite runs 2x faster with parallelization
- [ ] Zero secrets detected by gitleaks
- [ ] All dead code removed or documented

---

## 🔍 MONITORING COMMANDS

```bash
# Check training progress (read-only)
tmux attach -t tuev_training -r

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check latest metrics
tail -f experiments/eegpt_linear_probe/logs/latest.log | grep -E "loss|acc|auroc"

# Verify no file locks
lsof | grep "cache.*tuev"
```

---

## ⚠️ RISKS & MITIGATIONS

### Risk 1: Disrupting Active Training
- **Mitigation**: Wait for training completion
- **Check**: `tmux ls` shows no active sessions

### Risk 2: Channel Mapper Overfitting
- **Mitigation**: Use dropout, small learning rate
- **Check**: Monitor val vs train accuracy gap

### Risk 3: Breaking Other Datasets
- **Mitigation**: Conditional mapper only for TUEV
- **Check**: Run full test suite after changes

---

## 📝 NOTES

1. **Sprint 4 is OPTIONAL** - Current zero-fill works (99% training success)
2. **Sprint 5 is NICE-TO-HAVE** - Current quality gates are strong
3. **DO NOT interrupt training** - Let it complete first
4. **Backup everything** before implementing changes
5. **A/B test** any accuracy-critical changes

---

## DECISION CHECKPOINT

**After reviewing this plan:**

- [ ] Implement Sprint 4 (Channel Mapper) - IF accuracy < 80%
- [ ] Implement Sprint 5 (Polish) - IF moving to production
- [ ] Defer both - IF current state is sufficient

**Rationale**: _________________________________

**Target completion**: _________________________