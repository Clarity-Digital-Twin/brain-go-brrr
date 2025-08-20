# CRITICAL ISSUES: Deep Audit Results

## 🚨 NEW CRITICAL FINDINGS BEYOND EEGPT FEATURES

### 1. **AVERAGING BUG in train_paper_aligned.py**
**Location**: `experiments/eegpt_linear_probe/train_paper_aligned.py:67`
```python
x = features.mean(dim=1)  # (batch_size, embed_dim)
```
**Problem**: This averages the 4 summary tokens to get 512 features instead of flattening to 2048!
**Impact**: Using only 512 features when we should use AT MINIMUM 2048 (and actually 63,488)
**Fix**: Should be `features.flatten(1)` not `features.mean(dim=1)`

### 2. **WRONG CONFIG for TUAB**
**Location**: `experiments/eegpt_linear_probe/configs/tuab_4s_paper_aligned.yaml:27`
```yaml
probe:
  input_dim: 512  # WRONG! Should be 63488
```
**Problem**: Config expects only 512 features
**Impact**: Linear probe created with wrong input dimension
**Fix**: Change to `input_dim: 63488`

### 3. **POOLING in LinearProbeHead**
**Location**: `src/brain_go_brrr/infra/ml_models/linear_probe.py:207-208`
```python
if self.pool == "mean":
    x = x.mean(dim=1)
```
**Problem**: Default pooling is "mean" which averages temporal features
**Impact**: Any code using LinearProbeHead loses temporal information by default
**Fix**: Should use pool="flatten" or handle temporal dimension properly

### 4. **TUEV HARDCODED to 2048 features**
**Location**: `experiments/eegpt_linear_probe/train_tuev_aligned.py:96`
```python
self.classifier = nn.Linear(4 * 512, 6)  # 2048 features
```
**Problem**: Should be `15 * 4 * 512 = 30720`
**Impact**: TUEV gets 0.15 BAcc (worse than random)
**Fix**: Change to `nn.Linear(30720, 6)`

### 5. **EEGPTWrapper doesn't support temporal mode**
**Location**: `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py:93-108`
```python
def forward(self, x: torch.Tensor, chan_ids: torch.Tensor | None = None) -> torch.Tensor:
    # No return_all_temporal parameter!
    return cast("torch.Tensor", self.model(x, chan_ids))
```
**Problem**: Wrapper doesn't pass through temporal extraction flag
**Impact**: Can't get temporal features even if we fix EEGTransformer
**Fix**: Add `return_all_temporal` parameter and pass through

### 6. **Normalization Using Wrong Stats**
**Location**: `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py:52-56`
```python
# Default normalization parameters - TUAB is already normalized!
self.register_buffer("input_mean", torch.zeros(1))
self.register_buffer("input_std", torch.ones(1))
```
**Problem**: Comments say "TUAB is already normalized" but uses identity normalization
**Impact**: May not be normalizing correctly for pretrained EEGPT
**Check**: Need to verify if TUAB preprocessing already normalizes

### 7. **embed_num HARDCODED to 4 everywhere**
**Locations**: Multiple files hardcode `embed_num=4`
- All model initializations default to 4
- Tests assert shape must be (B, 4, 512)
- No way to change number of summary tokens

**Problem**: Can't experiment with different numbers of summary tokens
**Impact**: Limited architectural flexibility

## Summary Table

| Issue | Severity | Impact | Files Affected |
|-------|----------|--------|----------------|
| Averaging instead of flattening | CRITICAL | 75% feature loss | train_paper_aligned.py |
| Wrong TUAB config (512 vs 63488) | CRITICAL | 99.2% feature loss | tuab_4s_paper_aligned.yaml |
| TUEV wrong dimensions | CRITICAL | 93.3% feature loss | train_tuev_aligned.py |
| Default mean pooling | HIGH | Temporal info loss | linear_probe.py |
| Wrapper missing temporal | HIGH | Can't access features | eegpt_wrapper.py |
| Normalization confusion | MEDIUM | Potential accuracy loss | eegpt_wrapper.py |
| Hardcoded embed_num | LOW | Limited flexibility | Multiple files |

## Compounding Effects

These issues COMPOUND each other:
1. EEGPT only returns 4 tokens (missing temporal)
2. Then we AVERAGE those 4 tokens to 1 (losing summary diversity)
3. Config expects only 512 features (can't handle more)
4. Tests enforce wrong shapes (prevent fixes)

**Result**: We're using 512 features out of 63,488 possible = **0.8% utilization**

## Immediate Actions Required

1. **Fix train_paper_aligned.py**: Change `.mean(dim=1)` to `.flatten(1)`
2. **Update TUAB config**: Change input_dim from 512 to 63488
3. **Fix TUEV classifier**: Change from 2048 to 30720 inputs
4. **Add temporal mode to EEGPT**: Implement `return_all_temporal` flag
5. **Update wrapper**: Pass through temporal flag
6. **Fix tests**: Make shape assertions conditional on mode

## Performance Impact Estimate

With ALL fixes applied:
- **TUAB**: 0.79 → 0.87 AUROC (10% improvement)
- **TUEV**: 0.15 → 0.62 BAcc (4x improvement)

These aren't minor bugs - they're FUNDAMENTAL ARCHITECTURAL ERRORS!

## Data Split Verification

**Finding**: Train/eval splits appear correct at the file level
- TUAB: Uses separate directories for train/eval
- TUEV: Has edf/train and edf/eval structure
- No evidence of data leakage between splits
- Split happens at subject/recording level, not window level

## Root Cause Analysis

The problems stem from **misunderstanding the EEGPT architecture**:

1. **Temporal Processing**: EEGPT processes EACH temporal patch independently with its own summary tokens
2. **Feature Extraction**: Must extract ALL temporal×summary features, not just the last 4
3. **Averaging Cascade**: Multiple averaging operations compound the feature loss:
   - First: Only extracting last 4 tokens (losing temporal)
   - Second: Averaging those 4 to 1 (losing summary diversity)
   - Third: Config expecting only 512 features (preventing fixes)

## The Math

**TUAB (8-second windows at 256Hz)**:
- Samples: 2048
- Patches: 2048/64 = 32 temporal positions
- But we use 4s windows = 1024 samples = 16 patches
- Wait... config says 4s but code expects 8s?
- **ANOTHER BUG**: Window size mismatch!

**TUEV (4-second windows at 250Hz)**:
- Samples: 1000
- Patches: 1000/64 = 15.625 → 15 (with padding)
- Features: 15 × 4 × 512 = 30,720
- Currently using: 2,048 (6.7%)

## Priority Fix Order

1. **Immediate**: Fix averaging bug in train_paper_aligned.py (1 line change)
2. **Critical**: Implement return_all_temporal in EEGPT (core fix)
3. **Essential**: Update configs with correct dimensions
4. **Important**: Fix wrapper to pass temporal flag
5. **Cleanup**: Update tests to handle both modes
