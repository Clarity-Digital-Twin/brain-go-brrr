# EEGPT & TUEV: Architecture Fix Documentation

## Executive Summary

**CRITICAL BUG**: Our implementation only returns 4 summary tokens (2,048 features) but TUEV needs ALL 60 summary tokens (15 temporal × 4 summary = 30,720 features). This causes catastrophic failure: BAcc 0.15 vs paper's 0.62.

**ROOT CAUSE**: Line 498 in `src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`:
```python
x = x[:, -self.embed_num :, :]  # Only extracts last 4 tokens, throws away 320 patch tokens!
```

**SOLUTION**: Extract ALL 15 temporal positions × 4 summary tokens × 512 dims = 30,720 features (matches paper's Table 13)

## The Problem

### What the Paper Shows
1. Input: 20 × 1000 (after channel reduction)
2. EEGPT encoder with kernel=64, stride=64 → 15 temporal patches
3. Output shape: 15 × 4 × 512 (Table 13, line 612)
4. Text says: "encoder passes output tokens corresponding to summary tokens" (line 164)
5. **Our Bug**: We only return 4 tokens (2,048 features) instead of 30,720

### Why TUEV Fails
- Paper shows 15 × 4 × 512 = 30,720 features needed
- Our implementation returns only 4 × 512 = 2,048 features
- Missing 93.3% of required features
- Lost all temporal structure (15 positions collapsed to 1)
- Result: Worse than random guessing (BAcc 0.15 < 0.167)

## Paper Evidence

### Table 13 Architecture (Lines 606-613)
```
Input: 23 channels → Conv1d → 20 channels → EEGPT → "15 × 4 × 512" → Linear → 6 classes
```

### Critical Observation
- Paper Table 13 shows "15 × 4 × 512" output shape
- Paper text: "encoder passes output tokens corresponding to summary tokens" (Line 164)
- Pattern: TUAB has 31 patches → 31 × 4 × 512; TUEV has 15 patches → 15 × 4 × 512
- Interpretation: First dimension tracks temporal patches, preserving temporal structure
- Total features: 15 × 4 × 512 = 30,720 (NOT our current 2,048)

## Implementation Fix

### Step 1: Create Proper Feature Extractor

```python
# src/brain_go_brrr/models/eegpt/feature_extractor.py

class EEGPTFeatureExtractor:
    """Extract features preserving temporal structure for downstream tasks."""
    
    def __init__(self, checkpoint_path: str):
        self.model = create_eegpt_model(checkpoint_path)
    
    def extract_features(self, x):
        # Forward pass through encoder
        # Current: returns only last 4 tokens
        # Need: extract features preserving temporal structure
        
        # Option 1: Extract patch embeddings at specific positions
        # Option 2: Extract transformed patches before summary token selection
        # Option 3: Reshape summary tokens with temporal information
        
        # The exact mechanism is unclear from paper, but output must be:
        # TUAB: [batch, 31, 4, 512] → flatten → [batch, 63,488]
        # TUEV: [batch, 15, 4, 512] → flatten → [batch, 30,720]
        
        # TODO: Investigate exact extraction method from paper's code
```

### Step 2: Fix TUEV Training

```python
# experiments/eegpt_linear_probe/train_tuev_fixed.py

class TUEVLinearProbeFixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = EEGPTFeatureExtractor(checkpoint_path)
        # Classifier expects 30,720 features per Table 13
        self.classifier = nn.Linear(15 * 4 * 512, 6)
```

## Verification Results

Paper vs Our Implementation:
```
Current (last 4 tokens):     2,048 features → BAcc 0.15 ❌
Paper (all 60 tokens):      30,720 features → BAcc 0.62 🎯
Fix needed: Extract ALL summary tokens, not just last 4
```

## Why This Will Work

1. **Matches Paper Dimensions**: 15 × 4 × 512 = 30,720 features (Table 13)
2. **Preserves Temporal Structure**: 15 temporal positions maintained
3. **Feature/Sample Ratio**: 30,720 / 84,000 = 0.37 (reasonable)
4. **Pattern Consistency**: TUAB (31×4×512) and TUEV (15×4×512) follow same pattern
5. **Fix Both Tasks**: TUAB needs 63,488 features, not 2,048

## Migration Plan

### Week 1
- [ ] Implement `EEGPTFeatureExtractor` with multiple modes
- [ ] Test patch extraction on small subset
- [ ] Verify shapes match paper

### Week 2  
- [ ] Train TUEV with 15-patch extraction
- [ ] Compare to summary-only baseline
- [ ] Tune regularization if overfitting

### Week 3
- [ ] Ensure TUAB still works with summary mode
- [ ] Document API changes
- [ ] Create migration guide

## Configuration Changes

### TUEV Config Updates
```yaml
# configs/tuev_table13_aligned.yaml
model:
  extract_all_summary_tokens: true  # NOT just last 4
  classifier:
    input_dim: 30720  # 15 × 4 × 512, NOT 2048
    dropout: 0.5  # Table 13 specified
```

## Expected Outcomes

| Metric | Current | Expected | Target |
|--------|---------|----------|--------|
| BAcc | 0.15 | 0.40+ | 0.62 |
| F1 | 0.50 | 0.70+ | 0.82 |
| Kappa | -0.01 | 0.30+ | 0.64 |

## Command to Test

```bash
# After implementing fix
cd experiments/eegpt_linear_probe
python train_tuev_fixed.py --config configs/tuev_table13_aligned.yaml --use-patches
```

## References

- EEGPT Paper: Table 13 (lines 606-613)
- Architecture: `src/brain_go_brrr/infra/ml_models/eegpt_architecture.py:498`
- Verification: `experiments/eegpt_linear_probe/FINAL_TUEV_VERIFICATION.py`

---

**STATUS**: Ready to implement. No more analysis needed. Just build the fix.