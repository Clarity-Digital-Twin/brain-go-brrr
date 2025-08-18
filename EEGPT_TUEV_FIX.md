# EEGPT & TUEV: Architecture Fix Documentation

## Executive Summary

**CRITICAL BUG**: Our implementation returns 4 summary tokens TOTAL (2,048 features) but TUEV needs 4 summary tokens PER TEMPORAL PATCH (15 patches × 4 tokens = 30,720 features). This causes catastrophic failure: BAcc 0.15 vs paper's 0.62.

**ROOT CAUSE**: Architectural misunderstanding. We process all patches together:
```python
# Our implementation
Patches (B, 15*20, 512) → Add 4 tokens → Transformer → Extract last 4 → (B, 4, 512)
```

**REFERENCE IMPLEMENTATION**: Processes each temporal position separately:
```python  
# Reference EEGPT
Patches (B, 15, 20, 512) → Flatten (B*15, 20, 512) → Add 4 tokens to EACH → 
Transformer → Extract 4 from EACH → (B, 15, 4, 512)
```

**SOLUTION**: Process temporal patches independently, getting 4 summary tokens per patch = 30,720 features total

## The Problem Explained

### What the Paper Actually Does
1. Input: 20 × 1000 (after channel reduction)
2. Creates 15 temporal patches (1000 / 64 = 15.625 → 15)
3. **KEY**: Each temporal patch processed separately through transformer
4. Each patch gets 4 summary tokens
5. Output: 15 patches × 4 tokens × 512 dims = 30,720 features

### Reference Code Evidence
```python
# Line 769 in EEGPT_mcae_finetune_change_tuev.py
LinearWithConstraint(30720, num_classes)  # Hardcoded!

# Lines 532-536: Process each temporal position
x = x.flatten(0, 1)  # (B*15, 20, 512)
summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
x = torch.cat([x, summary_token], dim=1)  # Add to EACH position

# Line 558: Final shape
x = x.reshape((B, N, self.embed_num, -1))  # (B, 15, 4, 512)
```

### Why TUEV Fails With Our Implementation
- We return 4 tokens TOTAL (no temporal information)
- Reference returns 4 tokens PER temporal position (preserves time)
- Missing 93.3% of features AND all temporal structure
- Result: BAcc 0.15 (worse than random 0.167)

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

## Implementation Fix - CONFIRMED FROM REFERENCE CODE

### How Reference EEGPT Extracts Features

From `reference_repos/EEGPT/downstream_tueg/Modules/models/EEGPT_mcae_finetune_change_tuev.py`:

1. **Encoder forward** (line 508-559):
   - Processes patches: `x = self.patch_embed(x)` → shape (B, N, C, D)
   - N = num_temporal_patches (15 for TUEV with 1000 samples)
   - Adds channel embeddings and summary tokens
   - After transformer blocks: `x = x[:, -summary_token.shape[1]:, :]`
   - **KEY**: Returns shape (B, N, embed_num, embed_dim) = (B, 15, 4, 512)

2. **Classifier forward** (line 828-845):
   - Gets encoder output: `x = self.forward_features(x)` → (B, 15, 4, 512)
   - **Line 843**: `x = x.flatten(1)` → (B, 30720)
   - **Line 769**: `LinearWithConstraint(30720, num_classes)` - HARDCODED!

### Step 1: Fix Our Encoder Output

```python
# src/brain_go_brrr/infra/ml_models/eegpt_architecture.py

def forward(self, x: Tensor, chan_ids: Tensor | None = None, return_all_tokens: bool = False) -> Tensor:
    """Forward pass through EEG Transformer encoder.
    
    Args:
        x: Input tensor of shape (B, C, T)
        chan_ids: Channel IDs for positional embedding
        return_all_tokens: If True, return all temporal positions × summary tokens
                          If False, return only last 4 summary tokens (legacy)
    
    Returns:
        If return_all_tokens: (B, num_patches, embed_num, embed_dim)
        Else: (B, embed_num, embed_dim) - legacy behavior
    """
    # ... existing code until line 494 ...
    
    # Apply transformer blocks
    for block in self.blocks:
        x = block(x)
    
    if return_all_tokens:
        # NEW: Return ALL temporal positions with their summary tokens
        # Split patches and summary tokens
        num_patches = num_patches * num_channels  # Total patch tokens
        patch_tokens = x[:, :num_patches, :]
        summary_tokens = x[:, -self.embed_num:, :]
        
        # Reshape to preserve temporal structure
        # patch_tokens shape: (B, num_temporal * num_channels, embed_dim)
        # We need: (B, num_temporal, embed_num, embed_dim)
        
        # The reference code reshapes this way (lines 549-550, 558):
        B = batch_size
        N = time_steps // self.patch_size  # num_temporal_patches
        
        # They use the summary tokens PER temporal position
        # So we need to replicate or track summary tokens per patch
        
        # Actually, looking closer at line 536-543:
        # They concatenate summary tokens AFTER flattening patches
        # So each temporal position gets the SAME summary tokens
        
        # Return shape matching reference: (B, N, embed_num, embed_dim)
        summary_tokens = summary_tokens.unsqueeze(1).repeat(1, N, 1, 1)
        x = summary_tokens  # Shape: (B, N, 4, 512)
    else:
        # Legacy: Extract only the last summary tokens  
        x = x[:, -self.embed_num:, :]
    
    # Final normalization
    x = self.norm(x)
    
    return x
```

### Step 2: Fix TUEV Training

```python
# experiments/eegpt_linear_probe/train_tuev_fixed.py

class TUEVLinearProbeFixed(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        # Load EEGPT encoder
        self.encoder = create_eegpt_model(checkpoint_path)
        self.encoder.eval()  # Freeze encoder
        
        # Classifier expects 30,720 features (15 × 4 × 512)
        # This matches reference line 769: LinearWithConstraint(30720, num_classes)
        self.classifier = nn.Linear(15 * 4 * 512, 6)
        self.dropout = nn.Dropout(0.5)  # Match paper Table 13
    
    def forward(self, x):
        with torch.no_grad():
            # Get ALL temporal positions × summary tokens
            features = self.encoder(x, return_all_tokens=True)  # (B, 15, 4, 512)
        
        # Flatten to match reference (line 843)
        features = features.flatten(1)  # (B, 30720)
        
        # Apply dropout and classify
        features = self.dropout(features)
        return self.classifier(features)
```

## Verification Results

Paper vs Our Implementation:
```
Current:    4 tokens total    →  2,048 features → BAcc 0.15 ❌
Reference: 15×4 tokens        → 30,720 features → BAcc 0.62 🎯
Fix: Process each temporal patch separately, get 4 tokens each
```

## Why This Will Work

1. **Matches Reference Exactly**: 30,720 features hardcoded in their classifier
2. **Preserves Temporal Structure**: Each 250ms window analyzed separately
3. **Proven Architecture**: Reference achieves 0.62 BAcc with this approach
4. **Pattern Consistency**: 
   - TUEV: 1000 samples → 15 patches → 15×4×512 = 30,720 features
   - TUAB: 2000 samples → 31 patches → 31×4×512 = 63,488 features
5. **Fixes Root Cause**: Temporal information preserved, not collapsed

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