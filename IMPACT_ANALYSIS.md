# EEGPT Fix: Complete Impact Analysis

## Files That Need Changes

### 1. Core Architecture File
**`src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`**
- **Line 498**: Currently `x = x[:, -self.embed_num :, :]` - extracts only last 4 tokens
- **Change**: Add `return_all_temporal` parameter to forward method
- **Impact**: ALL downstream code needs to handle new output shape option

### 2. Linear Probe Implementations (3 files)
**`src/brain_go_brrr/infra/ml_models/eegpt_linear_probe.py`**
- **Line 80**: `feature_dim = embed_dim * n_summary_tokens  # 512 * 4 = 2048`
- **Change**: Need to handle temporal dimension, feature_dim depends on input size

**`src/brain_go_brrr/infra/ml_models/eegpt_linear_probe_robust.py`**
- Similar changes needed

**`src/brain_go_brrr/infra/ml_models/eegpt_two_layer_probe.py`**
- Similar changes needed

### 3. Wrapper Classes
**`src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py`**
- **Lines 93-120**: forward and extract_features methods
- **Change**: Pass through `return_all_temporal` parameter

### 4. Training Scripts
**`experiments/eegpt_linear_probe/train_paper_aligned.py`**
- **Config**: `input_dim: 512` should be `63488` for TUAB
- **Model**: Need to handle (B, 31, 4, 512) features

**`experiments/eegpt_linear_probe/train_tuev_aligned.py`**
- **Line 96**: `self.classifier = nn.Linear(4 * 512, 6)`
- **Change**: Should be `nn.Linear(15 * 4 * 512, 6)` = 30720

### 5. Test Files That Will Break (7 critical tests)
All these tests assert shape == (B, 4, 512):
- `tests/unit/test_models_eegpt_wrapper.py:53`
- `tests/unit/test_models_eegpt_model.py:156`
- `tests/unit/test_eegpt_model_loading.py:135`
- `tests/unit/test_eegpt_checkpoint_loading.py:191,198`
- `tests/unit/test_encoder_raw_output.py:41`
- `tests/unit/test_eegpt_summary_tokens.py:69`

### 6. Feature Extractors
**`src/brain_go_brrr/infra/adapters/eegpt_feature_extractor.py`**
- Needs to handle new output shape

**`src/brain_go_brrr/domain/preprocessing/features/extractor.py`**
- May need updates for temporal features

### 7. API Endpoints
**`src/brain_go_brrr/api/routers/eegpt.py`**
- May need to specify which mode to use

## Implementation Strategy

### Phase 1: Add Backward-Compatible Flag
```python
def forward(self, x: Tensor, chan_ids: Tensor | None = None, 
            return_all_temporal: bool = False) -> Tensor:
    """
    Returns:
        If return_all_temporal=False: (B, 4, 512) - legacy
        If return_all_temporal=True: (B, N_temporal, 4, 512) - new
    """
```

### Phase 2: Update Training Scripts
1. TUAB: Change config to use 63,488 features
2. TUEV: Change classifier to use 30,720 features
3. Add flatten operation after getting features

### Phase 3: Fix Tests
Create two test modes:
- Legacy tests: Use `return_all_temporal=False`
- New tests: Use `return_all_temporal=True`

## Risk Assessment

### High Risk
- **Breaking existing trained models**: Old checkpoints expect 2048 features
- **API compatibility**: External users may depend on current shape
- **Test suite**: 100+ tests may need updates

### Medium Risk
- **Performance**: More features = more memory usage
- **Training time**: Larger classifiers train slower
- **Overfitting**: More parameters with same data

### Low Risk
- **Accuracy**: Should only improve with more features
- **Code clarity**: New flag makes behavior explicit

## Migration Plan

### Step 1: Create Feature Branch
```bash
git checkout -b feature/eegpt-temporal-features
```

### Step 2: Implement Core Change
1. Add `return_all_temporal` to EEGTransformer.forward()
2. Implement temporal processing logic
3. Test both modes work

### Step 3: Update Training
1. Fix TUAB config and training script
2. Fix TUEV implementation
3. Test training doesn't crash

### Step 4: Fix Tests
1. Add parameter to test fixtures
2. Update shape assertions
3. Add new tests for temporal mode

### Step 5: Validation
1. Train TUAB with new features
2. Verify AUROC improves from 0.79 → 0.85+
3. Train TUEV with new features
4. Verify BAcc improves from 0.15 → 0.40+

## Code Snippets

### Fix for eegpt_architecture.py
```python
def forward(self, x: Tensor, chan_ids: Tensor | None = None,
            return_all_temporal: bool = False) -> Tensor:
    # ... existing code until line 461 ...
    x = self.patch_embed(x)
    B, N, C, D = x.shape  # (B, num_patches, num_channels, embed_dim)
    
    if return_all_temporal:
        # NEW PATH: Process each temporal position
        x = x.flatten(0, 1)  # (B*N, C, D)
        
        # Add summary tokens to each temporal position
        summary_tokens = self.summary_token.repeat(B * N, 1, 1)
        x = torch.cat([x, summary_tokens], dim=1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract summary tokens from each position
        x = x[:, -self.embed_num:, :]  # (B*N, 4, D)
        x = self.norm(x)
        
        # Reshape to preserve temporal
        x = x.reshape(B, N, self.embed_num, D)  # (B, N, 4, 512)
        return x
    else:
        # LEGACY PATH: Current behavior
        # ... existing code ...
```

### Fix for TUAB config
```yaml
probe:
  input_dim: 63488  # 31 * 4 * 512, was 512
```

### Fix for TUEV classifier
```python
self.classifier = nn.Linear(15 * 4 * 512, 6)  # 30720 features
```

## Success Criteria

1. **No Breaking Changes**: Legacy mode still returns (B, 4, 512)
2. **TUAB Improvement**: AUROC increases from 0.79 to 0.85+
3. **TUEV Success**: BAcc increases from 0.15 to 0.40+
4. **Tests Pass**: All tests work with appropriate mode
5. **Documentation**: Clear migration guide for users

## Files Summary

- **1 core file** to modify (eegpt_architecture.py)
- **3 probe files** to update
- **2 training scripts** to fix
- **7+ test files** to update
- **~10 wrapper/adapter files** to pass through parameter
- **Total: ~25 files** need changes for full implementation