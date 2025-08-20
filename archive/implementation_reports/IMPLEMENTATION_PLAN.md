# Implementation Plan: Fix EEGPT Feature Extraction

## Problem Summary

Our EEGPT returns only 4 summary tokens (2,048 features) but TUEV needs 60 summary tokens (30,720 features) - one set of 4 tokens per temporal patch.

## Architecture Difference

### Our Current Implementation
```python
# All patches processed together
Input (B, 20, 1000)
→ Patches (B, 15*20, 512)
→ Add 4 summary tokens
→ Transformer
→ Extract last 4 tokens
→ Output (B, 4, 512) = 2,048 features
```

### Reference Implementation
```python
# Each temporal position processed separately
Input (B, 20, 1000)
→ Patches (B, 15, 20, 512)
→ Flatten to (B*15, 20, 512)
→ Add 4 summary tokens to EACH
→ Transformer on (B*15, 24, 512)
→ Extract 4 tokens from EACH
→ Output (B, 15, 4, 512) = 30,720 features
```

## Implementation Steps

### Step 1: Modify EEGTransformer Forward Pass

**File**: `src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`

```python
def forward(self, x: Tensor, chan_ids: Tensor | None = None,
            return_all_temporal: bool = False) -> Tensor:
    """
    Args:
        return_all_temporal: If True, return (B, N_temporal, 4, 512)
                           If False, return (B, 4, 512) for backward compat
    """
    # ... existing patch embedding ...

    if return_all_temporal:
        # NEW PATH: Process each temporal position separately
        B, N, C, D = x.shape  # After patch_embed

        # Flatten batch and temporal dims
        x = x.flatten(0, 1)  # (B*N, C, D)

        # Add summary tokens to EACH temporal position
        summary_tokens = self.summary_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([x, summary_tokens], dim=1)  # (B*N, C+4, D)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract summary tokens from each position
        x = x[:, -self.embed_num:, :]  # (B*N, 4, D)
        x = self.norm(x)

        # Reshape to preserve temporal structure
        x = x.reshape(B, N, self.embed_num, -1)  # (B, N, 4, 512)
        return x
    else:
        # LEGACY PATH: Current behavior for TUAB
        # ... existing code ...
```

### Step 2: Create TUEV-Specific Training Script

**File**: `experiments/eegpt_linear_probe/train_tuev_fixed.py`

```python
class TUEVModel(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = create_eegpt_model(checkpoint_path)
        self.encoder.eval()

        # TUEV: 15 temporal × 4 summary × 512 dim = 30,720
        self.classifier = nn.Linear(30720, 6)
        self.dropout = nn.Dropout(0.5)  # Paper Table 13

    def forward(self, x):
        with torch.no_grad():
            # Get features for all temporal positions
            features = self.encoder(x, return_all_temporal=True)
            # Shape: (B, 15, 4, 512)

        # Flatten for classifier
        features = features.flatten(1)  # (B, 30720)
        features = self.dropout(features)
        return self.classifier(features)
```

### Step 3: Update TUAB to Use Legacy Mode

**File**: `experiments/eegpt_linear_probe/train_paper_aligned.py`

```python
# Ensure TUAB continues using the current mode
features = self.encoder(x, return_all_temporal=False)  # (B, 4, 512)
```

## Testing Plan

### 1. Unit Test New Mode
```python
def test_temporal_mode():
    model = create_eegpt_model(checkpoint_path)
    x = torch.randn(2, 20, 1000)  # TUEV input

    # Test new mode
    out = model(x, return_all_temporal=True)
    assert out.shape == (2, 15, 4, 512)  # 15 patches for 1000 samples

    # Test legacy mode
    out = model(x, return_all_temporal=False)
    assert out.shape == (2, 4, 512)
```

### 2. Verify Feature Count
```python
def test_feature_dimensions():
    model = TUEVModel(checkpoint_path)
    x = torch.randn(2, 20, 1000)

    with torch.no_grad():
        features = model.encoder(x, return_all_temporal=True)
        flat = features.flatten(1)
        assert flat.shape[1] == 30720  # Matches reference
```

### 3. Smoke Test Training
```python
# Quick test with small batch
model = TUEVModel(checkpoint_path)
optimizer = torch.optim.Adam(model.classifier.parameters())
criterion = nn.CrossEntropyLoss()

for i in range(10):
    x = torch.randn(4, 20, 1000)
    y = torch.randint(0, 6, (4,))

    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    print(f"Step {i}: Loss = {loss.item():.4f}")
```

## Migration Timeline

### Week 1
- [x] Understand reference implementation
- [ ] Implement `return_all_temporal` mode
- [ ] Create unit tests
- [ ] Verify shapes match reference

### Week 2
- [ ] Create `train_tuev_fixed.py`
- [ ] Test on small TUEV subset
- [ ] Compare to random baseline (0.167)
- [ ] Verify improvement over current (0.15)

### Week 3
- [ ] Full TUEV training
- [ ] Target: BAcc > 0.40 (halfway to paper's 0.62)
- [ ] Document results
- [ ] Update CLAUDE.md

## Success Criteria

1. **Shape Verification**: Output is (B, 15, 4, 512) for TUEV
2. **Feature Count**: 30,720 features passed to classifier
3. **Performance**: BAcc > 0.167 (better than random)
4. **Backward Compatibility**: TUAB still works with legacy mode

## Risk Mitigation

- Keep legacy mode as default to not break TUAB
- Use feature flag for new behavior
- Extensive shape testing before training
- Start with small subset to verify improvement

## Next Action

Start implementing Step 1 in `eegpt_architecture.py` with the `return_all_temporal` flag.
