# TUAB Fix: Missing 99.2% of Features

## The Problem

**Current Implementation**: Uses only 512 features (last 4 summary tokens averaged)
**Reference Implementation**: Uses 63,488 features (31 temporal × 4 summary × 512)

**Evidence**: `reference_repos/EEGPT/downstream_tueg/Modules/models/EEGPT_mcae_finetune_change.py` line 709:
```python
LinearWithConstraint(63488, num_classes)  # HARDCODED!
```

## Why TUAB "Seems" to Work

- Binary classification (abnormal/normal) can work with global features
- 512 features capture overall brain state
- But missing temporal dynamics limits performance
- Current: 0.79 AUROC
- Expected with fix: 0.87 AUROC (paper's result)

## The Fix

### 1. Update Config
```yaml
# experiments/eegpt_linear_probe/configs/tuab_4s_paper_aligned.yaml
probe:
  input_dim: 63488  # Was 512, now 31×4×512
```

### 2. Update Model Forward Pass
```python
# In train_paper_aligned.py or wherever TUAB model is defined
def forward(self, x):
    # Get ALL temporal features
    features = self.encoder(x, return_all_temporal=True)  # (B, 31, 4, 512)
    features = features.flatten(1)  # (B, 63488)
    logits = self.probe(features)
    return logits
```

### 3. Adjust Probe Architecture
```python
# Current probe expects 512 input
self.probe = nn.Sequential(
    nn.Linear(63488, 256),  # Was Linear(512, 128)
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 2)
)
```

## Expected Improvements

| Metric | Current | Expected | Paper Target |
|--------|---------|----------|--------------|
| AUROC | 0.79 | 0.85+ | 0.87 |
| BAcc | ~0.75 | 0.78+ | 0.80 |

## Quick Test

```python
# Verify dimensions
model = create_eegpt_model(checkpoint_path)
x = torch.randn(2, 20, 2000)  # TUAB input
features = model(x, return_all_temporal=True)
assert features.shape == (2, 31, 4, 512)
flat = features.flatten(1)
assert flat.shape == (2, 63488)
print("✓ TUAB dimensions correct!")
```

## Why This Matters

1. **Temporal patterns**: Abnormal EEG often has temporal evolution
2. **Localized events**: Brief abnormalities in specific time windows
3. **Rhythmic patterns**: Periodic discharges need temporal context
4. **Complete information**: Using 100% of features vs 0.8%

## Migration Notes

- Keep old checkpoint for comparison
- May need to reduce learning rate (more parameters)
- Consider larger hidden dim in probe
- Monitor for overfitting with more features