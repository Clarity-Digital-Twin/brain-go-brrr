# EEGPT Training Dossier: Analysis of Implementation Gaps

## Executive Summary

Our current EEGPT linear probe training achieves **~0.79 AUROC** vs the paper's **0.8718 AUROC** on TUAB. This 8-point gap stems from several critical implementation differences identified through careful analysis of the EEGPT paper and reference repository.

## 🚨 Critical Findings

### 1. **Window Size Mismatch**
- **Paper**: Uses **10-second windows** (2000 samples @ 200Hz)
- **Our Implementation**: Uses **8-second windows** (2048 samples @ 256Hz)
- **Impact**: Different temporal context affects feature extraction quality

### 2. **Sampling Rate Discrepancy**
- **Paper**: 200 Hz (TUAB preprocessing)
- **Our Implementation**: 256 Hz
- **Impact**: Model expects specific sampling rate for optimal patch alignment

### 3. **Channel Selection & Ordering**
- **Paper**: Uses specific 23-channel subset with OLD naming (T3, T4, T5, T6)
- **Our Implementation**: 20 channels with modern naming
- **Missing Channels**: A1-REF, A2-REF, T1-REF, T2-REF

### 4. **Preprocessing Pipeline**
```python
# Paper's preprocessing (from make_TUAB.py)
raw.filter(l_freq=0.1, h_freq=75.0)  # We use 0.5-50 Hz
raw.notch_filter(50.0)                # We may not have notch
raw.resample(200, n_jobs=5)           # We use 256 Hz
```

### 5. **Training Configuration**
| Parameter | Paper | Ours | Impact |
|-----------|-------|------|--------|
| Batch Size | 100 | 64 | Less stable gradients |
| Learning Rate | 5e-4 | 5e-4 | ✓ Same |
| Weight Decay | 0.05 | 0.01 | Less regularization |
| Warmup Epochs | 5 | 0 | No gradual start |
| Total Epochs | 50 | 20 | Less training time |
| Layer Decay | 0.65 | None | No differential LR |
| Mixed Precision | Yes | No (fp32) | Slower training |

### 6. **Model Architecture Differences**
- **Paper**: Uses channel-specific convolutional layer (22→19 channels)
- **Ours**: Direct channel mapping without learned adaptation
- **Paper**: 2-layer probe with specific architecture
- **Ours**: Single linear layer

### 7. **Data Augmentation**
- **Paper**: No mention of augmentation for linear probe
- **Ours**: No augmentation (correct for linear probe)

### 8. **Optimizer Settings**
- **Paper**: Uses AdamW with specific betas
- **Ours**: Standard Adam
- **Paper**: OneCycle LR schedule
- **Ours**: Fixed LR (no scheduling)

## 🔬 Deep Dive: Key Implementation Gaps

### A. Channel Architecture Mismatch

The paper uses a learnable channel adaptation layer:
```python
self.chan_conv = Conv1dWithConstraint(22, 19, 1, max_norm=1)
```

This suggests EEGPT was pretrained with 58 channels but TUAB uses a 22→19 mapping. Our direct 20-channel approach misses this adaptation layer.

### B. Linear Probe Architecture

**Paper's Implementation**:
```python
self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
self.linear_probe2 = LinearWithConstraint(16*16, 4, max_norm=0.25)
self.drop = torch.nn.Dropout(p=0.50)
```

**Key Differences**:
1. Two-layer MLP with 16 hidden units
2. 50% dropout between layers
3. Weight constraints (max_norm)
4. Flattening strategy between layers

### C. Window Processing

The paper processes 10-second windows with 50% overlap during training:
```python
for i in range(channeled_data.shape[1] // 2000):
    # Extract 10s windows (2000 samples @ 200Hz)
```

Our 8-second non-overlapping windows provide less temporal context.

### D. Evaluation Protocol

**Paper**: Uses strict train/val/test split following BIOT protocol
**Ours**: May have data leakage if not following exact subject splits

## 📊 Performance Gap Analysis

Based on Table 2 from the paper:
- EEGPT-Tiny (4.7M): 0.8716 AUROC
- EEGPT-Large (25M): 0.8718 AUROC
- Our Implementation: ~0.79 AUROC

The 8-point gap likely comes from:
1. **Window size** (3-4 points)
2. **Channel architecture** (2-3 points)
3. **Probe architecture** (1-2 points)
4. **Training duration** (1 point)

## 🎯 Recommended Experiments

### Experiment 1: Match Window Size & Sampling
```python
# Modify TUABDataset
window_duration=10.0  # Instead of 8.0
sampling_rate=200     # Instead of 256
```

### Experiment 2: Add Channel Adaptation Layer
```python
class ImprovedEEGPTProbe(nn.Module):
    def __init__(self):
        # Add learnable channel mapping
        self.chan_adapt = nn.Conv1d(20, 22, 1)
        self.chan_conv = nn.Conv1d(22, 19, 1)
```

### Experiment 3: Match Probe Architecture
```python
class TwoLayerProbe(nn.Module):
    def __init__(self, input_dim=2048):
        self.probe1 = nn.Linear(input_dim, 16)
        self.dropout = nn.Dropout(0.5)
        self.probe2 = nn.Linear(16*16, 2)  # Binary for TUAB
```

### Experiment 4: Implement Layer Decay
```python
# Use different learning rates for different layers
layer_decay = 0.65
lr_scales = {
    'backbone': lr * (layer_decay ** depth),
    'probe': lr
}
```

### Experiment 5: Extended Training
- Train for 50 epochs instead of 20
- Implement warmup (5 epochs)
- Add learning rate scheduling

## 🚀 Quick Wins (Implement First)

1. **Change window size to 10s** (biggest impact)
2. **Add 2-layer probe with dropout** (easy fix)
3. **Increase batch size to 100** (if GPU memory allows)
4. **Add warmup epochs** (training stability)

## 📈 Expected Improvements

With all fixes implemented:
- Window size fix: +3-4% AUROC
- Probe architecture: +2-3% AUROC  
- Training improvements: +1-2% AUROC
- **Total Expected**: 0.86-0.88 AUROC (matching paper)

## 🔍 Additional Observations

1. **Scaling Law**: Paper shows logarithmic improvement with model size
2. **Summary Tokens**: Paper uses 4 summary tokens vs our 1
3. **Evaluation Frequency**: Paper evaluates every 0.5 epochs
4. **Checkpoint Selection**: Paper saves top-3 by AUROC

## 💡 Implementation Priority

1. **HIGH**: Fix window size and sampling rate
2. **HIGH**: Implement 2-layer probe architecture
3. **MEDIUM**: Add channel adaptation layers
4. **MEDIUM**: Extend training to 50 epochs
5. **LOW**: Implement layer decay
6. **LOW**: Match exact preprocessing pipeline

## 📝 Code Snippets for Key Fixes

### Fix 1: Window Size
```python
# In experiments/eegpt_linear_probe/configs/tuab_config.yaml
data:
  window_duration: 10.0  # Change from 8.0
  sampling_rate: 200     # Change from 256
```

### Fix 2: Probe Architecture
```python
class EEGPTTwoLayerProbe(nn.Module):
    def __init__(self, backbone_dim=768, hidden_dim=16, num_classes=2):
        super().__init__()
        self.probe1 = nn.Linear(backbone_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.probe2 = nn.Linear(hidden_dim * 16, num_classes)
        
    def forward(self, features):
        # features: [B, 16, 768] from EEGPT
        x = self.probe1(features)  # [B, 16, 16]
        x = self.relu(x)
        x = self.dropout(x)
        x = x.flatten(1)  # [B, 256]
        x = self.probe2(x)  # [B, 2]
        return x
```

### Fix 3: Training Configuration
```python
# In train_enhanced.py
cfg.training.epochs = 50
cfg.training.batch_size = 100
cfg.training.warmup_epochs = 5
cfg.training.weight_decay = 0.05
cfg.training.layer_decay = 0.65

# Add warmup to optimizer
scheduler = OneCycleLR(
    optimizer,
    max_lr=5e-4,
    epochs=50,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # 10% warmup
)
```

## 🎓 Lessons Learned

1. **Window size is critical** - EEGPT expects specific temporal context
2. **Channel mapping matters** - Pretrained models need adaptation layers
3. **Probe complexity** - Two layers with dropout outperform single layer
4. **Training duration** - 50 epochs needed for convergence
5. **Exact preprocessing** - Small differences compound

## 📅 Next Steps

1. Implement window size fix immediately
2. Create `train_enhanced.py` with all improvements
3. Run ablation study on each fix
4. Document results in training log
5. Share findings with team

---

**Created**: 2025-08-01
**Author**: Claude + JJ
**Status**: Ready for implementation
**Target**: Match paper's 0.8718 AUROC on TUAB